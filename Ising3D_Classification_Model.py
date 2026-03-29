import os
import gc
import re
import numpy as np
import matplotlib.pyplot as plt

# Disable XLA / JIT early (pre-TF import)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, LeakyReLU, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# -------------------- seeds / GPU --------------------
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e: print(f"GPU error: {e}")
tf.config.optimizer.set_jit(False)  # extra guard

# -------------------- config --------------------
num_of_total_files = "3000_4000"
num_of_tr = 2000
num_of_ind = "3000_4000"
T_b = "Zero"; T_A = "Infinity"
L = 130
input_shape = (L, L, L, 1)
learning_rate = 1e-4
batch_size = 2        # safer for very large L
epochs = 50
data_folder = ".."
notebook_dir = os.getcwd()

# -------------------- data pipeline --------------------
def load_and_preprocess(file_path):
    def _load_numpy(path):
        arr = np.load(path.decode("utf-8")).astype(np.float32)
        return arr.reshape((L, L, L, 1))
    tensor = tf.numpy_function(_load_numpy, [file_path], tf.float32)
    tensor.set_shape((L, L, L, 1))
    return tensor

def create_dataset(file_paths, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(list(file_paths))
    if shuffle: ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # targets = inputs for AE
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def split_data(folder, split_ratios=(0.7, 0.15, 0.15)):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    np.random.shuffle(files)
    total = len(files)
    train_end = int(split_ratios[0] * total)
    val_end = train_end + int(split_ratios[1] * total)
    return files[:train_end], files[train_end:val_end], files[val_end:]

ferro_folder = os.path.join(data_folder, "Ferromagnetic")
ferro_train, ferro_val, ferro_test = split_data(ferro_folder)
train_files = [os.path.join(ferro_folder, f) for f in ferro_train]
val_files   = [os.path.join(ferro_folder, f) for f in ferro_val]
test_files  = [os.path.join(ferro_folder, f) for f in ferro_test]

train_gen = create_dataset(train_files, batch_size, shuffle=True)
val_gen   = create_dataset(val_files,   batch_size, shuffle=False)
test_gen  = create_dataset(test_files,  batch_size, shuffle=False)

# -------- (B) denoising corruption (UNSUPERVISED) ----------
def random_spin_flips(x, p=0.01):
    # flip a small independent fraction of voxels
    mask = tf.cast(tf.random.uniform(tf.shape(x)) < p, x.dtype)
    return x * (1.0 - 2.0 * mask)

def add_corruption(x, y):
    x_noisy = random_spin_flips(x, p=0.01)   # tiny; physics-agnostic
    return x_noisy, y

# Apply corruption only to training inputs
train_gen = train_gen.map(add_corruption, num_parallel_calls=tf.data.AUTOTUNE)\
                     .prefetch(tf.data.AUTOTUNE)

# -------------------- model --------------------
def build_3d_cae(input_shape):
    inputs = Input(shape=input_shape)

    # anti-alias prefilter before each downsample (stride-1 convs)
    x = Conv3D(16, 3, padding='same', strides=1)(inputs); x = LeakyReLU()(x)
    x = Conv3D(16, 3, padding='same', strides=1)(x);      x = LeakyReLU()(x)
    x = Conv3D(32, 3, padding='same', strides=2)(x);      x = LeakyReLU()(x)

    x = Conv3D(32, 3, padding='same', strides=1)(x);      x = LeakyReLU()(x)
    x = Conv3D(64, 3, padding='same', strides=2)(x);      x = LeakyReLU()(x)

    x = Dropout(0.2)(x)
    x = Conv3D(64, 3, padding='same', strides=1)(x);      x = LeakyReLU()(x)

    x = Conv3DTranspose(64, 3, padding='same', strides=1)(x); x = LeakyReLU()(x)
    x = Conv3DTranspose(32, 3, padding='same', strides=2)(x); x = LeakyReLU()(x)
    x = Conv3DTranspose(16, 3, padding='same', strides=2)(x); x = LeakyReLU()(x)

    x = Conv3D(1, 1, activation='linear', padding='same')(x)

    # crop in case transpose-convs overshoot 
    x = Lambda(lambda t: t[:, :input_shape[0], :input_shape[1], :input_shape[2], :],
               name="crop_to_L")(x)
    return Model(inputs, x, name="CAE_3D")

model = build_3d_cae(input_shape)

# pooled loss via 3D box-blur conv
def box_blur3d(x):
    k = tf.ones((3,3,3,1,1), dtype=x.dtype) / 27.0  # simple 3×3×3 mean filter
    return tf.nn.conv3d(x, k, strides=[1,1,1,1,1], padding='SAME')

@tf.function(jit_compile=False)
def pooled_mse(y_true, y_pred):
    ytb = box_blur3d(y_true)
    ypb = box_blur3d(y_pred)
    mse_blur = tf.reduce_mean(tf.square(ytb - ypb), axis=[1,2,3,4])
    mse_pix  = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2,3,4])
    return 0.7 * mse_blur + 0.3 * mse_pix

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss=pooled_mse, metrics=["mae"],
              jit_compile=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr      = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

# -------------------- train --------------------
history = model.fit(
    train_gen, validation_data=val_gen, epochs=epochs,
    callbacks=[early_stopping, reduce_lr], verbose=1
)

# -------------------- plots (training curves) --------------------
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('3D Ising one-class AE - Reconstruction Loss', fontsize=16)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout()
plt.savefig(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_'
            f'Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Model_Loss.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'],     label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('3D Ising one-class AE - Reconstruction MAE', fontsize=16)
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
plt.tight_layout()
plt.savefig(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_'
            f'Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Model_MAE.png')
plt.close()

# -------------------- held-out ferro test error --------------------
y_pred = []
for x_batch, _ in test_gen:
    recon = model(x_batch, training=False)
    mse_per_sample = tf.reduce_mean(tf.math.squared_difference(x_batch, recon), axis=[1,2,3,4])
    y_pred.extend(mse_per_sample.numpy().tolist())

test_loss = np.mean(y_pred) if len(y_pred) else float('nan')
test_std  = np.std(y_pred)  if len(y_pred) else float('nan')
eval_loss, eval_mae = model.evaluate(test_gen, verbose=0)

plt.figure(figsize=(10,6))
plt.hist(y_pred, bins=40, alpha=0.85)
plt.title('AE Reconstruction Error on Held-Out T=0 Ferro', fontsize=16)
plt.xlabel('Per-sample error'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_'
            f'Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AE_TestError_Hist.png')
plt.close()

with open(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_'
          f'Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AE_Report.txt', 'w') as f:
    f.write("One-Class (Ferromagnetic T=0) 3D Conv Autoencoder Report\n")
    f.write("="*65 + "\n")
    f.write(f"Lattice size: {L} x {L} x {L}\n")
    f.write(f"Train/Val/Test sizes: {len(train_files)}, {len(val_files)}, {len(test_files)}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Batch size:    {batch_size}\n")
    f.write(f"Epochs (max):  {epochs}\n")
    f.write("-"*65 + "\n")
    f.write(f"Final Train Loss:      {history.history['loss'][-1]:.6f}\n")
    f.write(f"Best Val Loss:         {np.min(history.history['val_loss']):.6f}\n")
    f.write(f"Test Mean Error:       {test_loss:.6f}\n")
    f.write(f"Test Std(Error):       {test_std:.6f}\n")
    f.write(f"Eval Loss (test_gen):  {eval_loss:.6f}\n")
    f.write(f"Eval MAE  (test_gen):  {eval_mae:.6f}\n")

model.save(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_'
           f'Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AE_Model.keras')

del train_gen, val_gen, test_gen, history
gc.collect()
# ==================== END FIRST PART ====================











# =========================
# Independent testing (parallel over temperature folders)
# =========================
def calculate_first_derivative(temps, avg_predictions, avg_errors):
    derivatives_of_AvgP = []
    errors_in_derivatives = []
    for i in range(len(temps)):
        if i == 0:
            dT = (temps[i+1] - temps[i])
            derivative = (avg_predictions[i+1] - avg_predictions[i]) / dT
            error = (avg_errors[i+1] / dT)**2 + (avg_errors[i] / dT)**2
        elif i == len(temps) - 1:
            dT = (temps[i] - temps[i-1])
            derivative = (avg_predictions[i] - avg_predictions[i-1]) / dT
            error = (avg_errors[i] / dT)**2 + (avg_errors[i-1] / dT)**2
        else:
            dT = (temps[i+1] - temps[i-1])
            derivative = (avg_predictions[i+1] - avg_predictions[i-1]) / dT
            error = (avg_errors[i+1] / dT)**2 + (avg_errors[i-1] / dT)**2
        errors_in_derivatives.append(np.sqrt(max(0.0, error)))
        derivatives_of_AvgP.append(derivative)
    return np.array(derivatives_of_AvgP), np.array(errors_in_derivatives)

def calculate_binder_cumulant(P):
    P = np.array(P, dtype=np.float64)
    avg_P2 = np.average(P**2)
    avg_P4 = np.average(P**4)
    if avg_P2 == 0: return np.nan
    return 1.0 - (avg_P4 / (3.0 * avg_P2**2))

directory = f"/project/ratti/Ahmed/Ising3D_Sim/{L}/Ising3D_Metro_CUDA_Sz{L}_Saved_Configurations"

def _to_float(s):
    try: return float(s)
    except Exception: return np.inf

# numeric sort helper
_num = lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else float('inf')

subdirectories = sorted(
    [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))],
    key=_to_float
)

os.makedirs("All_Predictions", exist_ok=True)
os.makedirs("All_Scores", exist_ok=True)

_model_lock = threading.Lock()
BATCH_PRED = 16  # adjust for memory
MAX_WORKERS = min(4, len(subdirectories))

# Use the same model already built/loaded in memory
def _batch_err(model, X_batch):
    recon = model(X_batch, training=False).numpy()
    # NOTE: error metric should match training objective scale — keep MSE over voxels
    return np.mean((X_batch - recon)**2, axis=(1,2,3,4))

def process_temperature_folder(temp_str, batch_pred=BATCH_PRED):
    """
    Returns: (T, [(filename, err), ...], n_files)
    Also writes per-file prediction errors to All_Predictions/<T>.txt
    """
    T = float(temp_str)
    subdir_path = os.path.join(directory, temp_str)
    file_names = sorted([f for f in os.listdir(subdir_path) if f.endswith(".npy")], key=_num)
    n_files = len(file_names)
    if n_files == 0: return (T, [], 0)

    pairs = []  # (filename, err)
    out_path = os.path.join("All_Predictions", f"{T:.6f}.txt")
    with open(out_path, "w") as fpred:
        for start in range(0, n_files, batch_pred):
            batch_names = file_names[start:start+batch_pred]
            X_batch = []
            for fname in batch_names:
                arr = np.load(os.path.join(subdir_path, fname)).astype(np.float32)
                X_batch.append(arr[..., None])
            X_batch = np.stack(X_batch, axis=0)

            with _model_lock:
                batch_err = _batch_err(model, X_batch)

            for fname, m in zip(batch_names, batch_err):
                fpred.write(f"{fname}\t{m:.10f}\n")   # tab delimiter
                pairs.append((fname, float(m)))

            del X_batch; gc.collect()

    return (T, pairs, n_files)

# Run over all temperatures in parallel
raw_results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(process_temperature_folder, d, BATCH_PRED)
            for d in subdirectories if _to_float(d) != np.inf]
    for fut in as_completed(futs):
        raw_results.append(fut.result())

raw_results.sort(key=lambda x: x[0])  # (T, [(fname, err), ...], n)

# -------- ECDF score mapping (UNSUPERVISED) ----------
all_err = np.array([m for (_T, pairs, _n) in raw_results for (_fname, m) in pairs], dtype=np.float64)
if all_err.size == 0:
    raise RuntimeError("No configurations found across temperatures to compute AE scores.")
all_err_sorted = np.sort(all_err)

def ecdf_tail_prob(m):
    idx = np.searchsorted(all_err_sorted, m, side='right')
    q   = idx / (len(all_err_sorted) + 1.0)  # empirical CDF
    return 1.0 - q                           # higher error ⇒ lower score

# Aggregate per temperature, write scores WITH filenames
results = []
for (T, pairs, n) in raw_results:
    if n == 0:
        results.append((T, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        continue

    fname_P = [(fname, ecdf_tail_prob(m)) for (fname, m) in pairs]

    score_path = os.path.join("All_Scores", f"{T:.6f}.txt")
    with open(score_path, "w") as fscore:
        for fname, p in fname_P:
            fscore.write(f"{fname}\t{p:.10f}\n")   # tab delimiter

    P = np.array([p for (_fname, p) in fname_P], dtype=np.float64)

    avg_P = float(np.mean(P))
    var_P = float(max(0.0, np.mean(P**2) - avg_P**2))
    err_P = float(np.sqrt(var_P) / np.sqrt(len(P)))

    B = P * (1.0 - P)
    avg_B = float(np.mean(B))
    var_B = float(np.var(B, ddof=0))
    err_B = float(np.sqrt(var_B) / np.sqrt(len(B)))

    beta = 1.0 / T if T != 0 else np.inf
    V = L**3
    pseudo_sus = float(beta * V * var_P) if np.isfinite(beta) else np.nan

    bc = calculate_binder_cumulant(P)

    results.append((T, avg_P, err_P, var_P, avg_B, err_B, pseudo_sus, bc))

# ---------- DataFrames / outputs ----------
df_Avg_Predictions = pd.DataFrame(
    [(T, AvgP, ErrP) for (T, AvgP, ErrP, *_rest) in results],
    columns=['Temperature', 'Average Prediction per Configuration', 'Error in Prediction']
)
df_Var_Predictions = pd.DataFrame(
    [(T, VarP) for (T, _AvgP, _ErrP, VarP, *_rest) in results],
    columns=['Temperature', 'Variance in Prediction']
)
df_BernoulliVar = pd.DataFrame(
    [(T, AvgBern, ErrBern) for (T, _AvgP, _ErrP, _VarP, AvgBern, ErrBern, *_rest) in results],
    columns=['Temperature', 'Average Bernoulli Variance per Configuration', 'Error in Bernoulli Variance']
)
df_PseudoSus = pd.DataFrame(
    [(T, PseudoSus) for (T, _AvgP, _ErrP, _VarP, _AvgBern, _ErrBern, PseudoSus, *_rest) in results],
    columns=['Temperature', 'PseudoSus']
)
df_BC = pd.DataFrame(
    [(T, BC) for (T, _AvgP, _ErrP, _VarP, _AvgBern, _ErrBern, _PseudoSus, BC) in results],
    columns=['Temperature', 'Binder_Cumulant']
)

# Save BC
df_BC.to_csv(
    f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_BC_vs_T.txt',
    sep='\t', index=False, header=False
)

# d<AvgP>/dT
temps = df_Avg_Predictions['Temperature'].values
avg_predictions = df_Avg_Predictions['Average Prediction per Configuration'].values
avg_errors      = df_Avg_Predictions['Error in Prediction'].values
derivatives_of_AvgP, errors_in_derivatives = calculate_first_derivative(temps, avg_predictions, avg_errors)
pd.DataFrame({'Temperature': temps, 'd<AvgP>/dT': derivatives_of_AvgP, 'Error in d<AvgP>/dT': errors_in_derivatives}).to_csv(
    f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_dAvgP_vs_T.txt',
    sep=' ', index=False, header=False
)

# <P> & <M> vs T (visual comparison only)
Metro_file_path = os.path.join(notebook_dir, f"Ising3D_Metro_CUDA_Sz{L}_Sim_Results_All_Temps.txt")
df_Metro = pd.read_csv(Metro_file_path, sep="\t", header=None,
                       names=['Temperature', 'Average Magnetization per Spin', 'Error in Magnetization'],
                       usecols=[0, 1, 2]).sort_values('Temperature')
df_merged_with_errors = pd.merge(df_Avg_Predictions, df_Metro, on='Temperature')

plt.figure(figsize=(10, 6))
plt.errorbar(df_merged_with_errors['Temperature'],
             df_merged_with_errors['Average Prediction per Configuration'],
             yerr=df_merged_with_errors['Error in Prediction'],
             fmt='o', linestyle='-', label='<P> (Average Prediction)')
plt.errorbar(df_merged_with_errors['Temperature'],
             df_merged_with_errors['Average Magnetization per Spin'],
             yerr=df_merged_with_errors['Error in Magnetization'],
             fmt='o', linestyle='-', label='<M> (Average Magnetization per Spin)')
plt.title('3D Ising one-class AE - <P> and <M> vs. Temperature')
plt.xlabel('T'); plt.ylabel('<P> and <M>'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AvgP_and_AvgM_vs_T.png')
plt.close()

# sigma^2 vs chi (visual only)
df_Metro_full = pd.read_csv(Metro_file_path, sep="\t", header=None,
                            names=['Temperature', 'Average Magnetization per Spin', 'Error in Magnetization', 'Magnetic Susceptibility'],
                            usecols=[0, 1, 2, 3]).sort_values('Temperature')
df_merged = pd.merge(df_Var_Predictions, df_Metro_full, on='Temperature')
max_var = df_merged['Variance in Prediction'].max()
max_chi = df_merged['Magnetic Susceptibility'].max()
scaling_factor = (max_var / max_chi) if max_chi > 0 else 1.0

plt.figure(figsize=(10, 6))
plt.plot(df_merged['Temperature'], df_merged['Variance in Prediction'],
         marker='o', linestyle='-', label=r'$\sigma^2$ (Variance in Prediction)')
plt.plot(df_merged['Temperature'], scaling_factor * df_merged['Magnetic Susceptibility'],
         marker='o', linestyle='-', label=r'$\chi$  (Magnetic Susceptibility)')
plt.figtext(0.3, 0.01, f"Magnetic Susceptibility is scaled with a factor {scaling_factor:.2f} for better visualization",
            fontsize=9, ha='left')
plt.title(r'3D Ising one-class AE - $\sigma^2$ and $\chi$ vs. Temperature')
plt.xlabel('T'); plt.ylabel(r'$\sigma^2$ , $\chi$'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Chi_vs_T.png')
plt.close()

# save tables
df_Avg_Predictions.to_csv(
    os.path.join(notebook_dir, f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AvgP_vs_T.txt'),
    sep=' ', index=False, header=False
)
df_Var_Predictions.to_csv(
    os.path.join(notebook_dir, f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_sigma_square_vs_T.txt'),
    sep=' ', index=False, header=False
)
df_BernoulliVar.to_csv(
    os.path.join(notebook_dir, f'Ising3D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_'
                               f'TempRangeB{T_b}A{T_A}_BernoulliVar_vs_T.txt'),
    sep=' ', index=False, header=False
)

# final cleanup
del model
gc.collect()
