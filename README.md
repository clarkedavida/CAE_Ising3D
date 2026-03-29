# CAE_Ising3D

In [2603.20157](https://arxiv.org/abs/2603.20157)
we used a convolutional neural network autoencoder (CAE)
to extract $T_c$ and $\nu$ for the 3-$d$ Ising model.
Here we collect our CAE along with the data and scripts needed
to produce the plots in that paper. 

The CAE was created by [A. Abuali](https://github.com/SabryPhys)
and uses [Tensorflow](https://github.com/tensorflow/tensorflow) with 
[Keras](https://github.com/keras-team/keras).

To reproduce the figures, you need the 
[AnalysisToolbox](https://github.com/LatticeQCD/AnalysisToolbox). 
You can install the Toolbox using
```
pip install latqcdtools
```
Then navigate to the `data` subfolder and try using the scripts there.
`README.md` in that subfolder explains the data in more detail.