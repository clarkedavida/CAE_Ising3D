# Data files 

Here we collect data files needed to reproduce our figures.
The figures can be plotted using Python (see the README.md
in the top level to see what packages are needed) by calling
```
python figure*.py
```
The labels `a` and `b` refer to (left) and (right) subplots,
respectively.

The file `Perr_ALGthorough_TMIN0_TMAX6.txt` contains pseudocritical
temperatures and the maximum value of the susceptibility for
each lattice size. It is needed as input for `figure3.py`.

## Analysis_Scale_* 
Each folder has results for our independent CAE runs.
Each subfolder gives the lattice extension.
In each lattice extension subfolder, there is a `reconstructionErrors`
subfolder that contains directly the MSE for each configuration.
These files are formatted as
```
configuration    MSE
```
The `Analysis_Scale_1` folder contains additionally some thermodynamic
information, which was used to look at the correlation between
average magnetization and the MSE. For each lattice size subfolder,
the thermodynamic information is collected in the `Observables_per_config`
subfolder. Each file in that subfolder is formatted as
```
configuration    |m|    e
```
where `e` is the energy per site.