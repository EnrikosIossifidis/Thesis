# Code Master Thesis 'Understanding synergistic information using constructed SRVs'

This is the code implementation (code) to generate the data (results) and plots (code/plot_notebooks) in the Master Thesis 'Understanding synergistic information using constructed SRVs'. 

The code includes python scripts to:
* compare the *jointpdf* (https://bitbucket.org/rquax/jointpdf/src/master/) and *syndisc* package (https://github.com/pmediano/syndisc) for multiple random systems of X (*code/run_syndiscjointpdf, code/run_jointpdfp2, code/run_jointpdfp3, code/run_syndisc*). The packages are also included in *code/syndisc* and *code/jointpdf_original*. Since jointpdf is implemented in Python 2, a new Python 3 version is included under *code/jointpdfpython3*;
*  to plot cost landscape of the jointpdf package (*code/plot_landscape*);
*  to construct SRVs for uniform X (*run_sudokus*);
*  to compare constructed SRVs with SRVs from the packages (*code/compare_srvs*);
*  to optimize SRVs with different initial guesses (*code/run_initialsym*).  

## Description

* run_syndiscjointpy.py: run for different systems of X the syndisc and jointpdf package. Additionally, for the jointpdf package, different model parameters can be run such as turning assertion statements on or off (python 3 only) or amount of times that the optimization procedure is repeated. The packages are tested by running *run_jointpdfp2.py* or *run_syndisc.py* and computing the runtime, synergistic information and other properties of the found Synergistic Random Variables (SRVs). 
* run_sudokus.py: different types of SRVs (exhaustive, PSRVs, lower order or noisy SRVs) can be constructed by selecting which type we want to construct. Additionally, for the *permutation* SRVs (PSRVs), code is included to cluster the exhaustively constructed PSRVs by redundancy. 
* run_initialsym.py: use the best constructed SRVs as initial guesses to test whether we can improve the random initial guess in the jointpdf optimization procedure. First, all non-oversized constructed SRVs are loaded, then for random systems of X, the constructed SRV with the lowest cost is computed using the function *helpers/group_helpers.py/addbestsym* and optimized using *jointpdfpython3/measures.py/symsyninfo*
* 
## Getting Started

### Dependencies

* Python 2 and Python 3 must be supported. Additionally, to run *run_syndiscjointpdf.py* with *run_syndisc.py* and *run_jointpdfp2.py*, Anaconda must be installed where the Python 3 Anaconda environment is named 'base' and the Python 2 environment 'python2'. 
* Standard Python packages such as *numpy, pandas, matplotlib, scipy*, etc.
* Additionally packages for the optimization procedure in syndisc must be installed such as *dit, networkx, pypoman, cvxopt*

* 
### Executing programs

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Authors

Contributors names and contact info

Enrikos Iossifidis
