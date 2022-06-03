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
* compare_srvs.py: To compare the constructed SRVs with optimized SRVs from syndisc for random systems X, first all data and constructed SRVs are loaded. Next for each system X, and therefore each found SRV *Sfound*, the conditional entropy *H(Sfound|SRV)*, for each constructed SRV using the function *helpers/compare_helpers.py/normcondentropy*. Finally, the conditional entropy of *Sfound* given top 5 of constructed SRVs with the lowest *H(Sfound|SRV)* is computed using the function *helpers/compare_helpers.py/bestoffive*
* plot_landscape.py: A 2D or 3D plot is generated of a random plane for the cost landscape of the jointpdf package. The plane in the cost landscape is computed for systems of X consisting of 2 discrete random variables, each with M states. The cost landscape therefore consists of $M^3 - M^2$ dimensions. First, the parameters of an uniform SRV are computed as middle point of the random plane. Next, the two random orthogonal vectors are computed using *helpers/planes.py/random_orthogonal_unit_vectors*, the coordinates of the random plane with *helpers/planes.py/get_plane_points* and finally the cost value for each coordinate of the plane with *helpers/planes.py/get_plane_values*.  

* code/plot_notebooks: Each notebooks generates the figures as showed in thesis for each research question.

## Getting Started

### Dependencies

* Python 2 and Python 3 must be supported. Additionally, to run *run_syndiscjointpdf.py* with *run_syndisc.py* and *run_jointpdfp2.py*, Anaconda must be installed where the Python 3 Anaconda environment is named '*base*' and the Python 2 environment '*python2*'. 
* Standard Python packages such as *numpy, pandas, matplotlib, scipy*, etc.
* Additionally packages for the optimization procedure in syndisc must be installed such as *dit, networkx, pypoman, cvxopt*
* An overview of all used packages and imports are listed in setup.py.
 
### Executing programs
Examples of how to run codes involving *args*
* How to run *run_sudokus.py*:
```
python run_sudokus.py --states=2 --runtype=lowerorders --folder=../results/test/
```
* How to run *run_syndisc.py*:
```
python run_syndisc.py --states=4 --systems=3 --dist_type=iid
```
* How to run *run_syndiscjointpdf.py* (ex. only run experiments for jointpdf Python 3):
```
python run_syndisc.py --states=3 --systems=10 --n_repeats=1,3,5 --code1=run_jointpdfp3.py --code2=''
```
## Authors
Enrikos Iossifidis (linkedin.com/in/enrikos-iossifidis-9286b0113)
