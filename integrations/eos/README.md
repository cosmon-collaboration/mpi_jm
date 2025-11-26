# Introduction  

The file jm_eos.py implements the integration of nuclear equation of state calculations into mpi\_jm.

# Example job file  

```python
# file Fit_10_t0.0_pf0.30_d0.14_3Body_3rd_all_no.yaml  
ctime: 2025-11-25 12:49:55.838250
density: '0.14'
diag: 3Body_3rd:all
epsabs: '0.020'
epsdiag: 0.0001,0.0002,0.0003,0.0004,0.0005
group: -1
ham: Fit:10
maxEval: 300000000
nIncrease: 26214
nStart: 262144
no3n: '16:14'
noopt: 'no'
pf: '0.30'
temp: '0.0'
type: eos
```

- Ctime is the creation time of the job.
- density is the nucleon density in $\textrm{fm}^{-3}$.
- diag is the diagram or diagram group to be integrated.
- epsabs is the absolute error in keV allowed in Monte-Carlo integration.
- epsdiag is the set of energy denominator cutoffs used to verify no dependence on cutoff.
- group is used in some runs to specify diagram groups by particle/antiparticle content.
- ham specifies the chiral interaction
- maxEval specifies the maximum number of integrand evals.  We don't hit this.
- nStart specifies the importance sampling sample size for the first iteration.
- nIncrease specifies how much larte the sample size is for successive interations.
- no3n specifies the quadrature rules sizes for radial and Lebedev spherical integration.
- noopt 'no' says to perform normal ordering (create effective 2-body interaction from 3-body).
- pf gives the proton fraction.
- temp is the temperature in MeV.
- type is the job type, which leads to it being processed by module jm_eos.  


