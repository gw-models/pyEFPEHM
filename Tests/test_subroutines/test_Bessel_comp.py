import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
import scipy.special
import time
from pyEFPEHM import analytical_Nlm_p

########################################################################

#compute Ji directly
def compute_Ji_direct(p, pe):
	dp = np.array([1, -1, 2, -2, 3, -3, 4, -4])
	pcomp = np.asarray(p)[None,...] + dp[(slice(None),) + (np.newaxis,)*np.asarray(p).ndim]
	return scipy.special.jv(pcomp,pe)

########################################################################

#number of points to test at each step
Ntest_min = 1
Ntest_max = 1000
N_Ntest = 50

#number of tries
Ntries = 100

#eccentricity range to explore
emin = 0
emax = 0.7

#range of p to explor
pmin = -30
pmax = 100

#confidence level to use for plots
CL = 0.9

#PN parameter range to explore
ymin = 0
ymax = 6**-0.5

#mass ratios to explore
qmin = 0
qmax = 1

mode_array = [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]

########################################################################

#compute the number of tests to perform
Ntests = np.geomspace(Ntest_min, Ntest_max, N_Ntest).astype(int)

#Loop over number of tests
Ji_direct_tottime = np.zeros((N_Ntest, Ntries))
Nlm_p_tottime = np.zeros((N_Ntest, Ntries))
for itest, Ntest in enumerate(Ntests):
	#try as many times as necessary
	for itry in range(Ntries):
	
		#compute parameters to explore
		p = np.random.randint(pmin, pmax, size=Ntest)
		e = np.random.uniform(emin, emax, size=Ntest)
		y = np.random.uniform(ymin, ymax, size=Ntest)
		q = np.random.uniform(qmin, qmax, size=Ntest)
		
		#compute derived parameters
		pe = p*e
		e2 = e*e
		nu = q/((1+q)**2)
		dmu = (1-q)/(1+q)

		#compute bessel functions directly
		Ji_direct_tottime[itest,itry] = time.time()
		Ji_direct = compute_Ji_direct(p, pe)
		Ji_direct_tottime[itest, itry] = time.time() - Ji_direct_tottime[itest,itry]

		#compute amplitudes functions directly
		Nlm_p_tottime[itest,itry] = time.time()
		Nlm_p = analytical_Nlm_p(p, e2, y, nu, dmu, dchi=0, mode_array=mode_array)
		Nlm_p_tottime[itest, itry] = time.time() - Nlm_p_tottime[itest,itry]


#compute the median and confidence level
percentages = 50*(1 + CL*np.array([-1, 0, 1]))
Ji_direct_time_low, Ji_direct_time_median, Ji_direct_time_high = np.percentile(Ji_direct_tottime/Ntests[:,None],percentages, axis=1)
Nlm_p_time_low, Nlm_p_time_median, Nlm_p_time_high = np.percentile(Nlm_p_tottime/Ntests[:,None], percentages, axis=1)

#make a plot
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

plt.figure(figsize=(12,8))
plt.plot(Ntests, Ji_direct_time_median, color='C0', label=r'$J_{p+n}(p e)$')
plt.fill_between(Ntests, Ji_direct_time_low, Ji_direct_time_high, color='C0', alpha=0.25)
plt.plot(Ntests, Nlm_p_time_median, color='C1', label=r'$N^{l m}_p$')
plt.fill_between(Ntests, Nlm_p_time_low, Nlm_p_time_high, color='C1', alpha=0.25)
plt.xlabel(r'Array Size N')
plt.ylabel(r'Runtime/N [s]')
plt.xscale('log')
plt.yscale('log')
plt.xlim(Ntests[0], Ntests[-1])
plt.legend()
plt.tight_layout()

plt.show()
