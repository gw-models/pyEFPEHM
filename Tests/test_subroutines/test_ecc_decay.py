import os
import numpy as np
from pyEFPEHM.waveform.functions import *
from tqdm import tqdm

#####################################################################################################################################

#function to solve evolution of log(e2) as a function of log(y)
def solve_ivp_dloge2_dlogy(loge20, logy0, logyf, m1, m2, sz_1, sz_2, q1=1, q2=1, o1=1, o2=1, pn_phase_order=6, pn_spin_order=6, rtol=1e-12, atol=1e-12):
	
	#initialize class to compute PN derivatives
	PN_derivatives = pyEFPE_PN_derivatives(m1, m2, sz_1*sz_1, sz_2*sz_2, q1=q1, q2=q2, o1=o1, o2=o2, pn_phase_order=pn_phase_order, pn_spin_order=pn_spin_order)
	
	#compute constants that will be used throught evolution
	dchi = sz_1 - sz_2
	chi_eff = sz_1 + sz_2
	dchi2 = dchi*dchi
	sperp2_prec_avg = 0
	
	#function to compute d\log(e2)/d\log{y}
	def dloge2_dlogy(logy, loge2):
		#compute e2 and y
		e2 = np.exp(loge2)
		y = np.exp(logy)
		
		#compute Radiation Reaction derivatives
		Dy, De2, Dl, Ddl = PN_derivatives.Dy_De2_Dl_Ddl(y, e2, chi_eff, dchi, dchi2, sperp2_prec_avg)
		
		#return dloge2/dlogy
		return [(y*De2)/(e2*Dy)]

	#solve system of differential equations
	return solve_ivp(dloge2_dlogy, [logy0, logyf], [loge20], dense_output=True, method='RK45', rtol=rtol, atol=atol)

#####################################################################################################################################

#Input parameter ranges to consider
N_tests = 10000

#range of ys to consider
logy0 = np.log(0.2)
logyf = np.log(6**-0.5)
Nys = 1000

#range of parameters to consider
loge20_min = np.log(0.01)
loge20_max = np.log(0.9)
q_min = 0
q_max = 1
s_max = 1

#####################################################################################################################################

#create random parameters to explore
loge20s, qs, sz_1s, sz_2s = np.random.uniform([loge20_min, q_min,-s_max,-s_max],
                                              [loge20_max, q_max, s_max, s_max],
                                              size=(N_tests,4)).T

#create array of log(y) to evaluate solution in
logys = np.linspace(logy0, logyf, Nys)
#create array to strore solution
loge2s = np.zeros((N_tests, Nys))
for ip, [loge20, q, sz_1, sz_2] in enumerate(tqdm(zip(loge20s, qs, sz_1s, sz_2s), total=N_tests)):
	
	#solve corresponding differential equation
	ivp_sol = solve_ivp_dloge2_dlogy(loge20, logy0, logyf, 1, q, sz_1, sz_2)

	#evaluate it in the required values of logy
	loge2s[ip] = ivp_sol.sol(logys)

#compute e2/e02
e2s_e20s = np.exp(loge2s - loge20s[:,None])
#compute ys
ys = np.exp(logys)

#compute the leading 0PN prediction for (e/e0)^2
e2s  = np.exp(loge2s)
e20s = np.exp(loge20s)
e0_e20_0PN_leading = ((ys/ys[0])**(-19/3))[None,:]*((1 + (121/304)*e20s)**(145/121))[:,None]

#compute the 0PN prediction for (e/e0)^2
e0_e20_0PN = ((ys/ys[0])**(-19/3))[None,:]*(((1 + (121/304)*e2s)/(1 + (121/304)*e20s)[:,None])**(-145/121))

#compute ratio between exact and prediction
e2_e20_comp_0PN_leading = e2s_e20s/e0_e20_0PN_leading
e2_e20_comp_0PN = e2s_e20s/e0_e20_0PN

#compute the minimum and maximum of ratio between exact and prediction
e2_e20_comp_0PN_leading_min = np.amin(e2_e20_comp_0PN_leading, axis=0)
e2_e20_comp_0PN_leading_max = np.amax(e2_e20_comp_0PN_leading, axis=0)
e2_e20_comp_0PN_leading_median = np.median(e2_e20_comp_0PN_leading, axis=0)

e2_e20_comp_0PN_min = np.amin(e2_e20_comp_0PN, axis=0)
e2_e20_comp_0PN_max = np.amax(e2_e20_comp_0PN, axis=0)
e2_e20_comp_0PN_median = np.median(e2_e20_comp_0PN, axis=0)

#compute the minimum and maximum e2s_e20s
e2_e20_min = np.amin(e2s_e20s, axis=0)
e2_e20_max = np.amax(e2s_e20s, axis=0)
e2_e20_median = np.median(e2s_e20s, axis=0)


#make a plot
from matplotlib import pyplot as plt
import matplotlib.collections as mc
plt.rcParams.update({
	'lines.linewidth': 2.5,
        'axes.grid': False,
        'axes.labelsize': 28,
        'axes.linewidth': 1.75,
        'axes.titlesize': 32,
        'font.size': 24,
        'legend.fontsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family': 'serif',
        'font.sans-serif': ['Bitstream Vera Sans'],
        'font.serif': ['Times New Roman'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}',
        'text.usetex':True,
        'patch.force_edgecolor':True,
})

fig, ax = plt.subplots(figsize=(9, 7))
ax.fill_between(ys, e2_e20_comp_0PN_min, e2_e20_comp_0PN_max, alpha=0.3, label=r'$\frac{e^2}{e_0^2 \left(\frac{1 + \frac{121}{304} e^2}{1 + \frac{121}{304} e_0^2} \right)^{-145/121} \left(\frac{y}{y_0}\right)^{-19/3}}$')
ax.fill_between(ys, e2_e20_comp_0PN_leading_min, e2_e20_comp_0PN_leading_max, alpha=0.3, label=r'$\frac{e^2}{e_0^2 \left(1 + \frac{121}{304} e_0^2\right)^{145/121} \left(\frac{y}{y_0}\right)^{-19/3}}$ ')
ax.set_xlim(ys.min(), ys.max())
ax.set_xlabel(r'$y$')
ax.axhline(y=1, linewidth=1.75, color='k')
ax.legend()
fig.tight_layout()
os.makedirs('Plots', exist_ok=True)
plt.savefig('Plots/e2_decay_3PN_vs_0PN.pdf')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ys, e2_e20_median, label='Median')
ax.fill_between(ys, e2_e20_min, e2_e20_max, alpha=0.3, label='Min-Max')
ax.plot(ys, (ys/ys[0])**(-19/3), label=r'$(y/y_0)^{-19/3}$', color='k', linestyle='--')
ax.set_xlim(ys.min(), ys.max())
ax.set_ylabel(r'$e^2/e_0^2$')
ax.set_xlabel(r'$y$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
fig.tight_layout()


plt.show()

