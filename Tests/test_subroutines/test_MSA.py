import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
import scipy.special
from scipy.integrate import solve_ivp

from pyEFPEHM.utils.utils import *
from pyEFPEHM.waveform.functions import *

################################################################################################################################################

#function to compute basic precession averaged quantities of 2106.10291
def basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True, min_2J1pmcthL=2e-10):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
	
	#compute chi_eff and dchi0
	chi_eff = sz_1 + sz_2
	dchi0 = sz_1 - sz_2

	#compute (y^2 q) and (y^3 q) from Eqs.(29,30). We have substituted the coefficients B, C and D from Eqs.(B1-B3) already
	#where we have substituted J**2 = L**2 + 2*L*(mu1*sz_1 + mu2*sz_2) + 2*nu*sz_1*sz_2 + (mu1**2)*s2_1 + (mu2**2)*s2_2 + nu*DJ2
	#compute stuff that will be used to simplyfy equations
	dmu2 = dmu*dmu
	j2a = (dmu2/y) + (chi_eff*chi_eff)*y #related to modulus of aligned angular momentum
	dChi = dmu*dchi0
	sp2_tot = sp2_1 + sp2_2 + DJ2 #total perpendicular spin |sp_1 + sp_2|^2
	bp = y*sp2_tot
	dp = y*dmu2*(4*sp2_1*sp2_2 - DJ2*DJ2)
	
	#compute part of p (Eq.(29)) that does not vanish in the aligned spin case
	pal = (j2a - 2*dChi + bp)/3
	pal2 = pal*pal
	#compute perpendicular part of p (i.e. that vanishes in the aligned spin case)
	pperp = bp*dChi - dmu*(DJ2*dmu + (sp2_1 - sp2_2)*y*chi_eff)
	pal_pal2pperp = pal*(pal2 + pperp)
	
	#compute y^6 times the cubic discriminant (p/3)**3 - (0.5*q)**2 of Eq.(33). Where py2 = 3*pal2 + 2*pperp and qy3 = -2*pal_pal2pperp + dp. We expand in terms of pal, pperp and dp to avoid numerical errors
	discy6 = (pperp*pperp)*(9*pal2 + 8*pperp)/27 + dp*(pal_pal2pperp - 0.25*dp)
	
	#compute arg(G)/3 from Eq.(33). The discriminant has to be larger than 0 since there are three real roots in cubic equation.
	argG_3 = my_where(discy6>=0, np.arctan2(discy6**0.5, pal_pal2pperp-0.5*dp)/3, 0)
	
	#compute the sine and cosine of arg(G)/3
	sargG_3, cargG_3 = np.sin(argG_3), np.cos(argG_3)
	
	#compute cos(argG/3 - pi/6)
	cargG_3_m_pi_6 = 0.5*((3**0.5)*cargG_3 + sargG_3)
	
	#compute \sqrt(y^2 p) = \sqrt{3*pal2 + 2*pperp} that appears in the Y's of Eqs.(31,32)
	py2 = 3*pal2 + 2*pperp
	sqpy2 = abs(py2)**0.5
	
	#compute dmu*(dchi_av - dchi0) and dmu*dchi_diff, where dchi_diff=(chi_+ - chi_-)/2 and dchi_av=(chi_+ + chi_-)/2, we write them in such a way that avoids numerical error
	dmudchiav_m_dchi0 = my_where((py2>0) & (pal>0), (pal2*sargG_3*sargG_3 - (2/3)*pperp*cargG_3*cargG_3)/(pal + (3**-0.5)*sqpy2*cargG_3), pal - (3**-0.5)*sqpy2*cargG_3)
	dmudchi_diff = sqpy2*sargG_3

	#compute also dchi_av and dchi_diff, since we have to divide by dmu, consider the case in which it is very small separately and use their taylor expansion there
	if dmu>1e-12:
		dchi_av = dchi0 + (dmudchiav_m_dchi0/dmu)
		dchi_diff = dmudchi_diff/dmu
	else:
		s2_tot = sp2_tot + chi_eff*chi_eff
		dchi_av   = my_where(s2_tot!=0, chi_eff*(sp2_1 - sp2_2 + dchi0*chi_eff)/s2_tot, 0)
		dchi_diff = my_where(s2_tot!=0, np.sqrt(sp2_tot*(4*(sp2_2*sz_1*sz_1 + sp2_1*(sz_2*sz_2 + sp2_2))-DJ2*(4*sz_1*sz_2 + DJ2)))/s2_tot, 0)

	#compute m from Eq.(38)
	m = sargG_3/cargG_3_m_pi_6
	
	#compute sqrt(Y3 - Y_-), which is what actually appears in equations
	sqY3mYm = (2*sqpy2*cargG_3_m_pi_6/y)**0.5
	
	#compute the newtonian angular momentum from Eq.(8)
	L = nu/y

	#Define sum of moduli of perpendicular part of spins S0_perp_1^2 + S0_perp_2^2
	Sperp2_1 = (mu1*mu1)*sp2_1
	Sperp2_2 = (mu2*mu2)*sp2_2
	
	#compute component of J0 parallel to L (J0 \cdot \hat{L})
	J0Lh = L + 0.5*(chi_eff + dChi)
	J0Lh2 = J0Lh*J0Lh
	
	#compute squared perpedicular component of J
	Jperp2 = Sperp2_1 + Sperp2_2 + nu*DJ2

	#compute J by adding parallel and perpendicular moduli
	J = (J0Lh2 + Jperp2)**0.5
	
	#compute expansion factor in J = \sqrt{1 + 2*x}
	xJ = 0.5*Jperp2/J0Lh2

	#compute also J \pm J0Lh, considering the cases where x is small
	small_x = (abs(xJ)<1e-6)
	dJ_small_x = J0Lh*xJ*(1 - 0.5*xJ*(1-xJ))
	J_p_J0Lh = my_where(small_x & (J0Lh<0),-dJ_small_x, J + J0Lh)
	J_m_J0Lh = my_where(small_x & (J0Lh>0), dJ_small_x, J - J0Lh)

	#compute Np=N_+ and Nm=N_- from Eqs.(41-42)
	muSz = 2*(mu1*mu1*sz_1 + mu2*mu2*sz_2)
	dmudSp2 = dmu*(Sperp2_1-Sperp2_2)
	Np = J_p_J0Lh*(J_p_J0Lh - muSz)-dmudSp2
	Nm = J_m_J0Lh*(J_m_J0Lh + muSz)-dmudSp2

	#compute (Bp-Cp) = 2J*min(1+cos(\theta_L)) from Eqs.(26,43-46)
	Bp_m_Cp = 2*J_p_J0Lh + dmudchiav_m_dchi0 - dmudchi_diff
	#to avoid singularities force it to be larger than min_2J1pmcthL
	Bp_m_Cp = my_where(Bp_m_Cp>min_2J1pmcthL, Bp_m_Cp, min_2J1pmcthL)
	
	#compute (Bm+Cm) = 2J*min(1-cos(\theta_L)) from Eqs.(26,43-46)
	Bm_p_Cm = 2*J_m_J0Lh - dmudchiav_m_dchi0 - dmudchi_diff
	#to avoid singularities, force it to be larger than min_2J1pmcthL
	Bm_p_Cm = my_where(Bm_p_Cm>min_2J1pmcthL, Bm_p_Cm, min_2J1pmcthL)
	#now compute (Bm-Cm)=(Bm+Cm)-2*Cm
	Bm_m_Cm = Bm_p_Cm + 2*dmudchi_diff

	#compute prefactor's to elliptic PI's appearing in Eq.(109) of 2106.10291
	PI_fact_p = Np/Bp_m_Cp
	PI_fact_m = Nm/Bm_m_Cm

	#compute also the arguments -2*C/(B-C) using that Cp = -Cm = dmu*dchi_diff
	PI_arg_p = -2*dmudchi_diff/Bp_m_Cp
	PI_arg_m =  2*dmudchi_diff/Bm_m_Cm

	#compute Pp and Pm appearing in Eq.(109) of 2106.10291
	Pp = PI_fact_p*my_ellipPI(PI_arg_p, m)
	Pm = PI_fact_m*my_ellipPI(PI_arg_m, m)

	#choose whether to return extra stuff not needed to compute Dv
	if only_for_Dv:
		return m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff
	else:
		return m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m

#function to compute the precesion average factors of Eq.(65) and Eq.(72) of 2106.10291 that appear in beta and sigma
def precesion_average_factors_betasigma(m, mthreas=0.3):
	
	#consider first the case that m is not an array
	if np.asarray(m).ndim == 0:
		if m<mthreas:
			return precesion_average_factors_betasigma_small_m(m)
		else:
			return precesion_average_factors_betasigma_large_m(m)
	#if m is a numpy array, the process is a bit more involved
	else:
		#initialize numpy arrays to put the result in
		m_factor_dchi_prec_avg, m_factor_sigma = np.zeros_like(m), np.zeros_like(m)
		
		#find indexes corresponding to small m's and put the corresponding results
		idxs_small = (m<mthreas)
		if np.any(idxs_small):
			m_factor_dchi_prec_avg[idxs_small], m_factor_sigma[idxs_small] = precesion_average_factors_betasigma_small_m(m[idxs_small])
		
		#do the same for the large m's
		idxs_large = np.logical_not(idxs_small)
		if np.any(idxs_large):
			m_factor_dchi_prec_avg[idxs_large], m_factor_sigma[idxs_large] = precesion_average_factors_betasigma_large_m(m[idxs_large])

		return m_factor_dchi_prec_avg, m_factor_sigma

#make a function for the large m case that uses the exact expressions of Eq.(65) and Eq.(72)
def precesion_average_factors_betasigma_large_m(m):

	#compute E(m)/K(m)
	E_m = scipy.special.ellipe(m)
	K_m = scipy.special.ellipk(m)
	E_K_m = E_m/K_m

	#compute the m factors as defined in Eq.(65) and Eq.(72) respectively
	m_factor_dchi_prec_avg = (E_K_m - 1 + 0.5*m)/m
	m_factor_sigma = ((1/3) + m*((-1/3) + m/8) + E_K_m*((2/3)*(m-2) + E_K_m))/(m*m)

	return m_factor_dchi_prec_avg, m_factor_sigma

#make a function for the small m case that uses Pade approximants of Eq.(65) and Eq.(72).
def precesion_average_factors_betasigma_small_m(m):

	#For the factor in Eq.(65) use a {3,3} Pade around m=0. We expect the absolute/relative error on this factor to be smaller than 1.3e-8/5.9e-7 for m<0.3
	m_factor_dchi_prec_avg = -m*(1 + m*(-1 + m*(71/384)))/(16 + m*(-24 + m*((59/6) - m*(11/12))))
	#For the factor in Eq.(72) use a {2,5} Pade around m=0. We expect the absolute/relative error on this factor to be smaller than 6.1e-10/4.9e-6 for m<0.3
	m2 = m*m
	m_factor_sigma = m2/(1024 + m*(-1024 + m*(96 - m2*(133/32)*(1+m))))

	return m_factor_dchi_prec_avg, m_factor_sigma

#function to compute the precession averages <dchi>, <dchi^2> and <sperp^2>
def compute_dchi_dchi2_sperp2_prec_avg(y, m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, DJ2, sp2_1, sp2_2):
	
	#compute the m factors as defined in Eq.(65) and Eq.(72) respectively
	m_factor_dchi_prec_avg, m_factor_sigma = precesion_average_factors_betasigma(m)
	
	#compute dchi_prec_avg = <dchi> from Eq.(65)
	dchi_prec_avg = dchi_av - 2*dchi_diff*m_factor_dchi_prec_avg
	
	#compute also dmu*(dchi_prec_avg - dchi0)
	dmudchi_prec_avg_m_dchi0 = dmudchiav_m_dchi0 - 2*dmudchi_diff*m_factor_dchi_prec_avg

	#compute dchi2_prec_avg = <dchi^2>, note that in Eqs.(69-70) of 2106.10291 there is a typo in the sign of the m_factor_sigma
	dchi2_prec_avg = dchi_prec_avg*dchi_prec_avg + dchi_diff*dchi_diff*(0.5 - 4*m_factor_sigma)

	#compute <s_\perp^2> = <\sigma_0^(1)> - chi_eff2 using Eq.(71)
	sperp2_prec_avg = sp2_1 + sp2_2 + DJ2 - dmudchi_prec_avg_m_dchi0/y

	return dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg

#function to compute the relevant precession averaged quantities of 2106.10291 needed for differential equations of Eqs.(101-109) of 2106.10291
def prec_avg_quantities_for_Dv(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2):

	#compute basic quantities coming from solving precession equation in 2106.10291
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True)
	
	#compute dchi_prec_avg, betas and sigmas from 2106.10291
	dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg = compute_dchi_dchi2_sperp2_prec_avg(y, m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, DJ2, sp2_1, sp2_2)
	
	#compute also K(m)
	K_m = scipy.special.ellipk(m)
	
	#return the stuff needed for the derivatives
	return J, L, chi_eff, K_m, sqY3mYm, Pp, Pm, dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg

#function to compute the variation of the Euler angles on precession time-scales following arXiv:2106.10291
def precesion_Euler_angles(bpsip, y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)

	#compute basic quantities coming from solving precession equation in 2106.10291
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)

	#compute also the prefactor of the first term in \delta\zeta of Eq.(52), we simplify
	#2*dmu*dchi_diff/(m*sqY3mYm) = y*sqY3mYm in C=2*dmu*dchi_diff/(3*m*(1-y*chi_eff)*sqY3mYm)
	dzeta_E_fact = y*sqY3mYm/(3*(1-y*chi_eff))
	
	#compute E(m) and K(m)
	E_m = scipy.special.ellipe(m)
	K_m = scipy.special.ellipk(m)
	
	#compute hbpsip_pi_2 = (2/pi)\hat{\overline{\psi}}_p (after Eq.49)
	hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
	
	#compute the value of \hat{\psi}_p from \overline{\psi}_p using Eq.(95)
	hpsip = K_m*hbpsip_pi_2

	#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
	sn, cn, dn, am = scipy.special.ellipj(hpsip, m)

	#compute the incomplete elliptic integral of the second kind (Eq.(52))
	E_m_inc = dzeta_E_fact*(scipy.special.ellipeinc(am, m) - hbpsip_pi_2*E_m)

	#compute the incomplete elliptic integral of the third kind (Eq.(52))
	nusqY3mYm = nu*sqY3mYm
	ellipPI_p_inc = my_where(nusqY3mYm!=0, (PI_fact_p*my_ellipPI(PI_arg_p, m, phi=am) - hbpsip_pi_2*Pp)/nusqY3mYm, 0)
	ellipPI_m_inc = my_where(nusqY3mYm!=0, (PI_fact_m*my_ellipPI(PI_arg_m, m, phi=am) - hbpsip_pi_2*Pm)/nusqY3mYm, 0)
	
	#compute\delta\phi_z from Eq.(49)
	dphiz = ellipPI_p_inc + ellipPI_m_inc
	
	#compute \delta\zeta from Eq.(52)
	dzeta = E_m_inc + ellipPI_p_inc - ellipPI_m_inc

	#compute cos(\theta_L) from Eqs.(15, 26)
	dchi = dchi_av - dchi_diff*(1 - 2*sn*sn)
	costhL = (L + 0.5*(chi_eff + dmu*dchi))/J

	#return variation of Euler angles on precession timescales, forcing costhL to be between -1 and 1
	return dphiz, dzeta, np.maximum(np.minimum(costhL,1),-1)

################################################################################################################################################

#number of tests to do
N_test = int(1e4)

################################################################################################################################################

#generate random spins uniform in direction and magnitude
spin1_2 = np.random.uniform(low=[-1,-1,-1], high=[1,1,1], size=(2, N_test, 3))
s_1_2 = np.random.uniform(low=0, high=1, size=(2,N_test))
spin1, spin2 = spin1_2*((s_1_2/np.linalg.norm(spin1_2, axis=-1))[:,:,np.newaxis])
spin1_norm, spin2_norm = s_1_2

#generate random mass ratios
m1s = np.random.uniform(1, 2, size=N_test)
m2s = np.random.uniform(0, m1s, size=N_test)

#generate random PN parameters
ys = np.exp(np.random.uniform(np.log(0.001), np.log(6**-0.5), size=N_test))

#generate random psips
bpsips = np.random.uniform(-10000, 10000, N_test)

#now compute mass related stuff
Ms = m1s + m2s              #total mass
mu1s, mu2s = m1s/Ms, m2s/Ms #reduced individual masses
nus = mu1s*mu2s             #symmetric mass ratio
dmus = mu1s - mu2s          #dimensionless mass diference

#compute the reduced spins as defined in Eq.(7). Note that S = mu^2 spin
s0_1, s0_2 = mu1s[:, np.newaxis]*spin1, mu2s[:, np.newaxis]*spin2

#compute the components of the spin we need
sz_1s, sz_2s = s0_1[:,2], s0_2[:,2]
sp2_1s, sp2_2s = np.sum(s0_1[:,:2]**2, axis=-1), np.sum(s0_2[:,:2]**2, axis=-1)
DJ2s = 2*(s0_1[:,0]*s0_2[:,0] + s0_1[:,1]*s0_2[:,1])

################################################################################################################################################

#loop over tests
clas_quants = []
func_quants = []
prec_avg_factors_clas = []
prec_avg_factors_func = []
prec_avg_quants_clas = []
prec_avg_quants_func = []
Euler_angs_clas = []
Euler_angs_func = []
for i in range(N_test):
	
	#initialize MSA class
	MSA = MultipleScaleAnalysis(m1s[i], m2s[i], sz_1s[i], sz_2s[i], sp2_1s[i], sp2_2s[i], pn_spin_order=4)
	
	#update it with y and DJ2
	MSA.update(ys[i], DJ2s[i])
	
	#save MSA quantities
	clas_quants.append((MSA.m, MSA.dchi_av, MSA.dchi_diff, MSA.chi_eff0, MSA.J, MSA.aJ, MSA.sqY3mYm, MSA.Pp/MSA.ADphiP, MSA.Pm/MSA.ADphiP, 2*MSA.cJmodxddchiav, 2*MSA.cJmodxdchidiff))

	#now compute basic prec quantities
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff = basic_prec_quantities(ys[i], DJ2s[i], m1s[i], m2s[i], sz_1s[i], sz_2s[i], sp2_1s[i], sp2_2s[i], only_for_Dv=True)
	
	#save them
	func_quants.append((m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff))

	#compute precesion_average_factors_betasigma with the two methods available
	prec_avg_factors_clas.append(MSA.precesion_average_factors_betasigma())
	prec_avg_factors_func.append(precesion_average_factors_betasigma(m))

	#compute quatities with the two methods available
	prec_avg_quants_clas.append((MSA.dchi_prec_avg, 2*MSA.cJmodxddchi_prec_avg, MSA.dchi2_prec_avg, MSA.sperp2_prec_avg))
	prec_avg_quants_func.append(compute_dchi_dchi2_sperp2_prec_avg(ys[i], m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, DJ2s[i], sp2_1s[i], sp2_2s[i]))

	#now compute Euler angles
	Euler_angs_clas.append(MSA.precession_Euler_angles(bpsips[i]))
	Euler_angs_func.append(precesion_Euler_angles(bpsips[i], ys[i], DJ2s[i], m1s[i], m2s[i], sz_1s[i], sz_2s[i], sp2_1s[i], sp2_2s[i]))
	
#now compare these
clas_quants = np.array(clas_quants)
func_quants = np.array(func_quants)
prec_avg_factors_clas = np.array(prec_avg_factors_clas)
prec_avg_factors_func = np.array(prec_avg_factors_func)
prec_avg_quants_clas = np.array(prec_avg_quants_clas)
prec_avg_quants_func = np.array(prec_avg_quants_func)
Euler_angs_clas = np.array(Euler_angs_clas)
Euler_angs_func = np.array(Euler_angs_func)

def rel_err(a, b, axis=None):
	return 2*np.linalg.norm(a - b, axis=axis)/np.linalg.norm(a + b, axis=axis)

print("Average difference between basic_prec_quantities computed in class and in function:", rel_err(clas_quants, func_quants, axis=0))
print("Average difference between prec_avg_factors computed in class and in function:", rel_err(prec_avg_factors_clas, prec_avg_factors_func, axis=0))
print("Average difference between dchi_dchi2_sperp2_prec_avg computed in class and in function:", rel_err(prec_avg_quants_clas, prec_avg_quants_func, axis=0))
print("Average difference between Euler Angles computed in class and in function:", rel_err(Euler_angs_clas, Euler_angs_func, axis=0))
print()

################################################################################################################################################

#now compared vectorized implementations

#class
DJ2_0 = DJ2s[0]*np.ones_like(DJ2s)
MSA = MultipleScaleAnalysis(m1s[0], m2s[0], sz_1s[0], sz_2s[0], sp2_1s[0], sp2_2s[0], pn_spin_order=4)
MSA.update(ys, DJ2_0)
clas_quants = (MSA.m, MSA.dchi_av, MSA.dchi_diff, MSA.J, MSA.aJ, MSA.sqY3mYm, MSA.Pp/MSA.ADphiP, MSA.Pm/MSA.ADphiP, 2*MSA.cJmodxddchiav, 2*MSA.cJmodxdchidiff)

#function
m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff = basic_prec_quantities(ys, DJ2_0, m1s[0], m2s[0], sz_1s[0], sz_2s[0], sp2_1s[0], sp2_2s[0], only_for_Dv=True)

func_quants = (m, dchi_av, dchi_diff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff)

#compute precesion_average_factors_betasigma with the two methods available
prec_avg_factors_clas = MSA.precesion_average_factors_betasigma()
prec_avg_factors_func = precesion_average_factors_betasigma(m)

#compute quatities with the two methods available
prec_avg_quants_clas = np.array([MSA.dchi_prec_avg, 2*MSA.cJmodxddchi_prec_avg, MSA.dchi2_prec_avg, MSA.sperp2_prec_avg])
prec_avg_quants_func = compute_dchi_dchi2_sperp2_prec_avg(ys, m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, DJ2_0, sp2_1s[0], sp2_2s[0])

#now compute Euler angles
Euler_angs_clas = MSA.precession_Euler_angles(bpsips)
Euler_angs_func = precesion_Euler_angles(bpsips, ys, DJ2_0, m1s[0], m2s[0], sz_1s[0], sz_2s[0], sp2_1s[0], sp2_2s[0])

clas_quants = np.transpose(clas_quants)
func_quants = np.transpose(func_quants)
prec_avg_factors_clas = np.transpose(prec_avg_factors_clas)
prec_avg_factors_func = np.transpose(prec_avg_factors_func)
prec_avg_quants_clas = np.transpose(prec_avg_quants_clas)
prec_avg_quants_func = np.transpose(prec_avg_quants_func)
Euler_angs_clas = np.transpose(Euler_angs_clas)
Euler_angs_func = np.transpose(Euler_angs_func)

print("Average difference between basic_prec_quantities computed in class and in function:", rel_err(clas_quants, func_quants, axis=0))
print("Average difference between prec_avg_factors computed in class and in function:", rel_err(prec_avg_factors_clas, prec_avg_factors_func, axis=0))
print("Average difference between dchi_dchi2_sperp2_prec_avg computed in class and in function:", rel_err(prec_avg_quants_clas, prec_avg_quants_func, axis=0))
print("Average difference between Euler Angles computed in class and in function:", rel_err(Euler_angs_clas, Euler_angs_func, axis=0))

