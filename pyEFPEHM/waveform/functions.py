import numpy as np
import math
import scipy.special
from scipy.integrate import solve_ivp
import warnings

from pyEFPEHM.utils.utils import *
#import my polynomial class with cython
from pyEFPEHM.utils.cython_utils import my_cpoly

#################################################################

#When not specified, equations come from 2106.10291
#For units we will always assume that [t] = [1/f] = [M]

#################################################################

#function to compute initial conditions in v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
def initial_conditions_for_RR_eqs(f0_orb, fref_orb, ff_orb, e_ref, m1, m2, spin1_ref, spin2_ref, inclination, phi_ref, phi_e_ref, DJ2_tol=1e-10, DJ2_max_iter=10, pn_spin_order=6):

	r'''
	Function to compute initial conditions and initialize our waveform. We follow the conventions of LAL.
	https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group__lalsimulation__inspiral.html
	Everything is done in the LAL source frame S at the reference frequency: the z axis is along the Newtonian orbital
	angular momentum L, and the x axis is along the unit vector \hat{n} pointing from body 2 to body 1 (for eccentric
	orbits we take \hat{n} at the *mean* orbital phase, i.e. \lambda(t_ref) is measured from it).
	The input spins are the components in this frame, so aligned spins have spin1x=spin1y=spin2x=spin2y=0.
	The inclination is the angle between L and the direction of observation N, whose components in S are
	N = (sin(inclination)*cos(pi/2 - phi_ref), sin(inclination)*sin(pi/2 - phi_ref), cos(inclination)),
	i.e. phi_ref is the LAL reference phase (the angle on the orbital plane from the ascending node to body 1).
	This gives the LAL polarizations h_+ - i h_x = sum_{lm} h_{lm} {}_{-2}Y_{lm}(inclination, pi/2 - phi_ref).
	Internally the waveform is computed in the J frame (z along the total angular momentum J, L on the x-z plane),
	which is where the returned (cos_theta_JN, sin_theta_JN, phi_JN) are expressed.
	--------------------------------------------------
	
	f0_orb: float
	        initial orbital frequency
	fref_orb: float
	        reference orbital frequency, at which e_ref, phi_ref, phi_e_ref, the spins and the inclination are defined.
	        If it is at or beyond the end of the evolution (final frequency or ISCO), it is clamped to the final y with a warning.
	ff_orb: float
	        final orbital frequency. If it is larger than ISCO frequency, it will be forced to be the ISCO frequency.
	e_ref: float
	    reference (time-)eccentricity e_t from Quasi-Keplerian parametrization
	m1: float
	    primary mass, assumed units are [1/f]
	m2: float
	    secondary mass, assumed units are [1/f]. If m2>m1, all properties of the components will be flip
	spin1_ref: numpy array, shape (3,)
	    reference dimensionless spin vector spin1=S1/mu1^2=s1/mu1 of the primary object, cartesian components in the LAL source frame defined above
	spin2_ref: numpy array, shape (3,)
	    dimensionless spin vector spin2=S2/mu2^2=s2/mu2 of the secondary object, cartesian components in the LAL source frame defined above
	phi_ref: float
	    LAL reference phase phiRef, i.e. the (mean) orbital phase measured from the ascending node
	phi_e_ref: float
	    reference mean anomaly \ell_0 of quasi keplerian parametrization
	inclination : float
		Angle between orbital angular momentum and vector from binary to observer (N), in [0, pi].
		Other values are folded with (inclination, phi_ref) -> (-inclination, phi_ref - pi), which leaves N unchanged.

	'''
	
	#compute mass related things
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)

	#compute reference squared eccentricity
	e2_ref = e_ref*e_ref

	#compute reference y using Eq.(5)
	y_ref = y_of_forb(fref_orb, e2_ref, M)

	#if final frequency is None, integrate to the ISCO
	yf_ISCO = 6**-0.5
	if ff_orb is None:
		yf = yf_ISCO
	#Otherwise, get final y from final orbital frequency.
	else:

		#Use the 0PN estimate for the final eccentricity. This is an upper bound on the final eccentricity (see 2604.11903)
		e2f = float(compute_e2_of_f_0PN(ff_orb, e2_ref, fref_orb))

		#The 0PN e2f leads to a slightly over-estimated yf (which is good to have some more padding)
		yf = y_of_forb(ff_orb, e2f, M)

		#check that we are not surpassing the ISCO
		if yf > yf_ISCO:
			warnings.warn("With input final orbital frequency, yf=%.3f and the ISCO is surpassed, setting yf=yISCO=%.3f"%(yf, yf_ISCO), UserWarning, stacklevel=2)
			yf = yf_ISCO

	#if the reference point is at or beyond the end of the evolution, it will be clamped to the final y
	#the effective reference frequency is then the orbital frequency at y=yf with the reference eccentricity (Eq.(5))
	ref_clamped = (y_ref >= yf)
	if ref_clamped: fref_orb_eff = forb_of_y(yf, e2_ref, M)
	else:           fref_orb_eff = fref_orb

	#estimate the squared eccentricity at the initial frequency with the 0PN evolution from the effective reference
	#when the reference is at the initial frequency (default), the eccentricity is the reference one and no solve is needed
	if fref_orb_eff == f0_orb: e2_0_est = e2_ref
	else:                      e2_0_est = float(compute_e2_of_f_0PN(f0_orb, e2_ref, fref_orb_eff))

	#estimate y at the initial frequency using Eq.(5) and raise if it is not below the final y, before warning about any clamping
	y0_est = y_of_forb(f0_orb, e2_0_est, M)
	if y0_est >= yf:
		raise ValueError("Initial orbital frequency f0_orb=%.3e is at or beyond the end of the evolution (estimated y0=%.3f >= yf=%.3f)"%(f0_orb, y0_est, yf))

	#now that the evolution is known to be viable, clamp the reference point with a warning
	if ref_clamped:
		warnings.warn("Reference orbital frequency fref_orb=%.3e is at or beyond the end of the evolution, clamping the reference point to y=yf=%.3f (orbital frequency %.3e)"%(fref_orb, yf, fref_orb_eff), UserWarning, stacklevel=2)
		y_ref = yf

	#compute the reduced spins as defined in Eq.(7). Note that S = mu^2 spin
	s0_1, s0_2 = mu1*spin1_ref, mu2*spin2_ref
	
	#compute spin components parallel to L0 
	sz_1 = s0_1[2]
	sz_2 = s0_2[2]
	
	#compute norms of perpendicular part of spin vectors
	sp2_1 = np.sum(s0_1[:2]**2)
	sp2_2 = np.sum(s0_2[:2]**2)

	#compute also the full norms of the spin vectors
	s2_1 = sp2_1 + sz_1*sz_1
	s2_2 = sp2_2 + sz_2*sz_2

	#compute initial \delta\chi from Eq.(11)
	dchi0 = sz_1 - sz_2

	############################## Initialize MSA ############################

	#compute initial guess of DJ20
	DJ20 = 2*(s0_1[0]*s0_2[0] + s0_1[1]*s0_2[1])

	#compute V as introduced in Eq.(138), the z component of the cross product of the perpendicular spins (L is the z axis of the source frame)
	Vol = s0_1[0]*s0_2[1] - s0_1[1]*s0_2[0]
	
	#initialize MSA class
	MSA = MultipleScaleAnalysis(m1, m2, sz_1, sz_2, sp2_1, sp2_2, pn_spin_order=pn_spin_order)
	
	#update this initial guess od \D J taking into account \delta J of Eq.(91)
	DJ20_OG = DJ20
	for iJ in range(DJ2_max_iter):
	
		#update MSA class with current values of y and DJ2
		MSA.update(y_ref, DJ20)

		#compute the cos(2*am0) using Eq.(26), i.e. dchi = dchi_av - dchi_diff*cos(2*am0)
		if MSA.dchi_diff != 0: cos2am0 = max(-1, min(1, MSA.ddchi_av/MSA.dchi_diff))
		else:                  cos2am0 = 1
		
		#compute initial am0 = am(phip0, m0) inverting Eq.(26)
		am0 = 0.5*np.arccos(cos2am0)

		#now obtain phip0 by using that psip0 = K(am0, m0)
		psip0 = scipy.special.ellipkinc(am0, MSA.m)
		
		#compute the correct sign of \psi_p from Eq.(138) and Eq.(26) (i.e. use that V<0 => Ddchi<0 => \psi_p<0)
		if Vol<0: psip0, am0 = -psip0, -am0

		#now compute variation of DJ2 with respect to precession averaged mean
		dDJ2 = MSA.compute_dDJ2(psip0, e2_ref)
		
		#compute new DJ20 using this \delta DJ2
		DJ20_new = DJ20_OG - dDJ2

		#if tolerance is reached, break the loop
		if abs(DJ20_new - DJ20) < DJ2_tol: break

		#update DJ2
		DJ20 = DJ20_new
	
	#compute initial \overline{\psi}_p from Eq.(95)
	bpsip0 = 0.5*np.pi*psip0/MSA.K_m

	#compute \delta\phi_z and \delta\zeta from Eq.(49) and (52) respectively
	MSA.update(y_ref, DJ20)
	dphiz0, dzeta0, _ = MSA.precession_Euler_angles(bpsip0)

	#set phiz(t0)=zeta(t0)=0, i.e., initially, L is set to the x'-z' plane of the J-frame
	phiz00 = -dphiz0
	zeta00 = -dzeta0

	#############################################################################################################

	#the direction of observation N in the source frame, following the LAL convention
	phi_N = 0.5*np.pi - phi_ref
	cinc, sinc = np.cos(inclination), np.sin(inclination)

	#fold an inclination outside [0, pi] using N(-inclination, phi_N) = N(inclination, phi_N + pi). cos is even, so only sinc and phi_N change
	if sinc < 0: sinc, phi_N = -sinc, phi_N + np.pi
	cphN, sphN = np.cos(phi_N), np.sin(phi_N)

	#When the spins are exactly aligned, the J frame is equal to the L frame, i.e. the source frame rotated by pi about z, and the angles below are analytic
	if MSA.no_precession:

		#the origin of the orbital phase is the source frame x axis itself
		l0 = 0.

		#k=-N in the J frame. At inclination=0,pi only {}_{-2}Y_{l,2}(0,phi) ~ e^{2 i phi} survives at one pole and {}_{-2}Y_{l,-2}(pi,phi) at
		#the other, and their e^{+-2 i phi_JN} with phi_JN = phi_N is what the LAL identity requires
		cos_theta_JN, sin_theta_JN, phi_JN = -cinc, sinc, phi_N

		#a rotation about z leaves the polarization basis alone, so the J frame one is already the LAL one
		gamma_pol = 0.

	#otherwise build the J frame from the direction of the total angular momentum
	else:

		#compute the total angular momentum vector in the LAL source frame and its direction
		J_vec = MSA.bJ*(s0_1 + s0_2) + MSA.cJ*(s0_1 - s0_2)
		J_vec[2] += MSA.aJ
		J_hat = J_vec/np.linalg.norm(J_vec)

		#azimuth of J and inclination of L with respect to J. \alpha is degenerate when J has no perpendicular part, but
		#it enters both the phase origin l0 and the J frame below, so its (arbitrary) value cancels out of the waveform
		alpha = np.arctan2(J_hat[1], J_hat[0])
		cos_thL, sin_thL = J_hat[2], np.hypot(J_hat[0], J_hat[1])
		ca, sa = np.cos(alpha), np.sin(alpha)

		#build the J frame: z along J and L on the x-z plane with positive x component, i.e. L_J = (sin\theta_L, 0, cos\theta_L)
		z_J = np.array([ sin_thL*ca,  sin_thL*sa, cos_thL])
		x_J = np.array([-cos_thL*ca, -cos_thL*sa, sin_thL])
		y_J = np.array([         sa,         -ca,      0.])

		#the origin of the orbital phase is the projection of J on the orbital plane, so the reference \lambda is the
		#azimuth of \hat{n} (the x axis of the source frame) measured from it
		l0 = -alpha

		#compute the spherical angles of the unit vector pointing from the observer to the binary (k=-N) in the J frame
		N_vec = np.array([sinc*cphN, sinc*sphN, cinc])
		k_hat_Jframe = -np.array([np.dot(N_vec, x_J), np.dot(N_vec, y_J), np.dot(N_vec, z_J)])
		cos_theta_JN, sin_theta_JN = k_hat_Jframe[2], np.hypot(k_hat_Jframe[0], k_hat_Jframe[1])
		#when N is parallel to J the azimuth is degenerate (as in the atan2tol of LALSimIMRPhenomP.c), take it to be 0
		phi_JN = arctan2_tol(k_hat_Jframe[1], k_hat_Jframe[0])

		#gamma_pol rotates the polarization basis (\hat{theta}, \hat{phi} at N) from the J frame to the source frame. \hat{theta}_J is built from the
		#same phi_JN as the spherical harmonics, so the value chosen by arctan2_tol at a degenerate direction cancels out, and being a unit vector
		#in the plane of (\hat{theta}_S, \hat{phi}_S) its arctan2 arguments are always O(1). N=-k has cos=-cos_theta_JN and azimuth phi_JN+pi in the J frame
		cpJ, spJ = np.cos(phi_JN), np.sin(phi_JN)
		th_J = cos_theta_JN*(cpJ*x_J + spJ*y_J) - sin_theta_JN*z_J
		th_S = np.array([cinc*cphN, cinc*sphN, -sinc])
		ph_S = np.array([-sphN, cphN, 0.])
		gamma_pol = np.arctan2(np.dot(th_J, ph_S), np.dot(th_J, th_S))

	#compute reference \delta\lambda from above Eq.(17a) of 1801.08542, so that l0 - dl0 is the mean anomaly
	dl0 = l0 - phi_e_ref

	#compute initial conditions on v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
	v_ini = np.array([y_ref,e2_ref,l0,dl0,DJ20,bpsip0,phiz00,zeta00])

	#return relevant parameters and initial conditions, together with the direction of k in the J frame (cos_theta_JN, sin_theta_JN, phi_JN), the polarization
	#rotation gamma_pol, the effective (possibly clamped) reference frequency, the 0PN estimate of the squared eccentricity at f0_orb and the flag indicating
	#whether the reference was clamped
	return v_ini, yf, s2_1, s2_2, MSA, cos_theta_JN, sin_theta_JN, phi_JN, gamma_pol, fref_orb_eff, e2_0_est, ref_clamped

#Function to compute constant part of h0 from Eq.(20) of 2402.06804, i.e. we want to compute  4\sqrt{\pi/5} M \nu/d_L
# to obtain h0 we have to multiply by (M\omega)**(2/3) = ((1-e2)*(y**2)) (Eq.(5))
def compute_h0_pref(m1, m2, dL):
	return 4*((np.pi/5)**0.5)*(m1*m2/(m1+m2))/dL

#function to solve time diferential equations for v(t)=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
#use scipy solve_ivp and dense output
#the initial conditions v_ini are given at the effective reference orbital frequency fref_orb (already clamped by initial_conditions_for_RR_eqs when ref_clamped)
#e2_0_est is the 0PN estimate of the squared eccentricity at f0_orb computed by initial_conditions_for_RR_eqs
def solve_ivp_RR_eqs_t(v_ini, PN_derivatives, MSA, f0_orb, fref_orb, yf, e2_0_est, ref_clamped, rtol=[1e-10, 1e-10, 1e-12, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12], atol=[1e-12, 1e-12,  1e-8,  1e-8, 1e-12,  1e-8,  1e-6,  1e-6], max_duration=None, e2_max=1-1e-12):

	#function to compute dv/dt
	#e2 is clamped to [0, e2_max], since trial stages can overshoot e2 slightly above 1 or below 0
	def dv_dt(t, v):
		return derivatives_prec_avg(v[0], min(max(v[1],0), e2_max), v[4], PN_derivatives, MSA)

	#function to see if y is larger than the termination y=yf
	def yf_surpassed(t, v):
		return yf - v[0]

	#assign attribute to this function so that termination occurs if y>yf
	yf_surpassed.terminal = True

	#event triggered when the orbital frequency crosses f0_orb, using Eq.(5)
	def f0_reached(t, v):
		return forb_of_y(v[0], min(v[1], 1.), MSA.M) - f0_orb
	f0_reached.terminal = True

	#for the reference time give the 0PN estimate
	t_ref = tLO_func(v_ini[0], v_ini[1], MSA.m1, MSA.m2)

	#evaluate the f0_orb-crossing event at the reference state to decide if a pre-evolution/backward leg is needed
	f_m_f0_ref = f0_reached(t_ref, v_ini)

	#dense backward solutions to stitch before the forward one
	sols = []
	#by default, start the forward integration from the reference state, with the initial frequency f0_orb reached at its start
	t0, v0 = t_ref, v_ini
	t_start = t_ref

	#if the reference frequency is below the initial frequency, evolve the reference state forward until the orbital frequency reaches f0_orb
	if (fref_orb < f0_orb) and (f_m_f0_ref < 0):
		f0_reached.direction = 1
		pre_sol = solve_ivp(dv_dt, [t_ref, -10*t_ref], v_ini, method='RK45', events=f0_reached, rtol=rtol, atol=atol)
		if pre_sol.status != 1:
			raise RuntimeError("Could not evolve the reference state (fref_orb=%.4e) forward to the initial orbital frequency f0_orb=%.4e (solve_ivp status=%s)"%(fref_orb, f0_orb, pre_sol.status))
		#take the state at f0_orb and re-anchor its time with the 0PN estimate (the time origin is irrelevant, since t is shifted at the end)
		v0 = pre_sol.y[:,-1]
		t0 = tLO_func(v0[0], v0[1], MSA.m1, MSA.m2)
		t_start = t0

	#if the reference frequency is above the initial frequency, integrate backwards until f0_orb and stitch with the forward solution
	elif (fref_orb > f0_orb) and (f_m_f0_ref > 0):
		#estimate the time at f0_orb from the 0PN eccentricity estimate e2_0_est to set a generous backward integration horizon
		t0_est = tLO_func(y_of_forb(f0_orb, e2_0_est, MSA.M), e2_0_est, MSA.m1, MSA.m2)

		#integrate backwards only when the estimated horizon genuinely precedes the reference time
		if t0_est < t_ref:

			#integrate backwards from the reference state until the orbital frequency reaches f0_orb
			f0_reached.direction = -1
			sol_back = solve_ivp(dv_dt, [t_ref, t_ref + 10*(t0_est - t_ref)], v_ini, dense_output=True, method='RK45', events=f0_reached, rtol=rtol, atol=atol)
			if sol_back.status==-1:
				raise RuntimeError("Backward ODE integration from the reference frequency fref_orb=%.4e stopped before reaching the initial orbital frequency f0_orb=%.4e because the required step size is below floating-point spacing"%(fref_orb, f0_orb))
			if sol_back.status!=1:
				raise RuntimeError("Backward ODE integration from the reference frequency fref_orb=%.4e did not reach the initial orbital frequency f0_orb=%.4e"%(fref_orb, f0_orb))

			#time at which the initial orbital frequency is reached
			t_start = sol_back.t[-1]

			#make sure that max_duration (measured from f0_orb) allows the evolution to reach the reference frequency
			if (max_duration is not None) and (t_start + max_duration <= t_ref):
				raise ValueError("max_duration=%.4es is too short to evolve from the initial frequency f0_orb=%.4e to the reference frequency fref_orb=%.4e (%.4es needed), increase max_duration or decrease f22_ref"%(max_duration, f0_orb, fref_orb, t_ref - t_start))

			#use the backward leg, unless the backward integration terminated immediately (fref_orb barely above f0_orb),
			#in which case we fall back to the plain forward integration from the reference state
			if t_start != t_ref:
				sols = [sol_back.sol]
				#if the reference point was clamped to the end of the evolution, the solution is the backward leg alone
				#the final time from coalescence is the 0PN estimate at the reference, i.e. t_ref
				if ref_clamped:
					return ivp_sol_interp(sols, t_final=t_ref)

	#make sure that initial value of y is smaller than yf so that stopping condition makes sense
	if v0[0]>=yf:
		raise ValueError("Initial PN parameter larger than final PN parameter (y0=%s >= yf=%s)"%(v0[0], yf))

	#solve system of differential equations, with the maximum duration measured from the time f0_orb is reached
	if max_duration is None: tmax = -10*t_start
	else:                    tmax = t_start + max_duration
	ivp_sol = solve_ivp(dv_dt, [t0, tmax], v0, dense_output=True, method='RK45', events=yf_surpassed, rtol=rtol, atol=atol)

	#check if integration was stopped by reaching the end of t_span
	if ivp_sol.status==0 and max_duration is None:
		warnings.warn(
			"ODE integration stopped because the limit time (t=%ss) was reached, with y=%s, yf=%s, "
			"v_ini=%s, m1=%s, m2=%s, sz_1=%s, sz_2=%s, sp2_1=%s, sp2_2=%s"
			%(tmax, ivp_sol.y[0,-1], yf, list(v0), MSA.m1, MSA.m2, MSA.sz_1, MSA.sz_2, MSA.sp2_1, MSA.sp2_2),
			RuntimeWarning, stacklevel=2)
	if ivp_sol.status==-1:
		warnings.warn("ODE integration stopped because the required step size is below floating-point spacing.", RuntimeWarning, stacklevel=2)

	#estimate the value of the final time from coalescence using final y and e2
	tf = tLO_func(ivp_sol.y[0,-1], ivp_sol.y[1,-1], MSA.m1, MSA.m2)

	#return an interpolant of the (stitched) solution using our ivp_sol_interp class setting t=0 to be the coalescence time
	return ivp_sol_interp(sols + [ivp_sol.sol], t_final=tf)

#solve the system of Eq.(127-128) of arXiv:2106.10291 for the SUA constants a_{k, k_\max}
#we move all factorials to the l.h.s. to have values closer to 1 in the Matrix of linear system
def compute_ak_SUA(kmax):

	#if kmax>9, there are problems with numerical precission, throw a warning
	if kmax>9: warnings.warn("For SUA_kmax>9, finite-precision errors make the solved a_{k,kmax} values untrustworthy.", UserWarning, stacklevel=2)

	#rewritte it as a linear system M*a = b
	M = np.zeros((kmax+1, kmax+1), dtype=np.complex128)
	#b is 0.5 instead of 1 to take into account the 1/2 of Eq.(39) of 1408.5158
	b = np.full(kmax+1, 0.5)
	
	#compute k
	k=np.arange(kmax)+1
	
	#compute the matrix M_{q k} = ((i k^2)^q/(2q-1)!!)
	M[1:,1:] = np.cumprod(((1j*k*k)[None,:])/((2*k-1)[:,None]), axis=0)
	M[ 0, 0] = 0.5
	M[ 0,1:] = 1

	#now solve the system
	return np.linalg.solve(M, b)

#function to compute the derivatives appearing in Eqs.(101-109) of 2106.10291
def derivatives_prec_avg(y, e2, DJ2, PN_derivatives, MSA):

	#update MSA with this y and DJ2
	MSA.update(y, DJ2)
	
	#compute Dy, D(e^2), D\lambda and D\delta\lambda from Eqs.(101-104), using our class
	Dy, De2, Dl, Ddl = PN_derivatives.Dy_De2_Dl_Ddl(y, e2, MSA.chi_eff_prec_avg, MSA.dchi_prec_avg, MSA.dchi2_prec_avg, MSA.sperp2_prec_avg)
	#compute D\deltaJ^2. This equation is only accurate to leading PN order
	DDJ2 = Dy*MSA.compute_DDJ2_Dy(MSA.cJmodxddchi_prec_avg)
	#compute stuff that will be needed for derivatives of Euler angles
	y6 = y**6
	#compute D\overline{\psi}_p
	Dbpsip = 0.375*np.pi*MSA.ADdch*y6*MSA.sqY3mYm*MSA.inv_K_m
	#compute D\phi_{z,0}
	Dphiz0 = y6*(MSA.ADphiJ*MSA.J + MSA.inv_K_m*(MSA.Pm + MSA.Pp))
	#compute D\zeta_0
	Dzeta0 =-y6*(MSA.ADphiJ*(MSA.aJ + MSA.bJ*MSA.chi0mod + MSA.cJmod*MSA.dchi_prec_avg) + MSA.ADphiP*(4*MSA.aJ*MSA.bJmod + 2*MSA.bJ2_m_cJ2*MSA.chi0mod) + MSA.inv_K_m*(MSA.Pm - MSA.Pp))

	#compute D \equiv M/(1-e^2)^{3/2} d/dt (Eq.(4))
	D_fact = ((1 - e2)**1.5)/MSA.M
	
	#if y is an array, give the correct shape to D_fact
	if not MSA.scalar_y: D_fact = D_fact[np.newaxis,:]

	#return time derivative of v(t)=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0] as numpy array
	return D_fact*np.array([Dy, De2, Dl, Ddl, DDJ2, Dbpsip, Dphiz0, Dzeta0])

#class to compute things related to the Multipole Scale Analysis (MSA). We largely follow 2106.10291 and 2502.03929
class MultipleScaleAnalysis:
	
	#initialize computing the things that will stay constant throught the evolution
	def __init__(self, m1, m2, sz_1, sz_2, sp2_1, sp2_2, pn_spin_order=6, dmu_small_threas=1e-12, N_cubic_solves=2, m_ellip_max=float(np.nextafter(1.0, 0.0))):

		#make sure all inputs are floats because they will be more efficient to operate with than np.floats
		m1, m2, sz_1, sz_2, sp2_1, sp2_2 = float(m1), float(m2), float(sz_1), float(sz_2), float(sp2_1), float(sp2_2)

		#save the input constants
		self.m1 = m1
		self.m2 = m2
		self.sz_1 = sz_1
		self.sz_2 = sz_2
		self.sp2_1 = sp2_1
		self.sp2_2 = sp2_2
		#flag for exactly aligned spins, in which case there is no precession and the J frame is the L frame.
		self.no_precession = (sp2_1 == 0) and (sp2_2 == 0)
		self.dmu_small_threas = dmu_small_threas
		self.m_ellip_max = m_ellip_max

		#set up the pn_spin_order
		if pn_spin_order==-1: self.pn_spin_order = 8
		else:                 self.pn_spin_order = pn_spin_order
		if self.pn_spin_order>8 : warnings.warn("pn_spin_order>8 not implemented. Input pn_spin_order: %s"%(self.pn_spin_order), UserWarning, stacklevel=2)

		#set up the number of times to solve cubic equation
		#for pn_spin_order<= 4, it is unnecessary to do it more than once
		if self.pn_spin_order <=4 : self.N_cubic_solves = 1
		else                      : self.N_cubic_solves = max(N_cubic_solves, 1)
		
		#initialize type as not scalar, to force correct functions to be set
		self.scalar_y = False
		
		#compute mass related stuff
		self.M, self.mu1, self.mu2, self.nu, self.dmu = mass_params_from_m1_m2(m1, m2)
		
		#compute chi_eff0 and dchi0
		self.chi_eff0 = sz_1 + sz_2
		self.dchi0 = sz_1 - sz_2

		#compute total spins
		self.s2_1 = sp2_1 + sz_1*sz_1
		self.s2_2 = sp2_2 + sz_2*sz_2

		#compute stuff that will be used to simplyfy equations
		self.sz1sz2 = self.sz_1*self.sz_2
		self.sp21psp22 = sp2_1 + sp2_2
		self.sp2_prod = 4*sp2_1*sp2_2
		self.sp21_chi0mdchi0 = 2*sp2_1*sz_2
		self.sp22_chi0pdchi0 = 2*sp2_2*sz_1
		self.sq3 = 3**0.5
		self.sq1_3 = 3**-0.5

		###### Constants in D\delta\chi and D\chi ######
		
		#Coefficient A in front of Ddchi
		self.ADdch1_chi   = -1.
		self.ADdch2_const = -3./4. + self.nu*(-3./2.)
		self.ADdch3_chi   = 203./48. + (13./12.)*self.nu
		self.ADdch3_dch   = (-31./48.)*self.dmu
		self.ADdch4_const = -11./16. + self.nu*(-16./3. + self.nu*(7./8.))
		self.ADdch5_chi   = -85./64. + self.nu*(-2281./576. + self.nu*(-61./144.))
		self.ADdch5_dch   = (65./64. + self.nu*(911./576.))*self.dmu

		#Coefficient A in front of Dchi
		self.ADchi2_const = (-3./4.)*self.dmu
		self.ADchi3_chi   = (49./48.)*self.dmu
		self.ADchi3_dch   = -5./48. - self.nu/4.
		self.ADchi4_const = (-7./16. + self.nu*(65./24.))*self.dmu
		self.ADchi5_chi   = ((-59./64.) + self.nu*(-1121./576.))*self.dmu
		self.ADchi5_dch   = 39./64. + self.nu*(-545./576. + self.nu*(77./144.))
		
		###### Constants in J = aJ*lN + bJ*(s1+s2) + cJ*(s1-s2) ######
		
		#LlN appearing in aJ = (nu/y)*(1 + y2*LlN)
		self.LlN2_const = 3./2. + self.nu/6.
		self.LlN3_chi   = -49./24.
		self.LlN3_dch   = (-7./24.)*self.dmu
		self.LlN3_ys1s2 = -1.
		self.LlN4_const = 27./8. + self.nu*(-19./8. + self.nu/24.) - 0.5*(self.s2_1 + self.s2_2)
		self.LlN4_ch2   = 1.
		self.LlN5_chi   = -121./32. + self.nu*(671./288.)
		self.LlN5_dch   = (-55./32. + self.nu*(11./288.))*self.dmu
		self.LlN5_ys1s2 = self.nu*(2./3.)

		#Ls appearing in bJ = 0.5*(1 - nu*y2*Ls)
		self.Ls2_const = 7./4.
		self.Ls3_chi   = -1.
		self.Ls4_const = 33./16. + self.nu*(-61./48.)
		self.Ls5_chi   = -119./48. + self.nu/2.
		self.Ls5_dch   = (7./48.)*self.dmu
		
		#Ld appearing in cJ = 0.5*dmu*(1 - nu*y2*Ld)
		self.Ld2_const = 1./4.
		self.Ld4_const = 15./16. - self.nu/48.
		self.Ld5_chi   = 7./48.
		self.Ld5_dch   = (1./48.)*self.dmu
		
		###### Constants in \D lN = y^6 (ws*(s1+s2) + wd*(s1-s2))xlN ######
		
		#Coefficient ws in front of (s1+s2)
		self.ws0_const = 7./4.
		self.ws1_chi   = -3./2.
		self.ws2_const = -9./8. + self.nu*(-19./8.)
		self.ws3_chi   = 203/32 + self.nu*(13./8.)
		self.ws3_dch   = (-31./32.)*self.dmu
		self.ws4_const = -27./32. + self.nu*(-207./32. + self.nu*(127./96.))
		self.ws5_chi   = -255./128. + self.nu*(-2281./384. + self.nu*(-61./96.))
		self.ws5_dch   = (195./128. + self.nu*(911./384.))*self.dmu
		
		#Coefficient wd in front of (s1-s2)
		self.wd0_const = (1./4.)*self.dmu
		self.wd2_const = (9./8. - self.nu/8.)*self.dmu
		self.wd3_chi   = (-49./32.)*self.dmu
		self.wd3_dch   = 5./32. + self.nu*(3./8.)
		self.wd4_const = (27./32. + self.nu*(-81./32. + self.nu/96.))*self.dmu
		self.wd5_chi   = (177./128. + self.nu*(1121./384.))*self.dmu
		self.wd5_dch   = -117./128. + self.nu*(545./384. + self.nu*(-77./96.))

		###### Constants in \kappa_{\D \Delta_{J^2}} = (\D \Delta_{J^2}/\Dy)/(-2*cJmod(dchi-dch0)/y^2) ######

		self.kDDJ2_0_const  = 1.
		self.kDDJ2_2_const  = -9./4. - self.nu/6.
		self.kDDJ2_3_chi    = 61./12.
		self.kDDJ2_3_dch    = (7./12.)*self.dmu
		self.kDDJ2_4_const  = -135./8. + self.nu*(27./8. + self.nu*(-7./16.))
		self.kDDJ2_4_sperp2 = 1.5
		self.kDDJ2_4_ch2    = -1.5
		
	#function to compute constants in J = aJ*lN + bJ*(s1+s2) + cJ*(s1-s2) and prefactors in \D\delta\chi, \D\chi
	def PN_compute_aJ_bJ_cJ_ADdch_ADch(self, y, DJ2, chi_eff, dchi):
		
		#if required, compute y*(s1.s2) at leading PN order
		if self.pn_spin_order>=6:
			ys1s2 = self.y*(self.sz1sz2 + 0.5*DJ2) - 0.5*(self.dmu*(dchi - self.dchi0) + (chi_eff - self.chi_eff0))

		#initialize constants
		LlN = 0
		Ls = 0
		Ld = 0
		ADchi = 0
		ADdch = 0

		#start adding PN terms
		if self.pn_spin_order>=8:
			LlN = self.y*(      chi_eff*self.LlN5_chi + dchi*self.LlN5_dch + ys1s2*self.LlN5_ys1s2)
			Ls  = self.y*(      chi_eff*self.Ls5_chi  + dchi*self.Ls5_dch)
			Ld  = self.y*(      chi_eff*self.Ld5_chi  + dchi*self.Ld5_dch)
			ADdch = self.y*(    chi_eff*self.ADdch5_chi + dchi*self.ADdch5_dch)
			ADchi = self.y*(    chi_eff*self.ADchi5_chi + dchi*self.ADchi5_dch)
		if self.pn_spin_order>=7:
			LlN = self.y*(LlN + self.LlN4_const + chi_eff*chi_eff*self.LlN4_ch2)
			Ls  = self.y*(Ls  + self.Ls4_const)
			Ld  = self.y2*(Ld + self.Ld4_const)
			ADdch = self.y*(ADdch + self.ADdch4_const)
			ADchi = self.y*(ADchi + self.ADchi4_const)
		if self.pn_spin_order>=6:
			LlN = self.y*(LlN + chi_eff*self.LlN3_chi + dchi*self.LlN3_dch + ys1s2*self.LlN3_ys1s2)
			Ls  = self.y*(Ls  + chi_eff*self.Ls3_chi)
			ADdch = self.y*(ADdch + chi_eff*self.ADdch3_chi + dchi*self.ADdch3_dch)
			ADchi = self.y*(ADchi + chi_eff*self.ADchi3_chi + dchi*self.ADchi3_dch)
		if self.pn_spin_order>=5:
			LlN = self.y2*(LlN + self.LlN2_const)
			Ls  = self.y2*(Ls  + self.Ls2_const)
			Ld  = self.y2*(Ld  + self.Ld2_const)
			ADdch = self.y*(ADdch + self.ADdch2_const)
			ADchi = self.y2*(ADchi + self.ADchi2_const)
		if self.pn_spin_order>=4:
			ADdch = self.y*(ADdch + chi_eff*self.ADdch1_chi)
		
		#reconstruct aJ, bJ, cJ and ADdch
		aJ = (self.nu/self.y)*(1 + LlN)
		bJ = 0.5*(1 - self.nu*Ls)
		cJ = 0.5*self.dmu*(1 - self.nu*Ld)
		ADdch = 1. + ADdch

		return aJ, bJ, cJ, ADdch, ADchi

	#function to compute constants in \Omega_{lN} = ws*(s1+s2) + wd*(s1-s2) appearing in \D lN = y^6  \Omega_{lN} x lN
	def PN_compute_ws_wd(self, y, chi_eff, dchi):

		#initialize constants
		ws = 0
		wd = 0

		#start adding PN terms
		if self.pn_spin_order>=8:
			ws = self.y*(     chi_eff*self.ws5_chi + dchi*self.ws5_dch)
			wd = self.y*(     chi_eff*self.wd5_chi + dchi*self.wd5_dch)
		if self.pn_spin_order>=7:
			ws = self.y*(ws + self.ws4_const)
			wd = self.y*(wd + self.wd4_const)
		if self.pn_spin_order>=6:
			ws = self.y*(ws + chi_eff*self.ws3_chi + dchi*self.ws3_dch)
			wd = self.y*(wd + chi_eff*self.wd3_chi + dchi*self.wd3_dch)
		if self.pn_spin_order>=5:
			ws = self.y*(ws + self.ws2_const)
			wd =self.y2*(wd + self.wd2_const)
		if self.pn_spin_order>=4:
			ws = self.y*(ws + chi_eff*self.ws1_chi)
		if self.pn_spin_order>=3:
			ws = ws + self.ws0_const
			wd = wd + self.wd0_const

		return ws, wd

	#function to compute the constants that appear in D\chi, both in front of J and the elliptic integrals of the third kind
	def compute_ADphiJ_ADphiP(self, y, chi_eff, dchi):
		
		#compute constants in \Omega_{lN} = ws*(s1+s2) + wd*(s1-s2) appearing in \D lN = y^6  \Omega_{lN} x lN
		ws, wd = self.PN_compute_ws_wd(y, chi_eff, dchi)
		
		#When cJmod->0, ADphiJ and ADphiP can diverge, but the equations behave as if ADphiJ=ws/bJ and ADphiP=0
		#constant in front of J in \D\phi
		ADphiJ = self.safe_divide(wd + self.kchi*ws                           , self.cJmod, min_den=self.dmu_small_threas, out=self.safe_divide(ws, self.bJ))
		#constant in front of elliptic integrals in \D\phi
		ADphiP = self.safe_divide(0.5*(self.cJ*ws - self.bJ*wd)/self.bJ2_m_cJ2, self.cJmod, min_den=self.dmu_small_threas)

		return ADphiJ, ADphiP

	#method to update class with everything that is required to characterize the MSA solution
	def update(self, y, DJ2, min_J1pmcthL=1e-10):
		
		#determine if y is a scalar or an array
		if np.isscalar(y):
			#for speed, make sure y and DJ2 are floats and not np.floats
			y, DJ2 = float(y), float(DJ2)
			#to save computational cost only change class if needed
			if not self.scalar_y:
				self.scalar_y = True
				#choose functions in math library to avoid np overheads
				self.sin, self.cos = math.sin, math.cos
				self.sqrt, self.arctan2 = math.sqrt, math.atan2
				self.maximum = max
				self.minimum = min
				self.safe_divide = safe_divide_by_scalar
		else:
			self.scalar_y = False
			#choose numpy functions to handle arrays
			self.sin, self.cos = np.sin, np.cos
			self.sqrt, self.arctan2 = np.sqrt, np.arctan2
			self.maximum = np.maximum
			self.minimum = np.minimum
			self.safe_divide = safe_divide_by_array

		#save input y and DJ2
		self.y = y
		self.DJ2 = DJ2

		#Useful variables
		self.y2 = self.y*self.y

		##################### Solution of (\D\delta\chi)^2 #####################
		
		#set up initial guesses for precession average chi_eff and dchi
		self.chi_eff_prec_avg = self.chi_eff0
		self.dchi_prec_avg    = self.dchi0
		
		#repeatedly solve precession equations to refine this guess
		for ic in range(self.N_cubic_solves):

			#compute aJ, bJ, cJ in J = aJ*lN + bJ*(s1+s2) + cJ*(s1-s2)
			self.aJ, self.bJ, self.cJ, self.ADdch, ADchi = self.PN_compute_aJ_bJ_cJ_ADdch_ADch(y, DJ2, self.chi_eff_prec_avg, self.dchi_prec_avg)

			#Compute kchi that relates chi_eff \approx chi_eff0 + kchi*(dchi - dchi0)
			self.kchi = ADchi/self.ADdch
			
			#compute combinations of these constants that commonly appear
			self.one_p_kchi = 1 + self.kchi
			self.one_m_kchi = 1 - self.kchi
			self.cJmod = self.cJ + self.kchi*self.bJ
			self.bJmod = self.bJ + self.kchi*self.cJ
			self.bJ2_m_cJ2 = self.bJ*self.bJ - self.cJ*self.cJ
			self.chi0mod  =  self.chi_eff0 - self.kchi*self.dchi0

			#setup and solve cubic equation appearing in differential equation for Ddchi^2
			self.setup_and_solve_Ddchi_cubic()

			#compute complete elliptic integrals of the first and second kind
			self.K_m = scipy.special.ellipk(self.m)
			self.E_m = scipy.special.ellipe(self.m)

			#make sure we do not propagate any unwanted np.floats
			if self.scalar_y: self.K_m, self.E_m = float(self.K_m), float(self.E_m)

			#compute 1/K(m), which will be widely used
			self.inv_K_m = 1./self.K_m
			
			#compute the m dependent factors for precession averaged spins
			m_factor_dchi_prec_avg, m_factor_sigma = self.precesion_average_factors_betasigma()

			#compute precession averaged <dchi> = dchi_prec_avg
			self.dchi_prec_avg = self.dchi_av - 2*self.dchi_diff*m_factor_dchi_prec_avg

			#compute precession averaged chi_eff as a function of precession averaged dchi
			self.chi_eff_prec_avg = self.chi0mod + self.kchi*self.dchi_prec_avg

		#compute also cJmod*(dchi_prec_avg - dchi0)
		self.cJmodxddchi_prec_avg = self.cJmodxddchiav - 2* self.cJmodxdchidiff*m_factor_dchi_prec_avg

		#compute dchi2_prec_avg = <dchi^2>
		self.dchi2_prec_avg = self.dchi_prec_avg*self.dchi_prec_avg + self.dchi_diff*self.dchi_diff*(0.5 - 4*m_factor_sigma)

		#compute <s_\perp^2> = <\sigma_0^(1)-chi_eff2> \approx <\sigma_0^(1)> - chi_eff20
		self.sperp2_prec_avg = self.sp21psp22 + self.DJ2 - self.kappac_cJmod*self.cJmodxddchi_prec_avg

		##################### Things related to Euler Angles #####################

		#compute parallel component of total angular momentum J
		J0lN = self.aJ + self.bJ*self.chi_eff0 + self.cJ*self.dchi0
		J0lN2 = J0lN*J0lN
		
		#compute squared perpedicular component of J
		bJpcJ = self.bJ + self.cJ
		bJpcJ2 = bJpcJ*bJpcJ
		bJmcJ = self.bJ - self.cJ
		bJmcJ2 = bJmcJ*bJmcJ
		Sp2_1 = bJpcJ2*self.sp2_1
		Sp2_2 = bJmcJ2*self.sp2_2
		Jperp2 = Sp2_1 + Sp2_2 + self.bJ2_m_cJ2*DJ2

		#compute J by adding parallel and perpendicular moduli
		self.J = self.sqrt(self.maximum(J0lN2 + Jperp2, 0))

		#compute expansion factor in J = \sqrt{1 + 2*x}
		xJ = 0.5*Jperp2/J0lN2

		#compute also J \pm J0lN, considering the cases where x is small
		small_x = (abs(xJ)<1e-6)
		dJ_small_x = J0lN*xJ*(1 - 0.5*xJ*(1-xJ))
		J_p_J0lN = my_where(small_x & (J0lN<0),-dJ_small_x, self.J + J0lN)
		J_m_J0lN = my_where(small_x & (J0lN>0), dJ_small_x, self.J - J0lN)

		#compute Np=N_+ and Nm=N_-
		muSz = self.one_p_kchi*bJpcJ2*self.sz_1 + self.one_m_kchi*bJmcJ2*self.sz_2
		dmudSp2 = self.cJmod*(Sp2_1 - Sp2_2)
		Np = J_p_J0lN*(self.bJmod*J_p_J0lN - muSz) - dmudSp2
		Nm = J_m_J0lN*(self.bJmod*J_m_J0lN + muSz) - dmudSp2
		
		#compute (Bp-Cp) = J*min(1+cos(\theta_L))
		Bp_m_Cp = J_p_J0lN + self.cJmodxddchiav - self.cJmodxdchidiff
		#to avoid singularities force it to be larger than min_J1pmcthL
		Bp_m_Cp = self.maximum(Bp_m_Cp, min_J1pmcthL)
		
		#compute (Bm+Cm) = J*min(1-cos(\theta_L))
		Bm_p_Cm = J_m_J0lN - self.cJmodxddchiav - self.cJmodxdchidiff
		#to avoid singularities, force it to be larger than min_J1pmcthL
		Bm_p_Cm = self.maximum(Bm_p_Cm, min_J1pmcthL)
		#now compute (Bm-Cm)=(Bm+Cm)-2*Cm
		Bm_m_Cm = Bm_p_Cm + 2*self.cJmodxdchidiff

		#compute the constants that appear in D\chi, both in front of J and the elliptic integrals of the third kind
		self.ADphiJ, self.ADphiP = self.compute_ADphiJ_ADphiP(self.y, self.chi_eff_prec_avg, self.dchi_prec_avg)

		#compute prefactor's to elliptic PI's
		self.PI_fact_p = self.ADphiP*Np/Bp_m_Cp
		self.PI_fact_m = self.ADphiP*Nm/Bm_m_Cm

		#compute also the arguments -2*C/(B-C) using that Cp = -Cm = cJmod*dchi_diff
		self.PI_arg_p = -2*self.cJmodxdchidiff/Bp_m_Cp
		self.PI_arg_m =  2*self.cJmodxdchidiff/Bm_m_Cm

		#compute Pp and Pm appearing in Eq.(109) of 2106.10291
		#to compute the elliptic integral of the third kind \Pi(n;\phi;m) use Carlsons symmetric form RJ and the elliptic integral of the first kind K(m)
		self.Pp = self.PI_fact_p*(self.K_m + (self.PI_arg_p/3)*scipy.special.elliprj(0, 1 - self.m, 1, 1 - self.PI_arg_p))
		self.Pm = self.PI_fact_m*(self.K_m + (self.PI_arg_m/3)*scipy.special.elliprj(0, 1 - self.m, 1, 1 - self.PI_arg_m))

	#method to set up and solve cubic equation appearing in \D\delta\chi
	def setup_and_solve_Ddchi_cubic(self,):

		#compute coefficients of cubic equation (ddchi = dchi - dchi0)
		#Ddchi = (9/4)*(ADchi**2)*(acT*ddchi3 - bcT*ddchi2 + ccp*ddchi + dcp)
		one_m_kchi2 = 1 - self.kchi*self.kchi
		dchi0mod = self.dchi0 - self.kchi*self.chi_eff0
		self.kappac_cJmod = 2*self.aJ/self.bJ2_m_cJ2
		kappac = self.kappac_cJmod*self.cJmod
		acT = one_m_kchi2*kappac
		bcT = self.one_m_kchi*self.one_m_kchi*self.sp2_1 + self.one_p_kchi*self.one_p_kchi*self.sp2_2 + one_m_kchi2*self.DJ2 + self.chi0mod*self.chi0mod + kappac*(-2*dchi0mod +kappac)
		ccp = 2*((kappac - dchi0mod)*self.DJ2 + self.one_m_kchi*self.sp21_chi0mdchi0 - self.one_p_kchi*self.sp22_chi0pdchi0)
		dcp = self.sp2_prod - self.DJ2*self.DJ2

		#useful combinations
		bcT2 = bcT*bcT
		acT2 = acT*acT
		
		#compute coefficients of depressed cubic
		#Ddchi = (9/4)*(ADchi**2)*(acT*(ddchi - dY)**3 - (pact2/acT)*(ddchi - dY) + (qact3/act2))
		#acT*dY = bcT/3.
		pacT2 = (1./3.)*bcT2 - acT*ccp
		qacT3 = acT2*dcp + bcT*((1./3.)*acT*ccp +bcT2*(-2./27.))
		
		#compute acT^6 times the discriminant of cubic equation ((pacT2/3)**3 - (qacT3/2)**2)
		#We expand in terms of coefficients of cubic equation to avoid numerical errors
		discacT6 = acT2*(dcp*((-1./4.)*acT2*dcp + bcT*((-1./6.)*acT*ccp + (1./27.)*bcT2)) + ccp*ccp*((1./108.)*bcT2 - (1./27.)*acT*ccp))
		
		#compute arg(G)/3, with G = -qacT3/2 + 1j*sqrt(discacT6)
		#The discriminant has to be larger than 0 since there are three real roots in cubic equation.
		argG_3 = my_where(discacT6>=0, self.arctan2(self.sqrt(abs(discacT6)), -0.5*qacT3)/3., 0.)

		#compute the sine and cosine of arg(G)/3
		sargG_3, cargG_3 = self.sin(argG_3), self.cos(argG_3)
		
		#compute cos(argG/3 - pi/6)
		cargG_3_m_pi_6 = 0.5*(self.sq3*cargG_3 + sargG_3)
		
		#compute the solutions to the depressed cubic
		sqpacT2 = self.sqrt(abs(pacT2))

		#compute acT*ddchi_av = acT*(dchi_av - dchi0) = acT*(dY + 0.5*(Yp+Ym))
		acTddchiav = bcT/3. - self.sq1_3*sqpacT2*cargG_3

		#compute acT*dchi_diff = 0.5*acT*(Yp-Ym)
		acTdchidiff = sqpacT2*sargG_3

		#compute cJmod*ddchi_av and cJmod*dchi_diff using that acT = 2*aJ*cJmod/bJ2_m_cJ2
		self.cJmod_acT = 1./(one_m_kchi2*self.kappac_cJmod)
		self.cJmodxddchiav = self.cJmod_acT*acTddchiav
		self.cJmodxdchidiff = self.cJmod_acT*acTdchidiff

		#compute also dchi_av - dchi0 and dchi_diff. When acT->0, these come from solution to quadratic equation
		self.ddchi_av  = self.safe_divide( acTddchiav, acT, min_den=self.dmu_small_threas, out=self.safe_divide(0.5*ccp, bcT))
		self.dchi_diff = self.safe_divide(acTdchidiff, acT, min_den=self.dmu_small_threas, out=self.safe_divide(0.5*self.sqrt(self.maximum(ccp*ccp + 4*bcT*dcp, 0)), bcT))

		#compute chi_av
		self.dchi_av = self.ddchi_av + self.dchi0

		#compute sqrt(Y3 - Y_-), which is what actually appears in equations
		self.sqY3mYm = self.sqrt(2*sqpacT2*cargG_3_m_pi_6)

		#compute the parameter m of the Jacobi functions
		self.m = self.minimum(sargG_3/cargG_3_m_pi_6, self.m_ellip_max)
	
	#method to compute the precesion average factors of Eq.(65) and Eq.(72) of 2106.10291 that appear in beta and sigma
	def precesion_average_factors_betasigma(self, mthreas=0.1):
		
		#consider first the case that m is not an array
		if self.scalar_y:
			if self.m<mthreas:
				return self.precesion_average_factors_betasigma_small_m()
			else:
				return self.precesion_average_factors_betasigma_large_m()
		#if m is a numpy array, the process is a bit more involved
		else:
			#initialize numpy arrays to put the result in
			m_factor_dchi_prec_avg, m_factor_sigma = np.zeros_like(self.m), np.zeros_like(self.m)
			
			#find indexes corresponding to small m's and put the corresponding results
			idxs_small = (self.m < mthreas)
			if np.any(idxs_small):
				m_factor_dchi_prec_avg[idxs_small], m_factor_sigma[idxs_small] = self.precesion_average_factors_betasigma_small_m(idxs=idxs_small)
			
			#do the same for the large m's
			idxs_large = np.logical_not(idxs_small)
			if np.any(idxs_large):
				m_factor_dchi_prec_avg[idxs_large], m_factor_sigma[idxs_large] = self.precesion_average_factors_betasigma_large_m(idxs=idxs_large)

			return m_factor_dchi_prec_avg, m_factor_sigma
	
	#make a function for the small m case that uses Pade approximants of Eq.(65) and Eq.(72).
	def precesion_average_factors_betasigma_small_m(self, idxs=None):

		#Choose only a subset of indices if this is required
		if idxs is None: m = self.m
		else:            m = self.m[idxs]

		#For the factor in Eq.(65) use a {3,3} Pade around m=0. We expect the absolute/relative error on this factor to be smaller than 1.3e-8/5.9e-7 for m<0.3
		m_factor_dchi_prec_avg = -m*(1 + m*(-1 + m*(71/384)))/(16 + m*(-24 + m*((59/6) - m*(11/12))))
		#For the factor in Eq.(72) use a {2,5} Pade around m=0. We expect the absolute/relative error on this factor to be smaller than 6.1e-10/4.9e-6 for m<0.3
		m2 = m*m
		m_factor_sigma = m2/(1024 + m*(-1024 + m*(96 - m2*(133/32)*(1+m))))

		return m_factor_dchi_prec_avg, m_factor_sigma

	#make a function for the large m case that uses the exact expressions of Eq.(65) and Eq.(72)
	def precesion_average_factors_betasigma_large_m(self, idxs=None):

		#Choose only a subset of indices if this is required
		if idxs is None:
			m = self.m
			E_K_m = self.inv_K_m*self.E_m
		else:
			m = self.m[idxs]
			E_K_m = self.inv_K_m[idxs]*self.E_m[idxs]
		
		#compute the m factors as defined in Eq.(65) and Eq.(72) respectively
		m_factor_dchi_prec_avg = (E_K_m - 1 + 0.5*m)/m
		m_factor_sigma = ((1/3) + m*((-1/3) + m/8) + E_K_m*((2/3)*(m-2) + E_K_m))/(m*m)

		return m_factor_dchi_prec_avg, m_factor_sigma

	#function to compute the variation of the Euler angles on precession time-scales following arXiv:2106.10291
	def precession_Euler_angles(self, bpsip):

		#compute hbpsip_pi_2 = (2/pi)\hat{\overline{\psi}}_p (after Eq.49)
		hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
		
		#compute the value of \hat{\psi}_p from \overline{\psi}_p using Eq.(95)
		hpsip = self.K_m*hbpsip_pi_2

		#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
		sn, cn, dn, am = scipy.special.ellipj(hpsip, self.m)

		#compute the prefactor of the first term in \delta\zeta. Use that cJmod_acT = bJ2_m_cJ2/(2*one_m_kchi2*aJ)
		dzeta_E_fact = (4./3.)*self.cJmod_acT*self.sqY3mYm*self.ADphiJ/self.ADdch
		
		#compute the incomplete elliptic integral of the second kind (Eq.(52))
		E_m_inc = dzeta_E_fact*(scipy.special.ellipeinc(am, self.m) - hbpsip_pi_2*self.E_m)

		#compute the incomplete elliptic integrals of the third kind \Pi(n;\phi;m) using Carlsons symmetric form RJ and the elliptic integral of the first kind K(phi;m)
		Km_inc = scipy.special.ellipkinc(am, self.m)
		sn2 = sn*sn
		PI_arg_p_sn2 = self.PI_arg_p*sn2
		PI_arg_m_sn2 = self.PI_arg_m*sn2
		one_m_sn2 = np.maximum(1 - sn2, 0)
		one_m_msn2 = np.maximum(1 - self.m*sn2, 0)
		PI_p_inc = Km_inc + PI_arg_p_sn2*sn*scipy.special.elliprj(one_m_sn2, one_m_msn2, 1, 1 - PI_arg_p_sn2)/3
		PI_m_inc = Km_inc + PI_arg_m_sn2*sn*scipy.special.elliprj(one_m_sn2, one_m_msn2, 1, 1 - PI_arg_m_sn2)/3
		
		#compute the factors that depend on the incomplete elliptic integrals of the third kind (Eq.(52))
		inv_sqY3mYm_fact = self.safe_divide(4., 3.*self.ADdch*self.sqY3mYm)
		Pp_inc = inv_sqY3mYm_fact*(self.PI_fact_p*PI_p_inc - hbpsip_pi_2*self.Pp)
		Pm_inc = inv_sqY3mYm_fact*(self.PI_fact_m*PI_m_inc - hbpsip_pi_2*self.Pm)

		#compute\delta\phi_z from Eq.(49)
		dphiz = Pp_inc + Pm_inc
		
		#compute \delta\zeta from Eq.(52)
		dzeta = E_m_inc + Pp_inc - Pm_inc

		#compute dchi from Eq.(26)
		dchi = self.dchi_av - self.dchi_diff*(1 - 2*sn2)

		#compute cos(\theta_L) from Eq.(15)
		costhL = self.safe_divide(self.aJ + self.bJ*self.chi0mod+ self.cJmod*dchi, self.J)

		#return variation of Euler angles on precession timescales, forcing costhL to be between -1 and 1
		return dphiz, dzeta, np.clip(costhL, -1, 1)

	#function to compute (DDJ2/Dy)/(-2*cJmodxddchi/y2)
	def compute_kDDJ2(self,):

		#initialize coefficient
		kDDJ2 = 0.
		
		#start adding PN terms
		if self.pn_spin_order>=7:
			kDDJ2 = self.y*(        self.kDDJ2_4_const + self.sperp2_prec_avg*self.kDDJ2_4_sperp2 + self.chi_eff_prec_avg*self.chi_eff_prec_avg*self.kDDJ2_4_ch2)
		if self.pn_spin_order>=6:
			kDDJ2 = self.y*(kDDJ2 + self.chi_eff_prec_avg*self.kDDJ2_3_chi + self.dchi_prec_avg*self.kDDJ2_3_dch)
		if self.pn_spin_order>=5:
			kDDJ2 = self.y2*(kDDJ2 + self.kDDJ2_2_const)
		if self.pn_spin_order>=3:
			kDDJ2 += self.kDDJ2_0_const
		
		return kDDJ2

	#function to compute DeltaJ2/Dy
	def compute_DDJ2_Dy(self, cJmodxddchi):
		
		#compute ratio between DDJ2/Dy and its leading order expression -2*cJmodxddchi/y2
		kDDJ2 = self.compute_kDDJ2()
		
		#reconstruct the expression for DDJ2/Dy
		return -2*cJmodxddchi*kDDJ2/self.y2

	#function to compute the variation of \Delta_{J^2}
	def compute_dDJ2(self, hpsip, e2):

		#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
		sn, cn, dn, am = scipy.special.ellipj(hpsip, self.m)

		#compute a_Dy = (Dy/(nu*y**9)), we only compute it to 1PN since higher order terms are very suppressed
		a_Dy = 32./5. + e2*(28./5.) + self.y2*(-1486./105. - (88./5.)*self.nu + e2*(12296./105. - (5258./45.)*self.nu + e2*(3007./84. - (244./9.)*self.nu)))

		#compute ratio between DDJ2/Dy and its leading order expression -2*cJmodxddchi/y2
		kDDJ2 = self.compute_kDDJ2()

		#compute RR contributions to \delta DJ2. Use that cJmod_acT = bJ2_m_cJ2/(2*one_m_kchi2*aJ)
		dDJ2 = (8./3.)*self.cJmod_acT*kDDJ2*(self.sqY3mYm/self.ADdch)*self.y*self.nu*a_Dy*(scipy.special.ellipeinc(am, self.m) - self.inv_K_m*self.E_m*hpsip)

		#for large PN spin orders, compute also SP corrections
		if self.pn_spin_order>=5:
			dDJ2 += (-1./3.)*self.y*self.dmu*self.nu*self.dchi_diff*cn*cn

		return dDJ2

#class to compute the precession averaged value of Dy, D(e^2), D\lambda and D\delta\lambda to 3PN in non-spining and aligned-spin and 2PN in fully spinning using the PN formulas derived in our EFPE paper. We can select the n-th PN order of the spinning/non-spinnig part with pn_phase_order/pn_spin_order = 2*n. If pn_xxxx_order=-1, set it to the maximum.
class pyEFPE_PN_derivatives:

	#initialize computing the constants that depend on nu
	def __init__(self, m1, m2, s2_1, s2_2, q1=1, q2=1, o1=1, o2=1,
	                   Lambda2_1=0, Lambda2_2=0, Lambda3_1=0, Lambda3_2=0, Sigma2_1=0, Sigma2_2=0,
	                   Lambda23_1=0, Lambda23_2=0, Lambda32_1=0, Lambda32_2=0, Sigma23_1=0, Sigma23_2=0, Sigma32_1=0, Sigma32_2=0,
	                   pn_phase_order=9, pn_spin_order=8, pn_tidal_order=0, horizon_absorption=True):
		
		#save pn orders, taking into account that an order of -1 means to take the maximum order
		if pn_phase_order==-1: self.pn_phase_order = 9
		else:                  self.pn_phase_order = pn_phase_order

		if pn_spin_order==-1: self.pn_spin_order = 8
		else:                 self.pn_spin_order = pn_spin_order

		if pn_tidal_order==-1: self.pn_tidal_order = 15
		else:                  self.pn_tidal_order = pn_tidal_order

		
		self.pn_max_order = max(self.pn_phase_order, self.pn_spin_order)
		
		#if we are requesting a pn order higher than what is implemented, throw a Warning
		if self.pn_phase_order>9 : warnings.warn("pn_phase_order>9 not implemented. Input pn_phase_order: %s"%(self.pn_phase_order), UserWarning, stacklevel=2)
		if self.pn_spin_order >8 : warnings.warn("pn_spin_order>8 not implemented. Input pn_spin_order: %s"%(self.pn_spin_order), UserWarning, stacklevel=2)
		if self.pn_tidal_order>15: warnings.warn("pn_tidal_order>15 not implemented. Input pn_tidal_order: %s"%(self.pn_tidal_order), UserWarning, stacklevel=2)

		#save weather or not horizon absorption effects are taken into account
		self.horizon_absorption = horizon_absorption

		#compute mass related stuff
		M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
		nu2 = nu*nu
		nu3 = nu2*nu
		pi2 = np.pi*np.pi
		ln2, ln3, ln5, ln7 = np.log([2., 3., 5., 7.])
		self.nu = nu
		
		#compute symetric and antisymetric combinations of quadrupole parameters
		dqS = q1 + q2 - 2
		dqA = q1 - q2
		dqAdmu = dqA*dmu

		#compute symetric and antisymetric combinations of octupole parameters
		doS = o1 + o2 - 2
		doA = o1 - o2
		
		#compute spin related stuff
		s2iS =  s2_1 + s2_2
		s2iA =  s2_1 - s2_2
		
		#store the e^{2n} coefficients that enter the tail terms of Eq (C4) of 1801.08542
		c_phiy  = np.array([1., 97./32., 49./128., -49./18432., -109./147456., -2567./58982400.])
		c_phie  = np.array([1., 5969./3940., 24217./189120., 623./4538880., -96811./363110400., -5971./4357324800.])
		c_psiy  = np.array([1., -207671./8318., -8382869./266176., -8437609./4791168., 10075915./306634752., -38077159./15331737600.])
		c_zetay = np.array([1., 113002./11907., 6035543./762048., 253177./571536., -850489./877879296., -1888651./10973491200.])
		c_psie  = np.array([1., -9904271./891056., -101704075./10692672., -217413779./513248256., 35703577./6843310080., -3311197679./9854366515200.])
		c_zetae = np.array([1., 11228233./2440576., 37095275./14643456., 151238443./1405771776., -118111./611205120., -407523451./26990818099300.])
		c_kappay = 244*ln2*np.array([0., 1., -18881./1098., 6159821./39528., -16811095./19764., 446132351./123525.])-243*ln3*np.array([0., 1., -39./4., 2735./64., 25959./512., -638032239./409600.])-(48828125./5184.)*ln5*np.array([0., 0., 0., 1., -83./8., 12637./256.])-(4747561509943./33177600.)*ln7*np.array([0., 0., 0., 0., 0., 1.])
		c_kappae = 6536*ln2*np.array([1., -22314./817., 7170067./19608., -10943033./4128., 230370959./15480., -866124466133./8823600.])-6561*ln3*np.array([1., -49./4., 4369./64., 214449./512., -623830739./81920., 76513915569./1638400.])-(48828125./64.)*ln5*np.array([0., 0., 1., -293./24., 159007./2304., -6631171./27648.])-(4747561509943./245760.)*ln7*np.array([0.,0.,0.,0.,1.,-259./20.])

		#store also the e^{2n} coefficients in the Spin-Orbit tail-terms
		c_thyc = np.array([1., 21263./3008., 52387./12032., 253973./1732608., -82103./13860864.])
		c_thyd = np.array([1., 1897./592., -461./2368., -42581./340992., -3803./1363968.])
		c_thec = np.array([1., 377077./92444., 7978379./4437312., 5258749./106495488.])
		c_thed = np.array([1., 37477./19748., 95561./947904., -631523./22749696.])

		#now store the e^{2n} coefficients appearing in the non-spinning part of dy/dt
		self.p_a0NS = my_cpoly(np.array([32./5., 28./5.]))
		
		self.p_a2NS = my_cpoly(np.array([-1486./105. - (88./5.)*nu, 12296./105. - (5258./45.)*nu, 3007./84. - (244./9.)*nu]))
		
		self.p_a3NS = my_cpoly((128./5.)*np.pi*c_phiy)
		
		self.p_a4NS = my_cpoly(np.array([34103./2835. + (13661./315.)*nu + (944./45.)*nu2, -489191./1890. - (209729./630.)*nu + (147443./270.)*nu2, 2098919./7560. - (2928257./2520.)*nu + (34679./45.)*nu2, 53881./2520. - (7357./90.)*nu + (9392./135.)*nu2]))
		self.sqrt_a4NS = my_cpoly(np.array([16. - (32./5.)*nu, 266. - (532./5.)*nu, -859./2. + (859./5.)*nu, -65. + 26*nu]))
		
		self.p_a5NS = my_cpoly(np.pi*(-(4159./105.)*c_psiy-(756./5.)*nu*c_zetay))
		
		self.p_a6NS = my_cpoly(np.array([16447322263./21829500. - (54784./525.)*np.euler_gamma + (512./15.)*pi2 + (-(56198689./34020.) + (902./15.)*pi2)*nu + (541./140.)*nu2 - (1121./81.)*nu3, 33232226053./10914750. - (392048./525.)*np.euler_gamma + (3664./15.)*pi2 + (-(588778./1701.) + (2747./40.)*pi2)*nu - (846121./1260.)*nu2 - (392945./324.)*nu3, -227539553251./58212000. - (93304./175.)*np.euler_gamma + (872./5.)*pi2 + ((124929721./12960.) - (41287./960.)*pi2)*nu + (148514441./30240.)*nu2 - (2198212./405.)*nu3, -300856627./67375. - (4922./175.)*np.euler_gamma + (46./5.)*pi2 + ((1588607./432.) - (369./80.)*pi2)*nu + (12594313./3780.)*nu2 - (44338./15.)*nu3, -243511057./887040. + (4179523./15120.)*nu + (83701./3780.)*nu2 - (1876./15.)*nu3, 0.]) + (1284./175.)*c_kappay)
		self.sqrt_a6NS = my_cpoly(np.array([-616471./1575. + ((9874./315.)- (41./30.)*pi2)*nu + (632./15.)*nu2, 2385427./1050. + (-(274234./45.) + (4223./240.)*pi2)*nu + (70946./45.)*nu2, 8364697./4200. + ((1900517./630.) - (32267./960.)*pi2)*nu - (47443./90.)*nu2, -167385119./25200. + ((4272491./504.) - (123./160.)*pi2)*nu - (43607./18.)*nu2, -65279./168. + (510361./1260.)*nu - (5623./45.)*nu2]))
		self.log_a6NS = my_cpoly(np.array([54784./525., 392048./525., 93304./175., 4922./175.]))

		#now store the e^{2n} coefficients appearing in the non-spinning part of d(e^2)/dt
		self.p_b0NS = my_cpoly(np.array([608./15., 242./15.]))
		
		self.p_b2NS = my_cpoly(np.array([-1878./35. - (8168./45.)*nu, 59834./105. - (7753./15.)*nu, 13929./140. - (3328./45.)*nu]))
		
		self.p_b3NS = my_cpoly((788./3.)*np.pi*c_phie)
		
		self.p_b4NS = my_cpoly(np.array([-949877./945. + (18763./21.)*nu + (1504./5.)*nu2, -3082783./1260. - (988423./420.)*nu + (64433./20.)*nu2, 23289859./7560. - (13018711./2520.)*nu + (127411./45.)*nu2, 420727./1680. - (362071./1260.)*nu + (1642./9.)*nu2]))
		self.sqrt_b4NS = my_cpoly(np.array([2672./3. - (5344./15.)*nu, 2321. - (4642./5.)*nu, 565./3. - (226./3.)*nu]))
		
		self.p_b5NS = my_cpoly(np.pi*(-(55691./105.)*c_psie-(610144./315.)*nu*c_zetae))

		self.p_b6NS = my_cpoly(np.array([61669369961./4365900. - (2633056./1575.)*np.euler_gamma + (24608./45.)*pi2 + ((50099023./56700.) + (779./5.)*pi2)*nu - (4088921./1260.)*nu2 - (61001./243.)*nu3, 66319591307./21829500. - (9525568./1575.)*np.euler_gamma + (89024./45.)*pi2 + ((28141879./450.) - (139031./480.)*pi2)*nu - (21283907./1512.)*nu2 - (86910509./9720.)*nu3, -1149383987023./58212000. - (4588588./1575.)*np.euler_gamma + (42884./45.)*pi2 + ((11499615139./453600.) - (271871./960.)*pi2)*nu + (61093675./2016.)*nu2 - (2223241./90.)*nu3, 40262284807./4312000. - (20437./175.)*np.euler_gamma + (191./5.)*pi2 + (-(5028323./280.) - (6519./320.)*pi2)*nu + (24757667./1260.)*nu2 - (11792069./1215.)*nu3, 302322169./887040. - (1921387./5040.)*nu + (41179./108.)*nu2 - (386792./1215.)*nu3, 0.]) + (428./1575.)*c_kappae)
		self.sqrt_b6NS = my_cpoly(np.array([-22713049./7875. + (-(11053982./945.) + (8323./90.)*pi2)*nu + (108664./45.)*nu2, 178791374./7875. + (-(38295557./630.) + (94177./480.)*pi2)*nu + (681989./45.)*nu2, 5321445613./189000. + (-(26478311./756.) + (2501./1440.)*pi2)*nu + (450212./45.)*nu2, 186961./168. - (289691./252.)*nu + (3197./9.)*nu2]))
		self.one_m_sqrt_b6NS = 1460336./23625
		self.log_b6NS = my_cpoly(np.array([2633056./1575., 9525568./1575., 4588588./1575., 20437./175.]))
		
		#now store the e^{2n} coefficients appearing in the Spin-Orbit part of dy/dt
		self.chi_p_a3SO = my_cpoly(np.array([-752./15., -138., -611./30.]))
		self.dch_p_a3SO = my_cpoly(dmu*np.array([-152./15., -154./15., 17./30.]))
		
		self.chi_p_a5SO = my_cpoly(np.array([-5861./45. + (4004./15.)*nu, -968539./630. + (259643./135.)*nu, -4856917./2520. + (943721./540.)*nu, -64903./560. + (5081./45.)*nu]))
		self.chi_e2sqrt_a5SO = my_cpoly(np.array([-1416./5. + (1652./15.)*nu, 2469./5. - (5761./30.)*nu, 222./5. - (259./15.)*nu]))
		
		self.dch_p_a5SO = my_cpoly(dmu*np.array([-21611./315. + (632./15.)*nu, -55415./126. + (36239./135.)*nu, -72631./360. + (12151./108.)*nu, 909./560. - (143./45.)*nu]))
		self.dch_e2sqrt_a5SO = my_cpoly(dmu*np.array([-472./5. + (236./15.)*nu, 823./5. - (823./30.)*nu, 74./5. - (37./15.)*nu]))
		
		self.chi_p_a6SO = my_cpoly(-(3008./15.)*np.pi*c_thyc)
		self.dch_p_a6SO = my_cpoly(-(592./15.)*np.pi*dmu*c_thyd)

		#now store the e^{2n} coefficients appearing in the Spin-Orbit part of d(e^2)/dt
		self.chi_p_b3SO = my_cpoly(np.array([-3272./9., -26263./45., -812./15.]))
		self.dch_p_b3SO = my_cpoly(dmu*np.array([-3328./45., -1993./45., 23./15.]))
		
		self.chi_p_b5SO = my_cpoly(np.array([-13103./35. + (289208./135.)*nu, -548929./63. + (61355./6.)*nu, -6215453./840. + (1725437./270.)*nu, -87873./280. + (13177./45.)*nu]))
		self.chi_sqrt_b5SO = my_cpoly(np.array([-1184. + (4144./9.)*nu, -13854./5. + (16163./15.)*nu, -626./5. + (2191./45.)*nu]))
		
		self.dch_p_b5SO = my_cpoly(dmu*np.array([-32857./105. + (52916./135.)*nu, -1396159./630. + (126833./90.)*nu, -203999./280. + (56368./135.)*nu, 5681./1120. - (376./45.)*nu]))
		self.dch_sqrt_b5SO = my_cpoly(dmu*np.array([-1184./3. + (592./9.)*nu, -4618./5. + (2309./15.)*nu, -626./15. + (313./45.)*nu]))
		
		self.chi_p_b6SO = my_cpoly(-(92444./45.)*np.pi*c_thec)
		self.dch_p_b6SO = my_cpoly(-(19748./45.)*np.pi*dmu*c_thed)

		#now store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of dy/dt
		c_s2iS_a4SS = s2iS*np.array([8./5. - 8*dqS, 24./5. - (108./5.)*dqS, 3./5. - (63./20.)*dqS])
		c_s2iA_a4SS = dqA*s2iA*np.array([-8., -108./5., -63./20.])
		self.const_a4SS = my_cpoly(c_s2iS_a4SS + c_s2iA_a4SS)
		self.chi2_a4SS = my_cpoly(np.array([156./5. + 12*dqS, 84. + (162./5.)*dqS, 123./10. + (189./40.)*dqS]))
		self.sperp2_a4SS = my_cpoly(np.array([-84./5., -228./5., -33./5.]))
		self.chidch_a4SS = my_cpoly(dqA*np.array([24., 324./5., 189./20.]))
		self.dch2_a4SS = my_cpoly(np.array([-2./5. + 12*dqS, -6./5. + (162./5.)*dqS, -3./20. + (189./40.)*dqS]))
		
		self.chi2_p_a6SS = my_cpoly(np.array([30596./105. + (2539./105.)*dqS + (443./30.)*dqAdmu +  (-(688./5.) - (172./5.)*dqS)*nu, 115078./45. + (21317./60.)*dqS + (3253./60.)*dqAdmu + (-(3962./3.) - (1981./6.)*dqS)*nu, 4476649./2520. + (133703./420.)*dqS + (481./48.)*dqAdmu + (-(53267./45.) - (53267./180.)*dqS)*nu, 17019./140. + (29831./1120.)*dqS + (29./160.)*dqAdmu + (-(1343./15.) - (1343./60.)*dqS)*nu, 0.]))
		self.chi2_sqrt_a6SS = my_cpoly(np.array([-(244./15.) - (52./15.)*dqS - (4./15.)*dqAdmu + (16./5. + (4./5.)*dqS)*nu, 6283./30. + (1339./30.)*dqS + (103./30.)*dqAdmu + (-(206./5.) - (103./10.)*dqS)*nu, -(48007./120.) - (10231./120.)*dqS - (787./120.)*dqAdmu + (787./10. + (787./40.)*dqS)*nu, -(183./20.) - (39./20.)*dqS - (3./20.)*dqAdmu + (9./5. + (9./20.)*dqS)*nu]))
		self.chidch_p_a6SS = my_cpoly(np.array([(3134./15. + (443./15.)*dqS)*dmu + (5078./105. - (344./5.)*nu)*dqA, (30421./45. + (3253./30.)*dqS)*dmu + (21317./30. - (1981./3.)*nu)*dqA, (-(111./5.) + (481./24.)*dqS)*dmu + (133703./210. - (53267./90.)*nu)*dqA, (-(149./40.) + (29./80.)*dqS)*dmu + (29831./560. - (1343./30.)*nu)*dqA, 0.]))
		self.chidch_sqrt_a6SS = my_cpoly(np.array([(-(104./15.) - (8./15.)*dqS)*dmu + (-(104./15.) + (8./5.)*nu)*dqA, (1339./15. + (103./15.)*dqS)*dmu + (1339./15. - (103./5.)*nu)*dqA, (-(10231./60.) - (787./60.)*dqS)*dmu + (-(10231./60.) + (787./20.)*nu)*dqA, (-(39./10.) - (3./10.)*dqS)*dmu + (-(39./10.) + (9./10.)*nu)*dqA]))
		self.dch2_p_a6SS = my_cpoly(np.array([39./5. + (2539./105.)*dqS + (443./30.)*dqAdmu + (-(1163./15.) - (172./5.)*dqS)*nu, 659./15. + (21317./60.)*dqS + (3253./60.)*dqAdmu + (-(2399./15.) - (1981./6.)*dqS)*nu, 1769./90. + (133703./420.)*dqS + (481./48.)*dqAdmu + (2021./72. - (53267./180.)*dqS)*nu, 19./10. + (29831./1120.)*dqS + (29./160.)*dqAdmu + (-(3./10.) - (1343./60.)*dqS)*nu]))
		self.dch2_sqrt_a6SS = my_cpoly(np.array([-(4./15.) - (52./15.)*dqS - (4./15.)*dqAdmu + (32./15. + (4./5.)*dqS)*nu, 103./30. + (1339./30.)*dqS + (103./30.)*dqAdmu + (-(412./15.) - (103./10.)*dqS)*nu, -(787./120.) - (10231./120.)*dqS - (787./120.)*dqAdmu +  (787./15. + (787./40.)*dqS)*nu, -(3./20.) - (39./20.)*dqS - (3./20.)*dqAdmu + (6./5. + (9./20.)*dqS)*nu]))
		
		#now store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of d(e^2)/dt
		c_s2iS_b4SS = s2iS*np.array([-4./3., 34./3. - (938./15.)*dqS, 49./2. - (595./6.)*dqS, 9./4. - (37./4.)*dqS])
		c_s2iA_b4SS = dqA*s2iA*np.array([0., -938./15., -595./6., -37./4.])
		self.const_b4SS = my_cpoly(c_s2iS_b4SS + c_s2iA_b4SS)
		self.chi2_b4SS = my_cpoly(np.array([2./3., 3667./15. + (469./5.)*dqS, 4613./12. + (595./4.)*dqS, 287./8. + (111./8.)*dqS]))
		self.sperp2_b4SS = my_cpoly(np.array([2./3., -1961./15., -2527./12., -157./8.]))
		self.chidch_b4SS = my_cpoly(dqA*np.array([0., 938./5., 595./2., 111./4.]))
		self.dch2_b4SS = my_cpoly(np.array([2./3., 1./3. + (469./5.)*dqS, -13./4. + (595./4.)*dqS, -3./8. + (111./8.)*dqS]))
		
		self.chi2_p_b6SS = my_cpoly(np.array([1468414./945. + (2852./105.)*dqS + (3461./30.)*dqAdmu + (-(57844./45.) - (14461./45.)*dqS)*nu, 47715853./3780. + (1464091./840.)*dqS + (11007./40.)*dqAdmu + (-(21865./3.) - (21865./12.)*dqS)*nu, 4255831./504. + (166844./105.)*dqS + (2941./48.)*dqAdmu + (-(222533./45.) - (222533./180.)*dqS)*nu, 414027./1120. + (365363./4480.)*dqS + (511./640.)*dqAdmu + (-(1287./5.) - (1287./20.)*dqS)*nu]))
		self.chi2_sqrt_b6SS = my_cpoly(np.array([49532./45. + (10556./45.)*dqS + (812./45.)*dqAdmu + (-(3248./15.) - (812./15.)*dqS)*nu, 140117./60. + (29861./60.)*dqS + (2297./60.)*dqAdmu + (-(2297./5.) - (2297./20.)*dqS)*nu, 3721./180. + (793./180.)*dqS + (61./180.)*dqAdmu + (-(61./15.) - (61./60.)*dqS)*nu]))
		self.chidch_p_b6SS = my_cpoly(np.array([(176426./135. + (3461./15.)*dqS)*dmu + (5704./105. - (28922./45.)*nu)*dqA, (387212./135. + (11007./20.)*dqS)*dmu + (1464091./420. - (21865./6.)*nu)*dqA, (2562./5. + (2941./24.)*dqS)*dmu + (333688./105. - (222533./90.)*nu)*dqA, (-(33./32.) + (511./320.)*dqS)*dmu + (365363./2240. - (1287./10.)*nu)*dqA]))
		self.chidch_sqrt_b6SS = my_cpoly(np.array([(21112./45. + (1624./45.)*dqS)*dmu + (21112./45. - (1624./15.)*nu)*dqA, (29861./30. + (2297./30.)*dqS)*dmu + (29861./30. - (2297./10.)*nu)*dqA, (793./90. + (61./90.)*dqS)*dmu + (793./90. - (61./30.)*nu)*dqA]))
		self.dch2_p_b6SS = my_cpoly(np.array([8887./135. + (2852./105.)*dqS + (3461./30.)*dqAdmu + (-(13127./27.) - (14461./45.)*dqS)*nu, 161077./540. + (1464091./840.)*dqS + (11007./40.)*dqAdmu + (-(185723./270.) - (21865./12.)*dqS)*nu, 14827./90. + (166844./105.)*dqS + (2941./48.)*dqAdmu + (-(45373./360.) - (222533./180.)*dqS)*nu, 283./32. + (365363./4480.)*dqS + (511./640.)*dqAdmu + (-(117./20.) - (1287./20.)*dqS)*nu]))
		self.dch2_sqrt_b6SS = my_cpoly(np.array([812./45. + (10556./45.)*dqS + (812./45.)*dqAdmu + (-(6496./45.) - (812./15.)*dqS)*nu, 2297./60. + (29861./60.)*dqS + (2297./60.)*dqAdmu + (-(4594./15.) - (2297./20.)*dqS)*nu, 61./180. + (793./180.)*dqS + (61./180.)*dqAdmu + (-(122./45.) - (61./60.)*dqS)*nu]))
		
		#compute the contribution of Horizon Absorption
		self.pref_a5H = my_cpoly(np.array([ -4./5., -12./5.,  -3./10.]))
		self.pref_b5H = my_cpoly(np.array([-44./5., -66./5., -11./10.]))
		
		self.chi_c5H     = 1. - 2.*nu + (9./8.)*s2iS
		self.dch_c5H     = dmu + (9./8.)*s2iA
		self.chidch2_c5H = 45./16.
		self.chi3_c5H    = 15./16.
		
		#store the e^{2n} coefficients of the non-spinning part of the periastron precession k
		self.k0NS = 3
		
		self.p_k2NS = my_cpoly(np.array([27./2. - 7*nu, 51./4. - (13./2.)*nu]))
		
		self.p_k4NS = my_cpoly(np.array([105./2. + (-(625./4.) + (123./32.)*pi2)*nu + 7*nu2, 573./4. + (-(357./2.) + (123./128.)*pi2)*nu + 40*nu2, 39./2. - (55./4.)*nu + (65./8.)*nu2]))
		self.sqrt_k4NS = my_cpoly(np.array([15. - 6*nu, 30. - 12*nu]))
		
		#store the e^{2n} coefficients of the spin-orbit part of the periastron precession k
		self.chi_k1SO = -7./2.
		self.dch_k1SO = -(1./2.)*dmu
		
		self.chi_p_k3SO = my_cpoly(np.array([-26. + 8*nu, -(105./4.) + (49./4.)*nu]))
		self.dch_p_k3SO = my_cpoly(dmu*np.array([-8. + (1./2.)*nu, -(15./4.) + (7./4.)*nu]))
		
		#store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of the periastron precession k
		c_s2iS_k2SS = -(3./8.)*dqS*s2iS
		c_s2iA_k2SS = -(3./8.)*dqA*s2iA
		self.const_k2SS = c_s2iS_k2SS + c_s2iA_k2SS
		self.chi2_k2SS = 3./2. + (9./16.)*dqS
		self.sperp2_k2SS = -3./4.
		self.chidch_k2SS = (9./8.)*dqA
		self.dch2_k2SS = (9./16.)*dqS
		
		self.chi2_p_k4SS = my_cpoly(np.array([181./8. + (33./8.)*dqS + (3./4.)*dqAdmu + (-(5./2.) - (5./8.)*dqS)*nu, 369./16. + (75./16.)*dqS + (3./16.)*dqAdmu + (-(29./4.) - (29./16.)*dqS)*nu]))
		self.chidch_p_k4SS = my_cpoly(np.array([(43./4. + (3./2.)*dqS)*dmu + (33./4. - (5./4.)*nu)*dqA, (21./8. + (3./8.)*dqS)*dmu + (75./8. - (29./8.)*nu)*dqA]))
		self.dch2_p_k4SS = my_cpoly(np.array([1./8. + (33./8.)*dqS + (3./4.)*dqAdmu + (-(7./2.) - (5./8.)*dqS)*nu, -(3./16.) + (75./16.)*dqS + (3./16.)*dqAdmu - (29./16.)*dqS*nu]))
		
		#now compute quasi-circular corrections at higher PN orders
		#non-spinning coefficients
		self.a7NSQC = np.pi*(-883./126. + nu*(71735./189. + nu*(73196./189.)))

		self.a8NSQC = 3959271176713./3972969000. - (5776./315.)*pi2 + (1995856./11025.)*np.euler_gamma + (2140336./11025.)*ln2 - (9477./49.)*ln3 + nu*(-317589100793./61122600. - (1472377./2520.)*pi2 - (372800./441.)*np.euler_gamma - (8561344./11025.)*ln2 + (37908./49.)*ln3 + nu*(504221849./51030. - (22099./60.)*pi2 + nu*(-1909807./9720. + nu*(917./180.))))
		self.log_a8NSQC = -(1995856./11025.) + nu*(372800./441.)

		self.a9NSQC = np.pi*(343801320119./116424000. - (219136./525.)*np.euler_gamma + nu*(-516333533./83160. + (3608./15.)*pi2 + nu*(-6821669./6480. + nu*(-24107249./41580.))))
		self.log_a9NSQC = np.pi*(219136./525.)

		#spin-orbit coefficients
		self.chi_a7SOQC = (-4323559./5670. + nu*(87341./42. + nu*(-17840./27.)))
		self.dch_a7SOQC = (-1932041./5670. + nu*(40289./90. + nu*(-10819./135.)))*dmu

		self.chi_a8SOQC = (-30542./45. + nu*(26536./15.))*np.pi
		self.dch_a8SOQC = (-93914./315. + nu*(34303./105.))*np.pi*dmu

		#spin-spin coefficients
		self.chi2_a7SSQC   = (128. + 32.*dqS)*np.pi
		self.chidch_a7SSQC = 64.*dqA*np.pi
		self.dch2_a7SSQC   = (4./5. + 32.*dqS)*np.pi

		self.chi2_a8SSQC   = (14931877./5670. + (1931813./11340.)*dqS + nu*(-45809./15. - (17305./84.)*dqS + nu*(12536./45. + (3134./45.)*dqS)) + (81919./1260. + nu*(-1327./20.))*dqAdmu)
		self.chidch_a8SSQC = ((1931813./5670. + nu*(-17305./42. + nu*(6268./45.)))*dqA + dmu*(1597856./945. + (81919./630.)*dqS + nu*(-19949./15. - (1327./10.)*dqS)))
		self.dch2_a8SSQC   = 144727./540. + (1931813./11340.)*dqS + nu*(-765353./756. - (17305./84.)*dqS + nu*(71707./180. + (3134./45.)*dqS)) + (81919/1260 + nu*(-1327./20.))*dqAdmu

		#cubic in spin coefficients
		self.chi3_a7S3QC    = (-4688./15. - (1436./15.)*dqS - (88./5.)*doS)
		self.chi2dch_a7S3QC = ((-1496./15. - (374./15.)*dqS)*dmu - (2078./15.)*dqA - (264./5.)*doA)
		self.chidch2_a7S3QC = (-43./15. + (152./15.)*dqS - (748./15.)*dqAdmu - (264./5.)*doS)
		self.dch3_a7S3QC    = (-3./5. - (374./15.)*dqS)*dmu + (794./15.)*dqA - (88./5.)*doA

		#perpendicular spin-spin coefficients
		self.sperp2_a6SSpQC = 3251./42. + nu*(6373./90.)
		sp2iS_a6SSpQC = 62./15. - (2539./105.)*dqS + nu*(-181./45. + (172./5.)*dqS) - (443./30.)*dqAdmu
		sp2iA_a6SSpQC = (-599./15. - (443./30.)*dqS)*dmu + (-2539./105. + nu*(172./5.))*dqA

		self.sperp2_a7SSpQC = (-192./5.)*np.pi
		sp2iS_a7SSpQC = (-96./5.)*np.pi*dqS
		sp2iA_a7SSpQC = (-96./5.)*np.pi*dqA

		self.sperp2_a8SSpQC = -9355721./22680. + nu*(-195697./280. + nu*(-162541./1080.))
		sp2iS_a8SSpQC = -563./60. - (1931813./11340.)*dqS + nu*(13427./420. + (17305./84.)*dqS + nu*(12109./540. - (3134./45.)*dqS)) + (-81919./1260. + nu*(1327./20.))*dqAdmu
		sp2iA_a8SSpQC = (-6302./105. - (81919./1260.)*dqS + nu*(1801./10. + (1327./20.)*dqS))*dmu + (-1931813./11340. + nu*(17305./84. + nu*(-3134./45.)))*dqA

		#parts of perpendicular spin-spin coefficients that are constant
		self.const_a6SSpQC = s2iS*sp2iS_a6SSpQC + s2iA*sp2iA_a6SSpQC
		self.const_a7SSpQC = s2iS*sp2iS_a7SSpQC + s2iA*sp2iA_a7SSpQC
		self.const_a8SSpQC = s2iS*sp2iS_a8SSpQC + s2iA*sp2iA_a8SSpQC

		#parts of perpendicular spin-spin coefficients that are proportional to chi_eff^2
		self.chi2_a6SSpQC  = -0.5*sp2iS_a6SSpQC
		self.chi2_a7SSpQC  = -0.5*sp2iS_a7SSpQC
		self.chi2_a8SSpQC  = -0.5*sp2iS_a8SSpQC

		#parts of perpendicular spin-spin coefficients that are proportional to chi_eff*dchi
		self.chidch_a6SSpQC = -sp2iA_a6SSpQC
		self.chidch_a7SSpQC = -sp2iA_a7SSpQC
		self.chidch_a8SSpQC = -sp2iA_a8SSpQC

		#parts of perpendicular spin-spin coefficients that are proportional to dchi^2
		self.dch2_a6SSpQC = -0.5*sp2iS_a6SSpQC
		self.dch2_a7SSpQC = -0.5*sp2iS_a7SSpQC
		self.dch2_a8SSpQC = -0.5*sp2iS_a8SSpQC

		#compute constants for tidal contributions
		m1_4 = 16*(mu1**4)
		m2_4 = 16*(mu2**4)
		m1_5 = mu1*m1_4
		m2_5 = mu2*m2_4
		m1_4_m2 = m1_4*mu2
		m2_4_m1 = m2_4*mu1
		
		L2Aav   = m1_4_m2*Lambda2_1 + m2_4_m1*Lambda2_2
		L2Adiff = m1_4_m2*Lambda2_1 - m2_4_m1*Lambda2_2
		L2Bav   =    m1_5*Lambda2_1 +    m2_5*Lambda2_2
		L2Bdiff =    m1_5*Lambda2_1 -    m2_5*Lambda2_2

		L3Aav   = 4*(mu1*mu1*m1_4_m2*Lambda3_1 + mu2*mu2*m2_4_m1*Lambda3_2)

		S2Aav   = m1_4_m2*Sigma2_1 + m2_4_m1*Sigma2_2
		S2Adiff = m1_4_m2*Sigma2_1 - m2_4_m1*Sigma2_2
		S2Bav   =    m1_5*Sigma2_1 +    m2_5*Sigma2_2
		S2Bdiff =    m1_5*Sigma2_1 -    m2_5*Sigma2_2

		L23Aav   = m1_4_m2*Lambda23_1 + m2_4_m1*Lambda23_2
		L23Adiff = m1_4_m2*Lambda23_1 - m2_4_m1*Lambda23_2
		L23Bav   =    m1_5*Lambda23_1 + m2_5*Lambda23_2
		L23Bdiff =    m1_5*Lambda23_1 - m2_5*Lambda23_2

		L32Aav   = m1_4_m2*Lambda32_1 + m2_4_m1*Lambda32_2
		L32Adiff = m1_4_m2*Lambda32_1 - m2_4_m1*Lambda32_2

		S23Aav   = m1_4_m2*Sigma23_1 + m2_4_m1*Sigma23_2
		S23Adiff = m1_4_m2*Sigma23_1 - m2_4_m1*Sigma23_2
		S23Bav   =    m1_5*Sigma23_1 +    m2_5*Sigma23_2
		S23Bdiff =    m1_5*Sigma23_1 -    m2_5*Sigma23_2

		S32Aav   = m1_4_m2*Sigma32_1 + m2_4_m1*Sigma32_2
		S32Adiff = m1_4_m2*Sigma32_1 - m2_4_m1*Sigma32_2
		
		#e^{2n} coefficients appearing in tidal part of dy/dt
		self.p_a10T    = my_cpoly(L2Aav*np.array([144/5, 836/5, 31047/160, 14937/320, 943/640])
		                        + L2Bav*np.array([12/5, 351/10, 549/8, 381/16, 99/128]))
		self.sqrt_a10T = my_cpoly(L2Aav*np.array([ -6., 137/20, -635/32, -431/320, 67/160]))

		self.p_a12T    = my_cpoly(L2Aav*(np.array([4421/140, 1319937/560, 985073/128, 204209/40, 12368995/14336, 45101/2560]) + nu*np.array([-571/10, -21935/12, -1335341/240, -1974391/480, -642493/960, -6329/384])) 
		                        + L2Bav*(np.array([38/35, 90621/280, 217845/112, 231639/112, 756747/1792, 57501/7168]) + nu*np.array([751/10, 885/8, -372337/320, -227927/128, -197483/512, -9.]))
		                        + S2Aav*(np.array([4148/15, 9978/5, 32397/10, 7791/8, 13431/64, 47/64]))
		                        + S2Bav*(np.array([-4/15, -54/5, -273/5, -119/2, -441/32, -21/64])))
		self.sqrt_a12T = my_cpoly(L2Aav*(np.array([-22467/280, 6215/112, 28723/280, -1102047/1120, 195451/17920, 57481/8960]) + nu*np.array([99/10, -12529/120, 147/10, 484171/960, 12743/256, -2009/384]))
		                        + L2Bav*nu*np.array([-35, 45/8, -3195/64, -7075/128, 1175/64, 0])
		                        + S2Aav*np.array([-168, 15, -34, -5689/8, 1979/32, -7/16]))
		
		self.p_a13T    = my_cpoly(np.pi*(L2Aav*(576/5)*np.array([1, 24679/2304, 36403/1536, 3092927/221184, 20138917/21233664, -242265263/283115520])
		                        + L2Bav*(48/5)*np.array([1, 1689/64, 13509/128, 3633127/36864, 1002787/49152, 52964281/117964800])))

		self.p_a14T    = my_cpoly(L2Aav*(np.array([6993499/15120, 15714269/1512, 1102179223/15120, 52322942411/483840, 3070450891/86016, 86768245/18432, 11317927/172032]) + nu*(np.array([18677/105, -4227115/672, -80707429/840, -4801163293/26880, -8385074287/107520, -768392641/86016, -16057529/107520]) + nu*np.array([-1239/10, 357469/80, 181318513/3840, 392908657/4608, 191582975/4608, 54700813/11520, 13669/144]))) 
		                        + L2Bav*(np.array([21011/630, 173864/105, 109168823/6720, 158511807/4480, 397260377/21504, 405549581/215040, 177965/14336]) + nu*(np.array([378769/560, 7906847/672, 19089725/1344, -399309413/10752, -1005710919/28672, -937810319/215040, -402673/7168]) + nu*np.array([-5231/12, -4250659/720, -27142819/5760, 256379107/11520, 193185643/9216, 376791/128, 1605/32])))
		                        + L3Aav*np.array([60, 7879/16, 59239/64, 107649/256, 163543/2048, 1015/512, 0])
		                        + S2Aav*(np.array([74966/105, 4203881/140, 12029685/112, 31471151/336, 12672169/896, 7781601/2560, 52095/7168]) + nu*np.array([-11854/15, -888734/45, -12419029/180, -11093731/180, -3679321/360, -2156525/1152, -1607/192]))
		                        + S2Bav*(np.array([-299/315, -9988/105, -315713/280, -83777/30, -222105/128, -9529/40, -3317/1024]) + nu*np.array([15506/15, 29308/3, 4011151/180, 10664423/720, 25953361/5760, 1714169/2880, 973/192])))
		self.sqrt_a14T = my_cpoly(L2Aav*(np.array([5313599/7560, 99753767/15120, 5004769589/241920, -10792769629/483840, -5486137117/161280, -132934883/80640, 11479/672]) + nu*(np.array([-791837/1680, -7014449/1680, -5758529/384, 9966857/672, 194204035/7168, 89192267/71680, -6706709/107520]) + nu*np.array([3437/30, 399151/480, 4898621/1280, -86360387/23040, -36300523/5760, -3070357/5760, 10193/288]))) 
		                        + L2Bav*(np.array([66, 7149/4, 78573/16, -99015/32, -1741125/256, -350625/256, -3015/128]) + nu*(np.array([-1309361/1680, -5539817/3360, -11194711/8960, -52850815/10752, 34937057/21504, 514863/512, 603/64]) + nu*np.array([1013/4, 967/36, -12503/1152, 6274115/2304, 4595645/4608, -67005/256, 0])))
		                        + L3Aav*np.array([-135/4, -735/32, -605/32, -22295/256, 31135/1024, 805/512, 0])
		                        + S2Aav*(np.array([-88273/70, -392771/140, 1021067/140, -2125539/112, -18058423/896, 2508803/1792, 5753/896]) + nu*np.array([2478/5, -8321/15, -125261/30, 139121/12, 1422131/96, -45341/96, -233/48]))
		                        + S2Bav*nu*np.array([-4284/5, -25109/30, 20111/60, -62713/16, -26299/192, 19159/96, 0]))

		self.p_a15T    = my_cpoly(np.pi*(L2Aav*((62199/280)*np.array([1, 8244257/124398, 3122110229/7961472, 8425780561/15922944, 72889899301/382150656, -321835060033/38215065600]) + nu*(-12469/20)*np.array([1, 14581339/523698, 543887075/4189584, 2873062081/16758336, 1259854466359/19305603072, 217583883851/965280153600]))
		                        + L2Bav*((152/35)*np.array([1, 519939/1216, 22964451/4864, 7445548135/700416, 11520947777/1867776, 1831512481007/2241331200]) + nu*(5139/20)*np.array([1, -185047/143892, -59491027/767424, -5288868773/27627264, -43436295845/378888192, -1564634629573/88407244800]))
		                        + S2Aav*(5528/5)*np.array([1, 27953/2764, 1750279/66336, 19833731/1592064, -234543767/50946048, -78228667513/15283814400])
		                        + S2Bav*(-8/15)*np.array([1, 87, 27385/32, 286957/144, 10492679/8192, 368318011/1843200])))

		#e^{2n} coefficients appearing in tidal part of d(e^2)/dt
		self.p_b10T    = my_cpoly(L2Aav*np.array([2049/10, 45143/40, 186103/160, 283703/1280, 135/32])
		                        + L2Bav*np.array([306/5, 4383/10, 2241/4, 4581/32, 117/32]))
		self.sqrt_b10T = my_cpoly(L2Aav*np.array([110., 7391/40, 4057/160, -221/160]))

		self.p_b12T    = my_cpoly(L2Aav*(np.array([-8119/14, 23909213/2240, 181564903/4480, 978752881/35840, 34386317/10240, 69129/1792]) + nu*np.array([-7517/15, -965723/80, -16085557/480, -14078923/640, -2249095/768, -2565/64]))
		                        + L2Bav*(np.array([-6927/70, 254567/80, 4185269/280, 11069089/896, 18209263/8960, 458709/14336]) + nu*np.array([3837/20, -194121/80, -3890003/320, -5607683/512, -998717/512, -2277/64]))
		                        + S2Aav*np.array([1516/3, 202949/30, 21875/2, 1013453/240, 198221/960, -225/128])
		                        + S2Bav*np.array([-76/3, -369, -1057, -6265/8, -4347/32, -329/128]))
		self.sqrt_b12T = my_cpoly(L2Aav*(np.array([301437/280, 1060691/160, 3173225/448, 2859989/8960, -191119/8960]) + nu*np.array([-13087/30, -324351/80, -59527/16, -254477/640, 33163/1920]))
		                        + L2Bav*nu*np.array([1555/3, 15065/16, 87575/192, -3625/64, 0])
		                        + S2Aav*np.array([2488, 11403/2, 20749/4, 171/16, 1/16]))

		self.p_b13T    = my_cpoly(np.pi*(L2Aav*(14219/8)*np.array([1, 549103/71095, 174556741/13650240, 527873381/81901440, 15318454303/26208460800, -10560207337/78625382400])
		                        + L2Bav*(1491/4)*np.array([1, 30789/2485, 5007661/159040, 121605541/5725440, 446499763/130867200, 27852151/458035200])))

		self.p_b14T    = my_cpoly(L2Aav*(np.array([-82743697/15120, -1216333963/17280, 2720493533/24192, 24485741279/46080, 344719420351/1290240, 6451484171/286720, 9003637/57344]) + nu*(np.array([1827601/210, 4490959/140, -2726590009/6720, -10076963933/10752, -578882411/1344, -2113698147/57344, -495969/1792]) + nu*np.array([-38227/144, 99997487/2880, 3287046031/11520, 43147973551/92160, 9464460739/46080, 225900263/11520, 6255/32]))) 
		                        + L2Bav*(np.array([-9445/7, -1435301/160, 130189411/1680, 1044284889/4480, 455325695/3584, 5928020881/430080, 3729585/28672]) + nu*(np.array([4234787/1680, 102715421/1920, -1426909/4480, -1962033783/7168, -27436469973/143360, -461104567/20480, -118927/512]) + nu*np.array([-195761/120, -326623/24, 84553123/1920, 2591459329/15360, 162144067/1536, 20565367/1536, 5655/32])))
		                        + L3Aav*np.array([2701/12, 467425/192, 137051/32, 717151/384, 945883/6144, 105/128, 0])
		                        + S2Aav*(np.array([-542939/210, 12939691/210, 585876119/1680, 1005358889/2688, 629048467/6720, 322666189/107520, -143573/7168]) + nu*np.array([9268/15, -438757/10, -4344581/20, -36017293/160, -85861273/1440, -19321867/11520, 825/32]))
		                        + S2Bav*(np.array([1099/15, -9062/15, -432463/40, -893949/40, -1419607/128, -38423/32, -12529/1024]) + nu*np.array([86654/45, 156453/4, 4671832/45, 126391667/1440, 137091827/5760, 6057949/2880, 2443/96])))
		self.sqrt_b14T = my_cpoly(L2Aav*(np.array([652145/63, 465567461/3360, 28726706081/80640, 265153067/1344, 181722059/17920, -443487/8960]) + nu*(np.array([-402401/112, -98244221/1120, -3567895967/13440, -4217364169/26880, -59563387/7168, 21617483/107520]) + nu*np.array([-70141/90, 49912583/2880, 246826639/3840, 48388043/1280, 17478931/5760, -16465/144]))) 
		                        + L2Bav*(np.array([2439, 123759/4, 607905/8, 3004065/64, 207675/32, 5805/64]) + nu*(np.array([4660647/560, 7545683/210, 167396401/5376, -25931313/3584, -14386763/3584, -1161/32]) + nu*np.array([-156737/36, -3704137/144, -33915575/1152, -16762745/2304, 621985/768, 0])))
		                        + L3Aav*np.array([1735/4, 60975/64, 98345/128, -29745/512, -2515/512, 0])
		                        + S2Aav*(np.array([3021359/210, 22628569/140, 47491121/168, 88288349/672, -217397/128, -95765/2688]) + nu*np.array([-174674/15, -1638601/15, -542659/3, -2167483/24, -16613/24, 1411/48]))
		                        + S2Bav*nu*np.array([165172/15, 1894837/60, 293335/8, 462077/96, -59857/96, 0]))

		self.p_b15T    = my_cpoly(np.pi*(L2Aav*((728991/280)*np.array([1, 797698129/15551808, 1907561477/7775904, 7471204533529/26873524224, 8150141836301/89578414080, 23051561741023/20638866604032]) + nu*(-2826623/240)*np.array([1, 385730049/22612984, 219572776067/3798981312, 5522175308437/91175551488, 16212773766297/810449346560, 190734337323941/250081512652800]))
		                        + L2Bav*((-144331/140)*np.array([1, -20724175/1154648, -1301470649/6927888, -449078422319/1330154496, -2809221442669/17735393280, -22149413239687/1276948316160]) + nu*(693365/336)*np.array([1, -214893997/13867300, -81817497077/665630400, -1552171760033/7987564800, -22417919049971/255602073600, -1599730171037041/153361244160000]))
		                        + S2Aav*(230774/15)*np.array([1, 2938101/461548, 223866919/22154304, 816033053/531703296, -105984782861/42536263680, -3074077668623/2552175820800])
		                        + S2Bav*(-1528/15)*np.array([1, 39095/1528, 2434063/18336, 354168539/1760256, 6621836617/70410240, 32059415041/2816409600])))

		#e^{2n} coefficients appearing in tidal part of k
		self.p_k8T     = my_cpoly(L2Aav*np.array([45/32, 135/64, 45/256]))

		self.p_k10T    = my_cpoly(L2Aav*(np.array([81/8, 2187/32, 5373/128, 855/512]) + nu*np.array([75/64, -2265/128, -9825/512, -615/512]))
		                        + L2Bav*nu*np.array([315/64, 1575/128, 1575/512, 0])
		                        + S2Aav*np.array([39/2, 135/2, 495/16, 75/64]))
		self.sqrt_k10T = my_cpoly(L2Aav*np.array([9/8, 81/32, 9/16]))

		self.p_k12T    = my_cpoly(L2Aav*(np.array([2061/64, 21393/32, 163395/128, 371457/1024, 28215/4096]) + nu*(np.array([7263/256, -27015/512, -1381617/2048, -83097/256, -18615/2048]) + nu*np.array([-665/64, -5005/128, 56235/512, 93035/1024, 8815/2048]))) 
		                        + L2Bav*nu*(np.array([17991/256, 256329/512, 978975/2048, 59985/1024, 0]) + nu*np.array([-945/64, -19215/128, -92925/512, -29925/1024, 0]))
		                        + L3Aav*np.array([105/32, 1575/128, 1575/256, 525/2048, 0])
		                        + S2Aav*(np.array([543/4, 5763/4, 7965/4, 14511/32, 4125/512]) + nu*np.array([-29/4, -429, -28245/32, -35465/128, -975/128]))
		                        + S2Bav*nu*np.array([289/4, 357, 8925/32, 4165/128, 0]))
		self.sqrt_k12T = my_cpoly(L2Aav*(np.array([1827/32, 2061/8, 42903/256, 1377/128]) + nu*np.array([-309/16, -6261/64, -2157/32, -6])) 
		                        + L2Bav*nu*np.array([75/16, 825/64, 225/32, 0])
		                        + S2Aav*np.array([45/2, 315/4, 1125/16, 45/8]))
		
		#spin-tidal coefficients
		self.chi_a13STQC = -(1292/5)*L2Aav - (64/5)*L2Bav - (986/3)*S2Aav + (2/15)*S2Bav - (856/5)*L23Aav + (272/5)*L32Aav + (833/15)*S23Aav - (204/5)*S32Aav - 8*L23Bav - (1/15)*S23Bav
		self.dch_a13STQC = dmu*(-(456/5)*L2Aav + (43/5)*L2Bav) + (2751/20)*L2Adiff - (51/5)*L2Bdiff + (4936/15)*S2Adiff + (4/15)*S2Bdiff - (856/5)*L23Adiff + (272/5)*L32Adiff + (833/15)*S23Adiff - (204/5)*S32Adiff - 8*L23Bdiff - (1/15)*S23Bdiff

	#function to compute Dy and De^2
	def Dy_De2_Dl_Ddl(self, y, e2, chi_eff, dchi, dchi2, sperp2):

		#compute different things that will be needed
		sqrt1me2 = (1-e2)**0.5
		one_m_sqrt = e2/(1 + sqrt1me2) #= 1 - np.sqrt(1 - e2)
		sqrt_a = one_m_sqrt/sqrt1me2
		e2sqrt = e2/sqrt1me2
		log_fact = np.log((1 + sqrt1me2)/(8*y*sqrt1me2*(1-e2)))
		y2 = y*y
		y8 = y**8
		chi_eff2 = chi_eff*chi_eff
		chi_effdchi = chi_eff*dchi

		#initialize the PN derivatives
		Dy  = 0.
		De2 = 0.
		k   = 0.
		
		#add the required terms
		if self.pn_phase_order>=9:
			Dy  += self.a9NSQC + log_fact*self.log_a9NSQC
			Dy  *= y
		if self.pn_max_order>=8:
			if self.pn_phase_order>=8:
				Dy  += self.a8NSQC + log_fact*self.log_a8NSQC
			if self.pn_spin_order>=8:
				Dy  += chi_eff*self.chi_a8SOQC + dchi*self.dch_a8SOQC + self.const_a8SSpQC + chi_eff2*(self.chi2_a8SSQC + self.chi2_a8SSpQC) + chi_effdchi*(self.chidch_a8SSQC + self.chidch_a8SSpQC) + dchi2*(self.dch2_a8SSQC + self.dch2_a8SSpQC) + sperp2*self.sperp2_a8SSpQC
			Dy  *= y
		if self.pn_max_order>=7:
			if self.pn_phase_order>=7:
				Dy  += self.a7NSQC
			if self.pn_spin_order>=7:
				dchi3 = dchi*(3*dchi2 - 2*dchi*dchi) #When precessing, this approximates <dchi^3> from <dchi^2> and <dchi>
				Dy  += chi_eff*self.chi_a7SOQC + dchi*self.dch_a7SOQC + self.const_a7SSpQC + chi_eff2*(self.chi2_a7SSQC + self.chi2_a7SSpQC) + chi_effdchi*(self.chidch_a7SSQC + self.chidch_a7SSpQC) + dchi2*(self.dch2_a7SSQC + self.dch2_a7SSpQC) + sperp2*self.sperp2_a7SSpQC + chi_eff2*(chi_eff*self.chi3_a7S3QC + dchi*self.chi2dch_a7S3QC) + chi_eff*dchi2*self.chidch2_a7S3QC + dchi3*self.dch3_a7S3QC
			Dy  *= y
		if self.pn_max_order>=6:
			if self.pn_phase_order>=6:
				Dy  += self.p_a6NS(e2) + sqrt_a*self.sqrt_a6NS(e2) + log_fact*self.log_a6NS(e2)
				De2 += e2*(self.p_b6NS(e2) + sqrt1me2*self.sqrt_b6NS(e2) + log_fact*self.log_b6NS(e2)) + one_m_sqrt*self.one_m_sqrt_b6NS
				k   += self.p_k4NS(e2) + sqrt1me2*self.sqrt_k4NS(e2)
			if self.pn_spin_order>=6:
				Dy  += chi_eff*self.chi_p_a6SO(e2) + dchi*self.dch_p_a6SO(e2) + chi_eff2*self.chi2_p_a6SS(e2) + chi_effdchi*self.chidch_p_a6SS(e2) + dchi2*self.dch2_p_a6SS(e2) + sqrt_a*(chi_eff2*self.chi2_sqrt_a6SS(e2) + chi_effdchi*self.chidch_sqrt_a6SS(e2) + dchi2*self.dch2_sqrt_a6SS(e2))
				Dy  += self.const_a6SSpQC + chi_eff2*self.chi2_a6SSpQC + chi_effdchi*self.chidch_a6SSpQC + dchi2*self.dch2_a6SSpQC + sperp2*self.sperp2_a6SSpQC #quasi-circular perpendicular spin corrections
				De2 += e2*(chi_eff*self.chi_p_b6SO(e2) + dchi*self.dch_p_b6SO(e2) + chi_eff2*self.chi2_p_b6SS(e2) + chi_effdchi*self.chidch_p_b6SS(e2) + dchi2*self.dch2_p_b6SS(e2) + sqrt1me2*(chi_eff2*self.chi2_sqrt_b6SS(e2) + chi_effdchi*self.chidch_sqrt_b6SS(e2) + dchi2*self.dch2_sqrt_b6SS(e2)))
				k   += chi_eff2*self.chi2_p_k4SS(e2) + chi_effdchi*self.chidch_p_k4SS(e2) + dchi2*self.dch2_p_k4SS(e2)
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=5:
			if self.pn_phase_order>=5:
				Dy  += self.p_a5NS(e2)
				De2 += e2*self.p_b5NS(e2)
			if self.pn_spin_order>=5:
				Dy  += chi_eff*(self.chi_p_a5SO(e2) + e2sqrt*self.chi_e2sqrt_a5SO(e2)) + dchi*(self.dch_p_a5SO(e2) + e2sqrt*self.dch_e2sqrt_a5SO(e2))
				De2 += e2*(chi_eff*(self.chi_p_b5SO(e2) + sqrt1me2*self.chi_sqrt_b5SO(e2)) + dchi*(self.dch_p_b5SO(e2) + sqrt1me2*self.dch_sqrt_b5SO(e2)))
				k   += chi_eff*self.chi_p_k3SO(e2) + dchi*self.dch_p_k3SO(e2)
				if self.horizon_absorption:
					cab5 = dchi*self.dch_c5H + chi_eff*(self.chi_c5H + dchi2*self.chidch2_c5H + chi_eff2*self.chi3_c5H)
					Dy  += self.pref_a5H(e2)*cab5
					De2 += e2*self.pref_b5H(e2)*cab5
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=4:
			if self.pn_phase_order>=4:
				Dy  += self.p_a4NS(e2) + sqrt_a*self.sqrt_a4NS(e2)
				De2 += e2*(self.p_b4NS(e2) + sqrt1me2*self.sqrt_b4NS(e2))
				k   += self.p_k2NS(e2)
			if self.pn_spin_order>=4:
				Dy  += self.const_a4SS(e2) + chi_eff2*self.chi2_a4SS(e2) + chi_effdchi*self.chidch_a4SS(e2) + dchi2*self.dch2_a4SS(e2) + sperp2*self.sperp2_a4SS(e2)
				De2 += self.const_b4SS(e2) + chi_eff2*self.chi2_b4SS(e2) + chi_effdchi*self.chidch_b4SS(e2) + dchi2*self.dch2_b4SS(e2) + sperp2*self.sperp2_b4SS(e2)
				k   += self.const_k2SS     + chi_eff2*self.chi2_k2SS     + chi_effdchi*self.chidch_k2SS     + dchi2*self.dch2_k2SS     + sperp2*self.sperp2_k2SS
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=3:
			if self.pn_phase_order>=3:
				Dy  += self.p_a3NS(e2)
				De2 += e2*self.p_b3NS(e2)
			if self.pn_spin_order>=3:
				Dy  += chi_eff*self.chi_p_a3SO(e2) + dchi*self.dch_p_a3SO(e2)
				De2 += e2*(chi_eff*self.chi_p_b3SO(e2) + dchi*self.dch_p_b3SO(e2))
				k   += chi_eff*self.chi_k1SO + dchi*self.dch_k1SO
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_phase_order>=2:
			Dy  = y2*(Dy  + self.p_a2NS(e2))
			De2 = y2*(De2 + e2*self.p_b2NS(e2))
			k   = y2*(k   + self.k0NS)
		
		#if required, add the tidal terms
		DyTidal  = 0.
		De2Tidal = 0.
		kTidal   = 0.
		if self.pn_tidal_order>=15:
			DyTidal  = y*(DyTidal  + self.p_a15T(e2))
			De2Tidal = y*(De2Tidal + self.p_b15T(e2))
		if self.pn_tidal_order>=14:
			DyTidal  = y*(DyTidal  + self.p_a14T(e2) + sqrt_a*self.sqrt_a14T(e2))
			De2Tidal = y*(De2Tidal + self.p_b14T(e2) + sqrt1me2*self.sqrt_b14T(e2))
			kTidal   =y2*(kTidal   + self.p_k12T(e2) + sqrt1me2*self.sqrt_k12T(e2))
		if self.pn_tidal_order>=13:
			DyTidal  = y*(DyTidal  + self.p_a13T(e2) + chi_eff*self.chi_a13STQC + dchi*self.dch_a13STQC)
			De2Tidal = y*(De2Tidal + self.p_b13T(e2))
		if self.pn_tidal_order>=12:
			DyTidal  =y2*(DyTidal  + self.p_a12T(e2) + sqrt_a*self.sqrt_a12T(e2))
			De2Tidal =y2*(De2Tidal + self.p_b12T(e2) + sqrt1me2*self.sqrt_b12T(e2))
			kTidal   =y2*(kTidal   + self.p_k10T(e2) + sqrt1me2*self.sqrt_k10T(e2))
		if self.pn_tidal_order>=10:
			Dy  +=    y2*y8*(DyTidal  + self.p_a10T(e2) + sqrt_a*self.sqrt_a10T(e2))
			De2 += e2*y2*y8*(De2Tidal + self.p_b10T(e2) + sqrt1me2*self.sqrt_b10T(e2))
			k   +=    y2*y8*(kTidal   + self.p_k8T(e2))

		#always take into account the 0PN terms of Dy and De2
		Dy = y*y8*self.nu*(self.p_a0NS(e2) + Dy)
		De2 = -y8*self.nu*(e2*self.p_b0NS(e2) + De2)

		#compute D\lambda from Eq.(103)
		Dl = y2*y

		#compute D\delta\lambda from Eq.(104)
		Ddl = k*Dl/(1+k)

		return Dy, De2, Dl, Ddl

#compute the series expansion of the tLO integral for x->0 (here x=e**2)
# F = (24/19)*x**(-24/19)*((1 + (121/304)*x)**(-3480/2299))*sqrt(1-x)*integral((x**(5/19))*((1 + (121/304)*x)**(1181/2299))*((1 - x)**(-3/2)))
def F_tLO_series_at_0(x):
	coefs = np.array([1.000000000000000, -0.1511627906976744, 0.2656836084021005, 0.007463780007501875, 0.08800790590714085, 0.03153077124184580, 0.04185392210761341, 0.02761371124737777, 0.02642686119635599, 0.02145414169943131, 0.01926563742403364, 0.01677217450181226, 0.01503898450331388, 0.01347003088516929, 0.01220348405026613, 0.01110346195121149, 0.01016749330223870, 0.009352447459389354, 0.008642114232972108, 0.008016709107501086, 0.007463457161231162,0.006970902964332086, 0.006530253812462707, 0.006134111124121483, 0.005776451398258840, 0.005452229768143720, 0.005157232928828976, 0.004887901117379935, 0.004641215172072290, 0.004414596725230578,0.004205833180162198, 0.004013015784562471, 0.003834490347048246,0.003668816871424952, 0.003514736646109113, 0.003371145103099870, 0.003237069368178693, 0.003111649567641214, 0.002994123194217794, 0.002883811964918853, 0.002780110725151045, 0.002682478039498533,0.002590428180410570, 0.002503524280447905, 0.002421372457429333,0.002343616756405161, 0.002269934780185545, 0.002200033902499887, 0.002133647975961232, 0.002070534461714599, 0.002010471919657125])
	return np.polyval(np.flip(coefs),x)

#compute the series expansion of the tLO integral for x->1 (here x=e**2)
def F_tLO_series_at_1(x):
	
	#we are going to approximate the integral of ((1 - 1/u**2)**(5/19))*(1 - (121/425)/u**2)**(1181/2299) - 1 as -f0 + sum_{n=1}^{nmax} cn*u**-(2*n-1)
	cns = np.array([0.40941176470588236, 0.02286320645905421, 0.008142925951557094, 0.004155878512401501, 0.0024847568765827364, 0.001633290511588348, 0.0011455395465032605, 0.000842577094447224, 0.0006427195446513564, 0.0005045715814217113, 0.00040544477831148924, 0.0003321116545345162, 0.0002764636962467965, 0.00023331895675436905, 0.0001992473575067662, 0.00017190919726595898, 0.00014966654751355843, 0.000131346469484909, 0.00011609207361015848, 0.00010326617525262309, 9.238740896587563e-05, 8.308691874613337e-05, 7.50784086790562e-05, 6.813705814151314e-05, 6.20844345948999e-05, 5.677753690123865e-05, 5.210072977646673e-05, 4.7959732146031274e-05, 4.427708468062032e-05, 4.0988696117937945e-05, 3.80411855904995e-05, 3.538981870077757e-05, 3.2996890965966614e-05, 3.083045152872919e-05, 2.886328796018351e-05, 2.707211306399751e-05, 2.543690918050699e-05, 2.3940396192787112e-05, 2.256759736012561e-05, 2.1305483020854193e-05, 2.014267666038337e-05, 1.906921121897629e-05, 1.8076326095558655e-05, 1.7156297290322297e-05, 1.6302294667351972e-05, 1.550826151746742e-05, 1.4768812541410176e-05, 1.407914711455732e-05])
	
	#evaluate polynomial
	u = 1-x
	return ((48/19)*((425/304)**(1181/2299)))*(1 - 1.4555165803216864*np.sqrt(u) + u*np.polyval(np.flip(cns),u))*(x**(-24/19))*((1 + (121/304)*x)**(-3480/2299)) 

#function to choose series at x=0 or at x=1 (here x=e**2)
def F_tLO_series(x, x_thr=0.4):
	
	#distinguish case in which x is an array or not
	if np.ndim(x)==0:
		#if x small, use series expansion at 0, otherwise use series expansion at 1
		if x<=x_thr: return F_tLO_series_at_0(x)
		else: return F_tLO_series_at_1(x)
		
	else:
		#make sure x is a numpy array of floats
		x = np.asarray(x, dtype=float)

		#if x small, use series expansion at 0, otherwise use series expansion at 1
		F = np.zeros_like(x)
		i_low = x<=x_thr
		i_high = np.logical_not(i_low)
		if np.any(i_low):  F[i_low] = F_tLO_series_at_0(x[i_low])
		if np.any(i_high): F[i_high] = F_tLO_series_at_1(x[i_high])
	
		return F

#function to compute Newtonian time
def tLO_func(y, e2, m1, m2):

	#compute mass related stuff
	M, _, _, nu, _ = mass_params_from_m1_m2(m1, m2)

	#return the LO time
	return -(5/256)*(M/nu)*(y**-8)*((1-e2)**-0.5)*F_tLO_series(e2)

#function to compute the PN expansion parameter y from the orbital frequency and the squared eccentricity using Eq.(5)
def y_of_forb(f_orb, e2, M):
	return ((2*np.pi*M*f_orb)**(1./3.))/np.sqrt(1 - e2)

#function to compute the orbital frequency from the PN expansion parameter y and the squared eccentricity inverting Eq.(5)
def forb_of_y(y, e2, M):
	return (y**3)*((1 - e2)**1.5)/(2*np.pi*M)

#function to compute squared eccentricity as a function of frequency using the leading order (0PN) expressions. See e.g. Eq.(1.2) of 1605.00304.
def compute_e2_of_f_0PN(f, e20, f0, atol=1e-14, maxiter_Newton=6):

	#circular binaries (e0=0) stay circular; mask them so f0bar!=0 avoids 0-division
	circular = np.equal(e20, 0.0)

	#if all circular, there is nothing to solve
	if np.all(circular): return np.zeros_like(e20)

	e20_safe = np.where(circular, 0.5, e20)

	#compute dimensionless frequency
	fbar = (f/f0)*(e20_safe**(-9/19))*((1 - e20_safe)**(3/2))*((1 + (121/304)*e20_safe)**(-1305/2299))

	#compute LO inverse for e2<<1
	x_0 = fbar**(-19/9)
	
	#compute LO inverse for (1-e2)<<1
	x_1 = ((425/304)**(870/2299))*(fbar**(2/3))

	#compute approximate inverse using {2,2} and {1,3} Pades of inverse series of x_0 and x_1 respectively
	e2 = np.where(fbar>1.57,
	              x_0*(1 + x_0*(1346419979/308389608))/(1 + x_0*((4940161597/616779216) + x_0*(6051616442057/562502644992))),
	              1 - x_1/(1 + x_1*(36/85 + x_1*(3018/36125 + x_1*185532/15353125))))
	
	#apply Newton rhapson
	for iNewton in range(maxiter_Newton):
		
		#compute f(e2) and df(e2)
		fi = (e2**(-9/19))*((1 - e2)**(3/2))*((1 + (121/304)*e2)**(-1305/2299)) - fbar
		dfi = (-3/608)*(96 + e2*(292 + e2*37))*(e2**(-28/19))*((1 - e2)**(1/2))*((1 + (121/304)*e2)**(-3604/2299))
		
		#compute how much we have to shift e2
		de2 = -(fi/dfi)
		
		#compute new value of e2
		e2 = e2 + de2
		
		#if all values of e2 are below tolerance, break loop
		if np.all(np.abs(de2)<atol): break

	return np.where(circular, 0.0, e2)

#function to compute the matter-effect parameters of a neutron star from its dimensionless mass quadrupolar tidal deformability Lambda2 using quasi-universal relations
#returns a dictionary with the pyEFPE keys ('q', 'o', 'Lambda2', 'Lambda3', 'Sigma2', 'Lambda23', 'Lambda32', 'Sigma23', 'Sigma32'), the compactness 'C' (=G m/(c^2 R)) and the dimensionless moment of inertia 'I' (=I_moment/m^3)
#if companion (1 or 2) is given, the keys follow the pyEFPE naming for that companion ('q1', 'o1', 'Lambda2_1', 'Lambda3_1', 'Sigma2_1', ..., 'C_1', 'I_1') so that they can be merged into the pyEFPE parameter dictionary, dropping 'C_i' and 'I_i', which pyEFPE does not use as parameters
#Lambda2=0 is interpreted as a black hole (q=o=1, all deformabilities 0, C=1/2, I=nan); the fits are valid for 2.3<=Lambda2<=1e4, and outside that range they are evaluated at its closest edge (with a warning)
#the pyEFPE Sigma2 is the current-quadrupole polarizability sigma^(2)/m^5 of Eq.(1.6b) of arXiv:2005.13367, which enters the phase as Eq.(18) of arXiv:1805.01487, i.e. it is the same quantity (sign included) as the Sigma2 of arXiv:1807.08016 and arXiv:2210.09425, and it is negative for the (physical) irrotational fluid
#the rotational tidal Love numbers Lambda23, Lambda32, Sigma23 and Sigma32 are set to 0, since there is no fit in terms of Lambda2 in the convention of arXiv:1805.01487
#a user having their own values can use the hidden symmetry Sigma32 = 4 Lambda23, Lambda32 = 2 Sigma23 (Eq.(28) of arXiv:2103.16595, Eq.(3) of arXiv:2204.12510) to fill the missing ones
def neutron_star_quasi_universal_relations(Lambda2, companion=None):

	#the fits are only valid for Lambda2_min<=Lambda2<=Lambda2_max, and extrapolated outside that range their polynomials diverge, so Lambda2 is clamped
	Lambda2_min, Lambda2_max = 2.3, 1e4
	Lambda2_fit = min(max(Lambda2, Lambda2_min), Lambda2_max)

	#check the input
	if Lambda2<0: raise ValueError('Lambda2 has to be non-negative, but Lambda2=%s was given'%(Lambda2))
	if (Lambda2>0) and ((Lambda2<Lambda2_min) or (Lambda2>Lambda2_max)): warnings.warn('The neutron star quasi-universal relations are only valid for %g<=Lambda2<=%g, but Lambda2=%s was given; evaluating them at Lambda2=%g instead'%(Lambda2_min, Lambda2_max, Lambda2, Lambda2_fit), UserWarning, stacklevel=2)

	#Lambda2=0 is a black hole (the moment of inertia of a black hole is not defined)
	if Lambda2==0:
		result = {'q': 1.0, 'o': 1.0, 'Lambda2': 0.0, 'Lambda3': 0.0, 'Sigma2': 0.0, 'Lambda23': 0.0, 'Lambda32': 0.0, 'Sigma23': 0.0, 'Sigma32': 0.0, 'C': 0.5, 'I': np.nan}
	else:
		#all the fits are polynomials P(x) = sum_k a_k x^k in the natural logarithm x = ln(Lambda2_fit), we use np.polyval which takes the coefficients in reverse order
		x = np.log(Lambda2_fit)

		#Love-Q relation ln(Q) = P(ln(Lambda2)) with Q = -Q_quad/(M^3 chi^2) the Kerr normalized quadrupole: Eq.(15) and Table 1 (Q-lambda2 row) of arXiv:1608.02582
		lnq = np.polyval([1.245e-4, -4.283e-3, 0.04812, 0.09163, 0.1940], x)

		#S3-Q relation ln(S3) = P(ln(Q)) with S3 = -S_3/(M^4 chi^3) the Kerr normalized spin octupole: Eq.(15) form and Table 2 (S3-M2 row) of arXiv:1608.02582, also Eq.(39) of NRTidalv3 (arXiv:2311.07456)
		lno = np.polyval([-0.03309, 0.2458, -0.7152, 2.071, 3.131e-3], lnq)

		#Love-I relation ln(I) = P(ln(Lambda2)) with I = I_moment/M^3 the dimensionless moment of inertia: Eq.(15) and Table 1 (I-lambda2 row) of arXiv:1608.02582
		lnI = np.polyval([8.345e-6, -6.953e-4, 0.02238, 0.05951, 1.496], x)

		#Lambda3-Lambda2 relation ln(Lambda3) = P(ln(Lambda2)): Eq.(9) and Table II ("This Work" row) of arXiv:2210.09425 (~6000 EoSs, 11% maximum error)
		lnLambda3 = np.polyval([3.494e-5, -1.117e-3, 1.374e-2, -8.170e-2, 2.492e-1, 9.442e-1, -1.163], x)

		#|Sigma2|-Lambda2 relation ln(|Sigma2|) = P(ln(Lambda2)) for an irrotational fluid: Eq.(9) and Table II ("This Work" row) of arXiv:2210.09425 (3% maximum error)
		lnSigma2 = np.polyval([-5.192e-7, 2.643e-5, -5.03e-4, 4.096e-3, 7.609e-4, 4.821e-1, -2.019], x)

		#C-Lambda2 relation C = P(ln(Lambda2)): Eq.(9) and Table III ("This Work" row) of arXiv:2210.09425 (4% maximum error)
		C = np.polyval([-1.606e-6, 4.463e-5, -4.491e-4, 1.941e-3, -2.243e-3, -3.727e-2, 3.663e-1], x)

		result = {'q': np.exp(lnq), 'o': np.exp(lno), 'Lambda2': Lambda2, 'Lambda3': np.exp(lnLambda3), 'Sigma2': -np.exp(lnSigma2), 'Lambda23': 0.0, 'Lambda32': 0.0, 'Sigma23': 0.0, 'Sigma32': 0.0, 'C': C, 'I': np.exp(lnI)}

	#if a companion is given, use the pyEFPE naming of its parameters, which has no underscore for q and o
	if companion is None: return result
	else: return {(key if key in ('q', 'o') else key+'_')+str(companion): value for key, value in result.items()}
