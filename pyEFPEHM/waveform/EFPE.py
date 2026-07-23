"""
Main waveform generation module.

Copyright (c) 2026 Gonzalo Morras (gonzalo.morras@aei.mpg.de)

This file is part of pyEFPEHM.
Licensed under the Apache License. See the LICENSE file in the project root for details.
"""

import numpy as np
import warnings

from pyEFPEHM.utils.utils import *
from pyEFPEHM.utils.wigner import *
from pyEFPEHM.utils.cython_utils import my_cgroup_idxs_by_vals
from pyEFPEHM.waveform.functions import *
from pyEFPEHM.waveform.amplitudes import *
from scipy.interpolate import CubicSpline

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]
Mpc_s = 1.02927125054339e14 #Mpc/c in [s]

#Class to compute a post-Newtonian eccentric and precessing waveform. We mostly follow arXiv:2604.11903, arxiv:2502.03929, arxiv:2106.10291 and arXiv:1801.08542, and for the amplitudes, arxiv:2507.00169
class pyEFPE:

	def __init__(self, parameters):
	
		r"""
		Initialize the class with the given parameters.
		
		``parameters`` is a dictionary in which we expect to find the following
		
		Parameters

		-----------  Physical parameters for waveform generation -----------

		mass1 : float
			Mass of companion 1, in solar masses - Required
		mass2 : float
			Mass of companion 2, in solar masses - Required
		eccentricity : float
			Eccentricity at reference frequency - Default: 0
		spin1x : float
			x-component of dimensionless spin of companion 1 at reference frequency - Default: 0
		spin1y : float
			y-component of dimensionless spin of companion 1 at reference frequency - Default: 0
		spin1z : float
			z-component of dimensionless spin of companion 1 at reference frequency - Default: 0
		spin2x : float
			x-component of dimensionless spin of companion 2 at reference frequency - Default: 0
		spin2y : float
			y-component of dimensionless spin of companion 2 at reference frequency - Default: 0
		spin2z : float
			z-component of dimensionless spin of companion 2 at reference frequency - Default: 0
		distance : float
			Distance to the source, in Mpc - Default: 100 Mpc
		inclination : float
			Angle between orbital angular momentum and vector from binary to observer (N) at reference frequency.
			Measured in radians, between 0 and Pi. - Default: 0
		f22_start : float
			Starting waveform generation frequency of 22 mode, in Hz 
			We have f_orbital_start = 0.5*f22_start - Default: 20 Hz
		f22_ref : float
			Reference waveform frequency of 22 mode, in Hz
			We have f_orbital_ref = 0.5*f22_ref. If None, then f22_ref = f22_start. - Default: None
		f22_end : float or None
			Maximum 22 mode frequency the orbital motion is computed up to, in Hz
			If it is None, we compute all the way to the ISCO. Otherwise, we have f_orbital_end = min(0.5*f22_end, f_ISCO) - Default: None
		max_duration : float or None
			Maximum duration (in seconds) up to which to compute the orbital motion, measured from f22_start
			If f22_ref > f22_start, max_duration must be large enough for the evolution to reach f22_ref, otherwise a ValueError is raised
			If it is None, we compute all the way to the ISCO - Default: None
		phase : float
			Reference orbital phase, in radians - Default: 0
		mean_anomaly : float
			Reference mean anomaly of quasi-Keplerian parametrization - Default: 0
			(The keys e_start, phi_start and mean_anomaly_start are accepted as deprecated aliases of eccentricity, phase and mean_anomaly)
		mode_array: array_like of shape (N,2) or None
			Array containing [l, m] GW modes to take into account. Only 1PN modes are implemented ([[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]])
			Repeated modes and modes that are inconsistent with the pn_amplitude_order are removed.
			If mode_array is None, choose all available GW modes.
			IMPORTANT: m here is the co-precessing-frame multipolar index (called m in Eq.(3) of arXiv:2604.11903), not the inertial-frame
			spin-weighted spherical-harmonic index m' (called mp in generate_hlm_modes/generate_tdomain_hlm_modes). For a non-precessing binary
			the two coincide; for a precessing binary they generally do not, since a single co-precessing (l,m) mode "twists up" into
			contributions at every inertial-frame m'=-l..l. harmonic_array's (l,m,n) triples use this same co-precessing m as their second entry.
			- Default: [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]

		-----------  Matter effects parameters for waveform generation -----------

		q1 : float
			quadrupole parameter of companion 1 (see arXiv:1801.08542) - Default: 1
		q2 : float
			quadrupole parameter of companion 2 - Default: 1
		o1 : float
			octupole parameter of companion 1 (see arXiv:1411.4118, where it is called $\lambda_1$) - Default: 1
		o2 : float
			octupole parameter of companion 2 - Default: 1
		Lambda2_1: float
			Dimensionless mass quadrupolar tidal deformability of companion 1 - Default: 0
		Lambda2_2: float
			Dimensionless mass quadrupolar tidal deformability of companion 2 - Default: 0
		Lambda3_1: float
			Dimensionless mass octupolar tidal deformability of companion 1 - Default: 0
		Lambda3_2: float
			Dimensionless mass octupolar tidal deformability of companion 2 - Default: 0
		Sigma2_1: float
			Dimensionless current quadrupolar tidal deformability of companion 1 - Default: 0
		Sigma2_2: float
			Dimensionless current quadrupolar tidal deformability of companion 2 - Default: 0
		Lambda23_1: float
			Dimensionless rotational magnetic octupole-induced mass quadrupolar tidal deformability of companion 1 - Default: 0
		Lambda23_2: float
			Dimensionless rotational magnetic octupole-induced mass quadrupolar tidal deformability of companion 2 - Default: 0
		Lambda32_1: float
			Dimensionless rotational magnetic quadrupole-induced mass octupolar tidal deformability of companion 1 - Default: 0
		Lambda32_2: float
			Dimensionless rotational magnetic quadrupole-induced mass octupolar tidal deformability of companion 2 - Default: 0
		Sigma23_1: float
			Dimensionless rotational electric octupole-induced current quadrupolar tidal deformability of companion 1 - Default: 0
		Sigma23_2: float
			Dimensionless rotational electric octupole-induced current quadrupolar tidal deformability of companion 2 - Default: 0
		Sigma32_1: float
			Dimensionless rotational electric quadrupole-induced current octupolar tidal deformability of companion 1 - Default: 0
		Sigma32_2: float
			Dimensionless rotational electric quadrupole-induced current octupolar tidal deformability of companion 2 - Default: 0
			
		----------- Optional parameters for solving and computing stuff -----------

		pn_phase_order: int
			Twice the Post-Newtonian order to use in the non-spinning part of the phasing (i.e. pn_phase_order=n corresponds to the (n/2)PN order).
			pn_phase_order=-1 selects the maximum pn order available (=9) - Default: 9
		pn_spin_order: int
			Twice the Post-Newtonian order to use in the spinning part of the phasing (i.e. pn_spin_order=n corresponds to the (n/2)PN order).
			pn_spin_order=-1 selects the maximum pn order available (=8) - Default: 8
		pn_amplitude_order: int
			Twice the Post-Newtonian order to use in the amplitudes of the waveform (i.e. pn_amplitude_order=n corresponds to the (n/2)PN order).
			pn_amplitude_order=-1 selects the maximum pn order available (=2) - Default: 2
		pn_tidal_order: int
			Twice the Post-Newtonian order to use in the tidal part of the phasing (i.e. pn_tidal_order=n corresponds to the (n/2)PN order).
			Tidal effects start at pn_tidal_order=10 and pn_tidal_order=-1 selects the maximum pn order available (=15) - Default: 0
		horizon_absorption: bool
			Flag to choose whether to take into account horizon absorption effects. Requires pn_spin_order>=5 (HA enters at 2.5PN); it has no effect otherwise - Default: True
		Amplitude_tol: float
			Tolerance in time-domain amplitude. It will control how many Fourier modes are taken into account - Default: 1e-4
		Amplitude_pmax: int
			Maximum eccentric harmonic index |p| considered when selecting Fourier modes (modes with p in [-Amplitude_pmax, Amplitude_pmax] are candidates) - Default: 100
		harmonic_array: array_like of shape (N,3) or None
			Array containing the (l, m, n) harmonics to compute (throught the whole waveform).
			If m is negative, the harmonic label in the mode by mode output is transformed by doing (l, m, n)->(l, -m, -n).
			If None, harmonics are chosen as a function of time by the model to satisfy Amplitude_tol - Default: None
		DJ2_tol : float
			Tolerance in J**2 estimate when setting initial conditions - Default: 1e-10
		RR_sol_rtol: float or array-like, shape (8,)
			Relative tolerance when integrating ODEs.
			If array, relative tolerances on [y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
			- Default: [1e-10, 1e-10, 1e-12, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12]
		RR_sol_atol: float or array-like, shape (8,)
			Absolute tolerance when integrating ODEs.
			If array, absolute tolerances on [y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
			- Default: [1e-12, 1e-12,  1e-8,  1e-8, 1e-12,  1e-8,  1e-6,  1e-6]
		NecessaryModes_UpdatePeriod: int
			Number of Runge-Kutta iterations between updates to necessary of modes - Default: 4
		Series_Reversion_Order: int
			Order up to which we perform series reversion when computing interpolant of stationary times - Default: 5
		Interpolate_Amplitudes: bool
			Choose whether or not to interpolate the amplitudes to speed up waveform evaluation - Default: True
		Interp_points_per_prec_cycle: int
			Number of points per precession cycle used to interpolate the precession amplitudes - Default: 40
		Extra_interp_points_Nlm_p: int
			Number of extra points over the Runge-Kutta segments used to interpolate each Fourier mode amplitude - Default: 0
		SPA_Frequency_rtol: float
			Minimum accepted value of (f(t_SPA) - f)/f when solving for t_SPA - Default: 1e-12
		SUA_kmax: int
			Maximum order kmax to compute the constants a_{k,kmax} (Eq.(127-128) of arXiv:2106.10291) that appear in the Shifted Uniform Asymptotics (SUA) method. - Default: 1
		Compute_hlm_Modes: bool
			Choose whether or not to also set up the interpolants needed by generate_hlm_modes/generate_tdomain_hlm_modes (the raw, pre spherical-harmonic-projection multipole moments).
			Only takes effect when Interpolate_Amplitudes=True; leave False if you never call those methods, to avoid the extra interpolation cost - Default: False
		"""
		#dictionary with default params
		params = {'eccentricity': 0,
		          'spin1x': 0,
		          'spin1y': 0,
		          'spin1z': 0,
		          'spin2x': 0,
		          'spin2y': 0,
		          'spin2z': 0,
		          'distance': 100,
		          'inclination': 0,
		          'f22_start': 20,
		          'f22_ref': None,
		          'f22_end': None,
		          'max_duration': None,
		          'phase': 0,
		          'mean_anomaly': 0,
		          'mode_array': [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]],
		          'q1': 1, 'q2': 1, 'o1': 1, 'o2': 1,
		          'Lambda2_1': 0, 'Lambda2_2': 0, 'Lambda3_1': 0, 'Lambda3_2': 0, 'Sigma2_1': 0, 'Sigma2_2': 0,
		          'Lambda23_1': 0, 'Lambda23_2': 0, 'Lambda32_1': 0, 'Lambda32_2': 0, 'Sigma23_1': 0, 'Sigma23_2': 0, 'Sigma32_1': 0, 'Sigma32_2': 0,
		          'pn_phase_order': 9,
		          'pn_spin_order': 8,
		          'pn_amplitude_order': 2,
		          'pn_tidal_order': 0,
		          'horizon_absorption': True,
		          'Amplitude_tol': 1e-4,
		          'Amplitude_pmax': 100,
		          'harmonic_array': None,
		          'DJ2_tol': 1e-10,
		          'RR_sol_rtol': [1e-10, 1e-10, 1e-12, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12],
		          'RR_sol_atol': [1e-12, 1e-12,  1e-8,  1e-8, 1e-12,  1e-8,  1e-6,  1e-6],
		          'NecessaryModes_UpdatePeriod': 4,
		          'Series_Reversion_Order': 5,
		          'Interpolate_Amplitudes': True,
		          'Interp_points_per_prec_cycle': 40,
		          'Extra_interp_points_Nlm_p': 0,
		          'SPA_Frequency_rtol': 1e-12,
		          'SUA_kmax': 1,
		          'Compute_hlm_Modes': False,
		         }
		
		#validate the input parameters and merge them with the defaults (maps deprecated *_start keys, warns on typos / out-of-range values, raises on invalid masses or eccentricity)
		params = self._validate_parameters(parameters, params)

		#save parameters
		self.params = params
		
		#extract things from the parameter dictionary
		#we convert the masses from Msun to seconds
		self.m1 = t_sun_s*params['mass1']
		self.m2 = t_sun_s*params['mass2']
		self.e_ref = params['eccentricity']
		#Extract the dimensionless component spins (S_i/mu_i^2 = s_i/mu_i)
		self.spin0_1 = np.array([params['spin1x'], params['spin1y'], params['spin1z']])
		self.spin0_2 = np.array([params['spin2x'], params['spin2y'], params['spin2z']])
		self.q1, self.q2 = params['q1'], params['q2']
		self.o1, self.o2 = params['o1'], params['o2']
		self.Lambda2_1,  self.Lambda2_2  =  params['Lambda2_1'],  params['Lambda2_2']
		self.Lambda3_1,  self.Lambda3_2  =  params['Lambda3_1'],  params['Lambda3_2']
		self.Sigma2_1,   self.Sigma2_2   =   params['Sigma2_1'],   params['Sigma2_2']
		self.Lambda23_1, self.Lambda23_2 = params['Lambda23_1'], params['Lambda23_2']
		self.Lambda32_1, self.Lambda32_2 = params['Lambda32_1'], params['Lambda32_2']
		self.Sigma23_1,  self.Sigma23_2  =  params['Sigma23_1'],  params['Sigma23_2']
		self.Sigma32_1,  self.Sigma32_2  =  params['Sigma32_1'],  params['Sigma32_2']
		#convert the input luminosity distance from Mpc to s
		self.dL = Mpc_s*params['distance']
		self.inclination = params['inclination']
		#obtain initial, reference, and final orbital frequencies from initial and final 22 mode GW frequencies (f_orb = 0.5*f22)
		self.f0_orb = 0.5*params['f22_start']
		if params['f22_ref'] is None: self.fref_orb = self.f0_orb
		else: self.fref_orb = 0.5*params['f22_ref']
		if params['f22_end'] is None: self.ff_orb = None
		else: self.ff_orb = 0.5*params['f22_end']
		self.phi_ref = params['phase']
		self.phi_e_ref = params['mean_anomaly']
		if params['mode_array'] is None: params['mode_array'] = [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]
		self.mode_array = clean_mode_array(params['mode_array'], pn_amplitude_order=params['pn_amplitude_order'])
		self.harmonic_array, self.mode_array = process_harmonic_array(params['harmonic_array'], self.mode_array, pn_amplitude_order=params['pn_amplitude_order'])
		
		#if m2>m1, flip everything
		if self.m2>self.m1:
			self.m1, self.m2 = self.m2, self.m1
			self.spin0_1, self.spin0_2 = self.spin0_2, self.spin0_1
			self.phi_ref = self.phi_ref + np.pi #flip orbital phase (r -> -r under relabeling). Mean anomaly phi_e_ref is unchanged (radial phase from periastron)
			self.q1, self.q2 = self.q2, self.q1
			self.o1, self.o2 = self.o2, self.o1
			self.Lambda2_1,  self.Lambda2_2  =  self.Lambda2_2,  self.Lambda2_1
			self.Lambda3_1,  self.Lambda3_2  =  self.Lambda3_2,  self.Lambda3_1
			self.Sigma2_1,   self.Sigma2_2   =   self.Sigma2_2,   self.Sigma2_1
			self.Lambda23_1, self.Lambda23_2 = self.Lambda23_2, self.Lambda23_1
			self.Lambda32_1, self.Lambda32_2 = self.Lambda32_2, self.Lambda32_1
			self.Sigma23_1,  self.Sigma23_2  =  self.Sigma23_2,  self.Sigma23_1
			self.Sigma32_1,  self.Sigma32_2  =  self.Sigma32_2,  self.Sigma32_1
		
		#compute initial conditions for RK, in the following, v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
		#also obtain the effective (possibly clamped) reference frequency, the 0PN squared eccentricity estimate at f22_start and the clamping flag
		self.v_ini, self.yf, self.s2_1, self.s2_2, self.MSA, self.cos_theta_JN, self.phi_JN, self.fref_orb_eff, self.e2_0_est, self.ref_clamped = initial_conditions_for_RR_eqs(self.f0_orb, self.fref_orb, self.ff_orb, self.e_ref, self.m1, self.m2, self.spin0_1, self.spin0_2, self.inclination, self.phi_ref, self.phi_e_ref, DJ2_tol=params['DJ2_tol'], pn_spin_order=params['pn_spin_order'])

		#initialize class to compute PN derivatives
		self.PN_derivatives = pyEFPE_PN_derivatives(self.m1, self.m2, self.s2_1, self.s2_2, q1=self.q1, q2=self.q2, o1=self.o1, o2=self.o2,
		  Lambda2_1=self.Lambda2_1, Lambda2_2=self.Lambda2_2, Lambda3_1=self.Lambda3_1, Lambda3_2=self.Lambda3_2, Sigma2_1=self.Sigma2_1, Sigma2_2=self.Sigma2_2,
		  Lambda23_1=self.Lambda23_1, Lambda23_2=self.Lambda23_2, Lambda32_1=self.Lambda32_1, Lambda32_2=self.Lambda32_2,
		  Sigma23_1=self.Sigma23_1, Sigma23_2=self.Sigma23_2, Sigma32_1=self.Sigma32_1, Sigma32_2=self.Sigma32_2,
		  pn_phase_order=params['pn_phase_order'], pn_spin_order=params['pn_spin_order'], pn_tidal_order=params['pn_tidal_order'], horizon_absorption=params['horizon_absorption'])
		
		#initialize class to compute Wigner D matrices
		self.Wigner = WignerD(self.mode_array)

		#list of spin -2 spherical harmonics _{-2}Y_{l,m'}
		l_array      = np.unique(self.mode_array[:,0])
		self.m2_Ylmp = compute_m2_Ylm(self.cos_theta_JN, self.phi_JN, l_array=l_array)
		#convert to dictionary where the keys are the values of l
		self.m2_Ylmp = {l: self.m2_Ylmp[il] for il, l in enumerate(l_array)}
		#loop over modes to compute the projector into polarizations of each
		self.Apc_projs = []
		for (l, m) in self.mode_array:
			
			#compute also (-1)^{l + m' + m} conj(Y_{l -m'}), which will be used to compute polarizations
			m2_Ylmp_mod = np.conj(self.m2_Ylmp[l][::-1])
			#apply (-1)^{m' + m + l} using that at i=0 -> m'=-l
			if (m%2)==0: m2_Ylmp_mod[1::2] = -m2_Ylmp_mod[1::2]
			else       : m2_Ylmp_mod[::2]  = -m2_Ylmp_mod[::2]

			#compute 0.5*[1,-1j]*(_{-2}Y_{l,m'} \pm (-1)^{m' + m + l} conj(Y^{l -m'})) to proyect h^{l m'} GW modes into [hp, hc] polarizations
			Ap_proj =   0.5*(self.m2_Ylmp[l] + m2_Ylmp_mod)
			Ac_proj = -0.5j*(self.m2_Ylmp[l] - m2_Ylmp_mod) #change the sign of hc to go to LAL convention
			self.Apc_projs.append(np.transpose([Ap_proj, Ac_proj]))
		
		#Compute constant part of h0 from Eq.(20) of 2402.06804, i.e. we want to compute  4\sqrt{\pi/5} M \nu/d_L
		self.h0_pref  = compute_h0_pref(self.m1, self.m2, self.dL)
		
		#compute the solution of v(t)=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0] from Eqs.(101-109) of 2106.10291
		self.sol      = solve_ivp_RR_eqs_t(self.v_ini, self.PN_derivatives, self.MSA, self.f0_orb, self.fref_orb_eff, self.yf, self.e2_0_est, self.ref_clamped,
		                              rtol=params['RR_sol_rtol'], atol=params['RR_sol_atol'], max_duration=params['max_duration'])
		
		#compute the SUA constants a_{k,k_max} by solving a system similar to Eq.(127-128) of arXiv:2106.10291
		self.ak_SUA = compute_ak_SUA(params['SUA_kmax'])

		#################### Computation of the necessary modes ####################

		#if a harmonic_array was provided, set the necessary modes from it
		if self.harmonic_array is not None:
		
			#extract the ps from harmonic_array
			ps = self.harmonic_array[:,2]
			
			#extract the mode indices
			index_map = {tuple(mode): i for i, mode in enumerate(self.mode_array)}
			mode_idxs = np.array([index_map[tuple(mode)] for mode in self.harmonic_array[:,:2]])
		
			#have the user-specified harmonics active at all times
			N_interp_segments = len(self.sol.ts)
			self.necessary_multipole_idxs = np.tile(mode_idxs, N_interp_segments)
			self.necessary_ps = np.tile(ps, N_interp_segments)
			self.mode_interp_idx = np.repeat(np.arange(N_interp_segments), len(ps))

		#Otherwise, finf the necessary modes as a function of time
		else:

			#compute the necessary modes at each segment of the interpolant
			self.necessary_multipole_idxs, self.necessary_ps, self.mode_interp_idx = [], [], []
			pmin, pmax = -params['Amplitude_pmax'], params['Amplitude_pmax']
			mmin, mmax = np.amin(self.mode_array[:,1]), np.amax(self.mode_array[:,1])
			
			#check in what interpolant segments we actually have to compute necessary modes, fixing the start to make things fit
			i_interp_check_0 = len(self.sol.ts)%params['NecessaryModes_UpdatePeriod']
			i_interp_check = np.arange(i_interp_check_0, len(self.sol.ts)+1, params['NecessaryModes_UpdatePeriod'])
			if i_interp_check_0 != 0: i_interp_check = np.append(0, i_interp_check)

			#extract the required y, e2 and DJ2
			ys, e2s, DJ2s = self.sol.ys[0][np.ix_(i_interp_check[:-1],[0,1,4])].T
			e2s = np.maximum(e2s, 0)
			
			#update MSA to compute precesion average value of dchi
			self.MSA.update(ys, DJ2s)
			
			#Loop over initial squared eccentricity and PN parameter at each segment, saving the modes selected there
			seg_modes_sets = []
			for y, e2, dchi_prec_avg in zip(ys, e2s, self.MSA.dchi_prec_avg):

				#obtain necessary modes [l,|m|,p] of N^{l m}_p that have to be taken into account
				mode_idxs, ps = Fourier_modes_needed(e2, y, self.MSA.nu, self.MSA.dmu, dchi=dchi_prec_avg, tol=params['Amplitude_tol'], pmin=pmin, pmax=pmax, mode_array=self.mode_array, pn_amplitude_order=self.params['pn_amplitude_order'])

				#update the pmin and pmax estimates, since e2 will be (generally) decreasing
				pmin = -1 + min(np.amin(ps), mmin)
				pmax =  1 + max(np.amax(ps), mmax)

				#save the modes selected at this segment as a set of tuples (mode_idx, p)
				seg_modes_sets.append(set(zip(mode_idxs.tolist(), ps.tolist())))

			#Since a mode may only be found to be important when we check the next segment, choose the modes to be the union of the modes selected at each segment and the next one
			n_seg = len(seg_modes_sets)
			for i_seg in range(n_seg):

				#extract the intepolant indices in this segment
				i_interp_0, i_interp_f = i_interp_check[i_seg], i_interp_check[i_seg+1]

				#compute the set of modes as the union between the current and next sets
				if i_seg+1 < n_seg: seg_mode_set = seg_modes_sets[i_seg] | seg_modes_sets[i_seg+1]
				else:               seg_mode_set = seg_modes_sets[i_seg]

				#convert set of tuples to two lists
				mode_idxs, ps = map(list, zip(*seg_mode_set))

				#save |m| and p on lists
				n_repeat = i_interp_f - i_interp_0
				self.necessary_multipole_idxs += n_repeat*mode_idxs
				self.necessary_ps             += n_repeat*ps

				#save the RK interpolant each mode is in
				self.mode_interp_idx += np.repeat(np.arange(i_interp_0, i_interp_f), len(ps)).tolist()

		#save stuff as numpy arrays
		self.necessary_multipole_idxs = np.asarray(self.necessary_multipole_idxs, dtype=int)
		self.necessary_ps = np.asarray(self.necessary_ps, dtype=int)
		self.mode_interp_idx  = np.asarray(self.mode_interp_idx, dtype=int)

		####################### Interpolation of the phases #######################

		#compute m for each mode
		mraws = self.mode_array[self.necessary_multipole_idxs,1]

		#make sure that the phase p\lambda + (m - p)\delta\lambda is positive by doing [p,m]->[-p,-m]
		ms = np.where(self.necessary_ps>=0, mraws, -mraws)
		pabs = np.abs(self.necessary_ps)
		
		#compute the interpolant of the phase of each mode: p\lambda + (m - p)\delta\lambda
		ms_pabs = ms - pabs

		#compute the interpolants of phase and the first two derivatives
		self.mode_phases_y0 = []
		self.mode_phases_Qs = []
		for der in range(3):
			#compute constant part for each mode
			self.mode_phases_y0.append(pabs*self.sol.ys[der][self.mode_interp_idx,2] + ms_pabs*self.sol.ys[der][self.mode_interp_idx,3])
			#compute polynomial part for each mode
			self.mode_phases_Qs.append(pabs[:,np.newaxis]*self.sol.Qs[der][self.mode_interp_idx,2] + ms_pabs[:,np.newaxis]*self.sol.Qs[der][self.mode_interp_idx,3])

		####################### Interpolation of the SPA times #######################
		
		#put the initial and final frequencies (\omega=2*pi*f) of the time interpolant in arrays
		self.ts_interp_w0 = self.mode_phases_y0[1]
		self.ts_interp_wf = self.ts_interp_w0 + np.sum(self.mode_phases_Qs[1], axis=-1)

		#compute the matrix of the interpolant for stationary time t(f)
		self.ts_interp_Q = series_reversion(self.mode_phases_Qs[1], order=params['Series_Reversion_Order'])

		################################# Amplitudes #################################
		
		#if amplitudes are going to be interpolated, set it up
		if params['Interpolate_Amplitudes']:
			#set up the interpolation of precesion amplitudes
			self.interpolate_Apc_prec()
			#set up the interpolation of Fourier Mode Amplitudes
			self.interpolate_Nlm_p()
			self.compute_Nlm_p = self.compute_Nlm_p_interpolated
		#otherwise, use exact amplitudes
		else:
			self.compute_Nlm_p = self.compute_Nlm_p_exact

	#validate the user-provided parameter dictionary and merge it with the defaults: map deprecated *_start keys, warn on unrecognised keys and check physical ranges
	#returns the merged parameter dictionary
	@staticmethod
	def _validate_parameters(parameters, params):

		#map the deprecated *_start keys to the new keys they correspond to
		legacy_keys = {'e_start': 'eccentricity', 'phi_start': 'phase', 'mean_anomaly_start': 'mean_anomaly'}
		parameters = dict(parameters)
		for old_key, new_key in legacy_keys.items():
			if old_key in parameters:
				if new_key in parameters: raise ValueError("Both '%s' and its deprecated alias '%s' were passed, use only '%s'."%(new_key, old_key, new_key))
				warnings.warn("pyEFPE parameter '%s' is deprecated, its value will be used as '%s'; pass '%s' directly in the future."%(old_key, new_key, new_key), DeprecationWarning, stacklevel=3)
				parameters[new_key] = parameters.pop(old_key)

		#the set of recognised parameter keys (the defaults plus mass1/mass2, which have no default)
		valid_keys = set(params) | {'mass1', 'mass2'}

		#warn about unrecognised parameter keys (likely typos), which would otherwise be silently ignored
		unknown_keys = set(parameters) - valid_keys
		if unknown_keys: warnings.warn("Ignoring unrecognised pyEFPE parameter(s) %s (possible typo); they have no effect."%(sorted(unknown_keys)), UserWarning, stacklevel=3)

		#update default parameters with input parameters
		params.update(parameters)

		#masses must be positive and the reference eccentricity in [0, 1)
		if (params['mass1']<=0) or (params['mass2']<=0): raise ValueError("mass1 and mass2 must be positive (got mass1=%s, mass2=%s)."%(params['mass1'], params['mass2']))
		if not (0<=params['eccentricity']<1): raise ValueError("eccentricity must be in [0, 1) (got eccentricity=%s)."%(params['eccentricity']))

		#checks on the reference frequency, if given
		if params['f22_ref'] is not None:
			if params['f22_ref']<=0: raise ValueError("f22_ref must be positive (got f22_ref=%s)."%(params['f22_ref']))
			if params['f22_ref']<params['f22_start']:
				warnings.warn("f22_ref=%s is below f22_start=%s; the reference state will be evolved forward to f22_start before starting waveform generation."%(params['f22_ref'], params['f22_start']), UserWarning, stacklevel=3)

		#warn if either dimensionless spin magnitude exceeds the Kerr bound |chi|<=1
		chi1_sq = params['spin1x']**2 + params['spin1y']**2 + params['spin1z']**2
		chi2_sq = params['spin2x']**2 + params['spin2y']**2 + params['spin2z']**2
		if (chi1_sq>1) or (chi2_sq>1): warnings.warn("Dimensionless spin magnitude exceeds the Kerr bound |chi|<=1 (|chi1|=%.3f, |chi2|=%.3f)."%(chi1_sq**0.5, chi2_sq**0.5), UserWarning, stacklevel=3)

		#warn if tidal deformabilities are provided but fully ignored (tidal terms switch on only at pn_tidal_order>=10)
		if (0<=params['pn_tidal_order']<10) and ((params['Lambda2_1']!=0) or (params['Lambda2_2']!=0)):
			warnings.warn("Nonzero tidal deformabilities were provided but pn_tidal_order<10, so tidal effects are ignored. Set pn_tidal_order>=10 (or -1) to include them.", UserWarning, stacklevel=3)

		#horizon absorption enters at 2.5PN (needs pn_spin_order>=5) and assumes black-hole components
		if params['horizon_absorption']:
			if (0<=params['pn_spin_order']<5):
				warnings.warn("horizon_absorption=True has no effect when pn_spin_order<5 (HA enters at 2.5PN).", UserWarning, stacklevel=3)
			elif (not (0<=params['pn_tidal_order']<10)) and ((params['Lambda2_1']!=0) or (params['Lambda2_2']!=0)):
				warnings.warn("horizon_absorption=True applies black-hole horizon flux to a component with nonzero tidal Lambda2.", UserWarning, stacklevel=3)

		#return the merged parameter dictionary
		return params

	#function to compute stationary times given an input array of frequencies (see Eq.(46) of arXiv:1801.08542)
	def stationary_times(self, freqs, rtol=1e-12, max_iter=3):
		
		#compute the omega associated with these frequencies
		ws = 2*np.pi*np.asarray(freqs, dtype=float)

		#check in which interpolant of the different modes these ws are in
		f_idxs, interp_idxs = sorted_vals_in_intervals(ws, self.ts_interp_w0, self.ts_interp_wf)
		
		#handle the case where there are no stationary times
		if len(f_idxs)==0:
			warnings.warn("Waveform has no stationary times for params=%s and freqs=%s"%(self.params, freqs), UserWarning, stacklevel=2)
			return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
		
		#we will need the ws at the f_idxs
		ws = ws[f_idxs]

		#compute the x = (t - t0)/h of the t interpolant
		dws = ws - self.ts_interp_w0[interp_idxs]
		xs = dws*self.ts_interp_Q[interp_idxs,-1]
		for iQ in reversed(range(self.ts_interp_Q.shape[1]-1)):
			xs = dws*(self.ts_interp_Q[interp_idxs,iQ] + xs)
		
		#initialize coefficients of polynomials to compute dws/dxs
		hs_mode_interp = self.sol.hs[self.mode_interp_idx]
		dw_dxs_Q = self.mode_phases_Qs[2]*hs_mode_interp[:,None]
		dw_dxs_y = self.mode_phases_y0[2]*hs_mode_interp
		#perform iterations of Newton-Rhapson until desired tolerance is reached
		idxs_update = np.ones(len(xs), dtype=bool)
		for iNewt in range(max_iter):

			#if there are no points to update, break the loop
			if not np.any(idxs_update): break
			
			#Select required indices
			interp_idxs_u = interp_idxs[idxs_update]
			x_u = xs[idxs_update]

			#compute the approximate \omega
			dws_x = x_u*self.mode_phases_Qs[1][interp_idxs_u,-1]
			for iQ in reversed(range(self.mode_phases_Qs[1].shape[1]-1)):
				dws_x = x_u*(self.mode_phases_Qs[1][interp_idxs_u,iQ] + dws_x)
			
			#compute the approximate d\omega/dx
			dw_dxs = x_u*dw_dxs_Q[interp_idxs_u,-1]
			for iQ in reversed(range(dw_dxs_Q.shape[1]-1)):
				dw_dxs = x_u*(dw_dxs_Q[interp_idxs_u,iQ] + dw_dxs)
			dw_dxs += dw_dxs_y[interp_idxs_u]
			
			#compute difference between target \omega and \omega(x)
			delta_ws = dws[idxs_update] - dws_x
			
			#compute the updated value of x using Newton-Rhapson
			xs[idxs_update] += delta_ws/dw_dxs

			#find the indexes that are above tolerance and have to be updated for next iteration
			idxs_update[idxs_update] = (np.abs(delta_ws) > np.abs(rtol*ws[idxs_update]))
			
		#now compute the corresponding stationary times
		t_SPA = self.sol.ts[self.mode_interp_idx][interp_idxs] + hs_mode_interp[interp_idxs]*xs
		
		#compute the stationary phase psi_SPA
		psi_SPA = xs*self.mode_phases_Qs[0][interp_idxs,-1]
		for iQ in reversed(range(self.mode_phases_Qs[0].shape[1]-1)):
			psi_SPA = xs*(self.mode_phases_Qs[0][interp_idxs,iQ] + psi_SPA)
		psi_SPA += self.mode_phases_y0[0][interp_idxs]
		
		#compute SPA time scale T_SPA
		T_SPA = xs*self.mode_phases_Qs[2][interp_idxs,-1]
		for iQ in reversed(range(self.mode_phases_Qs[2].shape[1]-1)):
			T_SPA = xs*(self.mode_phases_Qs[2][interp_idxs,iQ] + T_SPA)
		T_SPA += self.mode_phases_y0[2][interp_idxs]
		
		#we remove points where \ddot{\psi} vanishes or becomes negative
		#something fancier along the lines of 2102.02713 could be done
		idxs_ddpsi_pos = (T_SPA>0)
		if not np.all(idxs_ddpsi_pos):
			ws          = ws[idxs_ddpsi_pos]
			f_idxs      = f_idxs[idxs_ddpsi_pos]
			interp_idxs = interp_idxs[idxs_ddpsi_pos]
			t_SPA       = t_SPA[idxs_ddpsi_pos]
			psi_SPA     = psi_SPA[idxs_ddpsi_pos]
			T_SPA       = T_SPA[idxs_ddpsi_pos]
		
		#now compute actual T_SPA
		T_SPA = 1/np.sqrt(T_SPA)

		#take into account that \psi_s = 2 \pi f t_s - \phi(t_s) - pi/4
		psi_SPA = ws*t_SPA - psi_SPA - 0.25*np.pi

		#return all the SPA related things that will be needed in the future
		return f_idxs, interp_idxs, t_SPA, psi_SPA, T_SPA

	#function to compute Euler angles
	def compute_Euler_angles(self, times):

		#if the perpendicular spins are 0, do not allow precession (this could be done deeper in the code)
		if (self.MSA.sp2_1==0) and (self.MSA.sp2_2==0):
			return np.zeros_like(times), np.zeros_like(times), np.ones_like(times)
		#otherwise compute the full Euler angles
		else:

			#compute the average Euler angles and y, e at the input times
			y, e2, DJ2, bpsip, phiz, zeta = self.sol(times, idxs=[0,1,4,5,6,7])

			#compute precession Euler angles
			self.MSA.update(y, DJ2)
			dphiz, dzeta, costhL = self.MSA.precession_Euler_angles(bpsip)
			phiz += dphiz
			zeta += dzeta

			return phiz, zeta, costhL
	
	#function to compute exact precession amplitudes \mathsf{A}^{+,\times}_{l,m}
	def compute_Apc_prec_exact(self, times, im):

		#compute the Euler angles at the input times
		phiz, zeta, costhL = self.compute_Euler_angles(times)

		#update Wigner matrix computer with those angles
		self.Wigner.update_angles(phiz, costhL, zeta)

		#return the projected Wigner D-Matrix at the input times
		return np.tensordot(self.Wigner.D(*self.mode_array[im]), self.Apc_projs[im], axes=(0, 0))
		
	#method to interpolate the precession amplitudes \mathsf{A}^{+,\times}_{l,m} = h_0 \sum_{m'=-l}^{l} \mathsf{P}^{+,\times}_{l,m,m'}(\Theta, \Phi)  D^l_{m'm}(\phi_z,\theta_L,\zeta), where
	#\mathsf{P}^{+}_{l,m,m'} = \frac{1}{2}\left[{}_{-2}Y^{l m'} + (-1)^{l + m + m'} ({}_{-2}Y^{l -m'})^{*}\right]
	#\mathsf{P}^{\times}_{l,m,m'} = \frac{\rmi}{2}\left[ {}_{-2}Y^{l m'} - (-1)^{l + m + m'} ({}_{-2}Y^{l -m'})^{*}\right]
	def interpolate_Apc_prec(self, ):

		#compute the step in (bpsip+phiz) for precession amplitude interpolation (a precesion cycle is a change of pi in bpsip or in phiz).
		self.dEuler_interp = np.pi/(1 + self.params['Interp_points_per_prec_cycle'])

		#invert the (bpsip + |phiz|)(t) polynomial series (phiz can be decreasing when \vec{J}\cdot\vec{L}<0)
		Euler_Qs = self.sol.Qs[0][:,5] + np.sign(self.sol.Qs[0][:,6,[0]])*self.sol.Qs[0][:,6]
		t_dEuler_Qs = series_reversion(Euler_Qs, order=self.params['Series_Reversion_Order'])
		
		#compute how much (bpsip + phiz) varies in each interpolation region
		Delta_Euler = np.sum(Euler_Qs, axis=-1)

		#loop over interpolants and find how many points we need to put on each one. We always take the first point, to be as safe as possible.
		dEuler, idxs_dEuler = [], []
		for i_interp in range(len(self.sol.ts)):

			#compute the values of (psi + phiz) in which we are going to interpolate the precession amplitudes
			dEuler += np.arange(0, Delta_Euler[i_interp], self.dEuler_interp).tolist()
			
			#put the interpolant index of these angles also in an array
			idxs_dEuler += (len(dEuler) - len(idxs_dEuler))*[i_interp,]
		
		#compute the interpolation times corresponding to the (dbpsip + dphiz)
		prec_interp_xs = np.einsum('ij,ji->i',t_dEuler_Qs[idxs_dEuler], power_range(dEuler, t_dEuler_Qs.shape[-1]))
		#if x>=1, it is outside it's interpolation region and we neglect it
		idxs_xs_l_1 = (prec_interp_xs<1)
		idxs_dEuler = np.array(idxs_dEuler)[idxs_xs_l_1]
		#reconstruct time from x
		self.prec_interp_ts = self.sol.ts[idxs_dEuler] + self.sol.hs[idxs_dEuler]*prec_interp_xs[idxs_xs_l_1]

		#Add the final time to the interpolation times for precession
		self.prec_interp_ts = np.append(self.prec_interp_ts, self.sol.all_ts[-1])

		#take only the strictly increasing times
		self.prec_interp_ts = np.unique(self.prec_interp_ts)

		#compute the Euler angles at the interpolation times
		phiz, zeta, costhL = self.compute_Euler_angles(self.prec_interp_ts)

		#update Wigner matrix computer with those angles
		self.Wigner.update_angles(phiz, costhL, zeta)

		#Interpolate the precession amplitudes with m>=0, the ones with m<0 can be obtained from \mathsf{A}^{+,\times}_{l,-m} = (-1)^l conj(\mathsf{A}^{+,\times}_{l, m})
		self.Apc_prec_interp, self.Apc_prec_cspline = [], []
		#if the raw (pre spherical-harmonic-projection) hlm modes were requested, also prepare their interpolants
		compute_hlm = self.params['Compute_hlm_Modes']
		if compute_hlm: self.hlm_prec_cspline, self.hlm_prec_cspline_negm = [], []
		unique_necessary_multipole_idxs = np.unique(self.necessary_multipole_idxs)
		for im, mode in enumerate(self.mode_array):

			#if the mode is used, interpolate it for all available times. The time range for each could be taylored, taking into account SUA time-shifts.
			if im in unique_necessary_multipole_idxs:
				#compute raw twist-up Wigner-D matrix for this mode, D^l_{m'm}(t), shape (2l+1, Ntimes)
				D_im = self.Wigner.D(*mode)
				#compute precession amplitude for this mode
				self.Apc_prec_interp.append(np.tensordot(D_im, self.Apc_projs[im], axes=(0, 0)))
				#compute a cubic spline for this mode
				self.Apc_prec_cspline.append(CubicSpline(self.prec_interp_ts, self.Apc_prec_interp[-1], axis=0))

				#if requested, also interpolate the raw (pre-projection) hlm building block D^l_{m'm}(t) for this mode, m'=-l..l
				if compute_hlm:
					self.hlm_prec_cspline.append(CubicSpline(self.prec_interp_ts, np.transpose(D_im), axis=0))
					#for m>0 modes we also need D^l_{m'(-m)}(t), which shows up for the negative eccentric harmonics n<0 (physically the -m co-precessing mode).
					#Note this is NOT simply related to D^l_{m'm}(t) by conjugation (that shortcut only holds for the fully spherical-harmonic-projected Apc_prec above), so we interpolate it separately.
					if mode[1]>0: self.hlm_prec_cspline_negm.append(CubicSpline(self.prec_interp_ts, np.transpose(self.Wigner.D(mode[0], -mode[1])), axis=0))
					else        : self.hlm_prec_cspline_negm.append(None)
			else:
				self.Apc_prec_interp.append(None)
				self.Apc_prec_cspline.append(None)
				if compute_hlm:
					self.hlm_prec_cspline.append(None)
					self.hlm_prec_cspline_negm.append(None)

	#function to compute Apc_prec for a given multipole index and time
	def raw_Apc_prec_im(self, times, im, m_neg_idxs):
	
		#compute the amplitudes with interpolated or exact formulas
		if self.params['Interpolate_Amplitudes']:
			Apc_prec_im = self.Apc_prec_cspline[im](times)
		else:
			Apc_prec_im = self.compute_Apc_prec_exact(times, im)

		#for modes with negative m, use that \mathsf{A}^{+,\times}_{l,-m} = (-1)^l conj(\mathsf{A}^{+,\times}_{l, m})
		if len(m_neg_idxs)>0:
			Apc_prec_im[m_neg_idxs] = np.conj(Apc_prec_im[m_neg_idxs])
			if (self.mode_array[im,0]%2)!=0: Apc_prec_im[m_neg_idxs] = -Apc_prec_im[m_neg_idxs]

		return Apc_prec_im

	#function to compute the precession amplitudes
	def compute_Apc_prec(self, times, multipole_idxs, ps, SUA_kmax=0, T_SPA=None, ak_SUA=None):
		
		#Initialize the precession amplitudes with zeros
		Apc_prec = np.zeros((len(times),2), dtype=np.complex128)
		
		#Find where p is negative
		p_neg_idxs = (ps < 0)
		
		#group by multipole_idxs
		grouped_idxs = my_cgroup_idxs_by_vals(multipole_idxs)

		#loop over modes
		for im, im_idxs in enumerate(grouped_idxs):
			
			#if there are any, compute corresponding amplitudes
			if len(im_idxs)>0:

				#Track the points that correspond to negative m modes
				m_neg_idxs = np.nonzero(p_neg_idxs[im_idxs])[0]
				
				#case without SUA
				if SUA_kmax==0:
					
					#just compute raw Apc amplitudes
					Apc_prec[im_idxs] = self.raw_Apc_prec_im(times[im_idxs], im, m_neg_idxs)

				#case with SUA
				else:

					#initialize Amplitudes with value at central stencil point
					t_im = times[im_idxs]
					Apc_prec_im = ak_SUA[0]*self.raw_Apc_prec_im(t_im, im, m_neg_idxs)

					#select relevant SPA time-scale for this mode
					T_SPA_im = T_SPA[im_idxs]

					#Add the values at the other stencil points
					for k_SUA in range(1, 1 + SUA_kmax):
						
						#Compute raw Amplitude at symetric stencil points, multiply them by SUA weight and add to total
						Apc_prec_im += ak_SUA[k_SUA]*(self.raw_Apc_prec_im(t_im + k_SUA*T_SPA_im, im, m_neg_idxs) + self.raw_Apc_prec_im(t_im - k_SUA*T_SPA_im, im, m_neg_idxs))

					#save the amplitudes of this mode in the main array
					Apc_prec[im_idxs] = Apc_prec_im

		return Apc_prec

	#function to compute the exact raw (pre spherical-harmonic-projection) hlm building block D^l_{m' m_signed}(t), m'=-l..l, for a given multipole index and (signed) m
	def compute_hlm_prec_exact(self, times, im, m_signed):

		#compute the Euler angles at the input times
		phiz, zeta, costhL = self.compute_Euler_angles(times)

		#update Wigner matrix computer with those angles
		self.Wigner.update_angles(phiz, costhL, zeta)

		#return the raw (un-projected) Wigner D-matrix at the input times, D^l_{m' m_signed}(t), shape (len(times), 2l+1)
		return np.transpose(self.Wigner.D(self.mode_array[im,0], m_signed))

	#function to compute the raw hlm building block for a given multipole index and time.
	#Unlike raw_Apc_prec_im, we cannot use the \mathsf{A}^{+,\times}_{l,-m} = (-1)^l conj(\mathsf{A}^{+,\times}_{l, m}) shortcut here:
	#that identity holds for the fully spherical-harmonic-projected amplitude, not for the raw D^l_{m'm}(t) building block
	#(D^l_{m',-m}(t) is not simply related to D^l_{m',m}(t) by conjugation), so the -m case is interpolated/computed separately.
	def raw_hlm_prec_im(self, times, im, m_neg_idxs):

		l, m = self.mode_array[im]

		#compute the amplitudes with interpolated or exact formulas
		if self.params['Interpolate_Amplitudes']:
			hlm_im = self.hlm_prec_cspline[im](times)
			if len(m_neg_idxs)>0: hlm_im[m_neg_idxs] = self.hlm_prec_cspline_negm[im](times[m_neg_idxs])
		else:
			hlm_im = self.compute_hlm_prec_exact(times, im, m)
			if len(m_neg_idxs)>0: hlm_im[m_neg_idxs] = self.compute_hlm_prec_exact(times[m_neg_idxs], im, -m)

		return hlm_im

	#function to compute the SUA'd raw hlm building block for a given multipole index (mirrors the per-mode SUA logic inside compute_Apc_prec, but for the raw, un-projected D^l_{m'm}(t) building block)
	def SUA_hlm_prec_im(self, t_im, im, m_neg_idxs, T_SPA_im, SUA_kmax, ak_SUA):

		#case without SUA
		if SUA_kmax==0:
			return self.raw_hlm_prec_im(t_im, im, m_neg_idxs)

		#initialize amplitudes with value at central stencil point
		hlm_im = ak_SUA[0]*self.raw_hlm_prec_im(t_im, im, m_neg_idxs)

		#add the values at the other (symmetric) stencil points
		for k_SUA in range(1, 1 + SUA_kmax):
			hlm_im += ak_SUA[k_SUA]*(self.raw_hlm_prec_im(t_im + k_SUA*T_SPA_im, im, m_neg_idxs) + self.raw_hlm_prec_im(t_im - k_SUA*T_SPA_im, im, m_neg_idxs))

		return hlm_im

	#function to compute the exact Fourier Mode Amplitudes
	def compute_Nlm_p_exact(self, times, interp_idxs):
		
		#extract the modes being requested
		multipole_idxs = self.necessary_multipole_idxs[interp_idxs]
		ps = self.necessary_ps[interp_idxs]

		#Find where p is negative
		p_neg_idxs = (ps < 0)

		#compute y, e2 and DJ2 at the input times
		ys, e2s, DJ2s = self.sol(times, idxs=[0, 1, 4])
		e2s = np.maximum(e2s, 0)
		
		#update MSA to compute precesion average value of dchi
		self.MSA.update(ys, DJ2s)
		
		#loop over multipole modes
		Nlm_p = np.zeros(len(e2s), dtype=np.complex128)
		for i_lm, mode in enumerate(self.mode_array):

			#find the indexes for this mode
			lm_idxs = np.nonzero(multipole_idxs == i_lm)[0]
			
			#if necessary, compute the mode amplitudes for this mode
			if len(lm_idxs)>0:
				Nlm_p[lm_idxs] = analytical_Nlm_p(ps[lm_idxs], e2s[lm_idxs], ys[lm_idxs], self.MSA.nu, self.MSA.dmu, dchi=self.MSA.dchi_prec_avg[lm_idxs], mode_array=[mode], pn_amplitude_order=self.params['pn_amplitude_order'])[0]
				
				#Negative p modes are actually negative m modes, and N^{l -m}_p = (-1)^l (N^{l m}_{-p})^{*}
				m_neg_idxs = lm_idxs[p_neg_idxs[lm_idxs]]
				if len(m_neg_idxs)>0:
					Nlm_p[m_neg_idxs] = np.conj(Nlm_p[m_neg_idxs])
					if (mode[0]%2)!=0: Nlm_p[m_neg_idxs] = -Nlm_p[m_neg_idxs]

		#Add the factor (M\omega)^{2/3} = (1-e^2)*y^2, and return the Fourier mode amplitudes
		return (1 - e2s)*np.square(ys)*Nlm_p
		
	#method to interpolate Fourier mode amplitudes
	def interpolate_Nlm_p(self,):

		#compute full time array
		all_interp_ts = np.linspace(self.sol.all_ts[:-1], self.sol.all_ts[1:], num=self.params['Extra_interp_points_Nlm_p']+1, endpoint=False, axis=1)
		all_interp_ts = np.append(all_interp_ts.flatten(), self.sol.all_ts[-1])

		#compute y, e2 and DJ2 at these times
		ys, e2s, DJ2s = self.sol(all_interp_ts, idxs=[0, 1, 4])
		e2s = np.maximum(e2s, 0)
		
		#update MSA to compute precesion average value of dchi
		self.MSA.update(ys, DJ2s)
		
		#compute factor (M\omega)^{2/3} = (1-e^2)*y^2
		omega_factor = (1 - e2s)*np.square(ys)
		
		#find the unique values of p that contribute to the amplitude
		self.unique_ps, unique_ps_idxs = np.unique(self.necessary_ps, return_inverse=True)

		#loop over each unique value of p
		self.Nlm_p_csplines = []
		self.interp_idx_to_Nlm_p_idx = np.zeros_like(self.necessary_ps)
		Nlm_p_idx = 0
		for i_p, p in enumerate(self.unique_ps):

			#compute the RK indices at which this value of p is present
			interp_idxs_p = np.nonzero(unique_ps_idxs == i_p)[0]
			Nlm_p_RK_idxs_p = self.mode_interp_idx[interp_idxs_p]
			
			#find the start and end time for this p
			idx_t0_p, idx_tf_p = np.searchsorted(all_interp_ts, [self.sol.all_ts[Nlm_p_RK_idxs_p[0]], self.sol.all_ts[Nlm_p_RK_idxs_p[-1]+1]], side='left')
			
			#padd it just to be sure
			idx_t0_p = max(idx_t0_p - 1, 0)
			idx_tf_p = min(idx_tf_p + 2, len(all_interp_ts))

			#select the times considered
			interp_ts_p = all_interp_ts[idx_t0_p:idx_tf_p]

			#find the lm modes that happen in this p
			unique_lm_idxs, unique_lm_idxs_idxs = np.unique(self.necessary_multipole_idxs[interp_idxs_p], return_inverse=True)
			unique_lm_modes = self.mode_array[unique_lm_idxs]
			
			#compute the Fourier mode amplitudes to be interpolated, applying the (M\omega)^{2/3} factor
			Nlm_p_interp = omega_factor[None,idx_t0_p:idx_tf_p]*analytical_Nlm_p(np.full(idx_tf_p - idx_t0_p, p, dtype=int), e2s[idx_t0_p:idx_tf_p], ys[idx_t0_p:idx_tf_p], self.MSA.nu, self.MSA.dmu, dchi=self.MSA.dchi_prec_avg[idx_t0_p:idx_tf_p], mode_array=unique_lm_modes, pn_amplitude_order=self.params['pn_amplitude_order'])

			#loop over lm mode indexes
			for i_lm_idx, Nlm_p_interp_i_lm in enumerate(Nlm_p_interp):

				#Negative p modes are actually negative m modes, and N^{l -m}_p = (-1)^l (N^{l m}_{-p})^{*}
				if p<0:
					Nlm_p_interp_i_lm =  np.conj(Nlm_p_interp_i_lm)
					if (unique_lm_modes[i_lm_idx,0]%2)!=0: Nlm_p_interp_i_lm = -Nlm_p_interp_i_lm

				#compute the RK indices at which this lm mode is present
				interp_idxs_lm = (unique_lm_idxs_idxs == i_lm_idx)
				Nlm_p_RK_idxs_lm = Nlm_p_RK_idxs_p[interp_idxs_lm]

				#find the start and end time for this lm mode
				idx_t0_lm, idx_tf_lm = np.searchsorted(interp_ts_p, [self.sol.all_ts[Nlm_p_RK_idxs_lm[0]], self.sol.all_ts[Nlm_p_RK_idxs_lm[-1]+1]], side='left')

				#padd it just to be sure
				idx_t0_lm = max(idx_t0_lm - 1, 0)
				idx_tf_lm = min(idx_tf_lm + 2, len(interp_ts_p))

				#create cubic splines
				self.Nlm_p_csplines.append(CubicSpline(interp_ts_p[idx_t0_lm:idx_tf_lm], Nlm_p_interp_i_lm[idx_t0_lm:idx_tf_lm], axis=0))
			
				#save the interp idxs that correspond to this spline
				self.interp_idx_to_Nlm_p_idx[interp_idxs_p[interp_idxs_lm]] = Nlm_p_idx
				
				#update the index of the next spline
				Nlm_p_idx += 1
			
	#function to compute the interpolated Fourier mode amplitudes
	def compute_Nlm_p_interpolated(self, times, interp_idxs):

		#initialize Fourier Mode amplitudes
		Nlm_p = np.zeros(len(times), dtype=np.complex128)

		#extract the modes being requested
		Nlm_p_idxs = self.interp_idx_to_Nlm_p_idx[interp_idxs]

		#group by Nlm_p_idxs
		grouped_idxs = my_cgroup_idxs_by_vals(Nlm_p_idxs)
		
		#loop over each different spline
		for iNlm_p, idxs in enumerate(grouped_idxs):

			#compute the interpolated amplitudes associated with this mode
			if len(idxs)>0: Nlm_p[idxs] = self.Nlm_p_csplines[iNlm_p](times[idxs])
	
		return Nlm_p

	#function to compute the amplitudes using the SUA
	def SUA_Amplitudes(self, times, interp_idxs, T_SPA):
		
		#compute the Fourier mode amplitudes. We do not SUA these since from Eqs.(18,19,34) of 1408.5158, they are 1.5PN order corrections, while Wigner-SUA are 0.5PN
		Nlm_p = self.compute_Nlm_p(times, interp_idxs)
		
		#compute the times for which the stencil points are in range and therefore we can do SUA
		iSUA = ((times - self.params['SUA_kmax']*T_SPA) >=  self.sol.all_ts[0]) & ((times + self.params['SUA_kmax']*T_SPA) <=  self.sol.all_ts[-1])
		
		#initialize precession amplitudes
		Apc_prec = np.zeros((len(times), 2), dtype=np.complex128)

		#extract the modes being requested
		multipole_idxs = self.necessary_multipole_idxs[interp_idxs]
		ps = self.necessary_ps[interp_idxs]
		
		#When SUA times are out of range, we do SPA
		iSPA = np.nonzero(np.logical_not(iSUA))[0]
		if len(iSPA)>0: Apc_prec[iSPA] = self.compute_Apc_prec(times[iSPA], multipole_idxs[iSPA], ps[iSPA], SUA_kmax=0)
		
		#Compute SUA'd precession amplitudes
		Apc_prec[iSUA] = self.compute_Apc_prec(times[iSUA], multipole_idxs[iSUA], ps[iSUA], SUA_kmax=self.params['SUA_kmax'], ak_SUA=self.ak_SUA, T_SPA=T_SPA[iSUA])

		#return the SUA'd amplitudes
		return Apc_prec*Nlm_p[:,np.newaxis]
		
	#function to compute the waveform h_{+,\times} polarizations given an input array of frequencies (see Eq.(46) of arXiv:1801.08542)
	def generate_waveform(self, freqs):
		"""
		Compute the frequency-domain gravitational-wave polarizations [h_+(f), h_x(f)] using the SUA approximation.

		This implements Eq. (46) of arXiv:1801.08542.

		Parameters
		----------
		freqs : array-like
			Sorted array of frequencies at which to evaluate the waveform.

		Returns
		-------
		waveform : ndarray of shape (2, len(freqs))
			Complex frequency-domain waveform:
			- waveform[0, :] : plus polarization h_+(f)
			- waveform[1, :] : cross polarization h_x(f)
			If, for a given frequency, no stationary time within the simulated time [t_start, t_end] is found, the waveform at that frequency is zero.
		"""
		#first compute the SPA related things
		f_idxs, interp_idxs, t_SPA, psi_SPA, T_SPA = self.stationary_times(freqs, rtol=self.params['SPA_Frequency_rtol'])

		#handle the case where there are no stationary times
		if len(f_idxs)==0:
			return np.zeros((2, len(freqs)), dtype=np.complex128)
		
		#compute SPA factor = T*exp(1j*\psi) (Eq.(46c) of 1801.08542)
		SPA_fact    = T_SPA * (np.cos(psi_SPA) + 1j*np.sin(psi_SPA))
		del psi_SPA
		
		#compute waveform at each stationary time = T*exp(1j*\psi)*ASUA (using Eq.(46) of 1801.08542)
		waveform_ts = SPA_fact[:,np.newaxis]*self.SUA_Amplitudes(t_SPA, interp_idxs, T_SPA)
		del SPA_fact, interp_idxs, t_SPA, T_SPA
		
		#initialize array to store result of adding terms corresponding to the same frequency
		waveform = np.zeros((2, len(freqs)), dtype=waveform_ts.dtype)
		
		#add values of the waveform corresponding to the same frequency together. Since np.bincount only takes real 1D weights,
		#transpose to have correct shape (2, frequencies) and conjugate to use standard Fourier transform definition
		for iw in range(len(waveform)):
			waveform.real[iw] =  np.bincount(f_idxs, weights=waveform_ts.real[:,iw], minlength=len(freqs))
			waveform.imag[iw] = -np.bincount(f_idxs, weights=waveform_ts.imag[:,iw], minlength=len(freqs))

		#now return the result, multiplying by the h0 prefactor and the \sqrt{2\pi} coming from the SPA
		return (((2*np.pi)**0.5)*self.h0_pref)*waveform

	#function to compute the modes h_{l,m,n}(f) given an input array of frequencies
	def generate_modes(self, freqs, return_waveform_pieces=False):
		r"""
		Compute the frequency-domain (l, m, n) waveform modes (see Eq.(18) of 2502.03929).
		Note that in the code, `n` is called `p`.

		For each mode (l,m,n) mode, quantities are evaluated at a given subset of input frequencies. The modes included at each frequency are controlled by the requested `Amplitude_tol` (see Sec.IID of 2502.03929).

		Parameters
		----------
		freqs : array-like
			Sorted array of frequencies at which to evaluate the waveform.
		return_waveform_pieces : bool, optional
			If True, include intermediate quantities used to construct each mode polarization. - Default: False
		Returns
		-------
		result : dict
			Dictionary with structure:

			{
			'freqs': freqs,  # Input array of frequencies
			'modes': {
				(l, m, n): { # l, m, n values for this mode
				'freq_idxs': ndarray of int of shape (N_mode, )
					Indices into `freqs` where this mode contributes.
				'polarizations': ndarray of shape (2, N_mode)
					Complex waveform polarization values [h_{+,l,m,n}(f), h_{x,l,m,n}(f)] for this mode.
				# Only if return_waveform_pieces=True:
				't_SPA': ndarray of shape (N_mode,)
					Stationary times t0_{l,m,n}(f) for this mode (See Eq.(104) of 2502.03929)
				'T_SPA': ndarray of shape (N_mode,)
					SPA time-scale T0_{l,m,n}(f) for this mode (See Eq.(106) of 2502.03929)
				'psi_SPA': ndarray of shape (N_mode,)
					SPA phase psi_{l,m,n}(f) = 2 pi f t0_{l,m,n}(f) - \phi_{l,m,n}(t0_{l,m,n}(f)) - pi/4 for this mode (See Eq.(107) of 2502.03929)
				'A_SUA': ndarray of shape (N_mode,2)
					SUA amplitude \mathcal{A}^\mathrm{SUA}_{l,m,n}(f) for this mode (See Eq.(18) and Eq.(109) of 2502.03929).
					The first (second) column contains the amplitude for the plus (cross) polarization.
				},
				}
			}
		"""

		#initialize dictionary to store the result
		result = {'freqs': freqs, 'modes': {}}

		#first compute the SPA related things
		f_idxs, interp_idxs, t_SPA, psi_SPA, T_SPA = self.stationary_times(freqs, rtol=self.params['SPA_Frequency_rtol'])

		#handle the case where there are no stationary times
		if len(f_idxs)==0: return result
		
		#compute SPA factor = T*exp(1j*\psi) (Eq.(46c) of 1801.08542)
		SPA_fact = T_SPA*(np.cos(psi_SPA) + 1j*np.sin(psi_SPA))
		
		#compute waveform at each stationary time = T*exp(1j*\psi)*ASUA (using Eq.(46) of 1801.08542)
		A_SUA = self.SUA_Amplitudes(t_SPA, interp_idxs, T_SPA)
		waveform_ts = SPA_fact[:,np.newaxis]*A_SUA

		#multiply waveform by correct prefactor, and conjugate to use standard Fourier transform definition
		waveform_ts = (((2*np.pi)**0.5)*self.h0_pref)*np.conj(waveform_ts)
		
		#group by multipole_idxs
		multipole_idxs = self.necessary_multipole_idxs[interp_idxs]
		grouped_im_idxs = my_cgroup_idxs_by_vals(multipole_idxs)
		
		#loop over (l,m) modes
		for im, im_idxs in enumerate(grouped_im_idxs):

			#if there are no indexes for this mode, skip it
			if len(im_idxs)==0: continue
			
			#extract l and m
			l, m = self.mode_array[im]
			
			#compute the ps corresponding to this mode
			ps = self.necessary_ps[interp_idxs[im_idxs]]

			#minimum value of p
			pmin = np.amin(ps)

			#divide also by the value of p
			grouped_ip_idxs = my_cgroup_idxs_by_vals(ps - pmin)
			
			#loop over p modes
			for ip, p_idxs in enumerate(grouped_ip_idxs):

				#if there are no indexes for this mode, skip it
				if len(p_idxs)==0: continue
				
				#reconstruct p
				p = ip + pmin

				#define label for this mode
				if p<0: mode_label = (l, -m, -p)
				else:   mode_label = (l,  m,  p)

				#Compute indexes of this mode
				imp_idxs = im_idxs[p_idxs]

				#store (sorted) frequency indices corresponding to this mode and waveform polarizations in result dictionary
				result['modes'][mode_label] = {'freq_idxs': f_idxs[imp_idxs], 'polarizations': np.transpose(waveform_ts[imp_idxs])}

				#If required, store extra waveform pieces in dictionary
				if return_waveform_pieces:
					result['modes'][mode_label].update({'t_SPA': t_SPA[imp_idxs], 'T_SPA': T_SPA[imp_idxs], 'psi_SPA': psi_SPA[imp_idxs], 'A_SUA': A_SUA[imp_idxs]})


		#return the result dictionary
		return result

	#function to compute the raw (l, m, n) multipole moments h_{l,m}(f) given an input array of frequencies, without applying the spherical-harmonic projection to polarizations
	def generate_hlm_modes(self, freqs, return_waveform_pieces=False):
		r"""
		Compute the raw frequency-domain multipole moments h_{l,m}(f), before applying the spin-weighted spherical-harmonic projection to polarizations.

		Requires the class to have been initialized with Compute_hlm_Modes=True.

		Unlike generate_modes (which returns h_{+,x} for each co-precessing-frame (l, m, n) harmonic, already projected through this model's own fixed
		viewing angles (Theta, Phi) = (theta_JN, phi_JN) derived from the `inclination`/`phase` parameters), this method returns the un-projected
		complex multipole moments in the fixed (inertial, J-aligned) frame used internally by the precession solution.

		CONVENTION (pyefpehm_native_fd): 'hlm' here is *not* claimed to equal any external (e.g. LAL/PyCBC) inertial-frame h_{lm}(f). What is proven --
		analytically from the paper's Eq.(16) Fourier convention, and confirmed numerically (candidate comparison at multiple independent, non-degenerate
		parameter points; relative L2 error ~1e-15, all other candidates tested wrong by O(1)) -- is only the self-consistent reconstruction contract
		of this model's own polarizations. For a projector row Proj_{l,mp} = [Ap_{l,mp}, Ac_{l,mp}] built the same way as self.Apc_projs (see
		interpolate_Apc_prec), evaluated at any desired fixed viewing angles (Theta, Phi) instead of this model's own (theta_JN, phi_JN):

			[h_+, h_x](f) = sum_{mp=-l}^{l} hlm[mp](f) * conj(Proj_{l,mp}(Theta, Phi))

		Note the complex conjugate on the projector. It is *not* itself proof that 'hlm' equals a conjugated external h_lm(f): the conjugate is
		consistent with a Fourier-transform sign-convention difference between this paper's Eq.(16) (a +2*pi*i*f*t kernel) and the standard/LAL/PyCBC
		convention (a -2*pi*i*f*t kernel) applied to the *real* h_+(t)/h_x(t) -- that step is only valid for a real time-domain signal, and does not
		by itself establish the analogous relation for the individual complex-valued 'hlm' component. Establishing (or ruling out) equality with an
		external h_lm(f) convention is a separate, not-yet-attempted task. Use reconstruct_polarizations_from_native_modes() to apply the formula
		above rather than re-deriving it; do not apply a global conj() to 'hlm' on the assumption that it should "look like" another model's h_lm.
		(generate_tdomain_hlm_modes has no such conjugate: h_+(t)/h_x(t) are built there as 2*Re(Amps*e^{-i phi}) directly, without an outer
		conjugate -- see that method's docstring. Its 'hlm' is likewise only proven to satisfy its own, unconjugated, self-consistent contract.)

		Because a single fixed inertial-frame index mp generally receives contributions from *all* co-precessing (l, m) multipoles of that l (this is
		what precession "twist-up" means), and because the SPA/SUA machinery relies on each co-precessing (l, m, n) harmonic having a single well-defined
		instantaneous frequency, results are keyed by the co-precessing (l, m, n) label exactly as in generate_modes, and for each such key we return the
		raw contribution to *every* inertial-frame index mp=-l..l as a single array, rather than pre-summing over mp. Summing contributions that share the
		same (l, mp) across different keys (i.e. across different co-precessing m and/or harmonics n) is left to the caller.

		Parameters
		----------
		freqs : array-like
			Sorted array of frequencies at which to evaluate the waveform.
		return_waveform_pieces : bool, optional
			If True, include intermediate quantities used to construct each mode. - Default: False

		Returns
		-------
		result : dict
			Dictionary with structure:

			{
			'freqs': freqs,  # Input array of frequencies
			'modes': {
				(l, m, n): { # co-precessing l, m, n values for this harmonic (same convention as generate_modes)
				'freq_idxs': ndarray of int of shape (N_mode, )
					Indices into `freqs` where this harmonic contributes.
				'mp': ndarray of int of shape (2l+1,)
					Inertial-frame indices mp=-l..l labelling the columns of 'hlm' below.
				'hlm': ndarray of complex, shape (N_mode, 2l+1)
					Contribution of this (l, m, n) harmonic to h_{l,mp}(f), for each mp in 'mp' (column j <-> mp[j]).
				# Only if return_waveform_pieces=True:
				't_SPA', 'T_SPA', 'psi_SPA'
				},
				}
			}
		"""
		if not self.params['Compute_hlm_Modes']:
			raise ValueError("generate_hlm_modes requires the class to be initialized with Compute_hlm_Modes=True")

		#initialize dictionary to store the result. mode_convention='pyefpehm_native_fd' flags that 'hlm' satisfies only this model's own,
		#self-consistent reconstruction contract (see docstring above) -- not proven to equal any external convention's h_lm(f)
		result = {'freqs': freqs, 'mode_convention': 'pyefpehm_native_fd', 'modes': {}}

		#first compute the SPA related things
		f_idxs, interp_idxs, t_SPA, psi_SPA, T_SPA = self.stationary_times(freqs, rtol=self.params['SPA_Frequency_rtol'])

		#handle the case where there are no stationary times
		if len(f_idxs)==0: return result

		#compute SPA factor = T*exp(1j*\psi) (Eq.(46c) of 1801.08542)
		SPA_fact = T_SPA*(np.cos(psi_SPA) + 1j*np.sin(psi_SPA))

		#compute the Fourier mode amplitudes. We do not SUA these, since from Eqs.(18,19,34) of 1408.5158 they are a 1.5PN correction, while Wigner-SUA is 0.5PN (mirrors SUA_Amplitudes)
		Nlm_p = self.compute_Nlm_p(t_SPA, interp_idxs)

		#compute the points for which the stencil points are in range and therefore we can do SUA (mirrors SUA_Amplitudes)
		iSUA = ((t_SPA - self.params['SUA_kmax']*T_SPA) >= self.sol.all_ts[0]) & ((t_SPA + self.params['SUA_kmax']*T_SPA) <= self.sol.all_ts[-1])

		#extract the modes being requested
		multipole_idxs = self.necessary_multipole_idxs[interp_idxs]
		ps_all = self.necessary_ps[interp_idxs]
		p_neg_all = (ps_all < 0)

		#group by multipole_idxs
		grouped_im_idxs = my_cgroup_idxs_by_vals(multipole_idxs)

		#loop over (l,m) co-precessing modes
		for im, im_idxs in enumerate(grouped_im_idxs):

			#if there are no indexes for this mode, skip it
			if len(im_idxs)==0: continue

			#extract l and m
			l, m = self.mode_array[im]

			#compute the raw (un-projected) hlm building block, SUA'd where applicable
			t_im = t_SPA[im_idxs]
			m_neg_im = p_neg_all[im_idxs]
			iSUA_im = iSUA[im_idxs]
			hlm_im = np.zeros((len(im_idxs), 2*l+1), dtype=np.complex128)

			idxs_noSUA = np.nonzero(~iSUA_im)[0]
			if len(idxs_noSUA)>0:
				hlm_im[idxs_noSUA] = self.SUA_hlm_prec_im(t_im[idxs_noSUA], im, np.nonzero(m_neg_im[idxs_noSUA])[0], None, 0, None)

			idxs_SUA = np.nonzero(iSUA_im)[0]
			if len(idxs_SUA)>0:
				hlm_im[idxs_SUA] = self.SUA_hlm_prec_im(t_im[idxs_SUA], im, np.nonzero(m_neg_im[idxs_SUA])[0], T_SPA[im_idxs][idxs_SUA], self.params['SUA_kmax'], self.ak_SUA)

			#multiply by the Fourier mode amplitude and SPA factor, then by the overall prefactor, conjugating to use the standard Fourier transform definition (mirrors generate_modes)
			waveform_ts_im = (((2*np.pi)**0.5)*self.h0_pref)*np.conj(SPA_fact[im_idxs,np.newaxis]*hlm_im*Nlm_p[im_idxs,np.newaxis])

			#compute the ps corresponding to this mode
			ps = ps_all[im_idxs]

			#minimum value of p
			pmin = np.amin(ps)

			#divide also by the value of p
			grouped_ip_idxs = my_cgroup_idxs_by_vals(ps - pmin)

			#loop over p modes
			for ip, p_idxs in enumerate(grouped_ip_idxs):

				#if there are no indexes for this mode, skip it
				if len(p_idxs)==0: continue

				#reconstruct p
				p = ip + pmin

				#define label for this mode
				if p<0: mode_label = (l, -m, -p)
				else:   mode_label = (l,  m,  p)

				#Compute indexes of this mode
				imp_idxs = im_idxs[p_idxs]

				#store (sorted) frequency indices, inertial mp labels, and raw hlm values corresponding to this mode in result dictionary
				result['modes'][mode_label] = {'freq_idxs': f_idxs[imp_idxs], 'mp': np.arange(-l, l+1), 'hlm': waveform_ts_im[p_idxs]}

				#If required, store extra waveform pieces in dictionary
				if return_waveform_pieces:
					result['modes'][mode_label].update({'t_SPA': t_SPA[imp_idxs], 'T_SPA': T_SPA[imp_idxs], 'psi_SPA': psi_SPA[imp_idxs]})

		#return the result dictionary
		return result

	#method that returns start time of the waveform
	def return_start_time(self,):
		return self.sol.all_ts[0]

	#method that returns end time of the waveform
	def return_end_time(self,):
		return self.sol.all_ts[-1]

	#Function to compute the time domain waveform polarizations h_{+,\times}(t) given an input array of times (or a time spacing)
	#When there are many Fourier modes, the computation could be done more efficiently, since many things are the same for all times
	#However, for simplicity, we choose to do things as similarly as possible to the Fourier Domain computation
	#An implementation following self.generate_tdomain_modes would be more efficient when there are many Fourier modes
	def generate_tdomain_waveform(self, times=None, delta_t=None, return_time_array=False):
		"""
		Compute the time-domain gravitational-wave polarizations h_+(t), h_x(t).

		The waveform is reconstructed by summing contributions from all Fourier modes, evaluated directly in the time domain. (see Eq.(18) of 2502.03929)

		Parameters
		----------
		times : array-like, optional
			Sorted array of times at which to evaluate the waveform.
			If None, a uniform grid is generated using `delta_t`. - Default: None
		delta_t : float, optional
			Time step used to generate a uniform time array if `times` is None. - Default: None
		return_time_array : bool, optional
			If True, also return the time array used in the computation. - Default: False

		Returns
		-------
		waveform : ndarray of shape (2, len(times))
			Real time-domain waveform:
			- waveform[0, :] : h_+(t)
			- waveform[1, :] : h_x(t)
			The waveform is 0 outside the simulated domain [t_start, t_end]

		times : ndarray, optional
			Time array used in the computation. Returned only if `return_time_array=True`.

		Raises
		------
		ValueError
			If neither `times` nor `delta_t` is provided.
		"""
		
		#if no time-array is given, create it
		if times is None:
			if delta_t is None: raise ValueError("To compute time-domain waveform, please give either an array of times or a time spacing delta_t")
			#make an equally spaced array for all available times
			times = np.arange(self.sol.all_ts[0], self.sol.all_ts[-1], delta_t)
			#by construction, all times are valid
			i_valid = np.arange(len(times))
			valid_times = times
		else:
			#make sure it is a numpy array
			times = np.asarray(times)
			#find valid times (i.e. where the system has been simulated)
			i_valid = np.nonzero((times>=self.sol.all_ts[0]) & (times<=self.sol.all_ts[-1]))[0]
			valid_times = times[i_valid]

		#find the values of interp_idxs for each time
		t_idxs, interp_idxs = sorted_vals_in_intervals(valid_times, self.sol.all_ts[self.mode_interp_idx], self.sol.all_ts[self.mode_interp_idx+1])
		ts_compute = valid_times[t_idxs]
		
		#compute the Amplitudes
		Amps = self.compute_Apc_prec(ts_compute, self.necessary_multipole_idxs[interp_idxs], self.necessary_ps[interp_idxs], SUA_kmax=0)*self.compute_Nlm_p(ts_compute, interp_idxs)[:,None]

		#find segment of Runge-Kutta each interp_idx is in
		idxs_t_interp = self.mode_interp_idx[interp_idxs]
		
		#compute the phases
		xs = (ts_compute - self.sol.ts[idxs_t_interp])/self.sol.hs[idxs_t_interp]
		del idxs_t_interp, ts_compute
		phi_t = xs*self.mode_phases_Qs[0][interp_idxs,-1]
		for iQ in reversed(range(self.mode_phases_Qs[0].shape[1]-1)):
			phi_t = xs*(self.mode_phases_Qs[0][interp_idxs,iQ] + phi_t)
		phi_t+= self.mode_phases_y0[0][interp_idxs]
		del interp_idxs, xs

		#compute the waveform h(t) = Re(A(t)*e^{-i\phi(t)})
		waveform_ts = Amps.real*(np.cos(phi_t)[:,None]) + Amps.imag*(np.sin(phi_t)[:,None])
		del phi_t, Amps

		#initialize array to store result of adding terms corresponding to the same time
		waveform = np.zeros((2, len(times)), dtype=waveform_ts.dtype)

		#add values of the waveform corresponding to the same time together
		#Since np.bincount only takes 1D weights, take opportunity to transpose to have correct shape (2, frequencies)
		for iw in range(len(waveform)):
			waveform[iw,i_valid] = np.bincount(t_idxs, weights=waveform_ts[:,iw], minlength=len(valid_times))

		#multiply by 2*h0 prefactor
		waveform *= 2*self.h0_pref

		#if requested, return polarizations and time array
		if return_time_array:
			return waveform, times
		#otherwise, return just the polarizations
		else:
			return waveform

	#Function to compute the time domain waveform modes h_{l,m,n}(t) given an input array of times (or a time spacing)
	def generate_tdomain_modes(self, times=None, delta_t=None, return_waveform_pieces=False):
		r"""
		Compute the time-domain (l, m, n) waveform modes (see Eq.(18) of 2502.03929).
		Note that in the code, `n` is called `p`.

		For each mode (l,m,n) mode, quantities are evaluated at a given subset of input times. The modes included at each time are controlled by the requested `Amplitude_tol` (see Sec.IID of 2502.03929).

		Parameters
		----------
		times : array-like, optional
			Sorted array of times at which to evaluate the waveform.
			If None, a uniform grid is generated using `delta_t`. - Default: None
		delta_t : float, optional
			Time step used to generate a uniform time array if `times` is None. - Default: None
		return_waveform_pieces : bool, optional
			If True, include intermediate quantities used to construct each mode polarization, as well as phase derivatives. - Default: False

		Returns
		-------
		result : dict
			Dictionary with structure:

		{
		'times': times, # Input array of times (or array constructed from delta_t)
		'modes': {
			(l, m, n): {
			'time_idxs': ndarray of int of shape (N_mode, )
				Indices into `times` where this mode contributes.
			'polarizations': ndarray of shape (2, N_mode)
				Real waveform values [h_{+,l,m,n}(t), h_{x,l,m,n}(t)] for this mode
			# Only if return_waveform_pieces=True:
			'Apc_prec': ndarray of shape (N_mode, 2)
				Precession amplitudes \mathsf{A}^{+,x}_{l,m}(t) for this mode (See Eq.(14) of 2502.03929)
				The first (second) column contains the amplitude for the plus (cross) polarization.
			'Nlm_p': ndarray of shape (N_mode,)
				Fourier mode amplitudes N^{l m}_n for this mode (See Eq.(18,19) of 2502.03929. Note that we use the convention of 2507.00169)
			'phase': ndarray of shape (N_mode,)
				Phase phi_{n,m}(t) = n\lambda(t) + (m-n)\delta\lambda(t) for this mode (See Eq.(18) of 2502.03929)
			'omega': ndarray of shape (N_mode,)
				Instantaneous frequency \omega_{n m} = d(phi_{n,m})/dt for this mode
			'DomegaDt': ndarray of shape (N_mode,)
				Time derivative of the frequency d\omega_{n m}/dt for this mode.
			},
			}
		}

		Raises
		------
		ValueError
			If neither `times` nor `delta_t` is provided.
		"""
		
		#if no time-array is given, create it
		if times is None:
			if delta_t is None: raise ValueError("To compute time-domain waveform, please give either an array of times or a time spacing delta_t")
			#make an equally spaced array for all available times
			times = np.arange(self.sol.all_ts[0], self.sol.all_ts[-1], delta_t)
			#by construction, all times are valid
			i_valid = np.arange(len(times))
			valid_times = times
		else:
			#make sure it is a numpy array
			times = np.asarray(times)
			#find valid times (i.e. where the system has been simulated)
			i_valid = np.nonzero((times>=self.sol.all_ts[0]) & (times<=self.sol.all_ts[-1]))[0]
			valid_times = times[i_valid]

		#initialize dictionary to store the result
		result = {'times': times, 'modes': {}}
		
		#find the values of interp_idxs for each time
		t_idxs, interp_idxs = sorted_vals_in_intervals(valid_times, self.sol.all_ts[self.mode_interp_idx], self.sol.all_ts[self.mode_interp_idx+1])

		#compute \lambda and \delta\lambda at the valid times
		valid_l, valid_dl = self.sol(valid_times, derivative=0, idxs=[2,3])

		#if required, compute also the first and second derivatives of the phase
		if return_waveform_pieces:
			valid_DlDt  , valid_DdlDt   = self.sol(valid_times, derivative=1, idxs=[2,3])
			valid_DDlDt2, valid_DDdlDt2 = self.sol(valid_times, derivative=2, idxs=[2,3])

		#group by multipole_idxs
		multipole_idxs = self.necessary_multipole_idxs[interp_idxs]
		grouped_im_idxs = my_cgroup_idxs_by_vals(multipole_idxs)

		#loop over (l,m) modes
		for im, im_idxs in enumerate(grouped_im_idxs):

			#if there are no indexes for this mode, skip it
			if len(im_idxs)==0: continue
			
			#extract l and m
			l, m = self.mode_array[im]
			
			#compute the interp_idxs, ps and time_idxs corresponding to this mode
			interp_idxs_im = interp_idxs[im_idxs]
			ps = self.necessary_ps[interp_idxs_im]
			t_idxs_im = t_idxs[im_idxs]

			#compute the minimum and maximum time indexes for this mode
			t_idxs_im_min = np.amin(t_idxs_im)
			t_idxs_im_max = np.amax(t_idxs_im)
			
			#minimum value of p
			pmin = np.amin(ps)

			#compute the precession amplitudes for this mode in [tmin, tmax]
			Apc_prec_im = self.raw_Apc_prec_im(valid_times[t_idxs_im_min:(t_idxs_im_max+1)], im, np.array([]))

			#divide also by the value of p
			grouped_ip_idxs = my_cgroup_idxs_by_vals(ps - pmin)

			#loop over p modes
			for ip, p_idxs in enumerate(grouped_ip_idxs):

				#if there are no indexes for this mode, skip it
				if len(p_idxs)==0: continue
				
				#reconstruct p
				p = ip + pmin

				#Compute the (sorted) times corresponding to this mode
				t_idxs_ip = t_idxs_im[p_idxs]
				
				#Compute the precession amplitudes corresponding to this mode
				Apc_prec_ip = Apc_prec_im[t_idxs_ip-t_idxs_im_min]
				
				#compute the phase of this mode = p\lambda + (m - p)\delta\lambda
				phi_t = p*valid_l[t_idxs_ip] + (m - p)*valid_dl[t_idxs_ip]

				#handle when p is negative
				if p<0:
		
					#make sure the phase is positive doing [p,m]->[-p,-m]
					phi_t = -phi_t
					
					#for modes with negative m, use that \mathsf{A}^{+,\times}_{l,-m} = (-1)^l conj(\mathsf{A}^{+,\times}_{l, m})
					Apc_prec_ip = np.conj(Apc_prec_ip)
					if (self.mode_array[im,0]%2)!=0: Apc_prec_ip = -Apc_prec_ip

					#define label for this mode
					mode_label = (l, -m, -p)
				else:
					mode_label = (l,  m,  p)

				#compute Fourier mode amplitudes for this mode
				Nlm_p = self.compute_Nlm_p(valid_times[t_idxs_ip], interp_idxs_im[p_idxs])
				
				#compute time domain Amplitudes
				Amps = Apc_prec_ip*Nlm_p[:,None]
				
				#Compute the polarizations for this mode
				waveform_ts = (2*self.h0_pref)*(Amps.real*(np.cos(phi_t)[:,None]) + Amps.imag*(np.sin(phi_t)[:,None]))

				#store (sorted) time indices corresponding to this mode and waveform polarizations in result dictionary
				result['modes'][mode_label] = {'time_idxs': i_valid[t_idxs_ip], 'polarizations': np.transpose(waveform_ts)}

				#If required, store extra waveform pieces in dictionary
				if return_waveform_pieces:
					#compute the first and second derivatives of the phase
					omega    = p*valid_DlDt[t_idxs_ip]   + (m - p)*valid_DdlDt[t_idxs_ip]
					DomegaDt = p*valid_DDlDt2[t_idxs_ip] + (m - p)*valid_DDdlDt2[t_idxs_ip]
					#handle when p is negative, doing [p,m]->[-p,-m]
					if p<0: omega, DomegaDt = -omega, -DomegaDt
					#add stuff to dictionary that will be returned
					result['modes'][mode_label].update({'Apc_prec': Apc_prec_ip, 'Nlm_p': Nlm_p, 'phase': phi_t, 'omega': omega, 'DomegaDt': DomegaDt})

		#return result dictionary
		return result

	#Function to compute the time-domain raw (l, m, n) multipole moments h_{l,m}(t) given an input array of times (or a time spacing), without applying the spherical-harmonic projection to polarizations
	def generate_tdomain_hlm_modes(self, times=None, delta_t=None, return_waveform_pieces=False):
		r"""
		Compute the raw time-domain multipole moments h_{l,m}(t), before applying the spin-weighted spherical-harmonic projection to polarizations.

		Requires the class to have been initialized with Compute_hlm_Modes=True.

		See generate_hlm_modes for the conventions used and important caveats (results are keyed by the co-precessing (l, m, n) label, and for each such
		key the raw contribution to *every* inertial-frame index mp=-l..l is returned as a single array; summing contributions that share the same
		(l, mp) across different keys is left to the caller).

		Reconstruction (note: unlike generate_hlm_modes, there is NO conjugate on the projector here, since h(t) is built as 2*Re(Amps*e^{-i phi})
		directly, without an outer conjugate -- see generate_tdomain_waveform):

			[h_+, h_x](t) = sum_{mp=-l}^{l} hlm[mp](t) * Proj_{l,mp}(Theta, Phi)   (real part only; hlm already has the factor of 2 from 2*Re(...) built in)

		where Proj is built the same way as self.Apc_projs (see interpolate_Apc_prec) but at any desired fixed viewing angles.

		CONVENTION (pyefpehm_native_td -- note this is a *different* convention label from generate_hlm_modes' pyefpehm_native_fd; the FD and TD
		reconstruction contracts must not be mixed, e.g. do not apply the FD conj() here or vice versa): this 'hlm' is *not* claimed to equal any
		external (e.g. LAL/PyCBC) inertial-frame h_{lm}(t). What is proven -- confirmed numerically at multiple independent, non-degenerate
		parameter points -- is only the self-consistent reconstruction contract above, i.e. that it reproduces this model's own h_+(t), h_x(t)
		(as returned by generate_tdomain_waveform/generate_tdomain_modes) via the *un-conjugated* projector. Establishing (or ruling out) equality
		with an external h_lm(t) convention is a separate, not-yet-attempted task. Use reconstruct_polarizations_from_native_modes() to apply the
		formula above rather than re-deriving it.

		Parameters
		----------
		times : array-like, optional
			Sorted array of times at which to evaluate the waveform.
			If None, a uniform grid is generated using `delta_t`. - Default: None
		delta_t : float, optional
			Time step used to generate a uniform time array if `times` is None. - Default: None
		return_waveform_pieces : bool, optional
			If True, include intermediate quantities used to construct each mode. - Default: False

		Returns
		-------
		result : dict
			Dictionary with structure:

		{
		'times': times, # Input array of times (or array constructed from delta_t)
		'modes': {
			(l, m, n): { # co-precessing l, m, n values for this harmonic (same convention as generate_tdomain_modes)
			'time_idxs': ndarray of int of shape (N_mode, )
				Indices into `times` where this harmonic contributes.
			'mp': ndarray of int of shape (2l+1,)
				Inertial-frame indices mp=-l..l labelling the columns of 'hlm' below.
			'hlm': ndarray of complex, shape (N_mode, 2l+1)
				Contribution of this (l, m, n) harmonic to h_{l,mp}(t), for each mp in 'mp' (column j <-> mp[j]).
			# Only if return_waveform_pieces=True:
			'Nlm_p', 'phase', 'omega', 'DomegaDt'
			},
			}
		}

		Raises
		------
		ValueError
			If neither `times` nor `delta_t` is provided.
		"""
		if not self.params['Compute_hlm_Modes']:
			raise ValueError("generate_tdomain_hlm_modes requires the class to be initialized with Compute_hlm_Modes=True")

		#if no time-array is given, create it
		if times is None:
			if delta_t is None: raise ValueError("To compute time-domain waveform, please give either an array of times or a time spacing delta_t")
			#make an equally spaced array for all available times
			times = np.arange(self.sol.all_ts[0], self.sol.all_ts[-1], delta_t)
			#by construction, all times are valid
			i_valid = np.arange(len(times))
			valid_times = times
		else:
			#make sure it is a numpy array
			times = np.asarray(times)
			#find valid times (i.e. where the system has been simulated)
			i_valid = np.nonzero((times>=self.sol.all_ts[0]) & (times<=self.sol.all_ts[-1]))[0]
			valid_times = times[i_valid]

		#initialize dictionary to store the result. mode_convention='pyefpehm_native_td' flags that 'hlm' satisfies only this model's own,
		#self-consistent time-domain reconstruction contract (see docstring above) -- not proven to equal any external convention's h_lm(t).
		#Note this is a *different* label from generate_hlm_modes' pyefpehm_native_fd; the two conventions must not be mixed.
		result = {'times': times, 'mode_convention': 'pyefpehm_native_td', 'modes': {}}

		#find the values of interp_idxs for each time
		t_idxs, interp_idxs = sorted_vals_in_intervals(valid_times, self.sol.all_ts[self.mode_interp_idx], self.sol.all_ts[self.mode_interp_idx+1])

		#compute \lambda and \delta\lambda at the valid times
		valid_l, valid_dl = self.sol(valid_times, derivative=0, idxs=[2,3])

		#if required, compute also the first and second derivatives of the phase
		if return_waveform_pieces:
			valid_DlDt  , valid_DdlDt   = self.sol(valid_times, derivative=1, idxs=[2,3])
			valid_DDlDt2, valid_DDdlDt2 = self.sol(valid_times, derivative=2, idxs=[2,3])

		#group by multipole_idxs
		multipole_idxs = self.necessary_multipole_idxs[interp_idxs]
		grouped_im_idxs = my_cgroup_idxs_by_vals(multipole_idxs)

		#loop over (l,m) co-precessing modes
		for im, im_idxs in enumerate(grouped_im_idxs):

			#if there are no indexes for this mode, skip it
			if len(im_idxs)==0: continue

			#extract l and m
			l, m = self.mode_array[im]

			#compute the interp_idxs, ps and time_idxs corresponding to this mode
			interp_idxs_im = interp_idxs[im_idxs]
			ps = self.necessary_ps[interp_idxs_im]
			t_idxs_im = t_idxs[im_idxs]

			#compute the minimum and maximum time indexes for this mode
			t_idxs_im_min = np.amin(t_idxs_im)
			t_idxs_im_max = np.amax(t_idxs_im)

			#compute the raw (un-projected) hlm building blocks for this mode in [tmin, tmax]. Unlike raw_Apc_prec_im, there is no cheap conjugate
			#shortcut between the +m and -m building blocks (that shortcut only holds for the fully spherical-harmonic-projected Apc_prec), so we
			#need both variants (the -m one is only needed, and only exists, for m>0 modes; for m==0, ps>=0 always so it is never used)
			times_im_range = valid_times[t_idxs_im_min:(t_idxs_im_max+1)]
			hlm_im_pos = self.raw_hlm_prec_im(times_im_range, im, np.array([], dtype=int))
			if m>0: hlm_im_neg = self.raw_hlm_prec_im(times_im_range, im, np.arange(len(times_im_range)))
			else  : hlm_im_neg = hlm_im_pos

			#minimum value of p
			pmin = np.amin(ps)

			#divide also by the value of p
			grouped_ip_idxs = my_cgroup_idxs_by_vals(ps - pmin)

			#loop over p modes
			for ip, p_idxs in enumerate(grouped_ip_idxs):

				#if there are no indexes for this mode, skip it
				if len(p_idxs)==0: continue

				#reconstruct p
				p = ip + pmin

				#Compute the (sorted) times corresponding to this mode
				t_idxs_ip = t_idxs_im[p_idxs]

				#Compute the raw hlm building block corresponding to this mode (select the +m or -m variant depending on the sign of p)
				hlm_ip = (hlm_im_neg if p<0 else hlm_im_pos)[t_idxs_ip-t_idxs_im_min]

				#compute the phase of this mode = p\lambda + (m - p)\delta\lambda
				phi_t = p*valid_l[t_idxs_ip] + (m - p)*valid_dl[t_idxs_ip]

				#handle when p is negative
				if p<0:

					#make sure the phase is positive doing [p,m]->[-p,-m]
					phi_t = -phi_t

					#define label for this mode
					mode_label = (l, -m, -p)
				else:
					mode_label = (l,  m,  p)

				#compute Fourier mode amplitudes for this mode
				Nlm_p = self.compute_Nlm_p(valid_times[t_idxs_ip], interp_idxs_im[p_idxs])

				#compute the raw complex hlm(t), mirroring the h(t)=2 h0_pref Re(A(t) e^{-i phi(t)}) construction of generate_tdomain_modes, but without collapsing to the real part
				hlm_t = (2*self.h0_pref)*hlm_ip*Nlm_p[:,None]*(np.cos(phi_t) - 1j*np.sin(phi_t))[:,None]

				#store (sorted) time indices, inertial mp labels, and raw hlm values corresponding to this mode in result dictionary
				result['modes'][mode_label] = {'time_idxs': i_valid[t_idxs_ip], 'mp': np.arange(-l, l+1), 'hlm': hlm_t}

				#If required, store extra waveform pieces in dictionary
				if return_waveform_pieces:
					#compute the first and second derivatives of the phase
					omega    = p*valid_DlDt[t_idxs_ip]   + (m - p)*valid_DdlDt[t_idxs_ip]
					DomegaDt = p*valid_DDlDt2[t_idxs_ip] + (m - p)*valid_DDdlDt2[t_idxs_ip]
					#handle when p is negative, doing [p,m]->[-p,-m]
					if p<0: omega, DomegaDt = -omega, -DomegaDt
					#add stuff to dictionary that will be returned
					result['modes'][mode_label].update({'Nlm_p': Nlm_p, 'phase': phi_t, 'omega': omega, 'DomegaDt': DomegaDt})

		#return result dictionary
		return result

	#function to reconstruct [h_+, h_x] from the dict returned by generate_hlm_modes/generate_tdomain_hlm_modes,
	#applying the pyefpehm_native_fd/pyefpehm_native_td reconstruction contract documented in those methods
	#(they are DIFFERENT, non-interchangeable conventions -- see their docstrings), so callers do not need
	#to re-derive (or guess) the conj()/no-conj split themselves
	def reconstruct_polarizations_from_native_modes(self, result, Theta=None, Phi=None):
		r"""
		Reconstruct [h_+, h_x] from the dict returned by generate_hlm_modes(freqs, ...) or
		generate_tdomain_hlm_modes(times, ...), applying the documented pyefpehm_native_fd /
		pyefpehm_native_td reconstruction contract (see those methods' docstrings for the exact
		formula and what has and has not been proven about each convention -- they are different
		and must not be mixed, e.g. by passing a frequency-domain 'result' but expecting the
		time-domain, un-conjugated formula to apply, or vice versa).

		Parameters
		----------
		result : dict
			Output of generate_hlm_modes or generate_tdomain_hlm_modes (distinguished by the
			presence of a 'freqs' vs 'times' key). Must carry the 'mode_convention' key that both
			of those methods populate ('pyefpehm_native_fd' or 'pyefpehm_native_td' respectively);
			a mismatched or missing value raises ValueError rather than silently applying the wrong
			domain's formula.
		Theta, Phi : float, optional
			Fixed viewing angles (radians) at which to project the modes.
			Default: this model's own (theta_JN, phi_JN), i.e. the same angles
			generate_waveform/generate_tdomain_waveform use internally (derived from the
			`inclination`/`phase` parameters) - Default: None (uses this model's own angles)

		Returns
		-------
		waveform : ndarray of shape (2, N)
			[h_+, h_x]. Complex (frequency domain) if `result` came from generate_hlm_modes, real
			(time domain) if it came from generate_tdomain_hlm_modes - matching the return
			convention of generate_waveform/generate_tdomain_waveform respectively.

		Raises
		------
		ValueError
			If `result` carries neither a 'freqs' nor a 'times' key, or if its 'mode_convention'
			does not match the domain implied by which of those two keys is present.
		"""
		is_fd = 'freqs' in result
		is_td = 'times' in result
		if not (is_fd or is_td):
			raise ValueError("result must be the output of generate_hlm_modes (has a 'freqs' key) "
			                 "or generate_tdomain_hlm_modes (has a 'times' key); got neither.")
		expected_convention = 'pyefpehm_native_fd' if is_fd else 'pyefpehm_native_td'
		if result.get('mode_convention') != expected_convention:
			raise ValueError("result['mode_convention']=%r does not match the expected %r for this "
			                 "domain (%s); the frequency- and time-domain native reconstruction "
			                 "contracts are different and must not be mixed."
			                 % (result.get('mode_convention'), expected_convention, 'FD' if is_fd else 'TD'))

		n_out = len(result['freqs']) if is_fd else len(result['times'])
		idx_key = 'freq_idxs' if is_fd else 'time_idxs'

		#default to this model's own fixed viewing angles
		if Theta is None: Theta = np.arccos(self.cos_theta_JN)
		if Phi is None: Phi = self.phi_JN

		#build the [Ap,Ac] projector at the requested viewing angles for every l present, mirroring
		#interpolate_Apc_prec's construction of self.Apc_projs. The projector only depends on the
		#*parity* of the co-precessing m (not its sign, since (-1)**(-m)==(-1)**m), so it is keyed
		#here by (l, abs(m)) and is valid for both the m>=0 and (post relabeling) m<0 mode labels
		#that appear in `result['modes']`
		l_array = np.unique(self.mode_array[:,0])
		m2_Ylmp = compute_m2_Ylm(np.cos(Theta), Phi, l_array=l_array)
		m2_Ylmp = {l: m2_Ylmp[il] for il, l in enumerate(l_array)}
		Apc_projs_req = {}
		for l, m in self.mode_array:
			m2_Ylmp_mod = np.conj(m2_Ylmp[l][::-1])
			if (m%2)==0: m2_Ylmp_mod[1::2] = -m2_Ylmp_mod[1::2]
			else       : m2_Ylmp_mod[::2]  = -m2_Ylmp_mod[::2]
			Ap_proj =   0.5*(m2_Ylmp[l] + m2_Ylmp_mod)
			Ac_proj = -0.5j*(m2_Ylmp[l] - m2_Ylmp_mod)
			Apc_projs_req[(l, m)] = np.transpose([Ap_proj, Ac_proj])

		#accumulate each (l, m, n) harmonic's contribution to [h_+, h_x]
		waveform = np.zeros((n_out, 2), dtype=(np.complex128 if is_fd else float))
		for (l, m, n), mode in result['modes'].items():
			Proj = Apc_projs_req[(l, abs(m))]
			#frequency domain needs the *projector* conjugated before contracting with hlm (NOT the other
			#way around: hlm is complex, so conj(hlm*Proj) != hlm*conj(Proj) -- see generate_hlm_modes
			#docstring for why conj(Proj) is the correct one); time domain uses Proj unconjugated, then
			#takes the real part (see generate_tdomain_hlm_modes docstring)
			Proj_used = np.conj(Proj) if is_fd else Proj
			contrib = np.tensordot(mode['hlm'], Proj_used, axes=(1, 0)) #shape (N_mode, 2)
			if not is_fd: contrib = contrib.real
			np.add.at(waveform, mode[idx_key], contrib)

		return np.transpose(waveform)

