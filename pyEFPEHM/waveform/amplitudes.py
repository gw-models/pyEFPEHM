import numpy as np
import scipy.special
import warnings

#function take out modes from the mode array that are not consistent with the pn_amplitude_order
def clean_mode_array(mode_array, pn_amplitude_order=2):
	
	#select the maximum number of modes in the mode_array depending on the pn_amplitude_order
	if pn_amplitude_order==-1 or pn_amplitude_order>=2:
		max_mode_array = [(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),(4,0),(4,2),(4,4)]
	elif pn_amplitude_order>=1:
		max_mode_array = [(2,0),(2,1),(2,2),(3,1),(3,3)]
	else:
		max_mode_array = [(2,0),(2,2)]

	#convert input mode_array to a set of (l, |m|) tuples: negative-m modes are implied by their positive-m partner
	input_mode_array_set = {(l, abs(m)) for l, m in mode_array}

	#warn about requested modes that are not supported at this pn_amplitude_order
	unsupported_modes = sorted(input_mode_array_set - set(max_mode_array))
	if unsupported_modes:
		warnings.warn("Removing modes %s from mode_array since they are not supported at pn_amplitude_order=%s"%(unsupported_modes, pn_amplitude_order), UserWarning, stacklevel=2)

	#keep modes in max_mode_array that are also in the input mode_array
	mode_array = np.array([mode for mode in max_mode_array if mode in input_mode_array_set], dtype=int)

	#if no supported modes remain, raise an error
	if len(mode_array) == 0:
		raise ValueError("No supported modes remain in mode_array at pn_amplitude_order=%s. Supported modes: %s"%(pn_amplitude_order, max_mode_array))

	return mode_array

#function take process a user specified harmonic_array containing (l,m,p) modes and mode array to be used
def process_harmonic_array(harmonic_array, max_mode_array, pn_amplitude_order=2):

	#in case harmonic list is None, do nothing
	if harmonic_array is None: return harmonic_array, max_mode_array
	
	#Make sure input harmonic_array has the correct format (copy to avoid mutating the caller's array)
	harmonic_array = np.atleast_2d(np.array(harmonic_array, dtype=np.int64))
	if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 3:
		raise ValueError("harmonic_array must have shape (N, 3) with columns (l, m, p); got shape %s"%(harmonic_array.shape,))

	#if a mode has p=0, remove it
	idxs_p_is_0 = harmonic_array[:,2]==0
	if np.any(idxs_p_is_0):
		harmonic_array = harmonic_array[~idxs_p_is_0]
	
	#if a mode has a negative m, do (l, m, p)->(l, -m, -p)
	#for m=0 the p<0 harmonic is the conjugate partner of the p>0 one, so canonicalize it to (l, 0, -p) too
	idxs_neg_m = (harmonic_array[:,1]<0) | ((harmonic_array[:,1]==0) & (harmonic_array[:,2]<0))
	if np.any(idxs_neg_m):
		harmonic_array[idxs_neg_m,1:] = -harmonic_array[idxs_neg_m,1:]
	
	#consider unique (l,m,n) harmonics
	harmonic_array = np.unique(harmonic_array, axis=0)
	
	#consider unique (l,m) modes
	unique_modes = np.unique(harmonic_array[:,:2], axis=0)

	#build a set of supported (l,m) modes for row-wise membership testing
	supported_modes = set(map(tuple, max_mode_array))

	#loop over unique modes
	mode_list = []
	for mode in unique_modes:
		#if this mode is supported at the PN order considered, add it to the list
		if tuple(mode) in supported_modes:
			mode_list.append(mode)
		#otherwise, remove all harmonics that correspond to this mode
		else:
			warnings.warn("Removing harmonics for [l,m]=%s since they are not supported at pn_amplitude_order=%s"%(mode,pn_amplitude_order), UserWarning, stacklevel=2)
			harmonic_array = harmonic_array[~np.all(harmonic_array[:,:2]==mode, axis=1)]

	#if no supported harmonics remain, fall back to automatic mode selection
	if len(mode_list) == 0:
		warnings.warn("All requested harmonics were removed; falling back to automatic mode selection.", UserWarning, stacklevel=2)
		return None, max_mode_array

	#return harmonic_array and mode_array
	return harmonic_array, np.array(mode_list, dtype=int)

#function to apply the negative-m symmetry A^{l,-m} = (-1)^l conj(A^{l,m}) to an amplitude array
def apply_negative_m_symmetry(A, l):
	if (l%2)==0: return  np.conj(A)
	else:        return -np.conj(A)

#Function to compute \sum_{l,m,p>0} |N^{lm}_p|^2 approximating |\exp{-i a y^2}|^2 \approx |1 - i a y^2|^2
#We use that |N^{l, -m}_p|=|N^{l, m}_{-p}| and therefore
#for m!=0 -> \sum_{p>0} ||N^{l m}_p|| + ||N^{l -m}_p|| = \sum_{p \in Z} ||N^{l m}_p|| = ||H^{l m}||^2
#for m==0 -> \sum_{p>0} ||N^{l 0}_p|| = 0.5*||H^{l 0}||^2
def compute_H_norm2_PNconsistent(e2, y, nu, dmu, dchi=0, mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]], pn_amplitude_order=2):
	
	#check if 0.5PN and 1PN corrections are requestied
	if pn_amplitude_order == -1:
		pn_order_0p5, pn_order_1 = True, True
	else:
		pn_order_0p5 = (pn_amplitude_order>=1)
		pn_order_1   = (pn_amplitude_order>=2)
	
	#compute necessary powers of y and dmu
	y2 = y*y
	dmu2 = dmu*dmu
	y2dmu2 = y2*dmu2
	y2_one_m_3nu = y2*(1-3*nu)
	y4_one_m_3nu2 = y2_one_m_3nu*y2_one_m_3nu
	
	#compute necessary functions of e2
	one_m_e2 = 1-e2
	sqrt_1me2 = np.sqrt(one_m_e2)
	one_m_e2_2 = one_m_e2*one_m_e2

	#loop over mode array
	Hnorm = 0
	for im, mode in enumerate(mode_array):
		
		#convert the mode to a tuple for convenience
		mode = tuple(mode)
		
		#loop over implemented modes
		if   mode==(2,0):
			
			#compute the 0PN norm of (2,0) mode
			M20_0PN = (2/3)*(e2/((1 + sqrt_1me2)*sqrt_1me2))
			
			if pn_order_1:
				#compute the 1PN norm of (2,0) mode
				M20_1PN = e2*(38/21 - (34/63)*nu + (1/sqrt_1me2)*(-30/7 + (40/63)*nu + (1/(1 + sqrt_1me2))*(-6/7 - (34/63)*nu)))
				M20_2PN = e2*((2*(27 + 17*nu)**2)/(5292*(1 + sqrt_1me2)*sqrt_1me2) - 57/49 + nu*(-170/441 + nu*(289/1323)) + (-361/294 + nu*(323/441 - nu*(289/2646)))*e2 + (1/sqrt_1me2)*(1081/147 + nu*(143/147 - nu*(331/1323)) + (2363/294 + nu*(-313/147 + nu*(809/5292)))*e2))
				Hnorm += 0.5*(M20_0PN + y2*(M20_1PN + y2*M20_2PN))
			else:
				Hnorm += 0.5*M20_0PN
			
		elif mode==(2,1) and pn_order_0p5:

			#compute modified dmu in prefactor of (2,1) mode
			if pn_order_1: ydmod = y*dmu - 1.5*y2*dchi
			else         : ydmod = y*dmu

			#add the norm of (2,1) mode
			Hnorm += (2*(2 + e2)/(9*sqrt_1me2))*ydmod*ydmod
			
		elif mode==(2,2):
			
			#compute the 0PN norm of (2,2) mode
			M22_0PN = 5/sqrt_1me2 - 1
			if pn_order_1:
				#compute the 1PN norm of (2,2) mode
				M22_1PN = 9/7 + (17/21)*nu + (19/7 - (17/21)*nu)*e2 + (1/sqrt_1me2)*(-65/3 + (29/3)*nu + (5/3 + (16/3)*nu)*e2)
				M22_2PN = 72999/196 + nu*(-2851/98 + nu*(6767/1764)) + e2*(5625/98 + nu*(4115/147 - nu*(3239/882)) + e2*(-361/196 + nu*(323/294 - nu*(289/1764)))) + (1/sqrt_1me2)*(-611195/1764 + nu*(2119/882 + nu*(5333/1764)) +  e2*(223463/441 + nu*(-117743/882 + nu*(11677/882)) + e2*(-141737/3528 + nu*(11234/441 - nu*(5147/3528))))) + 48*(15 - (4 - nu)*sqrt_1me2)*np.log(0.5*(1 + 1/sqrt_1me2)) + 72*(5/sqrt_1me2 - 1)*scipy.special.spence(2*sqrt_1me2/(1 + sqrt_1me2))
				Hnorm += M22_0PN + y2*(M22_1PN + y2*M22_2PN)
			else:
				Hnorm += M22_0PN

		elif mode==(3,0) and pn_order_1:

			#add the norm of (3,0) mode
			Hnorm += 0.5*(e2*(1/84 + e2/336)/sqrt_1me2)*y4_one_m_3nu2
			
		elif mode==(3,1) and pn_order_0p5:

			#add the norm of (3,1) mode
			Hnorm += ((1/14)*one_m_e2 + (-5/72 + (145/1008)*e2)/sqrt_1me2)*y2dmu2
			
		elif mode==(3,2) and pn_order_1:
		
			#add the norm of (3,2) mode
			Hnorm += ((20/63 + e2*(485/504 + e2*(35/288)))/sqrt_1me2)*y4_one_m_3nu2
			
		elif mode==(3,3) and pn_order_0p5:
			
			#add the norm of (3,3) mode
			Hnorm += ((5/42)*one_m_e2 + (55/24 + (115/48)*e2)/sqrt_1me2)*y2dmu2
			
		elif mode==(4,0) and pn_order_1:
		
			#add the norm of (4,0) mode
			Hnorm += 0.5*((1/98 + e2*(-179/7056 + e2*(67/3136)))/sqrt_1me2 - one_m_e2_2/98)*y4_one_m_3nu2
			
		elif mode==(4,2) and pn_order_1:
		
			#add the norm of (4,2) mode
			Hnorm += ((65/3969 + e2*(-485/15876 + e2*(25/392)))/sqrt_1me2 - (5/441)*one_m_e2_2)*y4_one_m_3nu2
			
		elif mode==(4,4) and pn_order_1:
		
			#add the norm of (4,4) mode
			Hnorm += ((5165/2268 + e2*(119765/18144 + e2*(1035/896)))/sqrt_1me2 - (5/252)*one_m_e2_2)*y4_one_m_3nu2
			
		else:
			warnings.warn("Mode %s is not available at pn_amplitude_order=%s."%(mode, pn_amplitude_order), UserWarning, stacklevel=2)

	return Hnorm

#function to compute the coefficients of the sum in fbeta for the positive frequencies
# $sum_{n = n_0}^n_{max} a_n^pos \beta^{n-n0} J_{p + n}(p*e)$
def compute_an_pos(n, b2, p):
	return 24/(4 + n*n*(-5 + n*n))

#function to compute the coefficients of the sum in fbeta for the negative frequencies
# $sum_{n = n_0}^n_{max} a_n^neg \beta^{n-n0} J_{p - n}(p*e)$
def compute_an_neg(n, b2, p):
	one_m_b2 = 1 - b2
	return 24/(4 + n*n*(-5 + n*n)) + one_m_b2*(24/(-2 + n*(-1 + n*(2 + n))) + one_m_b2*(12/(2 + n*(3 + n)) - (p/((1 + b2)*(n+2)*(n+3)))*one_m_b2*(24/(n*(n+1)) + one_m_b2*(10/(n+1) + one_m_b2))))

# function to apply recurrence relations to make sums of the form
# $sum_{n = n_0}^n_{max} a_n \beta^{n-n0} J_{p + sign*n}(p*e)$
# converge faster by using that J_{p-n} = (p/n)*(J_{p-n} + (\beta/(1 + \beta^2))*(J_{p-n+1} + J_{p-n-1}))
# and then apply the relation J_{p-n} = ((1 + \beta^2)/\beta)*((p-(n-1))/p)*J_{p-(n-1)} - J_{p-(n-2)}
# to make it only depend on n0, n0+1
def a0a1_bessel_recurrence(an_func, n0, nmax, b2, p, sign='+', niter=1):

	#make sure that p is different from 0
	if np.asarray(p).ndim==0:
		if p == 0: return (0, 0)
	elif np.any(p==0):
		#compute where p is not 0
		idxs_p_not_0 = (p!=0)
		#set a to 0 where p is 0
		an = np.zeros((2,)+p.shape)
		#if b2 is also an array, take the values where p is not 0
		if np.asarray(b2).ndim!=0: b2 = b2[idxs_p_not_0]
		#return value of an where p is not 0
		an[:,idxs_p_not_0] = a0a1_bessel_recurrence(an_func, n0, nmax, b2, p[idxs_p_not_0], sign=sign, niter=niter)
		return an
	
	#if the sign is positive, change the sign of p
	if   sign == '+': p = -p
	
	#make sure number of iterations does not surpass maximum (=len(an)-1=nmax - n0)
	len_an =  nmax - n0 + 1
	niter = max(min(niter, len_an-1),0)
	
	#compute array of ns, there are niter extra at the end, required for recurrence relations
	n = n0 + np.arange(len_an + niter)

	#compute required functions of b2
	one_p_b2 = 1 + b2
	b2_1pb2 = b2/one_p_b2
	
	#reshape things to broadcast correctly
	if np.asarray(p).ndim!=0:
		n = n.reshape((len(n),) + (1,)*p.ndim)*np.ones(p.shape, dtype=n.dtype)[None,...]
		p = p[None,...]
		if np.asarray(b2).ndim!=0: b2, one_p_b2 = b2[None,...], one_p_b2[None,...]

	#compute an
	an = an_func(n, b2, p)
	
	#apply recurrence J_{p+n} = - (p/n)*(J_{p+n} + (\beta/(1 + \beta^2))*(J_{p+n+1} + J_{p+n-1}))
	#it could be applied to last niter terms to minimize computational cost but we want to avoid numerical errors
	p_n = p/n
	for i in range(1, niter+1):
		ai_n = p_n[i:len(an)]*an[i:]
		an = an[:-1]
		dai_n = ai_n[1:]-ai_n[:-1]
		an[i-1]   -= ai_n[0]*b2_1pb2
		an[i]      = ai_n[0] - ai_n[1]*b2_1pb2
		an[(i+1):] = (dai_n[:-1] - b2*dai_n[1:])/one_p_b2

	#now apply recurrence J_{p-n} = ((1 + \beta^2)/\beta)*((p-(n-1))/p)*J_{p-(n-1)} - J_{p-(n-2)} to reduce bessel functions
	pfact = ((p - n[2:len_an] + 1)/p)*one_p_b2
	annew = an[:-1].copy()
	for n in range(len_an,2,-1):
		#absorb [2:] bessel functions
		annew[1:n-1] = pfact[:n-2]*an[2:n]
		annew[:n-2] -= b2*an[2:n]
		annew[1]    += an[1]
		#store coefficients
		an[:n-1] = annew[:n-1]

	return an[:2]

#function to analytically compute Nlm_p
def analytical_Nlm_p(p, e2, y, nu, dmu, dchi=0, mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]], fbeta_n_neg_max=18, fbeta_n_pos_max=12, fbeta_niter=1, pn_amplitude_order=2):

	#check if 0.5PN and 1PN corrections are requestied
	if pn_amplitude_order == -1:
		pn_order_0p5, pn_order_1 = True, True
	else:
		pn_order_0p5 = (pn_amplitude_order>=1)
		pn_order_1   = (pn_amplitude_order>=2)
	
	#compute necessary functions of y and dmu
	y2 = y*y
	ydmu = y*dmu
	y2_one_m_3nu = y2*(1-3*nu)
	
	#compute necessary functions of e2
	e = np.sqrt(e2)
	pe = p*e
	one_m_e2 = 1-e2
	sqrt_1me2 = np.sqrt(one_m_e2)
	one_m_e2_3_2 = one_m_e2*sqrt_1me2
	b = e/(1 + sqrt_1me2)
	b2 = b*b
	b3 = b*b2
	b4 = b2*b2
	one_p_b2 = 1 + b2
	
	#compute required Bessel functions J_{p + i}(p*e)
	dp = np.array([1, -1, 2, -2, 3, -3, 4, -4])
	pcomp = np.asarray(p)[None,...] + dp[(slice(None),) + (np.newaxis,)*np.asarray(p).ndim]
	Ji = scipy.special.jv(pcomp,pe)
	
	#compute the symmetrized and antisymmetrized versions
	CJi = Ji[:8:2] + Ji[1:8:2]
	SJi = Ji[:8:2] - Ji[1:8:2]
	
	#loop over mode array
	Nlms = np.zeros((len(mode_array), *np.asarray(p).shape), dtype=np.complex128)
	for im, mode in enumerate(mode_array):

		#convert the mode to a tuple for convenience
		mode = tuple(mode)
		
		#loop over implemented modes
		if   mode==(2,0):
			if pn_order_1: N20_1PN = y2*(-(9/14 + (17/42)*nu + e2*(19/14 - (17/42)*nu))*CJi[0] + p*(1 - e2)*(26/7 - (1/7)*nu)*SJi[0])
			else         : N20_1PN = 0
			Nlms[im] = (6**-0.5)*e*(CJi[0] + N20_1PN)
		elif mode==(2,1) and pn_order_0p5:
			if pn_order_1: ydmod = ydmu - 1.5*y2*dchi
			else         : ydmod = ydmu
			Nlms[im] = (1j/3)*(p*ydmod*one_m_e2*(sqrt_1me2*CJi[0] - SJi[0]))
		elif mode==(2,2):
			Nlms[im] = 0.5*(e*(1 - p*sqrt_1me2)*CJi[0] + p*(2*e*SJi[0] - SJi[1] + sqrt_1me2*CJi[1]))
			#add 1PN correction
			if pn_order_1:
				Nlms[im] += 0.5*y2*(e*(-111/14 + (39/14)*nu + e2*(-19/14 + (17/42)*nu)+ p*(sqrt_1me2*(-115/14 - (19/14)*nu + e2*(356/21 - (11/21)*nu))-p*(1-3*nu)*(5/21 + e2*(-2/7 + e2*(2/21)))))*CJi[0] + e*(sqrt_1me2*(37/7 - (25/21)*nu) + p*(-262/21 + (65/21)*nu + e2*(23/21 + (8/21)*nu) + p*(1-3*nu)*sqrt_1me2*(2/21)*(2 - e2)))*SJi[0] + p*((37/14 - (67/42)*nu)*(SJi[1] - sqrt_1me2*CJi[1]) + p*(1-3*nu)*((CJi[1] - sqrt_1me2*SJi[1])/21)))
				#compute finite part of fbeta
				fbeta_nosum = (b3*(53 + b2*(4 - b2))*0.5*e*CJi[0] + (6 + b2*(-41/3 + b2*(-56/3)))*Ji[0] - (6 + b2*(13 + b2*(7 + b2*(2/3 - b2/3))))*Ji[1] + b4*Ji[4] + Ji[5])/one_p_b2 - b3*Ji[2]/6 - b*(8 + b2*(-6 + b2*(8/3 - b2/2)))*Ji[3]
				#fbeta sum uses backward Bessel recurrences that grow unstable for long windows; tested-stable limits scale with fbeta_niter
				if fbeta_n_pos_max > 22 + 2*fbeta_niter: warnings.warn("fbeta_n_pos_max>%d (at fbeta_niter=%d) makes the (2,2) positive-frequency Bessel recurrence numerically unstable; amplitudes can blow up."%(22+2*fbeta_niter, fbeta_niter), UserWarning, stacklevel=2)
				if fbeta_n_neg_max > 18 + 2*fbeta_niter: warnings.warn("fbeta_n_neg_max>%d (at fbeta_niter=%d) makes the (2,2) negative-frequency Bessel recurrence numerically unstable; amplitudes can blow up."%(18+2*fbeta_niter, fbeta_niter), UserWarning, stacklevel=2)
				#compute infinite part of fbeta
				an_pos = a0a1_bessel_recurrence(compute_an_pos, 3, fbeta_n_pos_max, b2, p, sign='+', niter=fbeta_niter)
				fbeta_sum = b4*(an_pos[0]*Ji[4] + b*an_pos[1]*Ji[6])
				an_neg = a0a1_bessel_recurrence(compute_an_neg, 3, fbeta_n_neg_max, b2, p, sign='-', niter=fbeta_niter)
				fbeta_sum += (p*((1-b2)**3)/one_p_b2)*((7/12 - b2/4)*Ji[5] - (2/5 - b2/5)*b*Ji[3]) + an_neg[0]*Ji[5] + b*an_neg[1]*Ji[7]
				#add fbeta
				Nlms[im] += p*y2*(3*b/(one_p_b2*one_p_b2))*(fbeta_nosum + fbeta_sum)
		elif mode==(3,0) and pn_order_1:
			Nlms[im] = (168**-0.5)*p*y2_one_m_3nu*one_m_e2_3_2*e*CJi[0]
		elif mode==(3,1) and pn_order_0p5:
			Nlms[im] = 1j*((56**-0.5)*ydmu*sqrt_1me2*(1 - p*(5/6)*sqrt_1me2)*(sqrt_1me2*CJi[0] - SJi[0]))
		elif mode==(3,2) and pn_order_1:
			Nlms[im] = ((5/1008)**0.5)*p*y2_one_m_3nu*one_m_e2_3_2*((e*(1 - p*sqrt_1me2))*CJi[0] + p*(sqrt_1me2*CJi[1] + 2*e*SJi[0] - SJi[1]))
		elif mode==(3,3) and pn_order_0p5:
			Nlms[im] = 1j*(((5/672)**0.5)*ydmu*sqrt_1me2*((2*sqrt_1me2 + p*(-13 + 5*e2 + p*sqrt_1me2*(5-4*e2)))*CJi[0] + (-6 + p*(7*sqrt_1me2 + p*(-7 + 4*e2)))*SJi[0]+p*p*(-sqrt_1me2*CJi[2] + SJi[2])))
		elif mode==(4,0) and pn_order_1:
			Nlms[im] = ((2**-1.5)/7)*y2_one_m_3nu*one_m_e2*e*(CJi[0] + p*(5/6)*SJi[0])
		elif mode==(4,2) and pn_order_1:
			Nlms[im] = (5**0.5/252)*y2_one_m_3nu*one_m_e2*(e*(6 + p*(-3*sqrt_1me2 + 2*p*one_m_e2))*CJi[0] + p*(e*(7 - 4*p*sqrt_1me2)*SJi[0] + (6*sqrt_1me2 - 2*p*one_m_e2)*CJi[1] + (-6 + 2*p*sqrt_1me2)*SJi[1]))
		elif mode==(4,4) and pn_order_1:
			Nlms[im] = (((5/7)**0.5)/72)*y2_one_m_3nu*one_m_e2*(e*(6 + p*sqrt_1me2*(-6 + p*(4*sqrt_1me2 + p*(-15 + 8*e2))))*CJi[0] + p*(e*(5 + p*(-20*sqrt_1me2 + 8*p*one_m_e2))*SJi[0] + sqrt_1me2*(12 + p*(-12*sqrt_1me2 + 8*p))*CJi[1] + p*((22*sqrt_1me2 - 6*p*one_m_e2)*SJi[1] + p*(-sqrt_1me2*CJi[3] + one_m_e2*SJi[3]))))
		else:
			warnings.warn("Mode %s is not available at pn_amplitude_order=%s."%(mode, pn_amplitude_order), UserWarning, stacklevel=2)

	return Nlms

#function to compute the Fourier modes needed to describe waveform at a given tolerance
def Fourier_modes_needed(e2, y, nu, dmu, dchi=0, tol=1e-4, pmin=-30, pmax=100, mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]], pn_amplitude_order=2):
	
	#compute analytical norm
	H_norm2 = compute_H_norm2_PNconsistent(e2, y, nu, dmu, dchi=dchi, mode_array=mode_array, pn_amplitude_order=pn_amplitude_order)
	
	#compute array of ps
	ps = np.arange(pmin, pmax+1)
	ps = ps[ps!=0]
	
	#compute Fourier modes
	Nlm = analytical_Nlm_p(ps, e2, y, nu, dmu, dchi=dchi, mode_array=mode_array, pn_amplitude_order=pn_amplitude_order).flatten()

	#set modes with m==0 and p<0 to 0 to avoid double counting them
	Nlm[((np.asarray(mode_array)[:,1]==0)[:,None] & (ps < 0)[None,:]).flatten()] = 0

	#compute the normalized amplitudes
	normed_Nlms = (np.square(Nlm.real) + np.square(Nlm.imag))/H_norm2

	#sort the elements from larger to smaller
	idxs_sort = np.argsort(-normed_Nlms)
	normed_Nlms_sorted = normed_Nlms[idxs_sort]

	#compute cumulative norm
	cum_norm = np.cumsum(normed_Nlms_sorted, axis=0)
	
	#check that total norm of modes makes sense
	if abs(1 - cum_norm[-1])>tol:
		warnings.warn('For e2=%.2g, y=%.2g, nu=%.2g, pmin=%s, pmax=%s, the relative error between the exact norm and the norm estimated with sum is larger than tol=%.3g. (exact - sum)/exact=%.4g'%(e2, y, nu, pmin, pmax, tol, 1 - cum_norm[-1]), UserWarning, stacklevel=2)
	
	#find how many Fourier modes have to be included
	N_fourier_needed = 1 + np.searchsorted(cum_norm, cum_norm[-1] - tol, side='right')

	#compute the orders needed
	idxs_needed = np.unravel_index(idxs_sort[:N_fourier_needed], (len(mode_array), len(ps)))

	#return two arrays, one with index of mode in mode array and other with the corresponding value of p of each
	return idxs_needed[0], ps[idxs_needed[1]]
