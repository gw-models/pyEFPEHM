import numpy as np
import time
import pyEFPEHM

#############################################################################################################################

#Function to compute arrays with necessary elements of Wigner (small) d matrix as a function of cos(\theta)
#For our purposes we need d^l_{m',m}, where [l, m] are the modes from the mode array
def compute_necessary_Wigner_small_dl_mpm(cth, mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]], dtype=np.complex128):
	
	#compute sin(n \theta) and cos(n \theta) up to the maximum n required
	cth2 = cth*cth
	sth = (1 - cth2)**0.5
	#sin(2\theta) and cos(2\theta)
	s2th = 2*cth*sth
	c2th = 2*cth2 - 1
	#sin(3\theta) and cos(3\theta)
	s3th = (4*cth2 - 1)*sth
	c3th = (4*cth2 - 3)*cth
	#sin(4\theta) and cos(4\theta)
	s4th = 2*s2th*c2th
	c4th = 2*c2th*c2th - 1
	
	#compute necessary square roots
	sq2 = 2**0.5
	sq3 = 3**0.5
	sq5 = 5**0.5
	sq7 = 7**0.5
	sq3_2 = sq3/sq2
	sq5_2 = sq5/sq2
	sq7_2 = sq7/sq2
	
	#loop over mode array and store d-matrices in a list
	dl_mpm = list()
	for mode in mode_array:

		#convert the mode to a tuple for convenience
		mode = tuple(mode)

		if   mode==(2,0):
			d2_m20 = (0.25*sq3_2)*(1 - c2th) #d^2_{-2,0}
			d2_m10 = (0.5*sq3_2)*s2th        #d^2_{-1,0}
			dl_mpm.append(np.array(
			[d2_m20          ,  #d^2_{-2,0}
			 d2_m10          ,  #d^2_{-1,0}
			 0.25 + 0.75*c2th,  #d^2_{ 0,0}
			 -d2_m10         ,  #d^2_{ 1,0} =-d^2_{-1,0}
			  d2_m20         ], #d^2_{ 2,0} = d^2_{-2,0}
			 dtype=dtype))
		elif mode==(2,1):
			dl_mpm.append(np.array(
			[0.5*sth - 0.25*s2th ,  #d^2_{-2,1}
			 0.5*cth - 0.5*c2th  ,  #d^2_{-1,1}
			 (0.5*sq3_2)*s2th    ,  #d^2_{ 0,1}
			 0.5*cth + 0.5*c2th  ,  #d^2_{ 1,1}
			 -0.5*sth - 0.25*s2th], #d^2_{ 2,1}
			 dtype=dtype))
		elif mode==(2,2):
			dl_mpm.append(np.array(
			[0.375 - 0.5*cth + 0.125*c2th,  #d^2_{-2,2}
			 0.5*sth - 0.25*s2th         ,  #d^2_{-1,2}
			 (0.25*sq3_2)*(1 - c2th)     ,  #d^2_{ 0,2}
			 0.5*sth + 0.25*s2th         ,  #d^2_{ 1,2}
			 0.375 + 0.5*cth + 0.125*c2th], #d^2_{ 2,2}
			 dtype=dtype))
		elif mode==(3,0):
			d3_m30 = (0.0625*sq5)*(3*sth - s3th)    #d^3_{-3,0}
			d3_m20 = (0.125*sq5*sq3_2)*(cth - c3th) #d^3_{-2,0}
			d3_m10 = (0.0625*sq3)*(sth + 5*s3th)    #d^3_{-1,0}
			dl_mpm.append(np.array(
			[d3_m30                ,  #d^3_{-3,0}
			 d3_m20                ,  #d^3_{-2,0}
			 d3_m10                ,  #d^3_{-1,0}
			 0.375*cth + 0.625*c3th,  #d^3_{ 0,0}
			 -d3_m10               ,  #d^3_{ 1,0} =-d^3_{-1,0}
			  d3_m20               ,  #d^3_{ 2,0} = d^3_{-2,0}
			 -d3_m30               ], #d^3_{ 3,0} =-d^3_{-3,0}
			 dtype=dtype))
		elif mode==(3,1):
			dl_mpm.append(np.array(
			[(0.03125*sq3*sq5)*(2 - cth - 2*c2th + c3th)      , #d^3_{-3,1}
			 (0.0625*sq5_2)*(sth + 4*s2th - 3*s3th)           , #d^3_{-2,1}
			 0.1875 - 0.03125*cth + 0.3125*c2th - 0.46875*c3th, #d^3_{-1,1}
			 (0.0625*sq3)*(sth + 5*s3th)                      , #d^3_{ 0,1}
			 0.1875 + 0.03125*cth + 0.3125*c2th + 0.46875*c3th, #d^3_{ 1,1}
			 (0.0625*sq5_2)*(sth - 4*s2th - 3*s3th)           , #d^3_{ 2,1}
			 (0.03125*sq3*sq5)*(2 + cth - 2*c2th - c3th)      ],#d^3_{ 3,1}
			 dtype=dtype))
		elif mode==(3,2):
			dl_mpm.append(np.array(
			[(0.0625*sq3_2)*(5*sth - 4*s2th + s3th) , #d^3_{-3,2}
			 0.3125*cth - 0.5*c2th + 0.1875*c3th    , #d^3_{-2,2}
			 (0.0625*sq5_2)*(sth + 4*s2th - 3*s3th) , #d^3_{-1,2}
			 (0.125*sq5*sq3_2)*(cth - c3th)         , #d^3_{ 0,2}
			 -(0.0625*sq5_2)*(sth - 4*s2th - 3*s3th), #d^3_{ 1,2}
			 0.3125*cth + 0.5*c2th + 0.1875*c3th    , #d^3_{ 2,2}
			 -(0.0625*sq3_2)*(5*sth + 4*s2th + s3th)],#d^3_{ 3,2}
			 dtype=dtype))
		elif mode==(3,3):
			dl_mpm.append(np.array(
			[0.3125 - 0.46875*cth + 0.1875*c2th - 0.03125*c3th, #d^3_{-3,3}
			 (0.0625*sq3_2)*(5*sth - 4*s2th + s3th)           , #d^3_{-2,3}
			 (0.03125*sq3*sq5)*(2 - cth - 2*c2th + c3th)      , #d^3_{-1,3}
			 (0.0625*sq5)*(3*sth - s3th)                      , #d^3_{ 0,3}
			 (0.03125*sq3*sq5)*(2 + cth - 2*c2th - c3th)      , #d^3_{ 1,3}
			 (0.0625*sq3_2)*(5*sth + 4*s2th + s3th)           , #d^3_{ 2,3}
			 0.3125 + 0.46875*cth + 0.1875*c2th + 0.03125*c3th],#d^3_{ 3,3}
			 dtype=dtype))
		elif mode==(4,0):
			d4_m40 = (0.015625*sq7*sq5_2)*(3 - 4*c2th + c4th)  #d^4_{-4,0}
			d4_m30 = (0.03125*sq5*sq7)*(2*s2th - s4th)  #d^4_{-3,0}
			d4_m20 = (0.03125*sq5_2)*(3 + 4*c2th - 7*c4th) #d^4_{-2,0}
			d4_m10 = (0.03125*sq5)*(2*s2th + 7*s4th)  #d^4_{-1,0}
			dl_mpm.append(np.array(
			[d4_m40                                ,  #d^4_{-4,0}
			 d4_m30                                ,  #d^4_{-3,0}
			 d4_m20                                ,  #d^4_{-2,0}
			 d4_m10                                ,  #d^4_{-1,0}
			 0.140625 + 0.3125*c2th + 0.546875*c4th,  #d^4_{ 0,0}
			 -d4_m10                               ,  #d^4_{ 1,0}
			  d4_m20                               ,  #d^4_{ 2,0}
			 -d4_m30                               ,  #d^4_{ 3,0}
			  d4_m40                               ],  #d^4_{ 4,0}
			 dtype=dtype))
		elif mode==(4,2):
			dl_mpm.append(np.array(
			[(0.015625*sq7)*(5 - 4*cth - 4*c2th + 4*c3th-c4th), #d^4_{-4,2}
			 (0.0625*sq7_2)*(sth + 2*s2th - 3*s3th + s4th)    , #d^4_{-3,2}
			 0.03125*(5 - 2*cth + 4*c2th - 14*c3th + 7*c4th)  , #d^4_{-2,2}
			 (0.03125*sq2)*(3*sth + 2*s2th + 7*s3th - 7*s4th) , #d^4_{-1,2}
			 (0.03125*sq5_2)*(3 + 4*c2th - 7*c4th)            , #d^4_{ 0,2}
			 (0.03125*sq2)*(3*sth - 2*s2th + 7*s3th + 7*s4th) , #d^4_{ 1,2}
			 0.03125*(5 + 2*cth + 4*c2th + 14*c3th + 7*c4th)  , #d^4_{ 2,2}
			 (0.0625*sq7_2)*(sth - 2*s2th - 3*s3th - s4th)    , #d^4_{ 3,2}
			 (0.015625*sq7)*(5 + 4*cth - 4*c2th - 4*c3th-c4th)],#d^4_{ 4,2}
			 dtype=dtype))
		elif mode==(4,4):
			dl_mpm.append(np.array(
			[0.0078125*(35-56*cth+28*c2th-8*c3th+c4th)  , #d^4_{-4,4}
			 (0.015625*sq2)*(14*sth-14*s2th+6*s3th-s4th), #d^4_{-3,4}
			 (0.015625*sq7)*(5-4*cth-4*c2th+4*c3th-c4th), #d^4_{-2,4}
			 (0.03125*sq7_2)*(6*sth-2*s2th-2*s3th+s4th) , #d^4_{-1,4}
			 (0.015625*sq7*sq5_2)*(3-4*c2th+c4th)       , #d^4_{ 0,4}
			 (0.03125*sq7_2)*(6*sth+2*s2th-2*s3th-s4th) , #d^4_{ 1,4}
			 (0.015625*sq7)*(5+4*cth-4*c2th-4*c3th-c4th), #d^4_{2,4}
			 (0.015625*sq2)*(14*sth+14*s2th+6*s3th+s4th), #d^4_{ 3,4}
			 0.0078125*(35+56*cth+28*c2th+8*c3th+c4th)  ],#d^4_{ 4,4}
			 dtype=dtype))
		else:
			print("Warning: mode %s not implemented"%(mode))
			dl_mpm.append(np.array([]))
		
	#return list of wigner d-matrices
	return dl_mpm

#Function to compute arrays with necessary elements of Wigner D matrix, as a function of cos(\theta)
#For our purposes we only need D^{l}_{m',m}, where [l,m] are the modes from the mode array
def compute_necessary_Wigner_Dl_mpm(phi, cos_theta, zeta, mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]):

	#compute necessary elements of Wigner (small) d matrix
	dl_mpm = compute_necessary_Wigner_small_dl_mpm(cos_theta, mode_array=mode_array, dtype=np.complex128)

	#compute exp(-i*phi) and exp(-i*zeta)
	exp_miph = np.cos(phi) - 1j*np.sin(phi)
	exp_miz = np.cos(zeta) - 1j*np.sin(zeta)

	#compute the exp(-i*m'*phi) and exp(-i*m*zeta) for the necessary values of m and m'
	lmax = 4
	exp_mimph = np.cumprod(np.broadcast_to(exp_miph[np.newaxis,...], (lmax, *exp_miph.shape)), axis=0)
	exp_mimpz = np.cumprod(np.broadcast_to(exp_miz[np.newaxis,...], (lmax, *exp_miz.shape)), axis=0)

	#loop over mode array and store D-matrices in a list
	Dl_mpm = list()
	for mode, dd in zip(mode_array, dl_mpm):
		
		#multiply d^l_{m',m} by exp(-i*m'*phi)
		dd[:mode[0]] *= np.conj(exp_mimph[(mode[0]-1)::-1]) #m'<0
		dd[(mode[0]+1):] *= exp_mimph[:mode[0]]             #m'>0
		
		#multiply d^l_{m',m} by exp(-i*m*phi) (if m!=0)
		if mode[1]!=0: dd *= exp_mimpz[[mode[1]-1]]
		
		#append it to list of Wigner D-matrices
		Dl_mpm.append(dd)
	
	#return list of Wigner D-matrices
	return Dl_mpm

#Function to compute D^l_{m',-m} = (-1)^{m' + m} (D^l_{-m', m})^{*}
def Dl_mpmm_from_Dl_mpm(Dl_mpm, l, m):

	#if m is 0 we do not have to do anything
	if m==0: return Dl_mpm
	
	#first write (D^l_{-m', m})^{*}
	Dl_mpmm = np.conj(Dl_mpm[::-1])
	
	#now apply (-1)^{m' + m} factor, using that Dl_mpmm[0] = D^l_{-l, m}
	if (m - l)%2==0: Dl_mpmm[1::2] = -Dl_mpmm[1::2]
	else           : Dl_mpmm[::2]  = -Dl_mpmm[::2]
	
	#return D^l_{m',-m} Wigner matrix
	return Dl_mpmm

#function to compute the spin weighted spherical harmonics with spin -2. From Eq.(F6) of 2502.03929:
#_{-2}Y_{l,m} = (-1)^m \sqrt{\frac{2 l + 1}{4 \pi}} D^l_{-m,-2}(\phi, \theta, 0) = \sqrt{\frac{2 l + 1}{4 \pi}} e^{i m \phi} d^l_{m,2}(\theta)
def compute_m2_Ylm(cos_theta, phi, l_array=[2,3,4]):

	#compute necessary elements of Wigner (small) d matrix
	dl_m2 = compute_necessary_Wigner_small_dl_mpm(cos_theta, mode_array=[[l, 2] for l in l_array], dtype=np.complex128)

	#compute exp(i m \phi)
	lmax = 4
	exp_iph = np.cos(phi) + 1j*np.sin(phi)
	exp_imph = np.cumprod(np.broadcast_to(exp_iph[np.newaxis,...], (lmax, *exp_iph.shape)), axis=0)
	
	#loop over values of l and store Spherical Harmonics in a list
	m2_Ylm = list()
	for l, dd in zip(l_array, dl_m2):
		
		#multiply d^l_{m,2} by exp(i*m*phi)
		dd[:l] *= np.conj(exp_imph[(l-1)::-1]) #m<0
		dd[(l+1):] *= exp_imph[:l]             #m>0

		#append it with the correct norm
		m2_Ylm.append((((0.5*l + 0.25)/np.pi)**0.5)*dd)

	#return list of _{-2}Y_{l,m}
	return m2_Ylm
		
		
#############################################################################################################################

#Inputs
Ntest = 1000000  #Number of points for tests
test_tol = 1e-12 #tolerance for tests
test_tol_std = 5 #tolerance for tests, as a function of error expected in montecarlo integration
mode_array = [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]
l_array = [2,3,4]

#############################################################################################################################

#random values of cos_theta, phi and zeta
cth = np.random.uniform(low=-1, high=1, size=Ntest)
phi = np.random.uniform(low=0, high=2*np.pi, size=Ntest)
zeta = np.random.uniform(low=0, high=2*np.pi, size=Ntest)

#initialize wigner D matrix
start_soltime = time.time()
W = pyEFPEHM.wigner.WignerD(mode_array)
print("pyEFPEHM time to initialize Wigner D matrix class: %s seconds" % (time.time() - start_soltime))

#compute small wigner d matrix
start_soltime = time.time()
dl_mpm = compute_necessary_Wigner_small_dl_mpm(cth, mode_array=mode_array)
print("\nTime to evaluate Wigner (small) d matrix         : %s seconds" % (time.time() - start_soltime))
start_soltime = time.time()
W = pyEFPEHM.wigner.WignerD(mode_array)
W.update_angles(np.zeros(Ntest), cth, np.zeros(Ntest))
dl_mpm_pyEFPEHM = [W.D(l, m) for (l, m) in mode_array]
print("pyEFPEHM time to evaluate Wigner (small) d matrix: %s seconds" % (time.time() - start_soltime))
dl_mpm_relerror = [2*np.linalg.norm(d1 - d2)/np.linalg.norm(d1 + d2) for (d1, d2) in zip(dl_mpm, dl_mpm_pyEFPEHM)]
print("Difference between implementation: %s\n"%(dl_mpm_relerror))

#compute Wigner D matrix
start_soltime = time.time()
Dl_mpm = compute_necessary_Wigner_Dl_mpm(phi, cth, zeta, mode_array=mode_array)
print("\nTime to evaluate Wigner D matrix         : %s seconds" % (time.time() - start_soltime))
start_soltime = time.time()
W = pyEFPEHM.wigner.WignerD(mode_array)
W.update_angles(phi, cth, zeta)
Dl_mpm_pyEFPEHM = [W.D(l, m) for (l, m) in mode_array]
print("pyEFPEHM time to evaluate Wigner D matrix: %s seconds" % (time.time() - start_soltime))
Dl_mpm_relerror = [2*np.linalg.norm(D1 - D2)/np.linalg.norm(D1 + D2) for (D1, D2) in zip(Dl_mpm, Dl_mpm_pyEFPEHM)]
print("Difference between implementation: %s\n"%(Dl_mpm_relerror))

#compute Spherical harmonic with its dedicated function
start_soltime = time.time()
m2_Ylm = compute_m2_Ylm(cth, phi, l_array=l_array)
print("\nTime to evaluate Spherical Harmonics         : %s seconds" % (time.time() - start_soltime))
start_soltime = time.time()
m2_Ylm_pyEFPEHM = pyEFPEHM.wigner.compute_m2_Ylm(cth, phi, l_array=l_array)
print("pyEFPEHM time to evaluate Spherical Harmonics: %s seconds" % (time.time() - start_soltime))
m2_Ylm_relerror = [2*np.linalg.norm(Y1 - Y2)/np.linalg.norm(Y1 + Y2) for (Y1, Y2) in zip(m2_Ylm, m2_Ylm_pyEFPEHM)]
print("Difference between implementation: %s\n"%(m2_Ylm_relerror))

#compute Spherical Harmonic from wigner matrices
start_soltime = time.time()
m2_Ylm_from_D = compute_necessary_Wigner_Dl_mpm(-phi, cth, 0, mode_array=[[l,2] for l in l_array])
m2_Ylm_from_D = [(((0.5*l + 0.25)/np.pi)**0.5)*YY for l, YY in zip(l_array, m2_Ylm_from_D)]
print("\nTime to evaluate Spherical Harmonics from D-matrix: %s seconds" % (time.time() - start_soltime))

m2_Ylm_err = np.amax([np.amax(np.abs(Y1 - Y2)) for Y1, Y2 in zip(m2_Ylm, m2_Ylm_from_D)])
print("Difference: Max: %.3g \n"%(np.amax(np.abs(m2_Ylm_err))))

####################  test orthogonality of Wigner D matrix  ####################

#loop over modes
errs_D_ortho = list()
errs_D_ortho_err = list()
for im1, mode1 in enumerate(mode_array):
	for im2, mode2 in enumerate(mode_array):
		
		#extract value of l
		l1, l2 = mode1[0], mode2[0]
		#compute prefactor that makes norm of D-matrix equal to 1
		pref = np.sqrt((2*l1+1)*(2*l1+1))
		
		#loop over signs of m1
		for sign_m1 in [1, -1]:
			
			#extract values of m1
			m1 = sign_m1*mode1[1]
			#extract Wigner matrix
			Dl_mpm_1 = Dl_mpm[im1]
			#change sign of m1 if necessary
			if sign_m1==-1: Dl_mpm_1 = Dl_mpmm_from_Dl_mpm(Dl_mpm_1, l1, mode1[1])
			
			#loop over signs of m2
			for sign_m2 in [1, -1]:
				
				#extract values of m2
				m2 = sign_m2*mode2[1]
				#extract Wigner matrix
				Dl_mpm_2 = Dl_mpm[im2]
				#change sign of m2 if necessary
				if sign_m2==-1: Dl_mpm_2 = Dl_mpmm_from_Dl_mpm(Dl_mpm_2, l2, mode2[1])
				
				#loop over matrix elements
				for iD1, D1 in enumerate(Dl_mpm_1):
					for iD2, D2 in enumerate(Dl_mpm_2):

						#compute the product of D1 and D2
						integrand = D1*np.conj(D2)
						#compute inner product and its error with montecarlo integration
						D1D2_innerprod = pref*np.mean(integrand, axis=-1)
						D1D2_innerprod_err = pref*np.std(integrand, axis=-1)/len(integrand)**0.5
						#if l1=l2, m1=m2 and m1'=m2' inner prod should be 1, else it should be 0
						if (l1==l2) and (m1==m2) and (iD1==iD2):
							err_D_ortho = abs(D1D2_innerprod - 1)
						else:
							err_D_ortho = abs(D1D2_innerprod)

						#append this error and its error
						errs_D_ortho.append(err_D_ortho)
						errs_D_ortho_err.append(abs(D1D2_innerprod_err))
						
						#if error is above tolerance, print it
						if err_D_ortho>test_tol_std*abs(D1D2_innerprod_err): print("l1=%s, l2=%s, m1=%s, m2=%s, m1p=%s, m2p=%s -> %.3g +- %.3g"%(l1, l2, m1, m2, iD1-l1, iD2-l2, err_D_ortho, D1D2_innerprod_err))

#convert to numpy arrays
errs_D_ortho  = np.array(errs_D_ortho)
errs_D_ortho_err = np.array(errs_D_ortho_err)

#print info
print('\nFrom %s tests of orthogonality of D-matrix: Max err: %.3g, Mean err: %.3g, Max err/std: %.3g, Mean err/std: %.3g'%(len(errs_D_ortho), np.amax(errs_D_ortho), np.mean(errs_D_ortho), np.amax(errs_D_ortho/errs_D_ortho_err), np.mean(errs_D_ortho/errs_D_ortho_err)))


####################  test completeness of Wigner D matrix  ####################

#loop over modes
errs_D_complete = list()
for im1, mode1 in enumerate(mode_array):
	for im2, mode2 in enumerate(mode_array):
		#extract value of l
		l1, l2 = mode1[0], mode2[0]
		#we need l1 = l2 to check completeness
		if l1==l2:
			#loop over signs of m1
			for sign_m1 in [1, -1]:
				
				#extract values of m1
				m1 = sign_m1*mode1[1]
				#extract Wigner matrix
				Dl_mpm_1 = Dl_mpm[im1]
				#change sign of m1 if necessary
				if sign_m1==-1: Dl_mpm_1 = Dl_mpmm_from_Dl_mpm(Dl_mpm_1, l1, mode1[1])
				
				#loop over signs of m2
				for sign_m2 in [1, -1]:
					
					#extract values of m2
					m2 = sign_m2*mode2[1]
					#extract Wigner matrix
					Dl_mpm_2 = Dl_mpm[im2]
					#change sign of m2 if necessary
					if sign_m2==-1: Dl_mpm_2 = Dl_mpmm_from_Dl_mpm(Dl_mpm_2, l2, mode2[1])

					#compute the value of the completeness sum
					delta_m1m2 = np.sum(np.conj(Dl_mpm_1)*Dl_mpm_2, axis=0)
					#if m1=m2 this should be 1, else it should be 0. Take maximum error
					if m1==m2:
						err_D_complete = np.amax(np.abs(delta_m1m2 - 1))
					else:
						err_D_complete = np.amax(np.abs(delta_m1m2))

					#append this error
					errs_D_complete.append(err_D_complete)

					#if error is above tolerance, print it
					if err_D_complete>test_tol: print("l1=%s, l2=%s, m1=%s, m2=%s -> %.3g"%(l1, l2, m1, m2, err_D_complete))

#print info
print('\nFrom %s tests of completeness of D-matrix: Max err: %.3g, Mean err: %.3g'%(len(errs_D_complete), np.amax(errs_D_complete), np.mean(errs_D_complete)))

#################### Test D^l_{m,m'}(0,0,0) = \delta_{m',m} ####################

#compute D^l_{m,m'}(0,0,0)
Dl_mpm_000 = compute_necessary_Wigner_Dl_mpm(0, np.cos(0), 0, mode_array=mode_array)

#compare with expectation
Dl_mpm_000_err = list()
for im, mode in enumerate(mode_array):
	for imp in range(len(Dl_mpm_000[im])):
		#if m'=m this has to be 1, otherwise 0
		if imp-mode[0] == mode[1]: Dl_mpm_000_err.append(abs(Dl_mpm_000[im][imp]-1))
		else:                      Dl_mpm_000_err.append(abs(Dl_mpm_000[im][imp]))
		
#print info
print('\nFrom %s tests of D^l_{m,mp}(0,0,0) = \delta_{mp,m}: Max err: %.3g, Mean err: %.3g'%(len(Dl_mpm_000_err), np.amax(Dl_mpm_000_err), np.mean(Dl_mpm_000_err)))

#################### Test orthogonormality of spherical harmonics ####################

#loop over modes
errs_Ylm_ortho = list()
errs_Ylm_ortho_err = list()
for il1, l1 in enumerate(l_array):
	for il2, l2 in enumerate(l_array):
		for im1, m1 in enumerate(range(-l1, l1+1)):
			for im2, m2 in enumerate(range(-l2, l2+1)):
				
				#compute the product of Y1 and Y2
				integrand = (4*np.pi)*m2_Ylm[il1][im1]*np.conj(m2_Ylm[il2][im2])
				#compute inner product and its error with montecarlo integration
				Y1Y2_innerprod = np.mean(integrand, axis=-1)
				Y1Y2_innerprod_err = np.std(integrand, axis=-1)/len(integrand)**0.5
				
				#if l1=l2 and m1=m2 inner prod should be 1, else it should be 0
				if (l1==l2) and (m1==m2):
					err_Ylm_ortho = abs(Y1Y2_innerprod - 1)
				else:
					err_Ylm_ortho = abs(Y1Y2_innerprod)

				#append this error and its error
				errs_Ylm_ortho.append(err_Ylm_ortho)
				errs_Ylm_ortho_err.append(abs(Y1Y2_innerprod_err))


				#if error is above tolerance, print it
				if err_Ylm_ortho>test_tol_std*abs(Y1Y2_innerprod_err): print("l1=%s, l2=%s, m1=%s, m2=%s-> %.3g +- %.3g"%(l1, l2, m1, m2, err_Ylm_ortho, Y1Y2_innerprod_err))

#convert to numpy arrays
errs_Ylm_ortho  = np.array(errs_Ylm_ortho)
errs_Ylm_ortho_err = np.array(errs_Ylm_ortho_err)

#print info
print('\nFrom %s tests of orthogonality of Ylm-matrix: Max err: %.3g, Mean err: %.3g, Max err/std: %.3g, Mean err/std: %.3g'%(len(errs_Ylm_ortho), np.amax(errs_Ylm_ortho), np.mean(errs_Ylm_ortho), np.amax(errs_Ylm_ortho/errs_Ylm_ortho_err), np.mean(errs_Ylm_ortho/errs_Ylm_ortho_err)))

