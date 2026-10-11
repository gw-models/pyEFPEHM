import numpy as np
from math import factorial, comb


class WignerD:
	"""Class to compute Wigner D-matrices of integer order

	Convention:
		D^l_{m'm}(a, b, g) = exp(-i m' a) * d^l_{m'm}(b) * exp(-i m g)

	The real small-d matrix is stored as a polynomial in cos(b):
		d^l_{m'm}(b) = (sin b)^delta * P^l_{m'm}(cos b),   delta = (m - m') mod 2.
	"""

	def __init__(self, modes):
		modes = np.asarray(modes, dtype=int).reshape(-1, 2)
		self.modes = modes
		self.max_l = int(modes[:, 0].max())

		# Constant coefficients (built once, stored):
		#   _coeffs[(l,m)] : real [2l+1, l+1] poly coeffs in cos(b); row=m'+l, col=power.
		#   _delta[(l,m)]  : int  [2l+1] sin(b) exponent (0/1) per m'.
		self._coeffs = {}
		self._delta = {}
		for l, m in modes:
			self._build(int(l), int(m))

		# angle buffers (filled by update_angles)
		self._cospow = None     # [max_l+1, n]   cos(b)^k  (row k), contiguous
		self._sin_beta = None   # [n]
		self._exp_a = None      # [2max_l+1, n]  exp(-i m' a), row = m'+max_l
		self._exp_g = None      # [max_l, n]     exp(-i k  g), row k-1 (k = 1..max_l)

	# Method to compute the constant coefficients that appear in P^l_{m'm}(cos b) polynomial
	def _build(self, l, m):
		
		# If this coefficient is already in the list, return
		if (l, m) in self._coeffs:
			return
		
		# Otherwise, construct it
		coeffs = np.zeros((2 * l + 1, l + 1))
		delta = np.zeros(2 * l + 1, dtype=int)
		for mp in range(-l, l + 1):
			d = (m - mp) % 2
			delta[mp + l] = d
			norm = np.sqrt(factorial(l + m)*factorial(l - m)*factorial(l + mp)*factorial(l - mp))
			poly = np.zeros(l + 1)
			for s in range(max(0, m - mp), min(l + m, l - mp) + 1):
				A = l - s + (m - mp - d) // 2
				B = s + (mp - m - d) // 2
				sign = -1.0 if (mp - m + s) % 2 else 1.0
				denom = factorial(l + m - s)*factorial(s)*factorial(mp - m + s)*factorial(l - mp - s)
				C = sign * norm / (2 ** l * denom)
				pa = np.array([comb(A, i) for i in range(A + 1)], dtype=float)
				pb = np.array([comb(B, j) * (-1.0) ** j for j in range(B + 1)])
				term = np.convolve(pa, pb)            # (1+x)^A (1-x)^B
				poly[:term.size] += C * term
			coeffs[mp + l] = poly
		self._coeffs[(l, m)] = coeffs
		self._delta[(l, m)] = delta

	# Method to update the Euler angles used to compute the Wigner matrix
	def update_angles(self, alpha, cos_beta, gamma, sin_beta=None):
		alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
		cos_beta = np.atleast_1d(np.asarray(cos_beta, dtype=float))
		gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
		alpha, cos_beta, gamma = np.broadcast_arrays(alpha, cos_beta, gamma)
		L, n = self.max_l, cos_beta.shape[0]

		# sin(b) from cos(b) unless it is given, which keeps its precision when cos(b) is within roundoff of +-1
		if sin_beta is None:
			self._sin_beta = np.sqrt(np.maximum(1.0 - cos_beta ** 2, 0.0))
		else:
			self._sin_beta = np.broadcast_to(np.atleast_1d(np.asarray(sin_beta, dtype=float)), cos_beta.shape)

		# powers of cos(b): [cos^0, cos^1, ..., cos^L] via cumprod -> [L+1, n]
		cospow = np.empty((L + 1, n))
		cospow[0] = 1.0
		cospow[1:] = cos_beta
		self._cospow = np.cumprod(cospow, axis=0)

		# phase ladders via cumprod of exp(-i ang) = cos - i sin
		# alpha: full ladder exp(-i m' a), m' = -L..L, stored as [2L+1, n]
		za = np.cos(alpha) - 1j * np.sin(alpha)
		pos = np.cumprod(np.broadcast_to(za, (L, n)), axis=0)
		self._exp_a = np.concatenate(
			[np.conj(pos[::-1]), np.ones((1, n), dtype=np.complex128), pos], axis=0)
		# gamma: store exp(-i m g) for m = 1..L (negative m handled by conj on use)
		zg = np.cos(gamma) - 1j * np.sin(gamma)
		self._exp_g = np.cumprod(np.broadcast_to(zg, (L, n)), axis=0)

	# Method to release the angle buffers. Any subsequent call to D() requires a new call to update_angles()
	def clear_angles(self, ):
		self._cospow, self._sin_beta, self._exp_a, self._exp_g = None, None, None, None

	# Method to compute the Wigner D-matrix
	def D(self, l, m, indxs=None):
		"""D^l_{m'm} for all m' in [-l, l].  Shape [2l+1, n]; row k is m'=k-l."""
		if l > self.max_l:
			raise ValueError(f"l={l} exceeds max_l={self.max_l}; add it to `modes`.")
		if (l, m) not in self._coeffs:
			self._build(l, m)
		L = self.max_l
		sl = slice(None) if indxs is None else indxs

		# small-d polynomial part (real)
		d = self._coeffs[(l, m)] @ self._cospow[:l + 1, sl]     # [2l+1, n]
		# Apply sin(b) factor where necessary
		odd = self._delta[(l, m)] == 1
		if odd.any():
			d[odd] *= self._sin_beta[sl]

		# Multiply by exp(-i m' a) for all m' (real*complex -> one complex allocation)
		D = d * self._exp_a[L - l:L + l + 1, sl]
		
		# Multiply by exp(-i m g)
		if m > 0:
			D *= self._exp_g[m - 1, sl]
		elif m < 0:
			D *= np.conj(self._exp_g[-m - 1, sl])
		
		return D

#function to compute the spin weighted spherical harmonics with spin -2. From Eq.(F6) of 2502.03929:
#_{-2}Y_{l,m} = (-1)^m \sqrt{\frac{2 l + 1}{4 \pi}} D^l_{-m,-2}(\phi, \theta, 0) = \sqrt{\frac{2 l + 1}{4 \pi}} e^{i m \phi} d^l_{m,2}(\theta)
def compute_m2_Ylm(cos_theta, phi, l_array=[2, 3, 4], sin_theta=None):

	#Initialize Wigner matrices that correspond to requested spherical harmonics
	W = WignerD([[l, 2] for l in l_array])
	#sin_theta is optional, and only matters for its precision when cos_theta is within roundoff of +-1
	W.update_angles(-np.asarray(phi, dtype=float), cos_theta, 0.0, sin_beta=sin_theta)

	#apply the correct normalization
	m2_Ylm = [np.sqrt((2 * l + 1) / (4 * np.pi)) * W.D(l, 2) for l in l_array]

	#if input was scalar, make sure m2_Ylm did not add extra dimentions
	if np.ndim(cos_theta) == 0 and np.ndim(phi) == 0:
		m2_Ylm = [Y[:, 0] for Y in m2_Ylm]
	
	return m2_Ylm
	
	
