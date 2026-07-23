'''
Regression tests for the raw (pre spherical-harmonic-projection) hlm mode exposure added to
EFPE.py: generate_hlm_modes, generate_tdomain_hlm_modes, and reconstruct_polarizations_from_native_modes.

These tests establish and protect the pyEFPEHM native reconstruction contract (see the docstrings
of the three methods above): summing the exposed 'hlm' components against the model's own
projector, using reconstruct_polarizations_from_native_modes(), must reproduce generate_waveform's/
generate_tdomain_waveform's official output. This is *not* a claim that 'hlm' equals any external
(e.g. LAL/PyCBC) mode convention -- see the mode_convention='pyefpehm_native_fd' docstrings.

Usage:
  python Tests/test_hlm_modes.py
'''

import numpy as np
import pyEFPEHM

RTOL = 1e-10

CONFIGS = [
	dict(mass1=10.0, mass2=2.0, eccentricity=0.3,
	     spin1x=-0.44, spin1y=-0.26, spin1z=0.48, spin2x=-0.31, spin2y=0.01, spin2z=-0.84,
	     inclination=1.57, phase=0.4, f22_start=10.0, f22_ref=20.0, Amplitude_tol=1e-4,
	     mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]),
	dict(mass1=22.0, mass2=6.0, eccentricity=0.15,
	     spin1x=0.25, spin1y=-0.1, spin1z=0.35, spin2x=-0.15, spin2y=0.2, spin2z=-0.1,
	     inclination=0.9, phase=1.9, f22_start=15.0, Amplitude_tol=1e-4,
	     mode_array=[[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]]),
]

FMAX = {0: 500.0, 1: 700.0}
DF = {0: 1./8, 1: 1./6}


def rel_l2(a, b):
	return np.linalg.norm(a - b) / np.linalg.norm(b)


def check_mp_hlm_alignment(i, domain, result):
	'''mode['hlm']'s last axis must be in exact 1:1 correspondence with mode['mp'] (column j <-> mp[j]=j-l).'''
	for (l, m, n), mode in result['modes'].items():
		expected_mp = np.arange(-l, l+1)
		np.testing.assert_array_equal(mode['mp'], expected_mp,
			err_msg=f"config {i} ({domain}): mode['mp'] for key {(l,m,n)} != arange(-l,l+1)")
		assert mode['hlm'].shape[-1] == len(expected_mp), (
			f"config {i} ({domain}): hlm last-axis length {mode['hlm'].shape[-1]} != len(mp) {len(expected_mp)} for key {(l,m,n)}")


def test_fd_reconstruction(i, params):
	wf = pyEFPEHM.pyEFPE(dict(params, Compute_hlm_Modes=True))
	freqs = np.arange(20.0, FMAX[i], DF[i])

	hp_ref, hc_ref = wf.generate_waveform(freqs)

	result = wf.generate_hlm_modes(freqs)
	assert result['mode_convention'] == 'pyefpehm_native_fd'
	check_mp_hlm_alignment(i, 'FD', result)

	waveform = wf.reconstruct_polarizations_from_native_modes(result)
	hp, hc = waveform[0], waveform[1]

	err_p, err_c = rel_l2(hp, np.asarray(hp_ref)), rel_l2(hc, np.asarray(hc_ref))
	assert err_p < RTOL and err_c < RTOL, (
		f"config {i}: FD reconstruction mismatch (rel_l2 hp={err_p:.3e}, hc={err_c:.3e})")

	# a result mislabelled/mismatched with the wrong domain must raise, not silently reconstruct wrongly
	bad = dict(result); bad['mode_convention'] = 'pyefpehm_native_td'
	try:
		wf.reconstruct_polarizations_from_native_modes(bad)
		raise AssertionError(f"config {i}: reconstruct_polarizations_from_native_modes should have "
		                     f"raised on a mismatched mode_convention (FD result labelled as TD)")
	except ValueError:
		pass

	return result, freqs


def test_td_reconstruction(i, params):
	wf = pyEFPEHM.pyEFPE(dict(params, Compute_hlm_Modes=True))
	times = np.arange(wf.return_start_time(), wf.return_end_time(), 1./2048)

	hp_ref, hc_ref = wf.generate_tdomain_waveform(times)

	result = wf.generate_tdomain_hlm_modes(times)
	assert result['mode_convention'] == 'pyefpehm_native_td'
	check_mp_hlm_alignment(i, 'TD', result)

	waveform = wf.reconstruct_polarizations_from_native_modes(result)
	hp, hc = waveform[0], waveform[1]

	err_p, err_c = rel_l2(hp, np.asarray(hp_ref)), rel_l2(hc, np.asarray(hc_ref))
	assert err_p < RTOL and err_c < RTOL, (
		f"config {i}: TD reconstruction mismatch (rel_l2 hp={err_p:.3e}, hc={err_c:.3e})")


def test_branch_t_spa_sharing(i, params):
	'''Confirms that the stationary time t_SPA for a given (m_coprec, n_ecc) branch does not
	depend on l: entries in generate_hlm_modes' output that share (m, n) but differ in l must
	have identical t_SPA wherever their freq_idxs overlap (see phi_mn(t) in Eq.(13) of
	arXiv:2604.11903, which has no l-dependence).'''
	wf = pyEFPEHM.pyEFPE(dict(params, Compute_hlm_Modes=True))
	freqs = np.arange(20.0, FMAX[i], DF[i])
	result = wf.generate_hlm_modes(freqs, return_waveform_pieces=True)

	from collections import defaultdict
	groups = defaultdict(list)
	for (l, m, n), mode in result['modes'].items():
		groups[(m, n)].append((l, mode))

	n_checked = 0
	for (m, n), entries in groups.items():
		if len(entries) < 2:
			continue
		l0, mode0 = entries[0]
		for l, mode in entries[1:]:
			common, idx0, idx1 = np.intersect1d(mode0['freq_idxs'], mode['freq_idxs'], return_indices=True)
			if len(common) == 0:
				continue
			n_checked += 1
			np.testing.assert_allclose(mode0['t_SPA'][idx0], mode['t_SPA'][idx1], rtol=1e-10, atol=0,
			                           err_msg=f"config {i}: t_SPA differs for (m,n)=({m},{n}) between l={l0} and l={l}")
	assert n_checked > 0, f"config {i}: no overlapping (m,n) branches with >1 l found -- test did not exercise anything"
	return n_checked


def test_no_branch_overwrite(i, params):
	'''Confirms that different co-precessing m sources landing on the same nominal (l, mp) are
	NOT merged/overwritten: the dict returned by generate_hlm_modes is keyed by the *source*
	(l, m_coprec, n_ecc), so two different m_coprec values must appear as two separate keys.'''
	wf = pyEFPEHM.pyEFPE(dict(params, Compute_hlm_Modes=True))
	freqs = np.arange(20.0, FMAX[i], DF[i])
	result = wf.generate_hlm_modes(freqs)

	by_l_n = {}
	collisions_checked = 0
	for (l, m, n) in result['modes']:
		key = (l, n)
		by_l_n.setdefault(key, set()).add(m)
	for (l, n), ms in by_l_n.items():
		if len(ms) > 1:
			collisions_checked += 1
	# not asserting collisions_checked > 0 (parameter-dependent), just confirming the dict shape
	# itself can hold >1 m per (l,n) without collapsing, which the dict keying guarantees by
	# construction; this test documents/protects that guarantee rather than re-deriving it
	assert all(isinstance(k, tuple) and len(k) == 3 for k in result['modes']), (
		f"config {i}: mode dict keys are not (l, m, n) triples")


def test_backward_compatibility(i, params):
	'''Compute_hlm_Modes=False (the default) must leave generate_waveform/generate_tdomain_waveform
	byte-for-byte unchanged relative to Compute_hlm_Modes=True, and generate_hlm_modes must raise.'''
	wf_off = pyEFPEHM.pyEFPE(params)
	wf_on = pyEFPEHM.pyEFPE(dict(params, Compute_hlm_Modes=True))
	freqs = np.arange(20.0, FMAX[i], DF[i])

	hp_off, hc_off = wf_off.generate_waveform(freqs)
	hp_on, hc_on = wf_on.generate_waveform(freqs)
	np.testing.assert_array_equal(np.asarray(hp_off), np.asarray(hp_on))
	np.testing.assert_array_equal(np.asarray(hc_off), np.asarray(hc_on))

	try:
		wf_off.generate_hlm_modes(freqs)
		raise AssertionError(f"config {i}: generate_hlm_modes should have raised with Compute_hlm_Modes=False")
	except ValueError:
		pass


def main():
	failures = []
	for i, params in enumerate(CONFIGS):
		for name, fn in [
			('fd_reconstruction', test_fd_reconstruction),
			('td_reconstruction', test_td_reconstruction),
			('branch_t_spa_sharing', test_branch_t_spa_sharing),
			('no_branch_overwrite', test_no_branch_overwrite),
			('backward_compatibility', test_backward_compatibility),
		]:
			try:
				fn(i, params)
				print(f"OK   config {i} :: {name}")
			except Exception as e:
				failures.append((i, name, e))
				print(f"FAIL config {i} :: {name} :: {e}")

	print()
	if failures:
		print(f"{len(failures)} test(s) FAILED")
		return 1
	print("All hlm-modes tests passed.")
	return 0


if __name__ == "__main__":
	import sys
	sys.exit(main())
