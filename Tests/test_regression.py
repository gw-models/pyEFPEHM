'''
Regression test: checks that pyEFPE waveforms are unchanged across code edits.
Run once to create the baseline (Tests/regression_reference.npz), then re-run after
any refactor that is supposed to preserve the output. Reports the relative L2
difference per configuration and fails if it exceeds the tolerance.

Usage:
  python Tests/test_regression.py            # compare against the baseline (or create it if missing)
  python Tests/test_regression.py --update    # (re)create the baseline from the current code
'''

import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import sys
import numpy as np
import scipy
import pyEFPEHM

#location of the stored baseline waveforms
REFERENCE_FILE = "./regression_reference.npz"

#segment length and maximum frequency
SEGLEN = 8
FMAX   = 512

#frequency and time grids used for every configuration
FREQS = np.arange(20.0, FMAX, 1./SEGLEN)
TIMES = np.arange(-SEGLEN, 0.0, 1./(2*FMAX))

#tolerance on the per-configuration relative L2 difference.
RTOL = 1e-12

#representative configurations spanning the physics regimes
def base_params(**kw):
	p = { 'eccentricity': 0.0, 'distance': 400.0, 'f22_start': 20.0,
	     'phase': 0.7, 'mean_anomaly': 1.3, 'inclination': 0.9,
	     'spin1x': 0.0, 'spin1y': 0.0, 'spin1z': 0.0, 'spin2x': 0.0, 'spin2y': 0.0, 'spin2z': 0.0}
	p.update(kw)
	return p

CONFIGS = {
	'equal_mass_non_spinning_circular' : base_params(mass1=10., mass2=10.),
	'equal_mass_non_spinning_eccentric' : base_params(mass1=9., mass2=9., eccentricity=0.3),
	'non_spinning_circular' : base_params(mass1=10., mass2=10., spin1z=0.4, spin2z=-0.3),
	'aligned_eccentric'  : base_params(mass1=20.0, mass2=5.0, eccentricity=0.3, spin1z=0.4,  spin2z=0.1),
	'precessing_circular': base_params(mass1=21.0, mass2=4.5, eccentricity=0.0, spin1x=0.4, spin1y=0.2, spin1z=0.3, spin2x=-0.2, spin2y=0.1, spin2z=0.1),
	'precessing_eccentric': base_params(mass1=22.0, mass2=4.0, eccentricity=0.2, spin1x=0.3, spin1y=0.1, spin1z=0.4, spin2x=-0.2, spin2y=0.15, spin2z=0.1),
	'tidal'              : base_params(mass1=1.6, mass2=1.4, eccentricity=0.1, spin1z=0.02, spin2z=0.01,
	                            Lambda2_1=400.0, Lambda2_2=600.0, pn_tidal_order=15, horizon_absorption=False, NS_quasi_universal_relations=False),
	#same binary with the default NS_quasi_universal_relations=True, so q, o, Lambda3 and Sigma2 are filled from Lambda2
	'tidal_quasi_universal': base_params(mass1=1.6, mass2=1.4, eccentricity=0.1, spin1z=0.02, spin2z=0.01,
	                            Lambda2_1=400.0, Lambda2_2=600.0, pn_tidal_order=15, horizon_absorption=False),
	#configurations with a reference frequency different from the starting frequency, exercising
	#the backward-stitched (f22_ref>f22_start) and pre-evolved (f22_ref<f22_start) solution paths
	'precessing_eccentric_f22_ref_above': base_params(mass1=22.0, mass2=4.0, eccentricity=0.2, spin1x=0.3, spin1y=0.1, spin1z=0.4, spin2x=-0.2, spin2y=0.15, spin2z=0.1, f22_ref=40.0),
	'aligned_eccentric_f22_ref_below'  : base_params(mass1=20.0, mass2=5.0, eccentricity=0.3, spin1z=0.4, spin2z=0.1, f22_ref=15.0),
}

#generate the frequency domain (h_plus, h_cross) waveform for a configuration
def generate_fdomain(params):
	return pyEFPEHM.pyEFPE(params).generate_waveform(FREQS)

#generate the time domain (h_plus, h_cross) waveform for a configuration
def generate_tdomain(params):
	return pyEFPEHM.pyEFPE(params).generate_tdomain_waveform(TIMES)

#the waveform domains to test: label -> generator function
DOMAINS = {'fdomain': generate_fdomain, 'tdomain': generate_tdomain}

#relative L2 difference between two complex arrays
def rel_l2(a, b):
	return 2*np.linalg.norm(a - b)/(np.linalg.norm(a) + np.linalg.norm(b))

#environment versions that can change regressions
def current_versions():
	return {
		'python'  : '%d.%d.%d' % sys.version_info[:3],
		'numpy'   : np.__version__,
		'scipy'   : scipy.__version__,
		'pyEFPEHM': pyEFPEHM.__version__,
	}

def main():
	update = ('--update' in sys.argv)

	#compute the current waveforms (frequency and time domain) for every configuration
	current = {}
	for name, params in CONFIGS.items():
		for domain, generate in DOMAINS.items():
			hp, hc = generate(params)
			current['%s__%s_hp'%(name, domain)] = hp
			current['%s__%s_hc'%(name, domain)] = hc

	#create the baseline if requested or if it does not exist yet
	if update or not os.path.exists(REFERENCE_FILE):
		#store the environment versions alongside the waveforms
		version_arrays = {'_version_%s' % k: np.array(v) for k, v in current_versions().items()}
		np.savez_compressed(REFERENCE_FILE, **current, **version_arrays)
		print("Baseline saved to %s (%d configurations)" % (REFERENCE_FILE, len(CONFIGS)))
		print("Recorded versions: %s" % ", ".join("%s %s" % (k, v) for k, v in current_versions().items()))
		return 0

	#otherwise compare against the baseline
	ref = np.load(REFERENCE_FILE)
	max_diff = 0.0
	failures = []
	print("%-40s  %-11s  %-11s  %-11s  %-11s" % ("Config", "fd h+", "fd hx", "td h+", "td hx"))
	for name in CONFIGS:
		#relative differences for each (domain, polarization): fd h+, fd hx, td h+, td hx
		diffs = [rel_l2(ref['%s__%s_%s'%(name, domain, pol)], current['%s__%s_%s'%(name, domain, pol)])
		         for domain in DOMAINS for pol in ('hp', 'hc')]
		max_diff = max(max_diff, *diffs)
		flag = '' if all(d <= RTOL for d in diffs) else '  <-- CHANGED'
		if flag: failures.append(name)
		print("%-40s  %-11.3e  %-11.3e  %-11.3e  %-11.3e%s" % (name, *diffs, flag))

	print("\nMax relative difference: %.3e (tol %.1e)" % (max_diff, RTOL))
	if failures:
		print("REGRESSION: waveform changed for: %s" % ", ".join(failures))
		#report any environment versions that differ from the baseline, as a likely cause
		cur = current_versions()
		changed = []
		for k, new in cur.items():
			key = '_version_%s' % k
			old = str(ref[key]) if key in ref.files else None
			if old != new:
				changed.append((k, old, new))
		if changed:
			print("\nEnvironment differs from the baseline (possible cause, not necessarily a real regression):")
			for k, old, new in changed:
				print("  %-9s %s -> %s" % (k, old if old is not None else '(not recorded)', new))
		return 1
	print("OK: all waveforms match the baseline.")
	return 0

if __name__ == "__main__":
	sys.exit(main())
