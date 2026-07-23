""" Plugins for external packages to access the pyEFPEHM waveform model(s)

Copyright (c) 2026 Gonzalo Morras (gonzalo.morras@aei.mpg.de)

This file is part of pyEFPEHM.
Licensed under the Apache License. See the LICENSE file in the project root for details.
"""

import numpy as np

# No imports should be placed here outside those required for the pyEFPEHM
# package itself to avoid increasing any install requirements for the
# model.

def rename_parameters(parameters):
    # Parameter name conversions, all others passed directly
    # This helps keep a consistent naming convention set for downstream
    # Where possible, sticking to
    # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/parameters.py
    #
    # Note that the native names may be provided in most cases in lieue of
    # these, except in the case of 'f_lower'.
    conversions = {'f_lower': 'f22_start',
                   'f_ref': 'f22_ref',
                   'f_final': 'f22_end',
                   'coa_phase': 'phase',
                   'mean_per_ano': 'mean_anomaly',
                   'dquad_mon1': 'q1',
                   'dquad_mon2': 'q2',
                   }

    # PyCBC's own convention for 'f_final' is that 0 (its default, see
    # pycbc.waveform.parameters.f_final) means "no cap -- use natural termination", same as it
    # being absent entirely. pyEFPEHM's own f22_end does NOT share that convention: 0 is treated as
    # a real cutoff, which shifts the ODE integration's stop time (observed as a several-ms shift in
    # return_start_time, i.e. a spurious linear-in-frequency FD phase error). So treat f_final=0 the
    # same as f_final=None here, unlike every other key (where only None means "not provided").
    unset_values = {'f_final': {None, 0}}

    renamed_params = {
        new_name: parameters[old_name]
        for old_name, new_name in conversions.items()
        if (old_name in parameters) and (parameters[old_name] not in unset_values.get(old_name, {None}))
    }
    parameters.update(renamed_params)


def _mode_array_to_pyEFPEHM(parameters):
    """ Convert a PyCBC-style mode_array (list of (l, m) or (l, m, n) tuples, where n is the
    eccentric harmonic index) into pyEFPEHM's own mode_array/harmonic_array parameters.

    (l, m) entries select the whole (l, m) co-precessing multipole, letting pyEFPEHM choose which
    eccentric harmonics n to include dynamically (controlled by Amplitude_tol); (l, m, n) entries
    pin down one specific harmonic via pyEFPEHM's `harmonic_array`.

    IMPORTANT: pyEFPEHM's `harmonic_array` is all-or-nothing -- as soon as any (l, m, n) entry is
    given, EFPE.pyEFPE.__init__ derives the *entire* set of active modes from harmonic_array alone
    (see process_harmonic_array in amplitudes.py) and no longer does its own dynamic per-(l,m)
    harmonic selection. So a plain (l, m) entry alongside any (l, m, n) entry would silently be
    dropped instead of getting "all its harmonics" as one might expect from mixing the two. To
    avoid silently returning an incomplete waveform, we require mode_array to be either all (l, m)
    or all (l, m, n) -- not a mix. If you need both (some modes with dynamically-chosen harmonics,
    others pinned to one specific harmonic), call get_fd_waveform_modes/get_td_waveform_modes twice
    (once per kind) and combine the results yourself.
    """
    mode_array = parameters.pop('mode_array', None)
    if mode_array is None:
        return

    lengths = {len(entry) for entry in mode_array}
    if lengths - {2, 3}:
        raise ValueError("mode_array entries must be (l, m) or (l, m, n), got %s"%(mode_array,))
    if lengths == {2, 3}:
        raise ValueError("mode_array mixes plain (l, m) entries with (l, m, n) entries (%s); "
                         "pyEFPEHM's harmonic_array is all-or-nothing, so this cannot be honoured "
                         "in one call -- see _mode_array_to_pyEFPEHM's docstring."%(mode_array,))

    if lengths == {2}:
        parameters['mode_array'] = [list(entry) for entry in mode_array]
    else:
        harmonic_array = [list(entry) for entry in mode_array]
        parameters['harmonic_array'] = harmonic_array
        parameters['mode_array'] = [entry[:2] for entry in harmonic_array]


def pycbc_fd_plugin(**parameters):
    """ Interface for the PyCBC package: frequency-domain, summed [h_+, h_x]
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    wf = EFPE.pyEFPE(parameters)
    M = parameters['mass1'] + parameters['mass2']
    delta_f = parameters['delta_f']

    # A pycbc.types.FrequencySeries has no independent frequency-offset field: sample k is defined
    # to be at f=k*delta_f, always starting from f=0. So the active band (f_lower..ISCO) must be
    # snapped onto that same k*delta_f grid and placed at the matching index, with genuine zeros
    # below f_lower -- simply wrapping an array that starts at f_lower would silently mislabel every
    # sample's frequency (off by f_lower), which corrupts anything relying on .sample_frequencies.
    kmin = int(np.ceil(parameters['f_lower']/delta_f))
    kmax = int(np.floor(f_SchwarzISCO(M)/delta_f)) + 1
    freqs = np.arange(kmin, kmax)*delta_f

    hp_active, hc_active = wf.generate_waveform(freqs)

    hp = np.zeros(kmax, dtype=np.complex128); hp[kmin:] = hp_active
    hc = np.zeros(kmax, dtype=np.complex128); hc[kmin:] = hc_active

    epoch = wf.return_start_time()
    hp = FrequencySeries(hp, epoch=epoch, delta_f=delta_f)
    hc = FrequencySeries(hc, epoch=epoch, delta_f=delta_f)
    return hp, hc


def pycbc_td_plugin(**parameters):
    """ Interface for the PyCBC package: time-domain, summed [h_+, h_x]
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import TimeSeries

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    wf = EFPE.pyEFPE(parameters)

    hp, hc = wf.generate_tdomain_waveform(delta_t=parameters["delta_t"])

    epoch = wf.return_start_time()
    hp = TimeSeries(hp, epoch=epoch, delta_t=parameters['delta_t'])
    hc = TimeSeries(hc, epoch=epoch, delta_t=parameters['delta_t'])
    return hp, hc


r""" Mode-by-mode plugins (feed PyCBC's get_fd_waveform_modes/get_td_waveform_modes)

pyEFPEHM natively decomposes the waveform by co-precessing-frame (l, m, n) harmonic (n is the
eccentric sub-harmonic index; see generate_hlm_modes/generate_tdomain_hlm_modes in EFPE.py). Each
such harmonic further contributes to *every* inertial-frame index mp=-l..l (this is what precession
"twist-up" means). Because two different co-precessing m's can independently produce a harmonic with
the same n that both contribute to the same target (l, mp), we key the returned dictionaries by the
4-tuple (l, mp, n, m) -- m being the *co-precessing* label the harmonic came from -- rather than
collapsing/summing same-(l, mp, n) contributions from different m together. Summing across m (and
across n, if desired) for a given (l, mp) is left to the caller.

CONVENTION (mode_convention="pyefpehm_native_fd" for pycbc_fd_modes_plugin, "pyefpehm_native_td" for
pycbc_td_modes_plugin -- these are DIFFERENT, non-interchangeable conventions, also present in
generate_hlm_modes'/generate_tdomain_hlm_modes' own returned dict; do not mix them): this 'hlm' is
*not* claimed to equal any external (e.g. LAL/PyCBC) inertial-frame h_lm. What is proven --
analytically from the pyEFPEHM paper's (arXiv:2604.11903) Eq.(16) Fourier convention, and confirmed
numerically at multiple independent, non-degenerate parameter points to ~1e-15 relative L2 error, see
Tests/test_hlm_modes.py -- is only this model's own, self-consistent reconstruction contract (see
generate_hlm_modes/generate_tdomain_hlm_modes docstrings in EFPE.py for the full derivation and what
has and has not been established):

    frequency domain:  [h_+, h_x](f) = sum_mp hlm[mp](f) * conj(Proj_{l,mp}(theta, phi))
    time domain:       [h_+, h_x](t) = Re[ sum_mp hlm[mp](t) * Proj_{l,mp}(theta, phi) ]

where Proj is built the same way as pyEFPE.pyEFPE.Apc_projs (see interpolate_Apc_prec in EFPE.py) at
the desired (theta, phi); pyEFPE.pyEFPE.reconstruct_polarizations_from_native_modes() applies this for
you. The conj() in the FD case is consistent with (but not proof of) a Fourier-transform sign-convention
difference between the paper's Eq.(16) (a +2*pi*i*f*t kernel) and the standard/LAL/PyCBC convention (a
-2*pi*i*f*t kernel): that argument is only valid for the *real* h_+(t)/h_x(t) polarizations (which is
what is proven here), not for an individual complex 'hlm' component in isolation. Do not apply a global
conj() to these arrays, or treat ulm/vlm below as another model's standard h_lm, on the assumption that
the FD case "should" work like the TD case, or vice versa -- use the formulas above as given.
"""

def pycbc_fd_modes_plugin(**parameters):
    """ Interface for the PyCBC package: frequency-domain (l, mp, n, m) modes, not summed.

    Returns
    -------
    ulm : dict
        (l, mp, n, m) -> Re(hlm(f)), as a pycbc.types.FrequencySeries
    vlm : dict
        (l, mp, n, m) -> Im(hlm(f)), as a pycbc.types.FrequencySeries
    """
    from . import EFPE
    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    parameters['Compute_hlm_Modes'] = True
    wf = EFPE.pyEFPE(parameters)

    M = parameters['mass1'] + parameters['mass2']
    delta_f = parameters['delta_f']

    # Same f=k*delta_f grid-snapping/zero-padding as pycbc_fd_plugin -- see its comment for why
    # (otherwise every mode's FrequencySeries would report frequencies mislabeled by -f_lower).
    kmin = int(np.ceil(parameters['f_lower']/delta_f))
    kmax = int(np.floor(f_SchwarzISCO(M)/delta_f)) + 1
    freqs = np.arange(kmin, kmax)*delta_f

    result = wf.generate_hlm_modes(freqs)
    epoch = wf.return_start_time()

    ulm, vlm = {}, {}
    for (l, m, n), mode in result['modes'].items():
        for j, mp in enumerate(mode['mp']):
            key = (l, int(mp), int(n), int(m))
            hlm_full = np.zeros(kmax, dtype=np.complex128)
            hlm_full[kmin + mode['freq_idxs']] = mode['hlm'][:, j]
            ulm[key] = FrequencySeries(hlm_full.real, epoch=epoch, delta_f=delta_f)
            vlm[key] = FrequencySeries(hlm_full.imag, epoch=epoch, delta_f=delta_f)
    return ulm, vlm


def pycbc_td_modes_plugin(**parameters):
    """ Interface for the PyCBC package: time-domain (l, mp, n, m) modes, not summed.

    Returns
    -------
    ulm : dict
        (l, mp, n, m) -> Re(hlm(t)), as a pycbc.types.TimeSeries
    vlm : dict
        (l, mp, n, m) -> Im(hlm(t)), as a pycbc.types.TimeSeries
    """
    from . import EFPE
    from pycbc.types import TimeSeries

    rename_parameters(parameters)
    _mode_array_to_pyEFPEHM(parameters)
    parameters['Compute_hlm_Modes'] = True
    wf = EFPE.pyEFPE(parameters)

    delta_t = parameters['delta_t']
    result = wf.generate_tdomain_hlm_modes(delta_t=delta_t)
    times = result['times']

    epoch = wf.return_start_time()

    ulm, vlm = {}, {}
    for (l, m, n), mode in result['modes'].items():
        for j, mp in enumerate(mode['mp']):
            key = (l, int(mp), int(n), int(m))
            hlm_full = np.zeros(len(times), dtype=np.complex128)
            hlm_full[mode['time_idxs']] = mode['hlm'][:, j]
            ulm[key] = TimeSeries(hlm_full.real, epoch=epoch, delta_t=delta_t)
            vlm[key] = TimeSeries(hlm_full.imag, epoch=epoch, delta_t=delta_t)
    return ulm, vlm
