""" Plugins for external packages to access the pyEFPEHM waveform model(s)
"""

import warnings

import numpy as np

# No imports should be placed here outside those required for the pyEFPEHM
# package itself to avoid increasing any install requirements for the
# model.

# Keys of the default parameter dictionary of pyEFPEHM, plus the required mass1/mass2.
# PyCBC calls a plugin with all of its ~59 waveform parameters and pyEFPEHM warns about
# every key it does not recognise, so they have to be filtered out before building the model.
PYEFPE_KEYS = frozenset((
    'mass1', 'mass2',
    'eccentricity', 'distance', 'inclination',
    'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z',
    'f22_start', 'f22_ref', 'f22_end', 'max_duration',
    'phase', 'mean_anomaly', 'mode_array',
    'q1', 'q2', 'o1', 'o2',
    'Lambda2_1', 'Lambda2_2', 'Lambda3_1', 'Lambda3_2', 'Sigma2_1', 'Sigma2_2',
    'Lambda23_1', 'Lambda23_2', 'Lambda32_1', 'Lambda32_2',
    'Sigma23_1', 'Sigma23_2', 'Sigma32_1', 'Sigma32_2', 'NS_quasi_universal_relations',
    'pn_phase_order', 'pn_spin_order', 'pn_amplitude_order', 'pn_tidal_order',
    'horizon_absorption', 'Amplitude_tol', 'Amplitude_pmax', 'harmonic_array',
    'DJ2_tol', 'RR_sol_rtol', 'RR_sol_atol', 'NecessaryModes_UpdatePeriod',
    'Series_Reversion_Order', 'Interpolate_Amplitudes',
    'Interp_points_per_prec_cycle', 'Extra_interp_points_Nlm_p',
    'SPA_Frequency_rtol', 'SUA_kmax',
))


def rename_parameters(parameters):
    # Parameter name conversions, all others passed directly
    # This helps keep a consistent naming convention set for downstream
    # Where possible, sticking to
    # https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/parameters.py
    #
    # Note that the native names may be provided in most cases in lieue of
    # these, except in the case of 'f_lower', and that they take precedence:
    # PyCBC injects a default for each of its own names into every call, so a
    # native name that is present can only have been set by the user.
    #
    # 'f_final' is absent on purpose: it is the ceiling of the frequency array,
    # not 'f22_end', and is used by pycbc_fd_plugin instead.
    # 'coa_phase' -> 'phase' identifies the coalescence phase with the reference
    # orbital phase at f22_ref (= f22_start by default), since pyEFPEHM has no
    # coalescence-phase parameterisation. Both are the LAL reference phase, and the
    # spins are the LAL source-frame components, so no rotation is needed here.
    # The four '*_order' are forwarded verbatim, including the -1 PyCBC defaults to,
    # which selects the maximum available order in pyEFPEHM as it does in LAL. That is
    # a no-op for a point-particle binary, where the tidal terms vanish identically.
    # 'anomaly' comes last since the comprehension keeps the last match, and
    # PyCBC injects mean_per_ano=0.0 into every call.
    # 'eccentricity', 'mode_array', the masses, spins, distance and inclination
    # are absent since pyEFPEHM already uses the PyCBC name for them.
    conversions = {'coa_phase': 'phase',
                   'lambda1': 'Lambda2_1',
                   'lambda2': 'Lambda2_2',
                   'lambda_octu1': 'Lambda3_1',
                   'lambda_octu2': 'Lambda3_2',
                   'phase_order': 'pn_phase_order',
                   'spin_order': 'pn_spin_order',
                   'amplitude_order': 'pn_amplitude_order',
                   'tidal_order': 'pn_tidal_order',
                   'mean_per_ano': 'mean_anomaly',
                   'anomaly': 'mean_anomaly',
                   }

    renamed_params = {
        new_name: parameters[old_name]
        for old_name, new_name in conversions.items()
        if (old_name in parameters) and (parameters[old_name] is not None)
    }
    for new_name, value in renamed_params.items():
        if parameters.get(new_name) is None:
            parameters[new_name] = value

    # 'f_lower' is -1 in the sequence plugin, where it is not a start frequency
    if (parameters.get('f_lower') is not None) and (parameters['f_lower'] > 0):
        parameters['f22_start'] = parameters['f_lower']

    # In the LAL convention f_ref = 0 means the start frequency, i.e. f22_ref = None
    if parameters.get('f_ref') and (parameters.get('f22_ref') is None):
        parameters['f22_ref'] = parameters['f_ref']

    # LAL's dQuadMon is the quadrupole parameter minus one, so 0 for a black hole
    for old_name, new_name in (('dquad_mon1', 'q1'), ('dquad_mon2', 'q2')):
        if (parameters.get(old_name) is not None) and (parameters.get(new_name) is None):
            parameters[new_name] = 1 + parameters[old_name]


def select_parameters(parameters):
    """ Keep only the parameters pyEFPEHM recognises, so that the ones PyCBC passes
    for other approximants do not each trigger a warning
    """
    return {key: value for key, value in parameters.items() if key in PYEFPE_KEYS}


def pycbc_fd_plugin(**parameters):
    """ Interface for the PyCBC package
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import FrequencySeries
    from pycbc.pnutils import f_SchwarzISCO

    rename_parameters(parameters)
    wf = EFPE.pyEFPE(select_parameters(parameters))
    delta_f = parameters['delta_f']
    delta_t = parameters.get('delta_t')

    # Ceiling of the returned array. PyCBC's f_final defaults to 0, leaving the
    # choice to the approximant: fall back to Nyquist, then to the ISCO.
    if parameters.get('f_final'):
        f_max = parameters['f_final']
    elif delta_t:
        f_max = 0.5/delta_t
    else:
        f_max = f_SchwarzISCO(parameters['mass1'] + parameters['mass2'])

    # A FrequencySeries places sample k at k*delta_f, and pyEFPEHM returns zero where
    # the waveform has no support, so no padding is needed. The array starts at 0
    # since with higher modes the m=1 and low-n harmonics radiate below f22_start.
    kmax = int(f_max/delta_f) + 1
    freqs = np.arange(kmax)*delta_f

    hp, hc = wf.generate_waveform(freqs)

    # As for TaylorF2 and the other inspiral-only approximants, the epoch labels the start
    # of the 1/delta_f segment, with t=0 (coalescence) at its cyclic wrap point.
    # generate_waveform already refers its times to coalescence, so no shift is needed here
    epoch = -1.0/delta_f

    # Beyond that segment the waveform wraps around and aliases onto its own end, and
    # the epoch no longer describes the data
    if wf.return_start_time() < epoch:
        warnings.warn("The pyEFPEHM waveform is %.1f s long, which does not fit in the "
                      "1/delta_f = %.1f s segment requested, so its start wraps around. "
                      "Use a smaller delta_f."%(-wf.return_start_time(), -epoch),
                      UserWarning, stacklevel=2)

    hp = FrequencySeries(hp, epoch=epoch, delta_f=delta_f)
    hc = FrequencySeries(hc, epoch=epoch, delta_f=delta_f)
    return hp, hc


def pycbc_td_plugin(**parameters):
    """ Interface for the PyCBC package
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import TimeSeries

    rename_parameters(parameters)
    wf = EFPE.pyEFPE(select_parameters(parameters))

    hp, hc = wf.generate_tdomain_waveform(delta_t=parameters["delta_t"])

    epoch = wf.return_start_time()
    hp = TimeSeries(hp, epoch=epoch, delta_t=parameters['delta_t'])
    hc = TimeSeries(hc, epoch=epoch, delta_t=parameters['delta_t'])
    return hp, hc


def pycbc_fd_sequence_plugin(**parameters):
    """ Interface for the PyCBC package, evaluating the waveform at an arbitrary
    sequence of frequency points (used e.g. by the relative binning likelihood)

    'f22_start' has to be passed explicitly, since PyCBC overwrites 'f_lower' with -1 here.
    """
    # Do imports here to avoid all possibility of circular imports that
    # could affect the overall package
    from . import EFPE

    from pycbc.types import Array

    rename_parameters(parameters)

    # PyCBC only checks 'required' when it calls the plugin itself, so guard direct calls
    if parameters.get('f22_start') is None:
        raise ValueError("Please provide f22_start")

    # PyCBC hands the sample points over as a pycbc Array
    freqs = np.asarray(parameters['sample_points'], dtype=float)

    wf = EFPE.pyEFPE(select_parameters(parameters))

    # generate_waveform needs sorted frequencies, so sort them and undo it afterwards
    i_sort = np.argsort(freqs)
    waveform = wf.generate_waveform(freqs[i_sort])

    hp, hc = np.empty_like(waveform[0]), np.empty_like(waveform[1])
    hp[i_sort], hc[i_sort] = waveform[0], waveform[1]

    # A sequence plugin returns plain Arrays, not FrequencySeries
    return Array(hp), Array(hc)


# Otherwise PyCBC would demand the f_lower and delta_f it just overwrote with -1, but
# that leaves no start frequency, so f22_start has to be given
pycbc_fd_sequence_plugin.required = ['mass1', 'mass2', 'approximant', 'f22_start']
