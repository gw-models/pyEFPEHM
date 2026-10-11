import sxs
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
        'axes.grid': False,
        'axes.labelsize': 28,
        'axes.linewidth': 1.75,
        'axes.titlesize': 32,
        'font.size': 24,
        'legend.fontsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family': 'serif',
        'font.sans-serif': ['Bitstream Vera Sans'],
        'font.serif': ['Times New Roman'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}',
        'text.usetex':True,
        'patch.force_edgecolor':True,})

#simulation to study
sxs_id = "SXS:BBH:4290"

#load catalog
df = sxs.load("dataframe", tag="3.0.0")

# Obtain and load the strain data
sim = sxs.load(sxs_id)
hraw = sim.h

#preprocess waveform
theta = 0.5*np.pi
phi   = 0
hprep = hraw.preprocess(evaluate_directions=(theta, phi))

#compute frequency domain waveform
delta_t = np.mean(np.diff(hprep.t))
hp_fft = delta_t*np.fft.rfft(hprep.real)
hc_fft = delta_t*np.fft.rfft(hprep.imag)
freqs = np.fft.rfftfreq(len(hprep.real), d=delta_t)

# Plot the real parts of the (2,2) and (2,0) modes
plt.figure(figsize=(13,8))
plt.plot(hraw.t, hraw.real[:, hraw.index(2,2)], label=r"(2,2)")
plt.plot(hraw.t, hraw.real[:, hraw.index(2,0)], label=r"(2,0)")
plt.xlabel("Time ($M$)")
plt.ylabel(r"$\Re\{h^{\ell, m}\}$")
plt.xlim(hraw.t[0], hraw.t[-1])
plt.title(sim.sxs_id)
plt.legend()

# Plot preprocessed h
plt.figure(figsize=(13,8))
plt.plot(hprep.t, hprep.real, label=r"$h_+$")
plt.plot(hprep.t, hprep.imag, label=r"$h_\times$")
plt.xlabel("Time ($M$)")
plt.ylabel(r"Strain")
plt.xlim(hprep.t[0], hprep.t[-1])
plt.title(sim.sxs_id)
plt.legend()

# Plot frequency domain h
plt.figure(figsize=(13,8))
plt.plot(freqs, np.abs(hp_fft), label=r"$|\tilde{h}_+|$")
plt.plot(freqs, np.abs(hc_fft), label=r"$|\tilde{h}_\times|$")
plt.xlabel("Frequency ($1/M$)")
plt.xlim(0.5*df.loc[sxs_id]["reference_orbital_frequency_mag"]/np.pi, 1)
plt.axvline(x=df.loc[sxs_id]["reference_orbital_frequency_mag"]/np.pi, ls='--', color='k', label=r'$f_\mathrm{ref}$')
plt.axvline(x=(6**-1.5)/np.pi, ls=':', color='k', label=r'$f_\mathrm{ISCO}$')
plt.title(sim.sxs_id)
plt.xscale('log')
plt.yscale('log')
plt.legend()


plt.show()
