import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import zeenmr as zn
import numpy as np


def plot_deconvolutions(spectra, deconvolutions, focus):
    fig = plt.figure(figsize=(12, 10), dpi=200)
    gs = gridspec.GridSpec(2, 1, figure=fig)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = itertools.cycle(prop_cycle.by_key()['color'])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlabel("Peak Center")
    ax1.set_ylabel("Half Width")

    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.set_xlabel("Chemical Shift")
    ax2.set_ylabel("Intensity")

    max_intensities = []
    for s, d in zip(spectra, deconvolutions):
        focus_idx = (
            (np.abs(s.chemical_shifts - focus[0])).argmin(),
            (np.abs(s.chemical_shifts - focus[1])).argmin()
        )
        x = s.chemical_shifts[focus_idx[0]:focus_idx[1]]
        lorentzians = d.lorentzians
        y = zn.Lorentzian.par_superposition_vec(x, lorentzians)
        max_intensities.append(np.max(y))

    offset_factor = np.mean(max_intensities) * 0.7

    for i, (s, d) in enumerate(zip(spectra, deconvolutions)):
        offset = (len(spectra) - i + 1) * offset_factor
        focus_idx = (
            (np.abs(s.chemical_shifts - focus[0])).argmin(),
            (np.abs(s.chemical_shifts - focus[1])).argmin()
        )
        x = s.chemical_shifts[focus_idx[0]:focus_idx[1]]
        lorentzians = d.lorentzians
        y = zn.Lorentzian.par_superposition_vec(x, lorentzians) + offset

        centers = np.array([l.maxp for l in lorentzians if focus[0] <= l.maxp <= focus[1]])
        maxima = zn.Lorentzian.par_superposition_vec(centers, lorentzians) + offset
        half_widths = np.array([l.hw for l in lorentzians if focus[0] <= l.maxp <= focus[1]])

        color = next(colors)
        ax1.scatter(centers, half_widths, color=color, label=f"Deconvolution {i + 1}")
        ax2.plot(x, y, color=color)
        ax2.scatter(centers, maxima, marker="x", color=color)

    plt.tight_layout()
    ax1.legend()
    plt.show()
    plt.close()


def main():
    spectra = zn.Spectrum.read_bruker_set("../../data/bruker/blood", 10, 10, (-2.2, 11.8))
    spectra = [
        zn.Spectrum(
            chemical_shifts=np.ascontiguousarray(np.flip(spectrum.chemical_shifts)),
            intensities=np.ascontiguousarray(np.flip(spectrum.intensities)),
            signal_boundaries=(-2.2, 11.8)
        )
        for spectrum in spectra
    ]

    deconvoluter = zn.Deconvoluter()
    deconvoluter.add_ignore_region((4.7, 4.9))
    deconvoluter.set_moving_average_smoother(5, 3)
    deconvoluter.set_noise_score_selector(7.0)
    deconvolutions = deconvoluter.par_deconvolute_spectra(spectra)
    aligner = zn.Aligner()
    aligner.set_pairwise_alignment()
    alignment = aligner.align_deconvolutions(deconvolutions)

    plot_deconvolutions(spectra, deconvolutions, (-0.01, 0.01))
    plot_deconvolutions(spectra, alignment.deconvolutions, (-0.01, 0.01))
    for i in range(0, 5):
        step = 0.25
        focus = (2.0 + i * step, 2.0 + (i + 1) * step)
        plot_deconvolutions(spectra, deconvolutions, focus)
        plot_deconvolutions(spectra, alignment.deconvolutions, focus)

if __name__ == "__main__":
    main()
