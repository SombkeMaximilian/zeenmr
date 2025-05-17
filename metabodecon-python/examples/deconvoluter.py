import matplotlib.pyplot as plt
import metabodecon as md
import time


def plot_deconvolution(spectrum, deconvolution, water_boundaries):
    x = spectrum.chemical_shifts
    y1 = spectrum.intensities
    y2 = deconvolution.par_superposition_vec(spectrum.chemical_shifts)
    s = spectrum.signal_boundaries
    w = water_boundaries
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(x, y1, label="Original Spectrum")
    plt.plot(x, y2, label="Deconvoluted Spectrum", linewidth=0.5)
    plt.gca().invert_xaxis()
    plt.axvline(x=s[0], color="black", label="Signal Boundaries")
    plt.axvline(x=s[1], color="black")
    plt.axvspan(w[0], w[1], color="cyan", alpha=0.3, label="Water Region")
    plt.xlabel("Chemical Shifts", fontsize=16)
    plt.ylabel("Intensity", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.show()


def main():
    signal = (-2.2, 11.8)
    blood = md.Spectrum.read_bruker("../../data/bruker/blood/blood_01", 10, 10, signal)
    water_boundaries = (4.7, 4.9)

    deconvoluter = md.Deconvoluter()
    deconvoluter.add_ignore_region(water_boundaries)
    t1 = time.time()
    deconvolution = deconvoluter.deconvolute_spectrum(blood)
    t2 = time.time()
    print(f"Sequential {(t2 - t1) * 1000:.3f}ms")
    t1 = time.time()
    deconvolution = deconvoluter.par_deconvolute_spectrum(blood)
    t2 = time.time()
    print(f"Parallel {(t2 - t1) * 1000:.3f}ms")

    deconvolution.write_json("blood_deconvolution.json")
    deconvolution_json = md.Deconvolution.read_json("blood_deconvolution.json")
    deconvolution.write_bin("blood_deconvolution.bin")
    deconvolution_bin = md.Deconvolution.read_bin("blood_deconvolution.bin")

    plot_deconvolution(blood, deconvolution, water_boundaries)
    plot_deconvolution(blood, deconvolution_json, water_boundaries)
    plot_deconvolution(blood, deconvolution_bin, water_boundaries)


if __name__ == "__main__":
    main()
