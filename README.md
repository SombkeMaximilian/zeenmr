# Metabodecon-Rust

Development was moved to [spang-lab/zeenmr](https://github.com/spang-lab/zeenmr).

Metabodecon is a work-in-progress project that provides functionality for handling and processing 1D NMR spectra. A more
limited version of this was already implemented in R [here](https://github.com/spang-lab/metabodecon/). This project
aims to improve on that by providing a more efficient and flexible implementation in Rust as well as additional features
that are not present in the R version.

[![Build status][build-badge]][build-link]

[build-badge]: https://github.com/SombkeMaximilian/metabodecon-rust/actions/workflows/rust.yml/badge.svg
[build-link]: https://github.com/SombkeMaximilian/metabodecon-rust/actions

## Features

Current planned and implemented features include (additional suggestions are welcome as issues):

- [x] Efficient representation of 1D NMR spectra
- [x] Read 1D NMR spectra from various formats
   - [x] Bruker
   - [x] JCAMP-DX
- [x] Serialization of the data structures with Serde
- [x] Peak detection in 1D NMR spectra
   - [x] Curvature analysis
- [x] Fitting of peaks to deconvolute 1D NMR spectra
   - [x] Lorentzian functions
     - [x] Analytical solution to the system of equations for the Lorentzian parameters using the detected peaks
- [ ] Alignment of 1D NMR spectra using the deconvoluted signals
- [ ] Python bindings
   - [x] Basic bindings
   - [ ] Complete Python package
- [ ] R bindings
   - [ ] Basic bindings
   - [ ] Complete R package
- [ ] CLI
- [ ] GUI

## Getting Started

### Installation

For now, the project is in a very early stage. However, you can already include the library crate in your own project by
adding the following to your `Cargo.toml`:

```toml
[dependencies]
metabodecon = { git = "https://github.com/SombkeMaximilian/metabodecon-rust" }
```

To install the Python bindings, follow these steps:
- activate your virtual environment
- install the `maturin` package using `pip install maturin`
- navigate to the `metabodecon-python` crate and run the following command:

  ```shell
  maturin develop --release
  ```

### Examples

Here is a simple example of how to use the library in Rust:

```rust
use metabodecon::{deconvolution, spectrum};

fn main() -> metabodecon::Result<()> {
    // Read a spectrum from Bruker TopSpin format
    let spectrum = Bruker::read_spectrum(
        "data/bruker/blood/blood_01",
        // Experiment Number
        10,
        // Processing Number
        10,
        // Signal Region
        (-2.2, 11.8),
    )?;
  
    // Deconvoluter with default settings
    let mut deconvoluter = Deconvoluter::default();
  
    // Ignore the water artifact
    deconvoluter.add_ignore_region((4.7, 4.9))?;
  
    // Deconvolute the spectrum
    let deconvoluted_spectrum = deconvoluter.deconvolute_spectrum(&spectrum)?;
  
    // WIP for now
}
```

Here is a simple example of how to use the library in Python:

```python
import matplotlib.pyplot as plt
import metabodecon as md

# Read a spectrum from Bruker TopSpin format
spectrum = md.Spectrum.read_bruker(
    "data/bruker/blood/blood_01",
    # Experiment Number
    10,
    # Processing Number
    10,
    # Signal Region
    (-2.2, 11.8)
)

# Deconvoluter with default options
deconvoluter = md.Deconvoluter()

# Ignore the water artifact
deconvoluter.add_ignore_region((4.7, 4.9))

# Deconvolute the spectrum
deconvolution = deconvoluter.deconvolute_spectrum(spectrum)

# Extract the chemical shifts, intensities, and signal boundaries
x = spectrum.chemical_shifts
y1 = spectrum.intensities
s = spectrum.signal_boundaries
w = (4.7, 4.9)

# Compute the superposition of the deconvoluted peaks
y2 = deconvolution.par_superposition_vec(spectrum.chemical_shifts)

# Plot the spectrum with overlaid deconvoluted signal superposition
plt.figure(figsize = (12, 8), dpi = 300)
plt.plot(x, y1, label = "Original Spectrum")
plt.plot(x, y2, label = "Deconvoluted Spectrum", linewidth=0.5)
plt.gca().invert_xaxis()
plt.axvline(x = s[0], color = "black", label = "Signal Boundaries")
plt.axvline(x = s[1], color = "black")
plt.axvspan(w[0], w[1], color = "cyan", alpha = 0.3, label = "Water Region")
plt.xlabel("Chemical Shifts", fontsize = 16)
plt.ylabel("Intensity", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend()
plt.show()
```

Here is a simple example of how to use the library in R:

```r
# WIP
```

## Developing

### Testing

Run tests:

  ```shell
  cargo test
  ```

### Benchmarking

Run benchmarks:

  ```shell
  cargo bench
  ```

## Contributing

Currently not accepting contributions, as this is part of my thesis. However, feel free to use it and open issues for
suggestions and bug reports. I will get back to them after my thesis is done.

## References

- Hyung-Won Koh et al. “An approach to automated frequency-domain feature extraction in nuclear magnetic resonance
  spectroscopy”.
  [[DOI]](https://doi.org/10.1016/j.jmr.2009.09.003)
  [[ScienceDirect]](https://www.sciencedirect.com/science/article/pii/S1090780709002584)
- Martina Häckl et al. “An R-Package for the Deconvolution and Integration of 1D NMR Data: MetaboDecon1D”.
  [[DOI]](https://doi.org/10.3390/metabo11070452)
  [[MDPI]](https://www.mdpi.com/2218-1989/11/7/452)
- JCAMP. JCAMP-DX: A Standard Form for the Exchange of Infrared Spectra in Computer Readable Form.
  [[JCAMP]](http://www.jcamp-dx.org/protocols/dxir01.pdf)
- JCAMP. JCAMP-DX for NMR.
  [[JCAMP]](http://www.jcamp-dx.org/protocols/dxnmr01.pdf)
- JCAMP. An Extension to the JCAMP-DX Standard File Format, JCAMP-DX V.5.01.
  [[JCAMP]](http://www.jcamp-dx.org/protocols/dx5-01-correctedv2.pdf)
- JCAMP. A Generic JCAMP-DX Standard File Format, JCAMP-DX V.6.00.
  [[JCAMP]](http://www.jcamp-dx.org/drafts/JCAMP6_2b%20Draft.pdf)
- Lampen, Peter. JCAMP-DX test files for NMR. ISAS.
  [[JCAMP]](http://www.jcamp-dx.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the MIT license, shall be licensed as MIT, without any additional terms or conditions.
