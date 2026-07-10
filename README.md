# ZeeNMR

ZeeNMR is a work-in-progress project that provides functionality for handling and processing NMR spectra.

[![Build status][build-badge]][build-link]

[build-badge]: https://github.com/SombkeMaximilian/zeenmr/actions/workflows/rust.yml/badge.svg
[build-link]: https://github.com/SombkeMaximilian/zeenmr/actions

## Features

Current planned and implemented features include (additional suggestions are welcome as issues):

- [x] Representation of 1D NMR spectra
- [x] Representation of nD NMR spectra
- [x] Read NMR spectra from various formats
    - [x] Bruker
    - [x] JCAMP-DX
- [x] Serialization of the data structures with Serde
- [x] Smoothing algorithms for 1D NMR spectra
    - [x] Simple moving average filter
    - [x] Savitzky-Golay filter
- [x] Peak detection in 1D NMR spectra
    - [x] Curvature analysis
- [x] Fitting of peaks to deconvolute 1D NMR spectra
    - [ ] Supported peak shapes 
       - [x] Lorentzian functions
       - [x] Gaussian functions
       - [ ] Voigt profiles
    - [ ] Supported algorithms
       - [x] Three point stencil with iterative refinement
       - [ ] Levenberg-Marquardt
- [ ] Alignment of 1D NMR spectra using the deconvoluted signals
- [ ] Python bindings
    - [ ] Basic bindings
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
zeenmr = { git = "https://github.com/SombkeMaximilian/zeenmr" }
```

To test the Python bindings, follow these steps:
- activate your virtual environment
- install the `maturin` package using `pip install maturin`
- navigate to the `zeenmr-python` crate and run the following command:

  ```shell
  maturin develop --release
  ```

### Examples

Here is a simple example of how to use the library in Rust:

```rust
// WIP
```

Here is a simple example of how to use the library in Python:

```python
# WIP
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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the MIT license, shall be licensed as MIT, without any additional terms or conditions.
