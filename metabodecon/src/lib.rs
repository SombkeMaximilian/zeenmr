#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! This crate provides data structures and a deconvolution algorithm for 1D
//! NMR spectra. The deconvolution algorithm is based on the work of
//! [Koh et al. (2009)].
//!
//! [Koh et al. (2009)]: https://doi.org/10.1016/j.jmr.2009.09.003.
//!
//! # Overview
//!
//! The primary type of the [`spectrum module`](spectrum) is the [`Spectrum`]
//! data structure. An instance of it can be created with [`Spectrum::new`],
//! where the data must satisfy the invariants described in the documentation of
//! the type. Usually, however, [`Spectrum`] instances are created by reading
//! data from various file formats, of which this crate currently supports the
//! following:
//! - **Bruker**: Use the [`Bruker`] interface to parse Bruker-formatted data.
//!   Requires the `bruker` feature.
//! - **JCAMP-DX**: Use the [`JcampDx`] interface to parse JCAMP-DX files.
//!   Requires the `jdx` feature.
//!
//! [`Spectrum`]: spectrum::Spectrum
//! [`Spectrum::new`]: spectrum::Spectrum::new
//! [`Bruker`]: spectrum::Bruker
//! [`JcampDx`]: spectrum::JcampDx
//!
//! The [`deconvolution module`](deconvolution) provides types and functions to
//! deconvolute a [`Spectrum`] into its constituent signals. The primary type in
//! this module is the [`Deconvoluter`] struct, which stores the configuration
//! of the deconvolution algorithm. The deconvolution algorithm consists of 3
//! modular parts, which can be configured individually, though currently only
//! one algorithm is implemented for each part:
//! 1. Smoothing the intensities of the [`Spectrum`]
//! 2. Peak detection and selection
//! 3. Fitting [Lorentzian functions] to the selected peaks
//!
//! [`Deconvoluter`]: deconvolution::Deconvoluter
//! [`Spectrum`]: spectrum::Spectrum
//! [Lorentzian functions]: https://en.wikipedia.org/wiki/Cauchy_distribution
//!
//! The most important methods of the [`Deconvoluter`] struct are:
//! - [`Deconvoluter::new`] create a new [`Deconvoluter`] with the provided
//!   settings. It also implements [`Default`] for a configuration that is
//!   suitable for most cases.
//! - [`Deconvoluter::add_ignore_region`] adds a ppm region to ignore, which can
//!   be useful for excluding, for example, compounds like stabilizing agents or
//!   a water signal.
//! - [`Deconvoluter::deconvolute_spectrum`] deconvolutes a [`Spectrum`] as
//!   described above.
//! - [`Deconvoluter::par_deconvolute_spectrum`] parallelizes certain parts of
//!   the deconvolution process using Rayon. This requires the `parallel`
//!   feature.
//! - [`Deconvoluter::deconvolute_spectra`] can be used to deconvolute multiple
//!   spectra at once.
//! - [`Deconvoluter::par_deconvolute_spectra`] parallelizes the deconvolution
//!   of multiple spectra across the spectra and within each deconvolution. This
//!   requires the `parallel` feature.
//!
//! [`Deconvoluter`]: deconvolution::Deconvoluter
//! [`Deconvoluter::new`]: deconvolution::Deconvoluter::new
//! [`Deconvoluter::add_ignore_region`]: deconvolution::Deconvoluter::add_ignore_region
//! [`Deconvoluter::deconvolute_spectrum`]: deconvolution::Deconvoluter::deconvolute_spectrum
//! [`Deconvoluter::par_deconvolute_spectrum`]: deconvolution::Deconvoluter::par_deconvolute_spectrum
//! [`Deconvoluter::deconvolute_spectra`]: deconvolution::Deconvoluter::deconvolute_spectra
//! [`Deconvoluter::par_deconvolute_spectra`]: deconvolution::Deconvoluter::par_deconvolute_spectra
//!
//! For a custom configuration of the deconvolution algorithm, the settings
//! enums are provided in the [`deconvolution`] module:
//!
//! - [`SmoothingSettings`](deconvolution::SmoothingSettings)
//! - [`SelectionSettings`](deconvolution::SelectionSettings)
//! - [`FittingSettings`](deconvolution::FittingSettings)
//!
//! The [`Deconvolution`] struct is the output of the deconvolution process. It
//! contains the [`Lorentzian`]s that were fitted to the selected peaks, as well
//! the settings that were used and the [MSE] between the superposition of the
//! fitted [`Lorentzian`]s and the original spectrum. The [`Lorentzian`] struct
//! is a data structure that represents a [Lorentzian function], and provides
//! methods to evaluate its value at a point or multiple points, as well as
//! computing the area under the peak.
//!
//! [`Deconvolution`]: deconvolution::Deconvolution
//! [`Lorentzian`]: deconvolution::Lorentzian
//! [MSE]: https://en.wikipedia.org/wiki/Mean_squared_error
//! [Lorentzian function]: https://en.wikipedia.org/wiki/Cauchy_distribution
//!
//! # Crate Status
//!
//! This crate is in an early stage of development and is not yet feature
//! complete. Features that are currently planned:
//! - More file formats for reading spectra
//! - More algorithms for smoothing, peak selection, and fitting
//! - An alignment algorithm for deconvolutions of multiple spectra, to make
//!   them comparable
//!
//! # Feature flags
//!
//! All feature flags are additive, meaning that they can be combined. The
//! following feature flags are available:
//!
//! | Flag      | Description                                         | Dependencies         |
//! |-----------|---------------------------------------------------- |----------------------|
//! | `bruker`  | Bruker file format interface for reading spectra.   | [Regex], [Byteorder] |
//! | `jdx`     | JCAMP-DX file format interface for reading spectra. | [Regex]              |
//! | `parallel`| Parallelization of the deconvolution process.       | [Rayon]              |
//! | `serde`   | Serialization and deserialization of data.          | [Serde]              |
//!
//! [Regex]: https://docs.rs/regex
//! [Byteorder]: https://docs.rs/byteorder
//! [Rayon]: https://docs.rs/rayon
//!
//! By default, all of these features are enabled.
//!
//! # Serialization with Serde
//!
//! All data structures in this crate can be serialized and deserialized using
//! [Serde] if the `serde` feature is enabled. Note that a [`Spectrum`] parsed
//! from a file will not contain all the information stored in said file.
//! Generally, only the relevant information for interpreting the spectrum is
//! retained.
//!
//! [Serde]: https://serde.rs/
//!
//! # Examples
//!
//! A [`Spectrum`] can be created manually from the chemical shifts, intensities
//! and signal region boundaries. Optionally, metadata can be provided as well.
//! Note that the nucleus and frequency do not affect the data and are not used
//! in the deconvolution process. A new reference compound can be set to shift
//! the spectrum to a new reference point.
//!
//! ```
//! use metabodecon::spectrum::Spectrum;
//!
//! # fn main() -> metabodecon::Result<()> {
//! // Generate 2^15 chemical shifts between 0 and 10 ppm.
//! let chemical_shifts = (0..2_u32.pow(15))
//!     .map(|i| i as f64 * 10.0 / (2_f64.powi(15) - 1.0))
//!     .collect::<Vec<f64>>();
//!
//! // Generate intensities using 2 Lorentzian peaks.
//! let intensities = chemical_shifts
//!     .iter()
//!     .map(|x| {
//!         // Left signal centered at 3 ppm.
//!         1.0 * 0.25 / (0.25_f64.powi(2) + (x - 3.0).powi(2))
//!             // Right signal centered at 7 ppm.
//!             + 1.0 * 0.25 / (0.25_f64.powi(2) + (x - 7.0).powi(2))
//!     })
//!     .collect::<Vec<f64>>();
//!
//! // Define the signal region.
//! let signal_boundaries = (1.0, 9.0);
//!
//! // Create a Spectrum object.
//! let mut spectrum = Spectrum::new(chemical_shifts, intensities, signal_boundaries)?;
//!
//! // Add metadata
//! spectrum.set_nucleus("13C");
//! spectrum.set_frequency(400.0);
//!
//! // Set the central point as the reference and shift it to 4.8.
//! spectrum.set_reference_compound((4.8, 2_usize.pow(14) - 1));
//! # Ok(())
//! # }
//! ```
//!
//! A [`Spectrum`] can also be parsed from the JCAMP-DX file format...
//!
//! [`Spectrum`]: spectrum::Spectrum
//!
//! ```
//! use metabodecon::spectrum::JcampDx;
//!
//! # fn main() -> metabodecon::Result<()> {
//! let path = "path/to/root";
//! # let path = "../data/jcamp-dx/blood";
//!
//! // Read all spectra from a directory of JCAMP-DX files.
//! let spectra = JcampDx::read_spectra(
//!     path,
//!     // Signal boundaries
//!     (-2.2, 11.8),
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ...and from the Bruker file format.
//!
//! ```
//! use metabodecon::spectrum::Bruker;
//!
//! # fn main() -> metabodecon::Result<()> {
//! let path = "path/to/root";
//! # let path = "../data/bruker/blood";
//!
//! // Read all spectra from Bruker TopSpin format directories within the root.
//! let spectra = Bruker::read_spectra(
//!     path,
//!     // Experiment number
//!     10,
//!     // Processing number
//!     10,
//!     // Signal boundaries
//!     (-2.2, 11.8),
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! The [`Deconvoluter`] can be used to deconvolute a [`Spectrum`]. The result
//! is a [`Deconvolution`] object that contains the fitted [`Lorentzian`]s.
//!
//! ```
//! use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
//! use metabodecon::spectrum::Bruker;
//!
//! # fn main() -> metabodecon::Result<()> {
//! // Read spectrum in Bruker TopSpin format.
//! let path = "path/to/root";
//! # let path = "../data/bruker/blood";
//! let spectra = Bruker::read_spectra(
//!     path,
//!     // Experiment number
//!     10,
//!     // Processing number
//!     10,
//!     // Signal boundaries
//!     (-2.2, 11.8),
//! )?;
//!
//! // Deconvolute the spectrum.
//! let deconvoluter = Deconvoluter::default();
//! let deconvolution = deconvoluter.deconvolute_spectra(&spectra)?;
//!
//! // in parallel!
//! let deconvolution = deconvoluter.par_deconvolute_spectra(&spectra)?;
//! # Ok(())
//! # }
//! ```
//!
//! [`Deconvoluter`] is modular and allows you to configure the smoothing, peak
//! selection, and fitting settings independently, though currently only one
//! method is implemented for each. Additionally, you can specify regions to be
//! ignored during the deconvolution. This may be useful for compounds like
//! stabilizing agents or a water signal.
//!
//! ```
//! use metabodecon::deconvolution::{
//!     Deconvoluter, FittingSettings, ScoringMethod, SelectionSettings, SmoothingSettings,
//! };
//!
//! # fn main() -> metabodecon::Result<()> {
//! // Create a new Deconvoluter with the desired settings.
//! let mut deconvoluter = Deconvoluter::new(
//!     SmoothingSettings::MovingAverage {
//!         iterations: 3,
//!         window_size: 3,
//!     },
//!     SelectionSettings::NoiseScoreFilter {
//!         scoring_method: ScoringMethod::MinimumSum,
//!         threshold: 5.0,
//!     },
//!     FittingSettings::Analytical { iterations: 20 },
//! )?;
//!
//! // Add a region to ignore during deconvolution.
//! deconvoluter.add_ignore_region((4.7, 4.9))?;
//! # Ok(())
//! # }
//! ```

#[macro_use]
pub(crate) mod macros;
pub(crate) const CHECK_PRECISION: f64 = 1.0e+3 * f64::EPSILON;

mod settings;
pub(crate) use settings::Settings;

pub mod spectrum;

pub mod deconvolution;

mod error;
pub use error::{Error, Result};
