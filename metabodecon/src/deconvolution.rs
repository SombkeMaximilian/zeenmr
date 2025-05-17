//! The Metabodecon deconvolution algorithm.
//!
//! # Key Components
//!
//! - [`Lorentzian`]: Data structure representing a Lorentzian function.
//! - [`Deconvolution`]: Container for results of the algorithm.
//! - [`Deconvoluter`]: Configuration for the deconvolution algorithm.
//!
//! # The Algorithm
//!
//! The deconvolution process is divided into 3 general steps:
//!
//! 1. Smoothing the intensities of the [`Spectrum`]
//! 2. Peak detection and selection
//! 3. Fitting Lorentzian functions to the selected peaks
//!
//! [`Spectrum`]: crate::spectrum::Spectrum
//!
//! ## Smoothing
//!
//! In order to reduce noise and improve the accuracy of peak detection, the
//! intensities of the [`Spectrum`] are smoothed using the selected method.
//! Currently, only the Moving Average filter is supported.
//!
//! [`Spectrum`]: crate::spectrum::Spectrum
//!
//! ## Peak Detection and Selection
//!
//! After applying the smoothing, peaks in the [`Spectrum`] are detected using
//! by analyzing the curvature of the [`Spectrum`] using the second derivative.
//! This approach has the advantage of being able to detect peaks that are
//! superposed with another, larger peak ("shoulders"). An individual peak is
//! represented by the maximum as well as the left and right inflection points.
//! Other methods are planned, but not yet implemented.
//!
//! [`Spectrum`]: crate::spectrum::Spectrum
//!
//! ## Fitting Lorentzian Functions
//!
//! Finally, the algorithm fits Lorentzian functions to the selected peaks, by
//! iteratively refining the parameters of Lorentzian functions for each peak.
//! This is done by solving a system of 3 equations for the Lorentzian function
//! parameters, using the 3 points that represent each peak. In each iteration,
//! the peak points are adjusted by multiplying their intensity by the ratio of
//! the superposition of the Lorentzian functions to the original [`Spectrum`]
//! at that point. A maximum number of iterations serves as a stopping
//! criterion. In the future, other fitting methods may be implemented.
//!
//! [`Spectrum`]: crate::spectrum::Spectrum
//!
//! # Example: Deconvoluting a [`Spectrum`]
//!
//! [`Spectrum`]: crate::spectrum::Spectrum
//!
//! ```
//! use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
//! use metabodecon::spectrum::Bruker;
//!
//! # fn main() -> metabodecon::Result<()> {
//! // Read a spectrum in Bruker TopSpin format.
//! let path = "path/to/spectrum";
//! # let path = "../data/bruker/blood/blood_01";
//! let spectrum = Bruker::read_spectrum(
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
//! let deconvolution = deconvoluter.deconvolute_spectrum(&spectrum)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Parallelized Deconvolution
//!
//! The most expensive parts of the deconvolution process can also be performed
//! in parallel by enabling the `parallel` feature (part of the `default`
//! features). This adds [Rayon] as a dependency.
//!
//! [rayon]: https://docs.rs/rayon
//!
//! ```
//! use metabodecon::deconvolution::Deconvoluter;
//! use metabodecon::spectrum::Bruker;
//!
//! # fn main() -> metabodecon::Result<()> {
//! // Read a spectrum in Bruker TopSpin format.
//! let path = "path/to/spectrum";
//! # let path = "../data/bruker/blood/blood_01";
//! let spectrum = Bruker::read_spectrum(
//!     path,
//!     // Experiment number
//!     10,
//!     // Processing number
//!     10,
//!     // Signal boundaries
//!     (-2.2, 11.8),
//! )?;
//!
//! // Deconvolute the spectrum in parallel.
//! let deconvoluter = Deconvoluter::default();
//! let deconvolution = deconvoluter.par_deconvolute_spectrum(&spectrum)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Configuring the [`Deconvoluter`]
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

mod deconvoluter;
pub use deconvoluter::Deconvoluter;

mod deconvolution;
pub use deconvolution::Deconvolution;

mod lorentzian;
pub use lorentzian::Lorentzian;

#[cfg(feature = "serde")]
mod serialized_representations;
#[cfg(feature = "serde")]
pub(crate) use serialized_representations::{SerializedDeconvolution, SerializedLorentzian};

mod fitting;
pub use fitting::FittingSettings;

mod peak_selection;
pub use peak_selection::{ScoringMethod, SelectionSettings};

mod smoothing;
pub use smoothing::SmoothingSettings;

pub mod error;
