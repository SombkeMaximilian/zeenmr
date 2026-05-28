#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures and algorithms for deconstructing NMR spectra into their
//! constituent component signals.

/// Small constant to check for non-zero parameters.
pub(crate) const CHECK_PRECISION: f64 = 1.0e+3 * f64::EPSILON;

pub mod error;

pub mod fitting;
pub mod peak_finding;
pub mod smoothing;

mod deconvolute;
#[cfg(feature = "rayon")]
pub use deconvolute::ParDeconvolute;
pub use deconvolute::{
    Deconvolute, Deconvoluter, MissingFitter, MissingPeakFinder, MissingSmoother,
};

pub mod iter {
    //! Iterator adapter traits for deconvolution.

    pub use crate::deconvolute::DeconvoluteMap;
    #[cfg(feature = "rayon")]
    pub use crate::deconvolute::ParDeconvoluteMap;
}

mod deconvolution;
pub use deconvolution::Deconvolution;
