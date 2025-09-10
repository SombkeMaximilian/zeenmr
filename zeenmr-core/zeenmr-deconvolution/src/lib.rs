#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Data structures and algorithms for deconstructing NMR spectra into their
//! constituent component signals.

pub mod error;

pub mod fitting;
pub mod peak_finding;
pub mod smoothing;

mod deconvoluter;
pub use deconvoluter::Deconvoluter;
