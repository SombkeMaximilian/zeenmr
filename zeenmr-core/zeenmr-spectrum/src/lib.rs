#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing NMR spectra and their properties.

pub mod frequency_axis;

mod nucleus;
pub use nucleus::Nucleus;
