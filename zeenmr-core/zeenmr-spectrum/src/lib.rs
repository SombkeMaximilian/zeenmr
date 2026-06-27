#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing NMR spectra and their properties.

pub mod error;

pub mod frequency_axis;

pub mod intensity_array;

mod nucleus;
pub use nucleus::Nucleus;

mod spectrum;
pub use spectrum::{DualChannel1D, Spectrum1D};

pub mod builder_1d {
    //! Builder pattern for spectra.

    pub use crate::spectrum::{Builder1D, HasAxis, HasRange, NeedsAxis, NeedsRange};
}
