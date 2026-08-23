#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing NMR spectra and their properties.

pub mod error;

pub mod iter;

pub mod axis;

pub mod dimension;

pub mod intensity_array;

mod nucleus;
pub use nucleus::Nucleus;

mod spectrum;
pub use spectrum::{Spectrum1D, SpectrumView1D};

pub mod builder_1d {
    //! Builder pattern for spectra.

    pub use crate::spectrum::{Builder1D, NeedsAxis, NeedsRange};
}
