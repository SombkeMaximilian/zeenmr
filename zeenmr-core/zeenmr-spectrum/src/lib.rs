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
pub use spectrum::{
    Spectrum, Spectrum1, Spectrum2, Spectrum3, SpectrumArc, SpectrumCow, SpectrumDyn,
    SpectrumOwned, SpectrumRc, SpectrumSliceView, SpectrumView,
};
