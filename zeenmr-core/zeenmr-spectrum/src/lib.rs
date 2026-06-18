#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing NMR spectra and their properties.

mod nucleus;
pub use nucleus::Nucleus;

mod reference;
pub use reference::ShiftReference;

mod spectral_linspace;
pub(crate) use spectral_linspace::SpectralLinspace;

mod spectral_range;
pub use spectral_range::{
    ChemicalShiftRange, FrequencyRange, IndexRange, RelativeRange, SpectralRange,
    TryFromIndexRange, TryIntoIndexRange,
};

mod spectrum;
pub use spectrum::Spectrum;

pub mod error;
