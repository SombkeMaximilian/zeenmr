#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! ZeeNMR spectrum-related data types.

mod nucleus;
pub use nucleus::Nucleus;

mod reference;
pub use reference::{ReferencingMethod, ShiftReference};

mod signal_boundaries;
pub use signal_boundaries::SignalBoundaries;

mod spectral_linspace;
pub(crate) use spectral_linspace::SpectralLinspace;

mod spectrum;
pub use spectrum::Spectrum;

pub mod error;
