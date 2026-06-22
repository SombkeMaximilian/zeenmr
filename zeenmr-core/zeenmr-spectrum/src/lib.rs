#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Data structures for representing NMR spectra and their properties.

mod nucleus;
pub use nucleus::Nucleus;

mod reference;
pub use reference::ShiftReference;
