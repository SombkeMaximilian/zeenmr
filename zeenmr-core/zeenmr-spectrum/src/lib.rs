#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! ZeeNMR spectrum-related data types.

mod nucleus;
pub use nucleus::Nucleus;

pub mod error;
