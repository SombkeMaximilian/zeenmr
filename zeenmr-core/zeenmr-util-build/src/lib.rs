#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Build script utilities for the ZeeNMR library.

mod cpu_cache;
pub use cpu_cache::cache_rs;
