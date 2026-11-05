#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Python bindings for the ZeeNMR project.

#[macro_use]
mod data_type;
pub use data_type::DataType;

mod array;
pub use array::BoundaryArray;

/// Version string matching the Rust crate's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// NMR data structures and algorithms.
#[pyo3::pymodule]
mod _zeenmr {
    use pyo3::prelude::*;

    /// Procedural initialization hook.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", crate::VERSION)?;

        Ok(())
    }

    #[pymodule_export]
    use crate::BoundaryArray;
}
