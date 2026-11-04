#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Python bindings for the ZeeNMR project.

#[macro_use]
mod data_type;
pub use data_type::DataType;

mod array;
pub use array::BoundaryArray;

use pyo3::prelude::*;

/// Version string matching the Rust crate's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// PyO3 main export module.
///
/// The `_` in the name avoids the name collision with the main library crate.
#[pymodule]
fn _zeenmr(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;

    m.add_class::<BoundaryArray>()?;

    Ok(())
}
