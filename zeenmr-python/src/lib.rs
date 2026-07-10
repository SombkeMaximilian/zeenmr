use pyo3::prelude::*;

#[pymodule]
fn _zeenmr(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
