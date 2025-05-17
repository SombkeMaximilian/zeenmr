use crate::Deconvolution;
use metabodecon::alignment;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct Alignment {
    inner: alignment::Alignment,
}

impl AsRef<alignment::Alignment> for Alignment {
    fn as_ref(&self) -> &alignment::Alignment {
        &self.inner
    }
}

impl From<alignment::Alignment> for Alignment {
    fn from(value: alignment::Alignment) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl Alignment {
    #[getter]
    pub(crate) fn deconvolutions<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyList>, PyErr> {
        PyList::new(
            py,
            self.inner
                .deconvolutions()
                .iter()
                .map(|deconvolution| deconvolution.clone().into())
                .collect::<Vec<Deconvolution>>(),
        )
    }
}
