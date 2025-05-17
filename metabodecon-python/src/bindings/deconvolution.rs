use crate::bindings::Lorentzian;
use crate::error::SerializationError;
use metabodecon::deconvolution;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct Deconvolution {
    inner: deconvolution::Deconvolution,
}

impl AsRef<deconvolution::Deconvolution> for Deconvolution {
    fn as_ref(&self) -> &deconvolution::Deconvolution {
        &self.inner
    }
}

impl From<deconvolution::Deconvolution> for Deconvolution {
    fn from(value: deconvolution::Deconvolution) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl Deconvolution {
    #[getter]
    pub(crate) fn lorentzians<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyList>, PyErr> {
        PyList::new(
            py,
            self.inner
                .lorentzians()
                .iter()
                .map(|lorentzian| (*lorentzian).into())
                .collect::<Vec<Lorentzian>>(),
        )
    }

    #[getter]
    pub(crate) fn mse(&self) -> f64 {
        self.inner.mse()
    }

    pub(crate) fn superposition(&self, chemical_shift: f64) -> f64 {
        deconvolution::Lorentzian::superposition(chemical_shift, self.inner.lorentzians())
    }

    pub(crate) fn superposition_vec<'py>(
        &self,
        py: Python<'py>,
        chemical_shifts: PyReadonlyArray1<'_, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(
            py,
            &deconvolution::Lorentzian::superposition_vec(
                chemical_shifts.as_slice().unwrap(),
                self.inner.lorentzians(),
            ),
        )
    }

    pub(crate) fn par_superposition_vec<'py>(
        &self,
        py: Python<'py>,
        chemical_shifts: PyReadonlyArray1<'_, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(
            py,
            &deconvolution::Lorentzian::par_superposition_vec(
                chemical_shifts.as_slice().unwrap(),
                self.inner.lorentzians(),
            ),
        )
    }

    pub(crate) fn write_json(&self, path: &str) -> PyResult<()> {
        let serialized = match serde_json::to_string_pretty(self.as_ref()) {
            Ok(serialized) => serialized,
            Err(error) => return Err(SerializationError::new_err(error.to_string())),
        };
        std::fs::write(path, serialized)?;

        Ok(())
    }

    #[staticmethod]
    pub(crate) fn read_json(path: &str) -> PyResult<Self> {
        let serialized = std::fs::read_to_string(path)?;

        match serde_json::from_str::<deconvolution::Deconvolution>(&serialized) {
            Ok(deserialized) => Ok(deserialized.into()),
            Err(error) => Err(SerializationError::new_err(error.to_string())),
        }
    }

    pub(crate) fn write_bin(&self, path: &str) -> PyResult<()> {
        let serialized = match rmp_serde::to_vec(self.as_ref()) {
            Ok(serialized) => serialized,
            Err(error) => return Err(SerializationError::new_err(error.to_string())),
        };
        std::fs::write(path, serialized)?;

        Ok(())
    }

    #[staticmethod]
    pub(crate) fn read_bin(path: &str) -> PyResult<Self> {
        let serialized = std::fs::read(path)?;

        match rmp_serde::from_slice::<deconvolution::Deconvolution>(&serialized) {
            Ok(deserialized) => Ok(deserialized.into()),
            Err(error) => Err(SerializationError::new_err(error.to_string())),
        }
    }
}
