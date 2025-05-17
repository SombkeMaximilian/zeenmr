use crate::error::{MetabodeconError, SerializationError};
use metabodecon::spectrum;
use numpy::PyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct Spectrum {
    inner: spectrum::Spectrum,
}

impl AsRef<spectrum::Spectrum> for Spectrum {
    fn as_ref(&self) -> &spectrum::Spectrum {
        &self.inner
    }
}

impl From<spectrum::Spectrum> for Spectrum {
    fn from(value: spectrum::Spectrum) -> Self {
        Spectrum { inner: value }
    }
}

#[pymethods]
impl Spectrum {
    #[new]
    pub(crate) fn new(
        chemical_shifts: Vec<f64>,
        intensities: Vec<f64>,
        signal_boundaries: (f64, f64),
    ) -> PyResult<Self> {
        match spectrum::Spectrum::new(chemical_shifts, intensities, signal_boundaries) {
            Ok(spectrum) => Ok(spectrum.into()),
            Err(error) => Err(MetabodeconError::from(error).into()),
        }
    }

    #[staticmethod]
    pub(crate) fn read_bruker(
        path: &str,
        experiment: u32,
        processing: u32,
        signal_boundaries: (f64, f64),
    ) -> PyResult<Self> {
        match spectrum::Bruker::read_spectrum(path, experiment, processing, signal_boundaries) {
            Ok(spectrum) => Ok(spectrum.into()),
            Err(error) => Err(MetabodeconError::from(error).into()),
        }
    }

    #[staticmethod]
    pub(crate) fn read_bruker_set(
        path: &str,
        experiment: u32,
        processing: u32,
        signal_boundaries: (f64, f64),
    ) -> PyResult<Vec<Self>> {
        match spectrum::Bruker::read_spectra(path, experiment, processing, signal_boundaries) {
            Ok(spectra) => Ok(spectra
                .into_iter()
                .map(|spectrum| spectrum.into())
                .collect()),
            Err(error) => Err(MetabodeconError::from(error).into()),
        }
    }

    #[staticmethod]
    pub(crate) fn read_jcampdx(path: &str, signal_boundaries: (f64, f64)) -> PyResult<Self> {
        match spectrum::JcampDx::read_spectrum(path, signal_boundaries) {
            Ok(spectrum) => Ok(spectrum.into()),
            Err(error) => Err(MetabodeconError::from(error).into()),
        }
    }

    #[staticmethod]
    pub(crate) fn read_jcampdx_set(
        path: &str,
        signal_boundaries: (f64, f64),
    ) -> PyResult<Vec<Self>> {
        match spectrum::JcampDx::read_spectra(path, signal_boundaries) {
            Ok(spectra) => Ok(spectra
                .into_iter()
                .map(|spectrum| spectrum.into())
                .collect()),
            Err(error) => Err(MetabodeconError::from(error).into()),
        }
    }

    #[getter]
    pub(crate) fn chemical_shifts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.chemical_shifts())
    }

    #[getter]
    pub(crate) fn intensities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.intensities())
    }

    #[getter]
    pub(crate) fn signal_boundaries(&self) -> (f64, f64) {
        self.inner.signal_boundaries()
    }

    #[getter]
    pub(crate) fn nucleus(&self) -> String {
        self.inner.nucleus().to_string()
    }

    #[getter]
    pub(crate) fn frequency(&self) -> f64 {
        self.inner.frequency()
    }

    #[getter]
    pub(crate) fn reference_compound<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let reference = self.inner.reference_compound();
        dict.set_item("chemical_shift", reference.chemical_shift())?;
        dict.set_item("index", reference.index())?;
        match reference.name() {
            Some(name) => dict.set_item("name", name)?,
            None => dict.set_item("name", py.None())?,
        };
        match reference.method() {
            Some(referencing_method) => dict.set_item("method", referencing_method.to_string())?,
            None => dict.set_item("method", py.None())?,
        };

        Ok(dict)
    }

    #[setter]
    pub(crate) fn set_signal_boundaries(&mut self, signal_boundaries: (f64, f64)) -> PyResult<()> {
        match self
            .inner
            .set_signal_boundaries(signal_boundaries)
        {
            Ok(_) => Ok(()),
            Err(e) => Err(MetabodeconError::from(e).into()),
        }
    }

    #[setter]
    pub(crate) fn set_nucleus(&mut self, nucleus: &str) {
        self.inner.set_nucleus(nucleus);
    }

    #[setter]
    pub(crate) fn set_frequency(&mut self, frequency: f64) {
        self.inner.set_frequency(frequency);
    }

    #[setter]
    pub(crate) fn set_reference_compound(&mut self, reference: Bound<'_, PyDict>) -> PyResult<()> {
        let reference = reference.as_any();
        let chemical_shift = reference
            .get_item("chemical_shift")?
            .extract::<f64>()?;
        let index = reference.get_item("index")?.extract::<usize>()?;
        let name = match reference.get_item("name") {
            Ok(name) => match name.is_none() {
                false => Some(name.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("reference compound name must be a string")
                })?),
                true => None,
            },
            Err(_) => None,
        };
        let method = match reference.get_item("method") {
            Ok(method) => match method.is_none() {
                false => Some(
                    method
                        .extract::<String>()
                        .map_err(|_| PyTypeError::new_err("referencing method must be a string"))?
                        .parse()
                        .map_err(|_| {
                            PyValueError::new_err(
                                "referencing method must be either 'external' or 'internal'",
                            )
                        })?,
                ),
                true => None,
            },
            Err(_) => None,
        };
        let reference = spectrum::meta::ReferenceCompound::new(chemical_shift, index, name, method);
        self.inner.set_reference_compound(reference);

        Ok(())
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

        match serde_json::from_str::<spectrum::Spectrum>(&serialized) {
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

        match rmp_serde::from_slice::<spectrum::Spectrum>(&serialized) {
            Ok(deserialized) => Ok(deserialized.into()),
            Err(error) => Err(SerializationError::new_err(error.to_string())),
        }
    }
}
