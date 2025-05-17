use metabodecon::deconvolution;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct Lorentzian {
    inner: deconvolution::Lorentzian,
}

impl AsRef<deconvolution::Lorentzian> for Lorentzian {
    fn as_ref(&self) -> &deconvolution::Lorentzian {
        &self.inner
    }
}

impl From<deconvolution::Lorentzian> for Lorentzian {
    fn from(value: deconvolution::Lorentzian) -> Self {
        Self { inner: value }
    }
}

#[pymethods]
impl Lorentzian {
    #[new]
    pub(crate) fn new(sf: f64, hw: f64, maxp: f64) -> Self {
        Lorentzian {
            inner: deconvolution::Lorentzian::new(sf * hw, hw.powi(2), maxp),
        }
    }

    #[staticmethod]
    pub(crate) fn from_transformed(sfhw: f64, hw2: f64, maxp: f64) -> Self {
        Lorentzian {
            inner: deconvolution::Lorentzian::new(sfhw, hw2, maxp),
        }
    }

    #[getter]
    pub(crate) fn sf(&self) -> f64 {
        self.inner.sf()
    }

    #[getter]
    pub(crate) fn hw(&self) -> f64 {
        self.inner.hw()
    }

    #[getter]
    pub(crate) fn maxp(&self) -> f64 {
        self.inner.maxp()
    }

    #[setter]
    pub(crate) fn set_sf(&mut self, sf: f64) {
        self.inner.set_sf(sf);
    }

    #[setter]
    pub(crate) fn set_hw(&mut self, hw: f64) {
        self.inner.set_hw(hw);
    }

    #[setter]
    pub(crate) fn set_maxp(&mut self, maxp: f64) {
        self.inner.set_maxp(maxp);
    }

    pub(crate) fn evaluate(&self, x: f64) -> f64 {
        self.inner.evaluate(x)
    }

    pub(crate) fn evaluate_vec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.evaluate_vec(x.as_slice().unwrap()))
    }

    pub(crate) fn integral(&self) -> f64 {
        self.inner.integral()
    }

    #[staticmethod]
    pub(crate) fn superposition(x: f64, lorentzians: Vec<Lorentzian>) -> f64 {
        deconvolution::Lorentzian::superposition(x, &lorentzians)
    }

    #[staticmethod]
    pub(crate) fn superposition_vec<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
        lorentzians: Vec<Lorentzian>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(
            py,
            &deconvolution::Lorentzian::superposition_vec(x.as_slice().unwrap(), &lorentzians),
        )
    }

    #[staticmethod]
    pub(crate) fn par_superposition_vec<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
        lorentzians: Vec<Lorentzian>,
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(
            py,
            &deconvolution::Lorentzian::par_superposition_vec(x.as_slice().unwrap(), &lorentzians),
        )
    }
}
