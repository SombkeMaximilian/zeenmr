use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};
use zeenmr::spectrum::axis::range::{FiniteBounds, FrequencyRange};
use zeenmr::spectrum::axis::{FrequencyAxis, Larmor, ShiftReference};

/// Type-erased chemical shift reference.
///
/// Holds a [`ShiftReference`] of one of the floating point data types.
#[derive(Copy, Clone, PartialEq, Debug)]
#[allow(missing_docs)]
enum ShiftReferenceDispatch {
    F32(ShiftReference<f32>),
    F64(ShiftReference<f64>),
}

/// Chemical shift reference for spectral frequency axes.
///
/// Anchors a chemical shift value to a frequency value for referencing.
#[pyclass(name = "ShiftReference", module = "zeenmr", eq)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct BoundaryShiftReference {
    /// Type-erased chemical shift reference.
    dispatch: ShiftReferenceDispatch,
}

impl From<ShiftReference<f32>> for BoundaryShiftReference {
    fn from(value: ShiftReference<f32>) -> Self {
        Self {
            dispatch: ShiftReferenceDispatch::F32(value),
        }
    }
}

impl From<ShiftReference<f64>> for BoundaryShiftReference {
    fn from(value: ShiftReference<f64>) -> Self {
        Self {
            dispatch: ShiftReferenceDispatch::F64(value),
        }
    }
}

#[pymethods]
impl BoundaryShiftReference {
    /// Creates a chemical shift reference from a chemical shift-frequency pair.
    ///
    /// Anchors `shift` ppm to `freq` Hz.
    #[new]
    pub fn new(shift: f64, freq: f64) -> PyResult<Self> {
        ShiftReference::new(shift, freq)
            .ok_or_else(|| {
                PyValueError::new_err("values must not be NaN or one of the infinities")
            })
            .map(Into::into)
    }

    /// Creates a chemical shift reference from a chemical shift.
    ///
    /// Anchors `shift` to 0 Hz.
    #[staticmethod]
    pub fn from_shift(shift: f64) -> PyResult<Self> {
        Self::new(shift, 0_f64)
    }

    /// Creates a chemical shift reference from a frequency.
    ///
    /// Anchors 0 ppm to `freq` Hz.
    #[staticmethod]
    pub fn from_freq(freq: f64) -> PyResult<Self> {
        Self::new(0_f64, freq)
    }

    /// Returns a copy of the chemical shift reference using `f32` as the data
    /// type.
    pub fn to_f32(&self) -> PyResult<Self> {
        match &self.dispatch {
            ShiftReferenceDispatch::F32(inner) => Ok((*inner).into()),
            ShiftReferenceDispatch::F64(inner) => {
                ShiftReference::new(inner.shift() as f32, inner.freq() as f32)
                    .ok_or_else(|| {
                        PyValueError::new_err("values must not saturate to +-inf during conversion")
                    })
                    .map(Into::into)
            }
        }
    }

    /// Returns a copy of the chemical shift reference using `f64` as the data
    /// type.
    pub fn to_f64(&self) -> Self {
        match &self.dispatch {
            ShiftReferenceDispatch::F32(inner) => {
                ShiftReference::new(inner.shift() as f64, inner.freq() as f64)
                    .expect("valid f32 values cannot saturate f64")
                    .into()
            }
            ShiftReferenceDispatch::F64(inner) => (*inner).into(),
        }
    }

    /// The chemical shift of this reference.
    #[getter]
    pub fn shift(&self) -> f64 {
        match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => inner.shift() as f64,
            ShiftReferenceDispatch::F64(inner) => inner.shift(),
        }
    }

    /// The frequency this reference anchors its chemical shift to.
    #[getter]
    pub fn freq(&self) -> f64 {
        match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => inner.freq() as f64,
            ShiftReferenceDispatch::F64(inner) => inner.freq(),
        }
    }

    /// The data type as a Rust type name (`"f32"` or `"f64"`).
    #[getter]
    pub fn data_type(&self) -> &str {
        match self.dispatch {
            ShiftReferenceDispatch::F32(_) => "f32",
            ShiftReferenceDispatch::F64(_) => "f64",
        }
    }

    /// Sets the chemical shift of this reference.
    #[setter]
    pub fn set_shift(&mut self, shift: f64) -> PyResult<()> {
        *self = match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => ShiftReference::new(shift as f32, inner.freq())
                .ok_or_else(|| {
                    PyValueError::new_err("shift must not be NaN or one of the infinities")
                })?
                .into(),
            ShiftReferenceDispatch::F64(inner) => ShiftReference::new(shift, inner.freq())
                .ok_or_else(|| {
                    PyValueError::new_err("shift must not be NaN or one of the infinities")
                })?
                .into(),
        };

        Ok(())
    }

    /// Sets the frequency this reference anchors its chemical shift to.
    #[setter]
    pub fn set_freq(&mut self, freq: f64) -> PyResult<()> {
        *self = match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => ShiftReference::new(inner.shift(), freq as f32)
                .ok_or_else(|| {
                    PyValueError::new_err("freq must not be NaN or one of the infinities")
                })?
                .into(),
            ShiftReferenceDispatch::F64(inner) => ShiftReference::new(inner.shift(), freq)
                .ok_or_else(|| {
                    PyValueError::new_err("freq must not be NaN or one of the infinities")
                })?
                .into(),
        };

        Ok(())
    }

    /// Returns basic information about the chemical shift reference.
    ///
    /// For example, `ShiftReference(0 ppm at 3000 Hz, type='f64')`
    pub fn __repr__(&self) -> String {
        let (shift, freq) = match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => (inner.shift() as f64, inner.freq() as f64),
            ShiftReferenceDispatch::F64(inner) => (inner.shift(), inner.freq()),
        };

        format!(
            "ShiftReference({} ppm at {} Hz, type = {})",
            shift,
            freq,
            self.data_type()
        )
    }

    /// Supports pickling and copying.
    pub fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, (Bound<'py, PyAny>,))> {
        let shift_reference = *(slf.borrow());

        Ok((
            slf.get_type(),
            (
                PyTuple::new(slf.py(), [shift_reference.shift(), shift_reference.freq()])?
                    .into_any(),
            ),
        ))
    }
}

/// Type-erased frequency axis.
///
/// Holds a [`FrequencyAxis`] of one of the floating point data types.
#[derive(Copy, Clone, PartialEq, Debug)]
#[allow(missing_docs)]
enum FrequencyAxisDispatch {
    F32(FrequencyAxis<f32>),
    F64(FrequencyAxis<f64>),
}

/// Spectral frequency axis.
///
/// This type holds the frequency information of a spectrum's axis. It models
/// the continuous frequency and chemical shift space of a dimension. Construct
/// one from a frequency range, larmor frequency, and chemical shift reference
/// with `FrequencyAxis((start, end), larmor, reference)`.
#[pyclass(name = "FrequencyAxis", module = "zeenmr", eq)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct BoundaryFrequencyAxis {
    dispatch: FrequencyAxisDispatch,
}

impl From<FrequencyAxis<f32>> for BoundaryFrequencyAxis {
    fn from(value: FrequencyAxis<f32>) -> Self {
        Self {
            dispatch: FrequencyAxisDispatch::F32(value),
        }
    }
}

impl From<FrequencyAxis<f64>> for BoundaryFrequencyAxis {
    fn from(value: FrequencyAxis<f64>) -> Self {
        Self {
            dispatch: FrequencyAxisDispatch::F64(value),
        }
    }
}

#[pymethods]
impl BoundaryFrequencyAxis {
    /// Creates a frequency axis from a frequency range, larmor frequency, and
    /// chemical shift reference.
    #[new]
    pub fn new<'py>(
        range: (f64, f64),
        larmor: f64,
        reference: &Bound<'py, BoundaryShiftReference>,
    ) -> PyResult<Self> {
        let range = FrequencyRange::new(range.0, range.1).ok_or_else(|| {
            PyValueError::new_err("range bounds must not be NaN or one of the infinities")
        })?;
        let larmor = Larmor::new(larmor).ok_or_else(|| {
            PyValueError::new_err("larmor must not be zero, NaN, or one of the infinities")
        })?;
        let reference = match reference.borrow().to_f64().dispatch {
            ShiftReferenceDispatch::F32(_) => unreachable!(),
            ShiftReferenceDispatch::F64(inner) => inner,
        };

        FrequencyAxis::new(range, larmor, reference).ok_or_else(|| {
            PyValueError::new_err("larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts")
        })
        .map(Into::into)
    }

    /// Returns a copy of the frequency axis using `f32` as the data type.
    pub fn to_f32(&self) -> PyResult<Self> {
        match &self.dispatch {
            FrequencyAxisDispatch::F32(inner) => Ok((*inner).into()),
            FrequencyAxisDispatch::F64(inner) => {
                let range = FrequencyRange::new(
                    inner.freq_range().start() as f32,
                    inner.freq_range().end() as f32,
                )
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "range bounds must not saturate to +-inf during conversion",
                    )
                })?;
                let larmor = Larmor::new(inner.larmor().get() as f32).ok_or_else(|| {
                    PyValueError::new_err("larmor must not saturate to +-inf during conversion")
                })?;
                let shift_reference = ShiftReference::new(
                    inner.reference().shift() as f32,
                    inner.reference().freq() as f32,
                )
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "shift reference must not saturate to +-inf during conversion",
                    )
                })?;

                FrequencyAxis::new(range, larmor, shift_reference)
                    .ok_or_else(|| PyValueError::new_err(
                        "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
                    ))
                    .map(Into::into)
            }
        }
    }

    /// Returns a copy of the frequency axis using `f64` as the data type.
    pub fn to_f64(&self) -> Self {
        match &self.dispatch {
            FrequencyAxisDispatch::F32(inner) => {
                let range = FrequencyRange::new(
                    inner.freq_range().start() as f64,
                    inner.freq_range().end() as f64,
                )
                .expect("valid f32 values cannot saturate f64");
                let larmor = Larmor::new(inner.larmor().get() as f64)
                    .expect("valid f32 values cannot saturate f64");
                let shift_reference = ShiftReference::new(
                    inner.reference().shift() as f64,
                    inner.reference().freq() as f64,
                )
                .expect("valid f32 values cannot saturate f64");

                FrequencyAxis::new(range, larmor, shift_reference)
                    .expect("valid f32 values cannot saturate f64")
                    .into()
            }
            FrequencyAxisDispatch::F64(inner) => (*inner).into(),
        }
    }

    /// The frequency range of this axis.
    #[getter]
    pub fn range(&self) -> (f64, f64) {
        match self.dispatch {
            FrequencyAxisDispatch::F32(inner) => (
                inner.freq_range().start() as f64,
                inner.freq_range().end() as f64,
            ),
            FrequencyAxisDispatch::F64(inner) => {
                (inner.freq_range().start(), inner.freq_range().end())
            }
        }
    }

    /// The larmor frequency of this axis.
    #[getter]
    pub fn larmor(&self) -> f64 {
        match self.dispatch {
            FrequencyAxisDispatch::F32(inner) => inner.larmor().get() as f64,
            FrequencyAxisDispatch::F64(inner) => inner.larmor().get(),
        }
    }

    /// The chemical shift reference of this axis.
    #[getter]
    pub fn shift_reference(&self) -> BoundaryShiftReference {
        match self.dispatch {
            FrequencyAxisDispatch::F32(inner) => inner.reference().into(),
            FrequencyAxisDispatch::F64(inner) => inner.reference().into(),
        }
    }

    /// The data type as a Rust type name (`"f32"` or `"f64"`).
    #[getter]
    pub fn data_type(&self) -> &str {
        match self.dispatch {
            FrequencyAxisDispatch::F32(_) => "f32",
            FrequencyAxisDispatch::F64(_) => "f64",
        }
    }

    /// Sets the chemical shift reference of this axis.
    #[setter]
    pub fn set_shift_reference<'py>(
        &mut self,
        reference: &Bound<'py, BoundaryShiftReference>,
    ) -> PyResult<()> {
        *self = match (self.dispatch, reference.borrow().dispatch) {
            (FrequencyAxisDispatch::F32(axis), ShiftReferenceDispatch::F32(reference)) => {
                axis.with_reference(reference)
                    .ok_or_else(|| PyValueError::new_err(
                        "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
                    ))?
                    .into()
            },
            (FrequencyAxisDispatch::F32(axis), ShiftReferenceDispatch::F64(reference)) => {
                axis.with_reference(
                    ShiftReference::new(reference.shift() as f32, reference.freq() as f32)
                        .ok_or_else(|| PyValueError::new_err(
                        "shift reference must not saturate to +-inf during conversion"
                        ))?
                )
                .ok_or_else(|| PyValueError::new_err(
                    "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
                ))?
                .into()
            },
            (FrequencyAxisDispatch::F64(axis), ShiftReferenceDispatch::F32(reference)) => {
                axis.with_reference(ShiftReference::new(
                    reference.shift() as f64,
                    reference.shift() as f64,
                ).expect("valid f32 values cannot saturate f64"))
                    .ok_or_else(|| PyValueError::new_err(
                        "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
                    ))?
                    .into()
            },
            (FrequencyAxisDispatch::F64(axis), ShiftReferenceDispatch::F64(reference)) => {
                axis.with_reference(reference)
                    .ok_or_else(|| PyValueError::new_err(
                        "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
                    ))?
                    .into()
            },
        };

        Ok(())
    }

    /// Returns basic information about the frequency axis.
    ///
    /// For example,
    ///
    /// ```text
    /// FrequencyAxis(range=(12000 Hz, 0 Hz), larmor=625 MHz,
    ///               reference=(0 ppm at 3000 Hz), type='f64')
    /// ```
    pub fn __repr__(&self) -> String {
        let (start, end) = self.range();
        let larmor = self.larmor();
        let (shift, freq) = match self.shift_reference().dispatch {
            ShiftReferenceDispatch::F32(inner) => (inner.shift() as f64, inner.freq() as f64),
            ShiftReferenceDispatch::F64(inner) => (inner.shift(), inner.freq()),
        };

        format!(
            "FrequencyAxis(range=({} Hz, {} Hz), larmor={} MHz, reference=({} ppm at {} Hz), type={})",
            start, end, larmor, shift, freq, self.data_type()
        )
    }

    /// Supports pickling and copying.
    pub fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (Bound<'py, PyAny>,))> {
        let (start, end) = slf.borrow().range();
        let larmor = slf.borrow().larmor();
        let (shift, freq) = match slf.borrow().shift_reference().dispatch {
            ShiftReferenceDispatch::F32(inner) => (inner.shift() as f64, inner.freq() as f64),
            ShiftReferenceDispatch::F64(inner) => (inner.shift(), inner.freq()),
        };

        Ok((
            slf.get_type(),
            (
                PyTuple::new(slf.py(), [start, end, larmor, shift, freq])?
                    .into_any(),
            )
        ))
    }
}
