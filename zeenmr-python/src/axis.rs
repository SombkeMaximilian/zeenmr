use crate::{BoundaryDataType, Reduce, into_reduce};
use num_traits::{AsPrimitive, Float};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
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

/// Matches on a [`ShiftReferenceDispatch`], binding the inner
/// [`ShiftReference`] to `$reference` and evaluating `$body` for each.
///
/// `$body` is expanded once per variant, so it must compile for every
/// data type and evaluate to the same type in each.
macro_rules! shift_reference_dispatch {
    ($dispatch:expr, $reference:ident => $body:expr) => {
        match $dispatch {
            ShiftReferenceDispatch::F32($reference) => $body,
            ShiftReferenceDispatch::F64($reference) => $body,
        }
    };
}

/// Converts a chemical shift reference from one float type to another.
fn convert_shift_reference<T1, T2>(reference: ShiftReference<T1>) -> PyResult<ShiftReference<T2>>
where
    T1: Float + AsPrimitive<T2>,
    T2: Float + 'static,
{
    ShiftReference::new(reference.shift().as_(), reference.freq().as_())
        .ok_or_else(|| PyValueError::new_err("values must not saturate to +-inf during conversion"))
}

/// Constructor arguments.
type ShiftReferenceArgs<'py> = (f64, f64, BoundaryDataType);

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

impl FromPyObject<'_, '_> for BoundaryShiftReference {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let data_type = object
            .getattr("data_type")?
            .extract::<BoundaryDataType>()
            .unwrap_or(BoundaryDataType::F64);

        if !matches!(data_type, BoundaryDataType::F32 | BoundaryDataType::F64) {
            return Err(PyTypeError::new_err("data_type must be 'f32' or 'f64'"));
        }
        let err = || PyValueError::new_err("values must be finite in the data type");

        match data_type {
            BoundaryDataType::F32 => {
                let shift = object.getattr("shift")?.extract::<f32>()?;
                let freq = object.getattr("freq")?.extract::<f32>()?;

                ShiftReference::new(shift, freq)
                    .ok_or_else(err)
                    .map(Into::into)
            }
            BoundaryDataType::F64 => {
                let shift = object.getattr("shift")?.extract::<f64>()?;
                let freq = object.getattr("freq")?.extract::<f64>()?;

                ShiftReference::new(shift, freq)
                    .ok_or_else(err)
                    .map(Into::into)
            }
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl BoundaryShiftReference {
    /// Creates a chemical shift reference from a chemical shift-frequency pair.
    ///
    /// Anchors `shift` ppm to `freq` Hz.
    #[new]
    #[pyo3(signature = (shift, freq, data_type=BoundaryDataType::F64))]
    pub fn new(shift: f64, freq: f64, data_type: BoundaryDataType) -> PyResult<Self> {
        let err = || PyValueError::new_err("values must be finite in the data type");

        match data_type {
            BoundaryDataType::F32 => ShiftReference::new(shift as f32, freq as f32)
                .ok_or_else(err)
                .map(Into::into),
            BoundaryDataType::F64 => ShiftReference::new(shift, freq)
                .ok_or_else(err)
                .map(Into::into),
            _ => Err(PyValueError::new_err("data_type must be 'f32' or 'f64'")),
        }
    }

    /// Creates a chemical shift reference from a chemical shift.
    ///
    /// Anchors `shift` to 0 Hz.
    #[staticmethod]
    #[pyo3(signature = (shift, data_type=BoundaryDataType::F64))]
    pub fn from_shift(shift: f64, data_type: BoundaryDataType) -> PyResult<Self> {
        Self::new(shift, 0_f64, data_type)
    }

    /// Creates a chemical shift reference from a frequency.
    ///
    /// Anchors 0 ppm to `freq` Hz.
    #[staticmethod]
    #[pyo3(signature = (freq, data_type=BoundaryDataType::F64))]
    pub fn from_freq(freq: f64, data_type: BoundaryDataType) -> PyResult<Self> {
        Self::new(0_f64, freq, data_type)
    }

    /// Returns a copy of the chemical shift reference using `f32` as the data
    /// type.
    pub fn to_f32(&self) -> PyResult<Self> {
        match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => Ok(inner.into()),
            ShiftReferenceDispatch::F64(inner) => {
                convert_shift_reference::<_, f32>(inner).map(Into::into)
            }
        }
    }

    /// Returns a copy of the chemical shift reference using `f64` as the data
    /// type.
    pub fn to_f64(&self) -> Self {
        match self.dispatch {
            ShiftReferenceDispatch::F32(inner) => convert_shift_reference::<_, f64>(inner)
                .expect("valid f32 values cannot saturate f64")
                .into(),
            ShiftReferenceDispatch::F64(inner) => inner.into(),
        }
    }

    /// The chemical shift of this reference.
    #[getter]
    pub fn shift(&self) -> f64 {
        shift_reference_dispatch!(self.dispatch, inner => inner.shift().as_())
    }

    /// The frequency this reference anchors its chemical shift to.
    #[getter]
    pub fn freq(&self) -> f64 {
        shift_reference_dispatch!(self.dispatch, inner => inner.freq().as_())
    }

    /// The data type as a Rust type name (`"f32"` or `"f64"`).
    #[getter]
    pub fn data_type(&self) -> BoundaryDataType {
        match self.dispatch {
            ShiftReferenceDispatch::F32(_) => BoundaryDataType::F32,
            ShiftReferenceDispatch::F64(_) => BoundaryDataType::F64,
        }
    }

    /// Sets the chemical shift of this reference.
    #[setter]
    pub fn set_shift(&mut self, shift: f64) -> PyResult<()> {
        let err = || PyValueError::new_err("shift must be finite");

        *self = shift_reference_dispatch!(self.dispatch, inner => {
            ShiftReference::new(shift.as_(), inner.freq())
                .ok_or_else(err)?
                .into()
        });

        Ok(())
    }

    /// Sets the frequency this reference anchors its chemical shift to.
    #[setter]
    pub fn set_freq(&mut self, freq: f64) -> PyResult<()> {
        let err = || PyValueError::new_err("freq must be finite");

        *self = shift_reference_dispatch!(self.dispatch, inner => {
            ShiftReference::new(inner.shift(), freq.as_())
                .ok_or_else(err)?
                .into()
        });

        Ok(())
    }

    /// Returns basic information about the chemical shift reference.
    ///
    /// For example, `ShiftReference(0 ppm at 3000 Hz, type='f64')`
    pub fn __repr__(&self) -> String {
        shift_reference_dispatch!(self.dispatch, inner => format!(
            "ShiftReference({} ppm at {} Hz, type='{}')",
            inner.shift(),
            inner.freq(),
            self.data_type(),
        ))
    }

    /// Supports pickling and copying.
    pub fn __reduce__<'py>(slf: &Bound<'py, Self>) -> Reduce<'py, ShiftReferenceArgs<'py>> {
        let reference = *(slf.borrow());

        into_reduce(
            slf.get_type(),
            (reference.shift(), reference.freq(), reference.data_type()),
        )
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

/// Matches on a [`FrequencyAxisDispatch`], binding the inner
/// [`FrequencyAxis`] to `$axis` and evaluating `$body` for each.
///
/// `$body` is expanded once per variant, so it must compile for every
/// data type and evaluate to the same type in each.
macro_rules! frequency_axis_dispatch {
    ($dispatch:expr, $axis:ident => $body:expr) => {
        match $dispatch {
            FrequencyAxisDispatch::F32($axis) => $body,
            FrequencyAxisDispatch::F64($axis) => $body,
        }
    };
}

/// Converts a frequency axis from one float type to another.
fn convert_frequency_axis<T1, T2>(axis: FrequencyAxis<T1>) -> PyResult<FrequencyAxis<T2>>
where
    T1: Float + AsPrimitive<T2>,
    T2: Float + 'static,
{
    let range = FrequencyRange::new(
        axis.freq_range().start().as_(),
        axis.freq_range().end().as_(),
    )
    .ok_or_else(|| {
        PyValueError::new_err("range bounds must not saturate to +-inf during conversion")
    })?;
    let larmor = Larmor::new(axis.larmor().get().as_()).ok_or_else(|| {
        PyValueError::new_err("larmor must not saturate to +-inf during conversion")
    })?;
    let shift_reference = convert_shift_reference::<_, T2>(axis.reference())?;

    FrequencyAxis::new(range, larmor, shift_reference)
        .ok_or_else(|| PyValueError::new_err(
            "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
        ))
}

/// Constructor arguments.
type FrequencyAxisArgs<'py> = (
    (f64, f64),
    f64,
    Bound<'py, BoundaryShiftReference>,
    BoundaryDataType,
);

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

impl FromPyObject<'_, '_> for BoundaryFrequencyAxis {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let range = object.getattr("range")?.extract::<(f64, f64)>()?;
        let larmor = object.getattr("larmor")?.extract::<f64>()?;
        let reference = object
            .getattr("shift_reference")?
            .extract::<BoundaryShiftReference>()?;
        let data_type = object
            .getattr("data_type")?
            .extract::<BoundaryDataType>()
            .unwrap_or(reference.data_type());

        if !matches!(data_type, BoundaryDataType::F32 | BoundaryDataType::F64) {
            return Err(PyTypeError::new_err("data_type must be 'f32' or 'f64'"));
        }

        match data_type {
            BoundaryDataType::F32 => {
                Self::new_with_static_type::<f32>(range, larmor, reference.dispatch)
            }
            BoundaryDataType::F64 => {
                Self::new_with_static_type::<f64>(range, larmor, reference.dispatch)
            }
            _ => unreachable!(),
        }
    }
}

impl BoundaryFrequencyAxis {
    /// Creates a type-erased frequency axis using `T` as its data type.
    fn new_with_static_type<T>(
        range: (f64, f64),
        larmor: f64,
        reference: ShiftReferenceDispatch,
    ) -> PyResult<Self>
    where
        T: Float + 'static,
        f32: AsPrimitive<T>,
        f64: AsPrimitive<T>,
        FrequencyAxis<T>: Into<BoundaryFrequencyAxis>,
    {
        let range = FrequencyRange::new(range.0.as_(), range.1.as_())
            .ok_or_else(|| PyValueError::new_err("range bounds must be finite"))?;
        let larmor = Larmor::new(larmor.as_()).ok_or_else(|| {
            PyValueError::new_err("larmor must not be zero, NaN, or one of the infinities")
        })?;
        let reference = shift_reference_dispatch!(
            reference,
            inner => convert_shift_reference::<_, T>(inner)?
        );

        FrequencyAxis::new(range, larmor, reference)
            .ok_or_else(|| PyValueError::new_err(
                "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
            ))
            .map(Into::into)
    }
}

#[pymethods]
impl BoundaryFrequencyAxis {
    /// Creates a frequency axis from a frequency range, larmor frequency, and
    /// chemical shift reference.
    #[new]
    #[pyo3(signature = (range, larmor, reference, data_type=BoundaryDataType::F64))]
    pub fn new<'py>(
        range: (f64, f64),
        larmor: f64,
        reference: &Bound<'py, BoundaryShiftReference>,
        data_type: BoundaryDataType,
    ) -> PyResult<Self> {
        let reference = reference.borrow().dispatch;

        match data_type {
            BoundaryDataType::F32 => Self::new_with_static_type::<f32>(range, larmor, reference),
            BoundaryDataType::F64 => Self::new_with_static_type::<f64>(range, larmor, reference),
            _ => Err(PyValueError::new_err("data_type must be 'f32' or 'f64'")),
        }
    }

    /// Returns a copy of the frequency axis using `f32` as the data type.
    pub fn to_f32(&self) -> PyResult<Self> {
        match self.dispatch {
            FrequencyAxisDispatch::F32(inner) => Ok(inner.into()),
            FrequencyAxisDispatch::F64(inner) => {
                convert_frequency_axis::<_, f32>(inner).map(Into::into)
            }
        }
    }

    /// Returns a copy of the frequency axis using `f64` as the data type.
    pub fn to_f64(&self) -> Self {
        match self.dispatch {
            FrequencyAxisDispatch::F32(inner) => convert_frequency_axis::<_, f64>(inner)
                .expect("valid f32 values cannot saturate f64")
                .into(),
            FrequencyAxisDispatch::F64(inner) => inner.into(),
        }
    }

    /// The frequency range of this axis.
    #[getter]
    pub fn range(&self) -> (f64, f64) {
        frequency_axis_dispatch!(self.dispatch, inner => {
            (inner.freq_range().start().as_(), inner.freq_range().end().as_())
        })
    }

    /// The larmor frequency of this axis.
    #[getter]
    pub fn larmor(&self) -> f64 {
        frequency_axis_dispatch!(self.dispatch, inner => inner.larmor().get().as_())
    }

    /// The chemical shift reference of this axis.
    #[getter]
    pub fn shift_reference(&self) -> BoundaryShiftReference {
        frequency_axis_dispatch!(self.dispatch, inner => inner.reference().into())
    }

    /// The data type as a Rust type name (`"f32"` or `"f64"`).
    #[getter]
    pub fn data_type(&self) -> BoundaryDataType {
        match self.dispatch {
            FrequencyAxisDispatch::F32(_) => BoundaryDataType::F32,
            FrequencyAxisDispatch::F64(_) => BoundaryDataType::F64,
        }
    }

    /// Sets the chemical shift reference of this axis.
    #[setter]
    pub fn set_shift_reference<'py>(
        &mut self,
        reference: &Bound<'py, BoundaryShiftReference>,
    ) -> PyResult<()> {
        *self = frequency_axis_dispatch!(self.dispatch, axis => {
            let reference = shift_reference_dispatch!(
                reference.borrow().dispatch,
                inner => convert_shift_reference(inner)?
            );

            axis.with_reference(reference)
                .ok_or_else(|| PyValueError::new_err(
                    "larmor is incompatible with the reference, or frequency range bounds map to non-finite chemical shifts"
                ))?
                .into()
        });

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
        frequency_axis_dispatch!(self.dispatch, axis => {
            let range = axis.freq_range();
            let larmor = axis.larmor().get();
            let reference = axis.reference();

            format!(
                "FrequencyAxis(range=({} Hz, {} Hz), larmor={} MHz, reference=({} ppm at {} Hz), type='{}')",
                range.start(),
                range.end(),
                larmor,
                reference.shift(),
                reference.freq(),
                self.data_type(),
            )
        })
    }

    /// Supports pickling and copying.
    pub fn __reduce__<'py>(slf: &Bound<'py, Self>) -> Reduce<'py, FrequencyAxisArgs<'py>> {
        let axis = slf.borrow();
        let reference = Bound::new(slf.py(), axis.shift_reference())?;

        into_reduce(
            slf.get_type(),
            (axis.range(), axis.larmor(), reference, axis.data_type()),
        )
    }
}
