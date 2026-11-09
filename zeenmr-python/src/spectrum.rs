use crate::{BoundaryArray, BoundaryDataType, BoundaryFrequencyAxis};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::fmt::Write;
use zeenmr::spectrum::SpectrumOwnedDyn;
use zeenmr::spectrum::axis::{FrequencyAxis, frequency_axes};
use zeenmr::spectrum::dimension::DynDim;

/// Defines the SpectrumDispatch enum and helper macros.
///
/// Takes any number of `Variant(Type) = "typestr"`. `typestr` is the NumPy
/// type ID string. For entry, it generates: a variant of [`IntensityDispatch`]
/// wrapping `SpectrumOwnedDyn<F, Type>`, which then serves as the inner types
/// of the `F32Frequency` and `F64Frequency` variants of [`SpectrumDispatch`],
/// and two arms (one for each frequency data type) in the `spectrum_dispatch`
/// helper macro.
macro_rules! define_spectrum_dispatch {
    ($($V:ident($T:ty)),* $(,)?) => {
        /// Owned spectrum with the intensity type erased.
        ///
        /// Holds a [`SpectrumOwnedDyn`] with the intensities as one of the
        /// supported data types.
        #[derive(Clone, PartialEq, Debug)]
        #[allow(missing_docs)]
        enum IntensityDispatch<F>
        where
            $(SpectrumOwnedDyn<F, $T>: PartialEq,)*
        {
            $($V(SpectrumOwnedDyn<F, $T>),)*
        }

        /// Fully type-erased owned spectrum.
        ///
        /// Holds a [`SpectrumOwnedDyn`] through [`IntensityDispatch`].
        /// Frequency information is stored as `f32` or `f64`.
        #[derive(Clone, PartialEq, Debug)]
        #[allow(missing_docs)]
        enum SpectrumDispatch {
            F32Frequency(IntensityDispatch<f32>),
            F64Frequency(IntensityDispatch<f64>),
        }

        $(
            impl From<SpectrumOwnedDyn<f32, $T>> for SpectrumDispatch {
                fn from(value: SpectrumOwnedDyn<f32, $T>) -> Self {
                    Self::F32Frequency(IntensityDispatch::$V(value))
                }
            }

            impl From<SpectrumOwnedDyn<f64, $T>> for SpectrumDispatch {
                fn from(value: SpectrumOwnedDyn<f64, $T>) -> Self {
                    Self::F64Frequency(IntensityDispatch::$V(value))
                }
            }
        )*

        impl SpectrumDispatch {
            /// Returns the data type of the frequency axes and intensities.
            fn data_type(&self) -> (BoundaryDataType, BoundaryDataType) {
                match self {
                    Self::F32Frequency(inner) => match inner {
                        $(IntensityDispatch::$V(_) => {
                            (BoundaryDataType::F32, BoundaryDataType::$V)
                        },)*
                    },
                    Self::F64Frequency(inner) => match inner {
                        $(IntensityDispatch::$V(_) => {
                            (BoundaryDataType::F64, BoundaryDataType::$V)
                        },)*
                    },
                }
            }
        }

        /// Matches on a [`SpectrumDispatch`], binding the inner
        /// [`SpectrumOwnedDyn`] to `$array` and evaluating `$body` for each.
        ///
        /// `$body` is expanded once per variant, so it must compile for every
        /// data type and evaluate to the same type in each.
        macro_rules! spectrum_dispatch {
            ($dispatch:expr, $spectrum:ident => $body:expr) => {
                match $dispatch {
                    SpectrumDispatch::F32Frequency(inner) => match inner {
                        $(IntensityDispatch::$V($spectrum) => $body,)*
                    }
                    SpectrumDispatch::F64Frequency(inner) => match inner {
                        $(IntensityDispatch::$V($spectrum) => $body,)*
                    }
                }
            };
        }
    };
}

with_data_types!(define_spectrum_dispatch);

/// Spectrum type used internally by the library's algorithms.
#[pyclass(name = "Spectrum", module = "zeenmr", eq)]
#[derive(Clone, PartialEq, Debug)]
pub struct BoundarySpectrum {
    /// Type-erased spectrum holding the spectral data and its data type.
    dispatch: SpectrumDispatch,
}

impl<F, T> From<SpectrumOwnedDyn<F, T>> for BoundarySpectrum
where
    SpectrumDispatch: From<SpectrumOwnedDyn<F, T>>,
{
    fn from(value: SpectrumOwnedDyn<F, T>) -> Self {
        Self {
            dispatch: value.into(),
        }
    }
}

#[pymethods]
impl BoundarySpectrum {
    /// Creates a spectrum from a list of frequency axes and a NumPy array
    /// containing the intensities.
    #[new]
    pub fn new<'py>(axes: &Bound<'py, PyList>, intensities: BoundaryArray) -> PyResult<Self> {
        let intensities_rank = array_dispatch!(&intensities.dispatch, array => array.rank());
        let axes_rank = axes.len();

        if intensities_rank != axes_rank {
            return Err(PyValueError::new_err(
                "axes and intensities must have the same rank (number of dimensions)",
            ));
        }

        let axes = axes
            .iter()
            .map(|axis| axis.extract::<BoundaryFrequencyAxis>())
            .collect::<PyResult<Vec<BoundaryFrequencyAxis>>>()?;
        let data_type = axes.iter().try_fold(
            if intensities.data_type() == BoundaryDataType::F64 {
                BoundaryDataType::F64
            } else {
                BoundaryDataType::F32
            },
            |data_type, axis| match axis.data_type() {
                BoundaryDataType::F32 => Ok(data_type),
                BoundaryDataType::F64 => Ok(BoundaryDataType::F64),
                _ => Err(PyTypeError::new_err(
                    "axis data_type must be 'f32' or 'f64'",
                )),
            },
        )?;

        Ok(match data_type {
            BoundaryDataType::F32 => array_dispatch!(intensities.dispatch, array => {
                let axes = axes
                    .into_iter()
                    .map(|axis| axis.to_inner_f32())
                    .collect::<PyResult<DynDim<FrequencyAxis<f32>>>>()?;

                SpectrumOwnedDyn::new(frequency_axes(axes), array)
                    .expect("rank check was performed before converting")
                    .into()
            }),
            BoundaryDataType::F64 => array_dispatch!(intensities.dispatch, array => {
                let axes = axes
                    .into_iter()
                    .map(|axis| axis.to_inner_f64())
                    .collect::<PyResult<DynDim<FrequencyAxis<f64>>>>()?;

                SpectrumOwnedDyn::new(frequency_axes(axes), array)
                    .expect("rank check was performed before converting")
                    .into()
            }),
            _ => unreachable!(),
        })
    }

    /// The number of dimensions.
    #[getter]
    pub fn rank(&self) -> usize {
        spectrum_dispatch!(&self.dispatch, spectrum => spectrum.rank())
    }

    /// The extent of the spectrum in each dimension.
    #[getter]
    pub fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        spectrum_dispatch!(&self.dispatch, spectrum => {
            PyTuple::new(py, spectrum.intensities().shape().as_slice())
        })
    }

    /// The frequency axes of the spectrum.
    #[getter]
    pub fn axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        spectrum_dispatch!(&self.dispatch, spectrum => {
            PyList::new(
                py,
                spectrum
                    .axes()
                    .as_slice()
                    .iter()
                    .map(|axis| BoundaryFrequencyAxis::from(*axis)),
            )
        })
    }

    /// A NumPy view of the intensities.
    ///
    /// Any writes through this view are observable in the Rust array.
    #[getter]
    pub fn intensities<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Self::__array__(slf)
    }

    /// The data type of the frequency axes and intensities.
    #[getter]
    pub fn data_type(&self) -> (BoundaryDataType, BoundaryDataType) {
        self.dispatch.data_type()
    }

    /// NumPy array interface (version 3).
    #[getter]
    pub fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        spectrum_dispatch!(&self.dispatch, spectrum => {
            crate::array::export(py, spectrum.intensities())
        })
    }

    /// Converts the intensity array to a NumPy array, sharing memory with it.
    pub fn __array__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("numpy")?
            .call_method1(intern!(slf.py(), "asarray"), (slf,))
    }

    /// The number of intensity values in this spectrum.
    pub fn __len__(&self) -> usize {
        spectrum_dispatch!(&self.dispatch, spectrum => spectrum.len())
    }

    /// Basic information about the spectrum.
    ///
    /// For example,
    ///
    /// ```text
    /// Spectrum(shape=[2, 3], type=(frequency='f64', intensity='f64')))
    /// ```
    pub fn __repr__(&self) -> String {
        let shape = spectrum_dispatch!(&self.dispatch, spectrum => {
            match spectrum.shape().as_slice().first() {
                Some(first) => {
                    let mut init = String::with_capacity(2 * (spectrum.rank().saturating_sub(1)));
                    write!(&mut init, "{first}").unwrap();

                    spectrum
                        .shape()
                        .as_slice()
                        .iter()
                        .skip(1)
                        .fold(init, |mut result, extent| {
                            result.push_str(", ");
                            write!(&mut result, "{extent}").unwrap();

                            result
                        })
                }
                None => String::new(),
            }
        });
        let (axis_type, intensity_type) = self.data_type();

        format!(
            "Spectrum(shape=[{}], type=(frequency='{}', intensity='{}')",
            shape, axis_type, intensity_type,
        )
    }
}
