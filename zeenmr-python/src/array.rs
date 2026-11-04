use crate::DataType;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyTuple, PyType};
use std::fmt::Write;
use zeenmr::spectrum::intensity_array::{
    ArrayOwned, ArrayOwnedDyn, ArrayViewDyn, RawStorage, shape,
};

/// Defines the array dispatch enum and helper macros.
///
/// Takes any number of `Variant(Type) = "typestr"`. `typestr` is the NumPy
/// type ID string. For each entry it generates: a variant of [`ArrayDispatch`]
/// wrapping `ArrayOwnedDyn<Type>`, accompanied by `From` impl, and an arm in
/// the `array_dispatch!` and `py_any_dispatch!` helper macros.
macro_rules! define_array_dispatch {
    ($($V:ident($T:ty) = $_:expr),* $(,)?) => {
        /// Type-erased owned array.
        ///
        /// Holds an [`ArrayOwnedDyn`] of one of the supported data types.
        #[derive(Clone, PartialEq, Debug)]
        #[allow(missing_docs)]
        enum ArrayDispatch {
            $($V(ArrayOwnedDyn<$T>),)*
        }

        $(
            impl From<ArrayOwnedDyn<$T>> for ArrayDispatch {
                fn from(value: ArrayOwnedDyn<$T>) -> Self {
                    Self::$V(value)
                }
            }
        )*

        impl ArrayDispatch {
            /// Returns the Rust name of the data type (e.g. `"f64"`, `"u8"`).
            fn type_str(&self) -> &'static str {
                match self {
                    $(Self::$V(_) => <$T>::TYPE_STR,)*
                }
            }
        }

        /// Matches on an [`ArrayDispatch`], binding the inner [`ArrayOwnedDyn`]
        /// to `$array` and evaluating `$body` for each.
        ///
        /// `$body` is expanded once per variant, so it must compile for every
        /// data type and evaluate to the same type in each.
        macro_rules! array_dispatch {
            ($dispatch:expr, $array:ident => $body:expr) => {
                match $dispatch {
                    $(ArrayDispatch::$V($array) => $body,)*
                }
            };
        }

        /// Extracts an arbitrary Python object as a NumPy array of any
        /// supported data type, binding the resulting [`PyReadonlyArrayDyn`]
        /// to `$array` and evaluating `$body`.
        ///
        /// Element types are tried in declaration order and the first
        /// successful extraction wins. `$body` is expanded once per type, and
        /// it must evaluate to a `PyResult`.
        macro_rules! py_any_dispatch {
            ($object:expr, $array:ident => $body:expr) => {{
                let object = $object;

                $(
                    if let Ok($array) = object.extract::<PyReadonlyArrayDyn<$T>>() {
                        $body
                    } else
                )*
                {
                    Err(PyValueError::new_err("expected a numpy array of a supported data type"))
                }
            }};
        }
    };
}

with_data_types!(define_array_dispatch);

impl ArrayDispatch {
    /// Converts a Python object into an `ArrayDispatch` by copying its data.
    ///
    /// The object must be a NumPy array of a supported data type; see
    /// [`py_any_dispatch!`]. Returns a `ValueError` if the dtype is unsupported
    /// or the array is empty or its shape overflows (see [`import`]).
    fn from_object(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        py_any_dispatch!(object, array => Ok(ArrayDispatch::from(import(array)?)))
    }
}

/// Array type used internally by the library's algorithms.
///
/// This type is the boundary between Python and Rust. Algorithms in `zeenmr`
/// operate on this type, while everything on the Python side uses the NumPy
/// interface. Construct one from any NumPy array of a supported data type
/// with `Array(obj)`, and convert back with `numpy.asarray(array)`.
///
/// Construction copies the data, while returning NumPy views does not.
#[pyclass(name = "Array", module = "zeenmr", eq)]
#[derive(Clone, PartialEq, Debug)]
pub struct BoundaryArray {
    /// Type-erased storage holding the array data and its data type.
    dispatch: ArrayDispatch,
}

#[pymethods]
impl BoundaryArray {
    /// Creates an array from a NumPy array of a supported data type, copying
    /// the data.
    #[new]
    pub fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            dispatch: ArrayDispatch::from_object(object)?,
        })
    }

    /// The extent of the array in each dimension.
    #[getter]
    pub fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        array_dispatch!(&self.dispatch, array => PyTuple::new(py, array.shape().as_slice().iter()))
    }

    /// The number of dimensions.
    #[getter]
    pub fn rank(&self) -> usize {
        array_dispatch!(&self.dispatch, array => array.rank())
    }

    /// The data type as a Rust type name (e.g., `"f64"`, `"u8"`).
    #[getter]
    pub fn data_type(&self) -> &str {
        self.dispatch.type_str()
    }

    /// A NumPy view of this array.
    ///
    /// Any writes through this view are observable in the Rust array.
    #[getter]
    pub fn data<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Self::__array__(slf)
    }

    /// Returns a dict containing the array's layout information.
    pub fn layout<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let layout = array_dispatch!(&self.dispatch, array => array.layout());
        let shape = layout.shape().as_slice().iter();
        let strides = layout.strides().as_slice().iter();
        let offset = layout.offset();
        let max_offset = layout.max_offset();

        let dict = PyDict::new(py);
        dict.set_item(intern!(py, "version"), crate::VERSION)?;
        dict.set_item(intern!(py, "shape"), PyTuple::new(py, shape)?)?;
        dict.set_item(intern!(py, "strides"), PyTuple::new(py, strides)?)?;
        dict.set_item(intern!(py, "offset"), offset)?;
        dict.set_item(intern!(py, "max_offset"), max_offset)?;

        Ok(dict)
    }

    /// NumPy array interface (version 3).
    #[getter]
    pub fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        array_dispatch!(&self.dispatch, array => export(py, array.view()))
    }

    /// Converts to a NumPy array, sharing memory with this array.
    pub fn __array__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("numpy")?
            .call_method1(intern!(slf.py(), "asarray"), (slf,))
    }

    /// Returns the number of elements in the array.
    pub fn __len__(&self) -> usize {
        array_dispatch!(&self.dispatch, array => array.len())
    }

    /// Indexes the array.
    ///
    /// This follows NumPy indexing and slicing rules.
    pub fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::data(slf)?.get_item(key)
    }

    /// Iterators over the array.
    ///
    /// This works exactly as NumPy iteration does.
    pub fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyIterator>> {
        Self::data(slf)?.try_iter()
    }

    /// Returns basic information about the array.
    ///
    /// For example, `Array(shape=[2, 3], type='f64')`.
    pub fn __repr__(&self) -> String {
        let shape = array_dispatch!(&self.dispatch, array => {
            match array.shape().as_slice().first() {
                Some(first) => {
                    let mut init = String::with_capacity(2 * (array.rank().saturating_sub(1)));
                    write!(&mut init, "{first}").unwrap();

                    array
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

        format!("Array(shape=[{}], type='{}')", shape, self.data_type())
    }

    /// Supports pickling and copying by reconstructing the array from its
    /// NumPy data.
    pub fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, (Bound<'py, PyAny>,))> {
        Ok((slf.get_type(), (Self::data(slf)?,)))
    }
}

/// Copies a NumPy array into an [`ArrayOwnedDyn`].
///
/// # Errors
///
/// Returns a `ValueError` if the array is empty or its shape overflows.
fn import<T>(array: PyReadonlyArrayDyn<'_, T>) -> PyResult<ArrayOwnedDyn<T>>
where
    T: DataType,
{
    let shape = shape(array.shape());
    let data = match array.as_slice() {
        Ok(slice) => Box::<[T]>::from(slice),
        Err(_) => array.as_array().iter().cloned().collect(),
    };

    ArrayOwned::new(data, shape).ok_or_else(|| PyValueError::new_err("empty or overflowing shape"))
}

/// Builds a NumPy [array interface] dictionary (version 3) describing `array`.
///
/// [array interface]: https://numpy.org/doc/stable/reference/arrays.interface.html
///
/// # Errors
///
/// Returns an error if adding elements to a dictionary or creating a Python
/// tuple fails.
fn export<'py, 's, T>(py: Python<'py>, array: ArrayViewDyn<'s, T>) -> PyResult<Bound<'py, PyDict>>
where
    T: DataType,
{
    let (storage, layout) = array.into_parts();
    let shape = layout.shape().as_slice().iter();
    let strides = layout
        .strides()
        .as_slice()
        .iter()
        .map(|stride| stride * size_of::<T>());

    // SAFETY: the layout offset is guaranteed to be within bounds of the
    // allocation by the array type.
    let ptr = unsafe { storage.as_ptr().add(layout.offset()) as usize };
    // Data races are left to the user to handle as we can't track numpy
    // borrowing from Rust.
    let read_only = false;

    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "version"), 3)?;
    dict.set_item(intern!(py, "typestr"), T::NUMPY_STR)?;
    dict.set_item(intern!(py, "shape"), PyTuple::new(py, shape)?)?;
    dict.set_item(intern!(py, "strides"), PyTuple::new(py, strides)?)?;
    dict.set_item(intern!(py, "data"), (ptr, read_only))?;

    Ok(dict)
}
