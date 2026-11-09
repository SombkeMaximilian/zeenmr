use numpy::Element;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// 32-bit complex type.
#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex32;

/// 64-bit complex type.
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;

/// Data type that can be exchanged with NumPy.
pub trait DataType: Element + Clone {
    /// Rust type name.
    const TYPE_STR: &'static str;

    /// NumPy array interface `typestr`, e.g. `"=f8"`.
    const NUMPY_STR: &'static str;
}

/// Passes all supported data types to the callback.
///
/// The macro must take list of types in the form:
///
/// `Variant(Type) = NUMPY_STR`
#[macro_export]
macro_rules! with_data_types {
    ($callback:ident) => {
        $callback! {
            U8(u8), U16(u16), U32(u32), U64(u64),
            I8(i8), I16(i16), I32(i32), I64(i64),
            F32(f32), F64(f64),
            C32($crate::c32), C64($crate::c64),
        }
    };
}

/// Implements the [`DataType`] trait for the chosen Rust types.
macro_rules! impl_data_type {
    ($($V:ident($T:ty)),* $(,)?) => {
        $(
            impl DataType for $T {
                const TYPE_STR: &'static str =
                    type_str::<{ stringify!($V).as_bytes()[0] | 32 }, { size_of::<$T>() }, false>();
                const NUMPY_STR: &'static str =
                    type_str::<{ stringify!($V).as_bytes()[0] | 32 }, { size_of::<$T>() }, true>();
            }
        )*
    };
}

with_data_types!(impl_data_type);

/// Defines the [`BoundaryDataType`] type with variants for the chosen Rust
/// types.
macro_rules! define_boundary_data_type {
    ($($V:ident($T:ty)),* $(,)?) => {
        /// Supported data types.
        #[pyclass(name = "DataType", module = "zeenmr", eq, eq_int, frozen, hash)]
        #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
        #[allow(missing_docs)]
        pub enum BoundaryDataType {
            $($V,)*
        }

        #[pymethods]
        impl BoundaryDataType {
            /// The Rust name of the data type (e.g. `"f64"`, `"u8"`).
            ///
            /// [`num_complex::Complex32`] and [`num_complex::Complex64`] are
            /// shortened to `c32` and `c64`.
            #[getter]
            pub fn type_str(&self) -> &'static str {
                match self {
                    $(BoundaryDataType::$V => <$T>::TYPE_STR,)*
                }
            }

            /// The NumPy `typestr`.
            #[getter]
            pub fn numpy_str(&self) -> &'static str {
                match self {
                    $(BoundaryDataType::$V => <$T>::NUMPY_STR,)*
                }
            }
        }
    };
}

with_data_types!(define_boundary_data_type);

impl FromPyObject<'_, '_> for BoundaryDataType {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(data_type) = object.extract::<PyRef<Self>>() {
            return Ok(*data_type);
        }

        if let Ok(data_type) = object.extract::<&str>().map(str::trim) {
            macro_rules! lookup {
                ($($V:ident($T:ty)),* $(,)?) => {
                    $(if data_type == <$T>::TYPE_STR || data_type == stringify!($V) { return Ok(Self::$V) })*
                };
            }
            with_data_types!(lookup);
        }

        let numpy_type = object
            .py()
            .import("numpy")?
            .getattr("dtype")?
            .call1((object,))?;
        let kind = numpy_type.getattr("kind")?.extract::<char>()?;
        let size = numpy_type
            .getattr("itemsize")?
            .extract::<usize>()?;
        let byte_order = numpy_type
            .getattr("byteorder")?
            .extract::<char>()?;
        let native = if cfg!(target_endian = "little") {
            '<'
        } else {
            '>'
        };

        if !(byte_order == '=' || byte_order == '|') && byte_order != native {
            return Err(PyTypeError::new_err("non-native byte order not supported"));
        }

        let numpy_str = format!("={kind}{size}");

        macro_rules! lookup {
            ($($V:ident($T:ty)),* $(,)?) => {
                $(if numpy_str == <$T>::NUMPY_STR { return Ok(Self::$V) })*
            };
        }
        with_data_types!(lookup);

        Err(PyTypeError::new_err(format!(
            "unsupported numpy dtype: {numpy_type}"
        )))
    }
}

impl std::fmt::Display for BoundaryDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.type_str())
    }
}

/// Type string using Rust-like names or NumPy `typestr` if `NUMPY` is `true`.
///
/// `KIND` should be either `b'u'`, `b'i'`, `b'f'`, or `b'c'`, where `'c'` is
/// special-cased to only consider how many bytes the complex parts contain
/// individually. That is, e.g., `c64` takes up 16 bytes (8 for the real, 8 for
/// the imaginary part).
const fn type_str<const KIND: u8, const BYTES: usize, const NUMPY: bool>() -> &'static str {
    if BYTES == 0 {
        panic!("ZST not supported");
    } else {
        let bytes = &const {
            let mut scratch = [0; (usize::MAX.ilog10() as usize) + 3];
            let mut exp = scratch.len();
            let mut curr =
                BYTES * if NUMPY { 1 } else { 8 } / if !NUMPY && KIND == b'c' { 2 } else { 1 };

            while exp > 0 && curr != 0 {
                exp -= 1;
                scratch[exp] = (curr % 10) as u8 + b'0';
                curr /= 10;
            }
            scratch[exp.saturating_sub(1)] = KIND;
            if NUMPY {
                scratch[exp.saturating_sub(2)] = b'=';
            }

            scratch
        };
        let size = BYTES * if NUMPY { 1 } else { 8 } / if !NUMPY && KIND == b'c' { 2 } else { 1 };
        let len = ((size.ilog10() as usize) + 1) + if NUMPY { 2 } else { 1 };
        // SAFETY: `bytes.len() - len` is at most be one past the end of `bytes`
        // (when `len == 0`). `ptr` is non-null, aligned, and valid for the
        // remaining elements as it was obtained from another slice, and `len`
        // is the remaining length.
        let non_zero = unsafe {
            // const equivalent to `&bytes[(bytes.len() - len)..]`.
            std::slice::from_raw_parts(bytes.as_ptr().add(bytes.len() - len), len)
        };

        match std::str::from_utf8(non_zero) {
            Ok(s) => s,
            _ => unreachable!(),
        }
    }
}
