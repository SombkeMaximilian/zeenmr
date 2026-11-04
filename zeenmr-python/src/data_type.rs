use numpy::Element;

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
            U8(u8) = "=u1", U16(u16) = "=u2", U32(u32) = "=u4", U64(u64) = "=u8",
            I8(i8) = "=i1", I16(i16) = "=i2", I32(i32) = "=i4", I64(i64) = "=i8",
            F32(f32) = "=f4", F64(f64) = "=f8",
            C32(::num_complex::Complex32) = "=c8", C64(::num_complex::Complex64) = "=c16",
        }
    };
}

/// Implements the [`DataType`] trait for the chosen Rust types.
macro_rules! impl_data_type {
    ($($V:ident($T:ty) = $numpy_str:expr),* $(,)?) => {
        $(
            impl DataType for $T {
                const TYPE_STR: &'static str = stringify!($T);
                const NUMPY_STR: &'static str = $numpy_str;
            }
        )*
    };
}

with_data_types!(impl_data_type);
