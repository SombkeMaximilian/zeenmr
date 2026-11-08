use pyo3::prelude::*;
use pyo3::types::PyType;

/// Return type of `__reduce__`.
///
/// `A` must match the constructor's signature.
pub type Reduce<'py, A> = PyResult<(Bound<'py, PyType>, A)>;

/// Creates the reduced form of a type whose constructor takes `A`.
pub fn into_reduce<A>(py_type: Bound<PyType>, args: A) -> Reduce<A> {
    Ok((py_type, args))
}
