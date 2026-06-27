use std::rc::Rc;
use std::sync::Arc;

/// 1D intensity array trait.
pub trait Array1D: AsRef<[Self::Elem]> {
    /// Array element type.
    type Elem;
}

impl<T> Array1D for Vec<T> {
    type Elem = T;
}

impl<T> Array1D for &[T] {
    type Elem = T;
}

impl<T> Array1D for &mut [T] {
    type Elem = T;
}

impl<T> Array1D for Box<[T]> {
    type Elem = T;
}

impl<T> Array1D for Rc<[T]> {
    type Elem = T;
}

impl<T> Array1D for Arc<[T]> {
    type Elem = T;
}
