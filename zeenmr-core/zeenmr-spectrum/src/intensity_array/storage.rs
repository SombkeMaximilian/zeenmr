use std::rc::Rc;
use std::sync::Arc;

/// Array storage trait.
pub trait Storage: AsRef<[Self::Elem]> {
    /// Array element type.
    type Elem;
}

impl<T> Storage for Vec<T> {
    type Elem = T;
}

impl<T> Storage for &[T] {
    type Elem = T;
}

impl<T> Storage for &mut [T] {
    type Elem = T;
}

impl<T> Storage for Box<[T]> {
    type Elem = T;
}

impl<T> Storage for Rc<[T]> {
    type Elem = T;
}

impl<T> Storage for Arc<[T]> {
    type Elem = T;
}

impl<A> Storage for &A
where
    A: Storage,
{
    type Elem = A::Elem;
}

impl<A> Storage for &mut A
where
    A: Storage,
{
    type Elem = A::Elem;
}
