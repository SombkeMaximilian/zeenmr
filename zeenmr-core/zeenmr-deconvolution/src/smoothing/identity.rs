use crate::smoothing::Smooth;
use std::borrow::Cow;

/// No-op smoother that does not modify the input values.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Identity;

impl<T> Smooth<T> for Identity
where
    T: Clone,
{
    fn smooth<'a>(&self, data: &'a [T]) -> Cow<'a, [T]> {
        data.into()
    }
}

impl Identity {
    /// Creates a new `Identity` smoother.
    pub fn new() -> Self {
        Self
    }
}
