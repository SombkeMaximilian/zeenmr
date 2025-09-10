use std::borrow::Borrow;
use crate::smoothing::Smooth;
use num_traits::Float;

/// No-op smoother that does not modify the input values.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Identity;

impl<F> Smooth<F> for Identity
where
    F: Copy + Float,
{
    fn smooth<I>(&self, data: I) -> impl Iterator<Item=F>
    where
        I: IntoIterator,
        I::Item: Borrow<F>,
    {
        data.into_iter()
    }
}

impl Identity {
    /// Creates a new `Identity` smoother.
    pub fn new() -> Self {
        Self
    }
}
