use crate::smoothing::Smooth;
use std::borrow::Cow;

/// No-op smoother that does not modify the input values.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Identity;

impl Smooth for Identity {
    fn smooth<'a>(&self, data: &'a [f64]) -> Cow<'a, [f64]> {
        data.into()
    }
}

impl Identity {
    /// Creates a new `Identity` smoother.
    pub fn new() -> Self {
        Self
    }
}
