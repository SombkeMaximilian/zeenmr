use crate::smoothing::Smooth;
use std::borrow::Borrow;

/// No-op smoother that does not modify the input values.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Identity;

impl Smooth for Identity {
    type Settings = ();

    fn settings(&self) -> Self::Settings {}

    fn smooth<I>(&self, data: I) -> Vec<f64>
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        data.into_iter()
            .map(|value| *value.borrow())
            .collect()
    }
}

impl Identity {
    /// Creates a new `Identity` smoother.
    pub fn new() -> Self {
        Self
    }
}
