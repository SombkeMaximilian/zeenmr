use crate::deconvolution::smoothing::{Smoother, SmoothingSettings};

/// Identity smoother that does not modify the input values.
#[derive(Debug)]
pub(crate) struct Identity;

impl<T> Smoother<T> for Identity {
    fn smooth_values(&self, _values: &mut [T]) {}

    fn settings(&self) -> SmoothingSettings {
        SmoothingSettings::Identity
    }
}

impl Identity {
    /// Creates a new `Identity` smoother.
    pub(crate) fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};

    #[test]
    fn thread_safety() {
        assert_send!(Identity);
        assert_sync!(Identity);
    }
}
