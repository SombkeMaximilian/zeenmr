#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of a deconvolution.
///
/// A `Deconvolution` contains the deconvoluted signals as peak shapes, the
/// settings used for deconvolution, and the mean squared error of between the
/// original spectrum and the superposition of peak shapes.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Deconvolution<P> {
    /// Deconvoluted peak shapes.
    peak_shapes: Box<[P]>,
}

impl<P> Deconvolution<P> {
    /// Creates a new `Deconvolution`.
    ///
    /// Normally, this type is only instantiated by the deconvolution functions
    /// of deconvoluters.
    pub fn new<I>(peak_shapes: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self {
            peak_shapes: peak_shapes.into_iter().collect(),
        }
    }

    /// Returns the deconvoluted peak shapes.
    pub fn peak_shapes(&self) -> &[P] {
        &self.peak_shapes
    }
}
