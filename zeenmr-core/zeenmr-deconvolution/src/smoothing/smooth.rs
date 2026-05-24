use std::borrow::Cow;

/// Trait for smoothing algorithms that process a sequence of values.
///
/// # Note for Implementors
///
/// [`Smooth::smooth`] should return the borrowed data in cases where no
/// smoothing is applied.
pub trait Smooth: Send + Sync {
    /// Smooth the provided data and return an owned instance of the smoothed
    /// values.
    fn smooth<'a>(&self, data: &'a [f64]) -> Cow<'a, [f64]>;
}
