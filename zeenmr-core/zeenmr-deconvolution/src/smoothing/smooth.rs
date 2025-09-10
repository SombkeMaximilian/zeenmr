use num_traits::Float;
use std::borrow::Borrow;

/// Trait for smoothing algorithms that process a sequence of values.
pub trait Smooth<F>
where
    F: Float,
{
    /// Smooth the provided data and return an iterator over the smoothed
    /// values.
    fn smooth<I>(&self, data: I) -> Vec<F>
    where
        I: IntoIterator,
        I::Item: Borrow<F>;
}
