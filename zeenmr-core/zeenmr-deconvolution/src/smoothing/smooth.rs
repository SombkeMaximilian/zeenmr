use num_traits::Float;
use std::borrow::Borrow;

/// Trait for smoothing algorithms that process a sequence of values.
pub trait Smooth<F>
where
    F: Copy + Float,
{
    /// Smooth the provided data in place.
    fn smooth<I>(&self, data: I) -> impl Iterator<Item = F>
    where
        I: IntoIterator,
        I::Item: Borrow<F>;
}
