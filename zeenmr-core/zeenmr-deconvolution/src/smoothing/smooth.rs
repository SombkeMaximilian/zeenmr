use std::borrow::Borrow;

/// Trait for smoothing algorithms that process a sequence of values.
pub trait Smooth {
    /// Smooth the provided data and return an iterator over the smoothed
    /// values.
    fn smooth<I>(&self, data: I) -> Vec<f64>
    where
        I: IntoIterator,
        I::Item: Borrow<f64>;
}
