use crate::axis::iter::AxisIter;
use num_traits::Float;

#[cfg(feature = "rayon")]
use crate::axis::iter::ParAxisIter;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Time axis of a free induction decay (FID).
///
/// Implicitly always starts at `t₀ = 0`.
///
/// # Invariants
///
/// The dwell must be finite, non-zero and positive.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// With the `serde` feature enabled, `TimeAxis` can be serialized using
/// `serde`. The two directions have different requirements:
///
/// - [`Serialize`] needs only `T: Serialize`.
/// - [`Deserialize`] needs `T: Float + Deserialize<'de>` so that the dwell can
///   be checked against the invariants.
///
/// [`Serialize`]: https://docs.rs/serde/latest/serde/trait.Serialize.html
/// [`Deserialize`]: https://docs.rs/serde/latest/serde/trait.Deserialize.html
///
/// Deserialization goes through [`TimeAxis::new`] and fails if the dwell is
/// invalid according to the invariants.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(
        try_from = "RawTimeAxis<T>",
        bound(deserialize = "T: Float + Deserialize<'de>")
    )
)]
pub struct TimeAxis<T> {
    /// Time step, also known as dwell.
    ///
    /// Stored in units of microseconds.
    dwell: T,
}

impl<T> TimeAxis<T>
where
    T: Float,
{
    /// Constructs a new `TimeAxis`.
    ///
    /// Returns `None` if `dwell` is non-finite, zero, or negative.
    pub fn new(dwell: T) -> Option<Self> {
        if dwell.is_finite() && dwell > T::zero() {
            Some(Self { dwell })
        } else {
            None
        }
    }

    /// Attaches a length to the axis, producing a grid.
    ///
    /// # Precision
    ///
    /// Due to floating point errors when adding and multiplying, passing a
    /// `len` that cannot be represented by `T` may lead to significant errors
    /// (e.g., `len > 2^24` for `f32`) in the downstream methods.
    pub fn grid(&self, len: usize) -> TimeGrid<'_, T> {
        TimeGrid { axis: self, len }
    }

    /// Returns the time step, also known as dwell.
    pub fn dwell(&self) -> T {
        self.dwell
    }

    /// Returns an iterator over `len` equally spaced time offsets spanning the
    /// axis.
    ///
    /// The first offset is always exactly `t₀ = 0`.
    ///
    /// Each call of this method recomputes the time offsets on the fly.
    ///
    /// # Precision
    ///
    /// Passing a `len` that cannot be represented by `T` may lead to
    /// significant errors (e.g., `len > 2^24` for `f32`).
    pub fn iter(&self, len: usize) -> AxisIter<T> {
        AxisIter::new(T::zero(), self.dwell, len)
    }
}

#[cfg(feature = "rayon")]
impl<T> TimeAxis<T>
where
    T: Float + Send,
{
    /// Returns a parallel iterator over `len` equally spaced time offsets
    /// spanning the axis.
    ///
    /// The first offset is always exactly `t₀ = 0`.
    ///
    /// Each call of this method recomputes the time offsets on the fly.
    ///
    /// # Precision
    ///
    /// Passing a `len` that cannot be represented by `T` may lead to
    /// significant errors (e.g., `len > 2^24` for `f32`).
    pub fn par_iter(&self, len: usize) -> ParAxisIter<T> {
        ParAxisIter::new(T::zero(), self.dwell, len)
    }
}

/// Convenience wrapper of an axis reference with an attached length.
///
/// An instance of this type can be obtained from [`TimeAxis::grid`].
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TimeGrid<'axis, T> {
    /// Reference to the original axis.
    ///
    /// This is not owned in order to enforce single source of truth.
    axis: &'axis TimeAxis<T>,
    /// Length of the 1D grid.
    len: usize,
}

impl<'axis, T> IntoIterator for TimeGrid<'axis, T>
where
    T: Float,
{
    type Item = T;
    type IntoIter = AxisIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'axis, T> IntoIterator for &TimeGrid<'axis, T>
where
    T: Float,
{
    type Item = T;
    type IntoIter = AxisIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'axis, T> IntoIterator for &mut TimeGrid<'axis, T>
where
    T: Float,
{
    type Item = T;
    type IntoIter = AxisIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'axis, T> TimeGrid<'axis, T>
where
    T: Float,
{
    /// Sets the length of the grid.
    pub fn with_len(self, len: usize) -> Self {
        Self { len, ..self }
    }

    /// Returns a reference to the original axis.
    pub fn axis(&self) -> &'axis TimeAxis<T> {
        self.axis
    }

    /// Returns the length of the grid.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the grid contains no points.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over the time offsets in the grid.
    ///
    /// See [`TimeAxis::iter`] for its exact behavior.
    pub fn iter(&self) -> AxisIter<T> {
        self.axis.iter(self.len)
    }
}

#[cfg(feature = "rayon")]
impl<'axis, T> IntoParallelIterator for TimeGrid<'axis, T>
where
    T: Float + Send,
{
    type Iter = ParAxisIter<T>;
    type Item = T;

    fn into_par_iter(self) -> Self::Iter {
        self.par_iter()
    }
}

#[cfg(feature = "rayon")]
impl<'axis, T> IntoParallelIterator for &TimeGrid<'axis, T>
where
    T: Float + Send,
{
    type Iter = ParAxisIter<T>;
    type Item = T;

    fn into_par_iter(self) -> Self::Iter {
        self.par_iter()
    }
}

#[cfg(feature = "rayon")]
impl<'axis, T> IntoParallelIterator for &mut TimeGrid<'axis, T>
where
    T: Float + Send,
{
    type Iter = ParAxisIter<T>;
    type Item = T;

    fn into_par_iter(self) -> Self::Iter {
        self.par_iter()
    }
}

#[cfg(feature = "rayon")]
impl<'axis, T> TimeGrid<'axis, T>
where
    T: Float + Send,
{
    /// Returns a parallel iterator over the time offsets in the grid.
    ///
    /// See [`TimeAxis::par_iter`] for its exact behavior.
    pub fn par_iter(&self) -> ParAxisIter<T> {
        self.axis.par_iter(self.len)
    }
}

/// Raw axis without invariants as an intermediate for deserialization.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct RawTimeAxis<T> {
    /// Time step, also known as dwell.
    ///
    /// Stored in units of microseconds.
    dwell: T,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<RawTimeAxis<T>> for TimeAxis<T>
where
    T: Float,
{
    type Error = &'static str;

    fn try_from(value: RawTimeAxis<T>) -> Result<Self, Self::Error> {
        Self::new(value.dwell).ok_or("dwell must be finite, non-zero and positive")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(TimeAxis<f32>: Send, Sync);
        assert_impl_all!(TimeAxis<f64>: Send, Sync);
    }

    #[test]
    fn invariants() {
        fn invariants_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let dwells = [
                T::zero(),
                -T::one(),
                T::nan(),
                T::infinity(),
                T::neg_infinity(),
            ];

            for dwell in dwells {
                assert!(TimeAxis::new(dwell).is_none());
            }
        }

        invariants_::<f32>();
        invariants_::<f64>();
    }
}
