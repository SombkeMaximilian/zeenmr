#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Trait for mathematical functions that can be evaluated at a given point.
///
/// A type implementing `Evaluate` is called a function and unless otherwise
/// specified, a `function` always refers to such a type.
pub trait Evaluate<T> {
    /// Evaluate the function at the specified point.
    fn evaluate(&self, at: T) -> T;
}

impl<T, E> Evaluate<T> for &E
where
    E: Evaluate<T>,
{
    fn evaluate(&self, at: T) -> T {
        (*self).evaluate(at)
    }
}

impl<T, E> Evaluate<T> for &mut E
where
    E: Evaluate<T>,
{
    fn evaluate(&self, at: T) -> T {
        (**self).evaluate(at)
    }
}

/// A function `f(x) = num(x) / den(x)` with strictly positive parts.
///
/// Implementing this trait enables fused evaluation that avoids 1 division
/// per fusion at the cost 1 addition and 3 multiplications, which is generally
/// worth it. However, these multiplications can lead to overflow if too many
/// fusions are performed.
///
/// # Correctness
///
/// Implementors must guarantee `num(x) > 0` and `den(x) > 0` for all finite
/// `x`. Violating this invalidates the error bounds of the fused kernels.
pub trait EvaluateParts<T>: Evaluate<T> {
    /// Returns `(num, den)`.
    fn parts(&self, at: T) -> (T, T);

    /// Bounds on `num` over the closed interval `[lo, hi]`.
    ///
    /// Must satisfy `0 < lo_bound <= num(x) <= hi_bound` for all `x` in range.
    fn num_bounds(&self, lo: T, hi: T) -> (T, T);

    /// Bounds on `den` over the closed interval `[lo, hi]`.
    ///
    /// Must satisfy `0 < lo_bound <= den(x) <= hi_bound` for all `x` in range.
    fn den_bounds(&self, lo: T, hi: T) -> (T, T);
}

impl<T, E> EvaluateParts<T> for &E
where
    E: EvaluateParts<T>,
{
    fn parts(&self, at: T) -> (T, T) {
        (**self).parts(at)
    }

    fn num_bounds(&self, lo: T, hi: T) -> (T, T) {
        (**self).num_bounds(lo, hi)
    }

    fn den_bounds(&self, lo: T, hi: T) -> (T, T) {
        (**self).den_bounds(lo, hi)
    }
}

impl<T, E> EvaluateParts<T> for &mut E
where
    E: EvaluateParts<T>,
{
    fn parts(&self, at: T) -> (T, T) {
        (**self).parts(at)
    }

    fn num_bounds(&self, lo: T, hi: T) -> (T, T) {
        (**self).num_bounds(lo, hi)
    }

    fn den_bounds(&self, lo: T, hi: T) -> (T, T) {
        (**self).den_bounds(lo, hi)
    }
}

/// Extension trait for iterators mapping its items through a function.
///
/// Given a function `f` implementing [`Evaluate`] and a sequence of input
/// values `(xᵢ)`, where `i≤n`, this adaptor produces an iterator of the
/// corresponding evaluations:
///
/// ```text
/// { f(x₁), f(x₂), ..., f(xₙ) }
/// ```
///
/// This is analogous to [`Iterator::map`], but uses the provided evaluator
/// instead of an arbitrary closure.
pub trait EvaluateMap<T>: Iterator {
    /// Apply the given evaluator to each item in the iterator, producing an
    /// iterator of evaluated results.
    fn evaluate<E>(self, evaluator: E) -> impl Iterator<Item = T>
    where
        E: Evaluate<T>;
}

impl<T, I> EvaluateMap<T> for I
where
    I: Iterator<Item = T>,
{
    fn evaluate<E>(self, evaluator: E) -> impl Iterator<Item = T>
    where
        E: Evaluate<T>,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}

/// Extension trait for parallel iterators mapping its items through a function.
///
/// Given a function `f` implementing [`Evaluate`] and a sequence of input
/// values `(xᵢ)`, where `i≤n`, this adaptor produces a parallel iterator of the
/// corresponding evaluations:
///
/// ```text
/// { f(x₁), f(x₂), ..., f(xₙ) }
/// ```
///
/// This is the parallel analogue of [`EvaluateMap`].
#[cfg(feature = "rayon")]
pub trait ParEvaluateMap<T: Send>: ParallelIterator {
    /// Apply the given evaluator to each item in the indexed parallel iterator,
    /// producing an indexed parallel iterator of evaluated results.
    fn evaluate<E>(self, evaluator: E) -> impl ParallelIterator<Item = T>
    where
        E: Evaluate<T> + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<T, I> ParEvaluateMap<T> for I
where
    T: Send,
    I: ParallelIterator<Item = T>,
{
    fn evaluate<E>(self, evaluator: E) -> impl ParallelIterator<Item = T>
    where
        E: Evaluate<T> + Send + Sync,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}
