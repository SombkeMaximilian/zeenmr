use num_traits::Zero;
use std::ops::Add;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Trait for mathematical functions that can be evaluated at a given point.
///
/// A type implementing `Evaluate` is called a function and unless otherwise
/// specified, a `function` always refers to such a type.
///
/// # Example
///
/// ```
/// use zeenmr_peakshape::Evaluate;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// assert_eq!(Power(2).evaluate(3.0), 9.0);
/// ```
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

/// Extension trait for iterators of functions that provides superposition of
/// their values at a point.
///
/// Given a family of functions `(fᵢ)`, where `i≤n`, their superposition `F` at
/// a point `x` is defined as the sum of the individual evaluations:
///
/// ```text
/// F(x) = f₁(x) + f₂(x) + ... + fₙ(x)
/// ```
///
/// This is analogous to [`Iterator::sum`], but instead of summing the items
/// directly, it sums their evaluations at a specified point.
///
/// # Example
///
/// ```
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::Superposition;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let powers = [Power(2), Power(2), Power(3)];
/// assert_eq!(powers.into_iter().superposition(3.0), 45.0);
/// ```
pub trait Superposition<T>: Iterator {
    /// Evaluate each function in the iterator at the given point and return
    /// the sum of the results.
    fn superposition(self, at: T) -> T
    where
        Self::Item: Evaluate<T>;
}

impl<T, E, I> Superposition<T> for I
where
    T: Copy + Add + Zero,
    E: Evaluate<T>,
    I: Iterator<Item = E>,
{
    fn superposition(self, at: T) -> T
    where
        Self::Item: Evaluate<T>,
    {
        self.fold(T::zero(), |acc, e| acc + e.evaluate(at))
    }
}

/// Extension trait for parallel iterators of functions that provides
/// superposition of their values at a point.
///
/// Given a family of functions `(fᵢ)`, where `i≤n`, their superposition `F` at
/// a point `x` is defined as the sum of the individual evaluations:
///
/// ```text
/// F(x) = f₁(x) + f₂(x) + ... + fₙ(x)
/// ```
///
/// This is the parallel analogue of [`Superposition`].
///
/// # Example
///
/// ```
/// use rayon::prelude::*;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::ParSuperposition;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let powers = vec![Power(2), Power(2), Power(3)];
/// assert_eq!(powers.into_par_iter().superposition(3.0), 45.0);
/// ```
#[cfg(feature = "rayon")]
pub trait ParSuperposition<T: Send + Sync>: ParallelIterator {
    /// Evaluate each function in the parallel iterator at the given point and
    /// return the sum of the results.
    fn superposition(self, at: T) -> T
    where
        Self::Item: Evaluate<T>;
}

#[cfg(feature = "rayon")]
impl<T, E, I> ParSuperposition<T> for I
where
    T: Copy + Send + Sync + Add + Zero,
    E: Evaluate<T> + Send + Sync,
    I: ParallelIterator<Item = E>,
{
    fn superposition(self, at: T) -> T
    where
        Self::Item: Evaluate<T>,
    {
        self.fold(|| T::zero(), |acc, e| acc + e.evaluate(at))
            .reduce(|| T::zero(), |a, b| a + b)
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
///
/// # Example
///
/// ```
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::EvaluateMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let squares = [2.0, 3.0, 4.0]
///     .into_iter()
///     .evaluate(Power(2))
///     .collect::<Vec<f64>>();
/// assert_eq!(squares[0], 4.0);
/// assert_eq!(squares[1], 9.0);
/// assert_eq!(squares[2], 16.0);
/// ```
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
///
/// # Example
///
/// ```
/// use rayon::prelude::*;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::ParEvaluateMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let squares = [2.0, 3.0, 4.0]
///     .into_par_iter()
///     .evaluate(Power(2))
///     .collect::<Vec<f64>>();
/// assert_eq!(squares[0], 4.0);
/// assert_eq!(squares[1], 9.0);
/// assert_eq!(squares[2], 16.0);
/// ```
#[cfg(feature = "rayon")]
pub trait ParEvaluateMap<T: Send>: IndexedParallelIterator {
    /// Apply the given evaluator to each item in the indexed parallel iterator,
    /// producing an indexed parallel iterator of evaluated results.
    fn evaluate<E>(self, evaluator: E) -> impl IndexedParallelIterator<Item = T>
    where
        E: Evaluate<T> + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<T, I> ParEvaluateMap<T> for I
where
    T: Send,
    I: IndexedParallelIterator<Item = T>,
{
    fn evaluate<E>(self, evaluator: E) -> impl IndexedParallelIterator<Item = T>
    where
        E: Evaluate<T> + Send + Sync,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}

/// Extension trait for iterators mapping its items through the superposition
/// of multiple functions.
///
/// Given a family of functions `(fⱼ)`, where `j≤m`, and a sequence of input
/// values `(xᵢ)`, where `i≤n`, this adaptor produces an iterator of their
/// superpositions:
///
/// ```text
/// { F(x₁), F(x₂), ..., F(xₙ) }
/// with   F(xᵢ) = f₁(xᵢ) + f₂(xᵢ) + ... + fₘ(xᵢ)
/// ```
///
/// This combines the behaviors of [`EvaluateMap`] and [`Superposition`]:
/// - Like [`EvaluateMap`], it maps over an iterator of input points.
/// - Like [`Superposition`], it sums the contributions of multiple evaluators.
///
/// # Example
///
/// ```
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::SuperpositionMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let evaluators = [Power(2), Power(3)];
/// let results = [2.0, 3.0, 4.0]
///     .into_iter()
///     .superposition(&evaluators)
///     .collect::<Vec<f64>>();
/// assert_eq!(results[0], 12.0);
/// assert_eq!(results[1], 36.0);
/// assert_eq!(results[2], 80.0);
/// ```
pub trait SuperpositionMap<T>: Iterator {
    /// Apply the superposition of the given evaluators to each item in the
    /// iterator, producing an iterator of evaluated results.
    fn superposition<E>(self, evaluators: &[E]) -> impl Iterator<Item = T>
    where
        E: Evaluate<T>;
}

impl<T, I> SuperpositionMap<T> for I
where
    T: Copy + Add + Zero,
    I: Iterator<Item = T>,
{
    fn superposition<E>(self, evaluators: &[E]) -> impl Iterator<Item = T>
    where
        E: Evaluate<T>,
    {
        let at = self.collect::<Vec<T>>();
        let init = if let Some(first) = evaluators.first() {
            at.iter()
                .copied()
                .evaluate(first)
                .collect::<Vec<T>>()
        } else {
            return Vec::new().into_iter();
        };

        evaluators[1..]
            .iter()
            .map(|e| at.iter().copied().evaluate(e))
            .fold(init, |mut acc, at| {
                acc.iter_mut()
                    .zip(at)
                    .for_each(|(acc, x)| *acc = *acc + x);

                acc
            })
            .into_iter()
    }
}

/// Extension trait for parallel iterators mapping its items through the
/// superposition of multiple functions.
///
/// Given a family of functions `(fⱼ)`, where `j≤m`, and a sequence of input
/// values `(xᵢ)`, where `i≤n`, this adaptor produces a parallel iterator of
/// their superpositions:
///
/// ```text
/// { F(x₁), F(x₂), ..., F(xₙ) }
/// with   F(xᵢ) = f₁(xᵢ) + f₂(xᵢ) + ... + fₘ(xᵢ)
/// ```
///
/// This is the parallel analogue of [`SuperpositionMap`].
///
/// # Example
///
/// ```
/// use rayon::prelude::*;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::ParSuperpositionMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate<f64> for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let evaluators = [Power(2), Power(3)];
/// let results = [2.0, 3.0, 4.0]
///     .into_par_iter()
///     .superposition(&evaluators)
///     .collect::<Vec<f64>>();
/// assert_eq!(results[0], 2.0f64.powi(2) + 2.0f64.powi(3));
/// assert_eq!(results[1], 3.0f64.powi(2) + 3.0f64.powi(3));
/// assert_eq!(results[2], 4.0f64.powi(2) + 4.0f64.powi(3));
/// ```
#[cfg(feature = "rayon")]
pub trait ParSuperpositionMap<T: Send + Sync>: IndexedParallelIterator {
    /// Apply the superposition of the given evaluators to each item in the
    /// parallel iterator, producing a parallel iterator of evaluated results.
    fn superposition<E>(self, evaluators: &[E]) -> impl IndexedParallelIterator<Item = T>
    where
        E: Evaluate<T> + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<T, I> ParSuperpositionMap<T> for I
where
    T: Copy + Send + Sync + Add + Zero,
    I: IndexedParallelIterator<Item = T>,
{
    fn superposition<E>(self, evaluators: &[E]) -> impl IndexedParallelIterator<Item = T>
    where
        E: Evaluate<T> + Send + Sync,
    {
        self.map(|x| evaluators.iter().superposition(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use num_traits::Pow;
    use static_assertions::assert_impl_all;

    // this isn't really a peak shape, but it serves as a simple test example
    #[derive(Copy, Clone, Debug)]
    struct Power(i32);

    const SQUARE: Power = Power(2);
    const CUBE: Power = Power(3);

    impl<T> Evaluate<T> for Power
    where
        T: Pow<i32, Output = T>,
    {
        fn evaluate(&self, at: T) -> T {
            at.pow(self.0)
        }
    }

    #[test]
    fn traits() {
        assert_impl_all!(Power: Evaluate<f32>, Evaluate<f64>, Send, Sync);
        assert_impl_all!(&Power: Evaluate<f32>, Evaluate<f64>, Send, Sync);
        assert_impl_all!(&mut Power: Evaluate<f32>, Evaluate<f64>, Send, Sync);
        assert_impl_all!(std::vec::IntoIter<Power>: Superposition<f32>, Superposition<f64>);
        assert_impl_all!(std::vec::IntoIter<f32>: EvaluateMap<f32>, SuperpositionMap<f32>);
        assert_impl_all!(std::vec::IntoIter<f64>: EvaluateMap<f64>, SuperpositionMap<f64>);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_traits() {
        assert_impl_all!(rayon::vec::IntoIter<Power>: ParSuperposition<f32>, ParSuperposition<f64>);
        assert_impl_all!(rayon::vec::IntoIter<f32>: ParEvaluateMap<f32>, ParSuperpositionMap<f32>);
        assert_impl_all!(rayon::vec::IntoIter<f64>: ParEvaluateMap<f64>, ParSuperpositionMap<f64>);
    }

    #[test]
    fn evaluate() {
        assert_approx_eq!(f64, SQUARE.evaluate(3.0), 9.0);
    }

    #[test]
    fn superposition() {
        let powers = [SQUARE, SQUARE, CUBE];

        let value = 3_f32;
        let superposition = powers.iter().superposition(value);

        assert_approx_eq!(f32, superposition, 2.0 * value.powi(2) + value.powi(3));

        let value = 3_f64;
        let superposition = powers.iter().superposition(value);

        assert_approx_eq!(f64, superposition, 2.0 * value.powi(2) + value.powi(3));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_superposition() {
        let powers = [SQUARE, SQUARE, CUBE];

        let value = 3_f32;
        let par_superposition = powers.par_iter().superposition(value);

        assert_approx_eq!(f32, par_superposition, 2.0 * value.powi(2) + value.powi(3));

        let value = 3_f64;
        let par_superposition = powers.par_iter().superposition(value);

        assert_approx_eq!(f64, par_superposition, 2.0 * value.powi(2) + value.powi(3));
    }

    #[test]
    fn evaluate_map() {
        let evaluate_map = (0..10).map(|i| i as f32).evaluate(SQUARE);
        let expected = (0..10).map(|i| (i as f32).powi(2));

        evaluate_map.zip(expected).for_each(|(a, b)| {
            assert_approx_eq!(f32, a, b);
        });

        let evaluate_map = (0..10).map(|i| i as f64).evaluate(SQUARE);
        let expected = (0..10).map(|i| (i as f64).powi(2));

        evaluate_map.zip(expected).for_each(|(a, b)| {
            assert_approx_eq!(f64, a, b);
        });
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_evaluate_map() {
        let evaluate_map = (0..10)
            .into_par_iter()
            .map(|i| i as f32)
            .evaluate(SQUARE)
            .collect::<Vec<f32>>();
        let expected = (0..10)
            .into_par_iter()
            .map(|i| (i as f32).powi(2))
            .collect::<Vec<f32>>();

        evaluate_map
            .into_iter()
            .zip(expected)
            .for_each(|(a, b)| {
                assert_approx_eq!(f32, a, b);
            });

        let evaluate_map = (0..10)
            .into_par_iter()
            .map(|i| i as f64)
            .evaluate(SQUARE)
            .collect::<Vec<f64>>();
        let expected = (0..10)
            .into_par_iter()
            .map(|i| (i as f64).powi(2))
            .collect::<Vec<f64>>();

        evaluate_map
            .into_iter()
            .zip(expected)
            .for_each(|(a, b)| {
                assert_approx_eq!(f64, a, b);
            });
    }

    #[test]
    fn superposition_map() {
        let powers = [SQUARE, CUBE, CUBE];

        let superposition_map = (0..10).map(|i| i as f32).superposition(&powers);
        let expected = (0..10).map(|i| (i as f32).powi(2) + 2.0 * (i as f32).powi(3));

        superposition_map
            .zip(expected)
            .for_each(|(a, b)| {
                assert_approx_eq!(f32, a, b);
            });

        let superposition_map = (0..10).map(|i| i as f64).superposition(&powers);
        let expected = (0..10).map(|i| (i as f64).powi(2) + 2.0 * (i as f64).powi(3));

        superposition_map
            .zip(expected)
            .for_each(|(a, b)| {
                assert_approx_eq!(f64, a, b);
            });
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_superposition_map() {
        let powers = [SQUARE, CUBE, CUBE];

        let superposition_map = (0..10)
            .into_par_iter()
            .map(|i| i as f32)
            .superposition(&powers)
            .collect::<Vec<f32>>();
        let expected = (0..10)
            .into_par_iter()
            .map(|i| (i as f32).powi(2) + 2.0 * (i as f32).powi(3))
            .collect::<Vec<f32>>();

        superposition_map
            .into_iter()
            .zip(expected)
            .for_each(|(a, b)| assert_approx_eq!(f32, a, b));

        let superposition_map = (0..10)
            .into_par_iter()
            .map(|i| i as f64)
            .superposition(&powers)
            .collect::<Vec<f64>>();
        let expected = (0..10)
            .into_par_iter()
            .map(|i| (i as f64).powi(2) + 2.0 * (i as f64).powi(3))
            .collect::<Vec<f64>>();

        superposition_map
            .into_iter()
            .zip(expected)
            .for_each(|(a, b)| assert_approx_eq!(f64, a, b));
    }
}
