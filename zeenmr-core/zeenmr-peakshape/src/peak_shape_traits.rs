#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Marker trait for types that can represent peak shapes in spectral data.
pub trait PeakShape: Evaluate {}

impl<E> PeakShape for E where E: Evaluate {}

/// Trait for mathematical functions that can be evaluated at a given point.
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use zeenmr_peakshape::Evaluate;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// assert_approx_eq!(f64, Power(2).evaluate(3.0), 9.0);
/// ```
pub trait Evaluate {
    /// Evaluate the function at the specified point.
    fn evaluate(&self, at: f64) -> f64;
}

impl<E> Evaluate for &E
where
    E: Evaluate,
{
    fn evaluate(&self, at: f64) -> f64 {
        (*self).evaluate(at)
    }
}

impl<E> Evaluate for &mut E
where
    E: Evaluate,
{
    fn evaluate(&self, at: f64) -> f64 {
        (**self).evaluate(at)
    }
}

/// Extension trait for iterators of [`Evaluate`] types that provides
/// superposition of their function values.
///
/// Given a family of functions `(f_i)_(i=1...n)`, their **superposition**
/// `F` at a point `x` is defined as the sum of the individual evaluations:
///
/// ```text
/// F(x) = f_1(x) + f_2(x) + ... + f_n(x)
/// ```
///
/// This is analogous to [`Iterator::sum`], but instead of summing the items
/// directly, it sums their evaluations at a specified point.
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::Superposition;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let powers = [Power(2), Power(2), Power(3)];
/// assert_approx_eq!(f64, powers.into_iter().superposition(3.0), 45.0);
/// ```
pub trait Superposition: Iterator {
    /// Evaluate each function in the iterator at the given point and return
    /// the sum of the results.
    fn superposition(self, at: f64) -> f64
    where
        Self::Item: Evaluate;
}

impl<E, I> Superposition for I
where
    E: Evaluate,
    I: Iterator<Item = E>,
{
    fn superposition(self, at: f64) -> f64
    where
        Self::Item: Evaluate,
    {
        self.map(|e| e.evaluate(at)).sum()
    }
}

/// Extension trait for parallel iterators of [`Evaluate`] types that provides
/// superposition of their function values.
///
/// Given a family of functions `(f_i)_(i=1...n)`, their **superposition**
/// `F` at a point `x` is defined as the sum of the individual evaluations:
///
/// ```text
/// F(x) = f_1(x) + f_2(x) + ... + f_n(x)
/// ```
///
/// This is the parallel analogue of [`Superposition`], powered by [rayon].
///
/// [rayon]: https://docs.rs/rayon
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use rayon::prelude::*;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::ParSuperposition;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let powers = vec![Power(2), Power(2), Power(3)];
/// assert_approx_eq!(f64, powers.into_par_iter().superposition(3.0), 45.0);
/// ```
#[cfg(feature = "rayon")]
pub trait ParSuperposition: ParallelIterator {
    /// Evaluate each function in the parallel iterator at the given point and
    /// return the sum of the results.
    fn superposition(self, at: f64) -> f64
    where
        Self::Item: Evaluate;
}

#[cfg(feature = "rayon")]
impl<E, I> ParSuperposition for I
where
    E: Evaluate + Send + Sync,
    I: ParallelIterator<Item = E>,
{
    fn superposition(self, at: f64) -> f64
    where
        Self::Item: Evaluate,
    {
        self.map(|e| e.evaluate(at)).sum()
    }
}

/// Extension trait for iterators of `f64` values that allows mapping them
/// through an [`Evaluate`] implementation.
///
/// Given a function `f` implementing [`Evaluate`] and a sequence of input
/// values `(x_i)_(i=1...n)`, this adaptor produces an iterator of the
/// corresponding evaluations:
///
/// ```text
/// { f(x_1), f(x_2), ..., f(x_n) }
/// ```
///
/// This is analogous to [`Iterator::map`], but uses the provided evaluator
/// instead of an arbitrary closure.
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::EvaluateMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let squares = [2.0, 3.0, 4.0]
///     .into_iter()
///     .evaluate(Power(2))
///     .collect::<Vec<f64>>();
/// assert_approx_eq!(f64, squares[0], 4.0);
/// assert_approx_eq!(f64, squares[1], 9.0);
/// assert_approx_eq!(f64, squares[2], 16.0);
/// ```
pub trait EvaluateMap: Iterator {
    /// Apply the given evaluator to each item in the iterator, producing an
    /// iterator of evaluated results.
    fn evaluate<E>(self, evaluator: E) -> impl Iterator<Item = f64>
    where
        E: Evaluate;
}

impl<I> EvaluateMap for I
where
    I: Iterator<Item = f64>,
{
    fn evaluate<E>(self, evaluator: E) -> impl Iterator<Item = f64>
    where
        E: Evaluate,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}

/// Extension trait for parallel iterators of `f64` values that allows mapping
/// them through an [`Evaluate`] implementation in parallel.
///
/// Given a function `f` implementing [`Evaluate`] and a sequence of input
/// values `(x_i)_(i=1...n)`, this adaptor produces a parallel iterator of the
/// corresponding evaluations:
///
/// ```text
/// { f(x_1), f(x_2), ..., f(x_n) }
/// ```
///
/// This is the parallel analogue of [`EvaluateMap`], powered by [rayon].
///
/// [rayon]: https://docs.rs/rayon
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use rayon::prelude::*;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::ParEvaluateMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
///     fn evaluate(&self, at: f64) -> f64 {
///         at.powi(self.0)
///     }
/// }
///
/// let squares = [2.0, 3.0, 4.0]
///     .into_par_iter()
///     .evaluate(Power(2))
///     .collect::<Vec<f64>>();
/// assert_approx_eq!(f64, squares[0], 4.0);
/// assert_approx_eq!(f64, squares[1], 9.0);
/// assert_approx_eq!(f64, squares[2], 16.0);
/// ```
#[cfg(feature = "rayon")]
pub trait ParEvaluateMap: IndexedParallelIterator {
    /// Apply the given evaluator to each item in the indexed parallel iterator,
    /// producing an indexed parallel iterator of evaluated results.
    fn evaluate<E>(self, evaluator: E) -> impl IndexedParallelIterator<Item = f64>
    where
        E: Evaluate + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<I> ParEvaluateMap for I
where
    I: IndexedParallelIterator<Item = f64>,
{
    fn evaluate<E>(self, evaluator: E) -> impl IndexedParallelIterator<Item = f64>
    where
        E: Evaluate + Send + Sync,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}

/// Extension trait for iterators of `f64` values that allows mapping them
/// through the **superposition** of evaluators.
///
/// Given a family of functions `(f_j)_(j=1...m)` and a sequence of input
/// values `(x_i)_(i=1...n)`, this adaptor produces an iterator of their
/// superpositions:
///
/// ```text
/// { F(x_1), F(x_2), ..., F(x_n) }
/// with   F(x_i) = f_1(x_i) + f_2(x_i) + ... + f_m(x_i)
/// ```
///
/// This combines the behaviors of [`EvaluateMap`] and [`Superposition`]:
/// - Like [`EvaluateMap`], it maps over an iterator of input points.
/// - Like [`Superposition`], it sums the contributions of multiple evaluators.
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::SuperpositionMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
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
/// assert_approx_eq!(f64, results[0], 12.0);
/// assert_approx_eq!(f64, results[1], 36.0);
/// assert_approx_eq!(f64, results[2], 80.0);
/// ```
pub trait SuperpositionMap: Iterator {
    /// Apply the superposition of the given evaluators to each item in the
    /// iterator, producing an iterator of evaluated results.
    fn superposition<E>(self, evaluators: &[E]) -> impl Iterator<Item = f64>
    where
        E: Evaluate;
}

impl<I> SuperpositionMap for I
where
    I: Iterator<Item = f64>,
{
    fn superposition<E>(self, evaluators: &[E]) -> impl Iterator<Item = f64>
    where
        E: Evaluate,
    {
        self.map(|x| evaluators.iter().superposition(x))
    }
}

/// Extension trait for parallel iterators of `f64` values that allows mapping
/// them through the **superposition** of multiple evaluators in parallel.
///
/// Given a family of functions `(f_j)_(j=1...m)` and a sequence of input
/// values `(x_i)_(i=1...n)`, this adaptor produces a parallel iterator of
/// their superpositions:
///
/// ```text
/// { F(x_1), F(x_2), ..., F(x_n) }
/// with   F(x_i) = f_1(x_i) + f_2(x_i) + ... + f_m(x_i)
/// ```
///
/// This is the parallel analogue of [`SuperpositionMap`], powered by [`rayon`].
///
/// # Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use rayon::prelude::*;
/// use zeenmr_peakshape::Evaluate;
/// use zeenmr_peakshape::iter::ParSuperpositionMap;
///
/// #[derive(Copy, Clone, Debug)]
/// struct Power(i32);
///
/// impl Evaluate for Power {
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
/// assert_approx_eq!(f64, results[0], 2.0f64.powi(2) + 2.0f64.powi(3));
/// assert_approx_eq!(f64, results[1], 3.0f64.powi(2) + 3.0f64.powi(3));
/// assert_approx_eq!(f64, results[2], 4.0f64.powi(2) + 4.0f64.powi(3));
/// ```
#[cfg(feature = "rayon")]
pub trait ParSuperpositionMap: IndexedParallelIterator {
    /// Apply the superposition of the given evaluators to each item in the
    /// parallel iterator, producing a parallel iterator of evaluated results.
    fn superposition<E>(self, evaluators: &[E]) -> impl IndexedParallelIterator<Item = f64>
    where
        E: Evaluate + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<I> ParSuperpositionMap for I
where
    I: IndexedParallelIterator<Item = f64>,
{
    fn superposition<E>(self, evaluators: &[E]) -> impl IndexedParallelIterator<Item = f64>
    where
        E: Evaluate + Send + Sync,
    {
        self.map(|x| evaluators.iter().superposition(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use static_assertions::assert_impl_all;

    // this isn't really a peak shape, but it serves as a simple test example
    #[derive(Copy, Clone, Debug)]
    struct Power(i32);

    const SQUARE: Power = Power(2);
    const CUBE: Power = Power(3);

    impl Evaluate for Power {
        fn evaluate(&self, at: f64) -> f64 {
            at.powi(self.0)
        }
    }

    #[test]
    fn traits() {
        assert_impl_all!(Power: PeakShape, Send, Sync);
        assert_impl_all!(&Power: PeakShape, Send, Sync);
        assert_impl_all!(&mut Power: PeakShape, Send, Sync);
        assert_impl_all!(std::vec::IntoIter<Power>: Superposition);
        assert_impl_all!(std::vec::IntoIter<f64>: EvaluateMap, SuperpositionMap);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_traits() {
        assert_impl_all!(rayon::vec::IntoIter<Power>: ParSuperposition);
        assert_impl_all!(rayon::vec::IntoIter<f64>: ParEvaluateMap, ParSuperpositionMap);
    }

    #[test]
    fn evaluate() {
        assert_approx_eq!(f64, SQUARE.evaluate(3.0), 9.0);
    }

    #[test]
    fn superposition() {
        let powers = [SQUARE, SQUARE, CUBE];
        let value = 3.0;
        let superposition = powers.iter().superposition(value);
        assert_approx_eq!(f64, superposition, 2.0 * value.powi(2) + value.powi(3));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_superposition() {
        let powers = [SQUARE, SQUARE, CUBE];
        let value = 3.0;
        let par_superposition = powers.par_iter().superposition(value);
        assert_approx_eq!(f64, par_superposition, 2.0 * value.powi(2) + value.powi(3));
    }

    #[test]
    fn evaluate_map() {
        let evaluate_map = (0..10).map(|i| i as f64).evaluate(SQUARE);
        let expected = (0..10).map(|i| (i as f64).powi(2));
        evaluate_map.zip(expected).for_each(|(a, b)| {
            assert_approx_eq!(f64, a, b);
        })
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_evaluate_map() {
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
            })
    }

    #[test]
    fn superposition_map() {
        let powers = [SQUARE, CUBE, CUBE];
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
