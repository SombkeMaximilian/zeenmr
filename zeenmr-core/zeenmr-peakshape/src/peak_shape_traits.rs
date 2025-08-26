use num_traits::Float;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub trait PeakShape: Evaluate {}

impl<T> PeakShape for T where T: Evaluate {}

pub trait Evaluate {
    type Scalar: Float;

    fn evaluate(&self, at: Self::Scalar) -> Self::Scalar;
}

impl<T> Evaluate for &T
where
    T: Evaluate + ?Sized,
{
    type Scalar = T::Scalar;

    fn evaluate(&self, at: Self::Scalar) -> Self::Scalar {
        (*self).evaluate(at)
    }
}

impl<T> Evaluate for &mut T
where
    T: Evaluate + ?Sized,
{
    type Scalar = T::Scalar;

    fn evaluate(&self, at: Self::Scalar) -> Self::Scalar {
        (&**self).evaluate(at)
    }
}

pub trait Superposition: Iterator {
    fn superposition<F>(self, at: F) -> F
    where
        Self: Sized,
        F: Float + std::iter::Sum,
        Self::Item: Evaluate<Scalar = F>,
    {
        self.map(|e| e.evaluate(at)).sum()
    }
}

impl<I> Superposition for I where I: Iterator {}

#[cfg(feature = "rayon")]
pub trait ParSuperposition: ParallelIterator {
    fn superposition<F>(self, at: F) -> F
    where
        Self: Sized,
        F: Float + std::iter::Sum + Send + Sync,
        Self::Item: Evaluate<Scalar = F>,
    {
        self.map(|e| e.evaluate(at)).sum()
    }
}

#[cfg(feature = "rayon")]
impl<I> ParSuperposition for I where I: ParallelIterator {}

pub trait EvaluateMap: Iterator {
    fn evaluate_map<E>(self, evaluator: E) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
        E: Evaluate<Scalar = Self::Item>,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}

impl<I> EvaluateMap for I where I: Iterator {}

#[cfg(feature = "rayon")]
pub trait ParEvaluateMap: ParallelIterator {
    fn evaluate_map<E>(self, evaluator: E) -> impl ParallelIterator<Item = Self::Item>
    where
        Self: Sized,
        E: Evaluate<Scalar = Self::Item> + Send + Sync,
    {
        self.map(move |x| evaluator.evaluate(x))
    }
}

#[cfg(feature = "rayon")]
impl<I> ParEvaluateMap for I where I: ParallelIterator {}

pub trait SuperpositionMap: Iterator {
    fn superposition_map<E>(self, evaluators: &[E]) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
        E: Evaluate<Scalar = Self::Item>,
        Self::Item: Float + std::iter::Sum,
    {
        self.map(|x| evaluators.iter().superposition(x))
    }
}

impl<I> SuperpositionMap for I where I: Iterator {}

#[cfg(feature = "rayon")]
pub trait ParSuperpositionMap: ParallelIterator {
    fn superposition_map<E>(self, evaluators: &[E]) -> impl ParallelIterator<Item = Self::Item>
    where
        Self: Sized,
        E: Evaluate<Scalar = Self::Item> + Sync,
        Self::Item: Float + std::iter::Sum,
    {
        self.map(|x| evaluators.iter().superposition(x))
    }
}

#[cfg(feature = "rayon")]
impl<I> ParSuperpositionMap for I where I: ParallelIterator {}

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
        type Scalar = f64;

        fn evaluate(&self, at: Self::Scalar) -> Self::Scalar {
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
        let evaluate_map = (0..10).map(|i| i as f64).evaluate_map(SQUARE);
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
            .evaluate_map(SQUARE)
            .collect::<Vec<f64>>();
        let expected = (0..10)
            .into_par_iter()
            .map(|i| (i as f64).powi(2))
            .collect::<Vec<f64>>();
        evaluate_map
            .into_iter()
            .zip(expected.into_iter())
            .for_each(|(a, b)| {
                assert_approx_eq!(f64, a, b);
            })
    }

    #[test]
    fn superposition_map() {
        let powers = [SQUARE, CUBE, CUBE];
        let superposition_map = (0..10)
            .map(|i| i as f64)
            .superposition_map(&powers);
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
            .superposition_map(&powers)
            .collect::<Vec<f64>>();
        let expected = (0..10)
            .into_par_iter()
            .map(|i| (i as f64).powi(2) + 2.0 * (i as f64).powi(3))
            .collect::<Vec<f64>>();
        superposition_map
            .into_iter()
            .zip(expected.into_iter())
            .for_each(|(a, b)| assert_approx_eq!(f64, a, b));
    }
}
