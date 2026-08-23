use crate::axis::{FrequencyAxis, TimeAxis};
use crate::dimension::{assert_rank_compatible, Dimension, DynDim, StaticDim, DimIndex, IntoDimension};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Frequency axes with a rank determined at runtime.
pub type DynFrequencyAxes<T> = Axes<DynDim<FrequencyAxis<T>>>;

/// Frequency axes with a rank determined at compile-time.
pub type StaticFrequencyAxes<T, const N: usize> = Axes<StaticDim<FrequencyAxis<T>, N>>;

/// Time axes with a rank determined at runtime.
pub type DynTimeAxes<T> = Axes<DynDim<TimeAxis<T>>>;

/// Time axes with a rank determined at compile-time.
pub type StaticTimeAxes<T, const N: usize> = Axes<StaticDim<TimeAxis<T>, N>>;

/// Convenience function for creating a collection of frequency axes.
pub fn frequency_axes<T, A>(axes: A) -> Axes<A::Dim>
where
    A: IntoDimension,
    A::Dim: Dimension<Elem = FrequencyAxis<T>>,
{
    Axes(axes.into_dim())
}

/// Convenience function for creating a collection of time axes.
pub fn time_axes<T, A>(axes: A) -> Axes<A::Dim>
where
    A: IntoDimension,
    A::Dim: Dimension<Elem = TimeAxis<T>>,
{
    Axes(axes.into_dim())
}

/// Axes of a spectrum or FID.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Axes<A>(A);

impl<A> Axes<A>
where
    A: Dimension,
{
    /// Creates a new collection of axes.
    pub fn new(axes: A) -> Self {
        Self(axes)
    }

    /// Returns the equivalent axes with a rank determined at runtime.
    pub fn into_dyn(self) -> Axes<DynDim<A::Elem>> {
        self.to_dimension()
            .expect("DynDim can represent any rank")
    }

    /// Returns the equivalent axes of rank `N`.
    ///
    /// Returns `None` if `self` does not have rank `N`.
    pub fn try_into_static<const N: usize>(self) -> Option<Axes<StaticDim<A::Elem, N>>> {
        self.to_dimension()
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns a reference to the axis at the specified `DimIndex`.
    pub fn get(&self, index: DimIndex) -> Option<&A::Elem> {
        self.0.as_slice().get(index.0)
    }

    /// Returns a slice containing all axes.
    pub fn as_slice(&self) -> &[A::Elem] {
        self.0.as_slice()
    }

    /// Returns a mutable slice containing all axes.
    pub fn as_mut_slice(&mut self) -> &mut [A::Elem] {
        self.0.as_mut_slice()
    }
}

impl<A1> Axes<A1>
where
    A1: Dimension,
{
    /// Returns the equivalent axis collection over `D2`.
    ///
    /// Returns `None` if `D2` cannot represent the rank of `self`.
    pub fn to_dimension<A2>(&self) -> Option<Axes<A2>>
    where
        A2: Dimension<Elem = A1::Elem>,
    {
        const { assert_rank_compatible::<A1, A2>() };

        Some(Axes(A2::from_dimension(&self.0)?))
    }
}
