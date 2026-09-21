use crate::axis::{FrequencyAxis, TimeAxis};
use crate::dimension::{
    DimIndex, Dimension, DynDim, IntoDimension, StaticDim, assert_rank_compatible,
};
use crate::intensity_array::Shape;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for NMR spectrum or FID axis types.
pub trait Axis {
    /// 1D grid type of the axis with a length attached.
    type Grid<'a>
    where
        Self: 'a;

    /// Attaches a length to the axis, producing a 1D grid.
    ///
    /// # Precision
    ///
    /// Due to floating point errors when adding and multiplying, passing a
    /// `len` that cannot be represented by `T` may lead to significant errors
    /// (e.g., `len > 2^24` for `f32`) in the downstream methods.
    fn grid(&self, len: usize) -> Self::Grid<'_>;
}

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
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Axes<A>(A);

impl<A> Axes<A>
where
    A: Dimension,
    A::Elem: Axis,
{
    /// Creates a new collection of axes.
    pub fn new(axes: A) -> Self {
        Self(axes)
    }

    /// Returns the equivalent axes with a rank determined at runtime.
    pub fn to_dyn(self) -> Axes<DynDim<A::Elem>> {
        self.to_dimension()
            .expect("DynDim can represent any rank")
    }

    /// Returns the equivalent axes of rank `N`.
    ///
    /// Returns `None` if `self` does not have rank `N`.
    pub fn try_to_static<const N: usize>(self) -> Option<Axes<StaticDim<A::Elem, N>>> {
        self.to_dimension()
    }

    /// Returns the equivalent axis collection over `A2`.
    ///
    /// Returns `None` if `A2` cannot represent the rank of `self`.
    pub fn to_dimension<A2>(&self) -> Option<Axes<A2>>
    where
        A2: Dimension<Elem = A::Elem>,
    {
        const { assert_rank_compatible::<A, A2>() };

        Some(Axes(A2::from_dimension(&self.0)?))
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns a reference to the axis at `index`.
    pub fn get(&self, index: DimIndex) -> Option<&A::Elem> {
        self.0.get(index)
    }

    /// Returns a mutable reference to the axis at `index`.
    pub fn get_mut(&mut self, index: DimIndex) -> Option<&mut A::Elem> {
        self.0.get_mut(index)
    }

    /// Returns a slice containing all axes.
    pub fn as_slice(&self) -> &[A::Elem] {
        self.0.as_slice()
    }

    /// Returns a mutable slice containing all axes.
    pub fn as_mut_slice(&mut self) -> &mut [A::Elem] {
        self.0.as_mut_slice()
    }

    /// Returns a grid over the axes.
    pub fn grid<'a, D, G>(&'a self, shape: Shape<D>) -> Option<GridAxes<G>>
    where
        D: Dimension<Elem = usize>,
        G: Dimension<Elem = <A::Elem as Axis>::Grid<'a>>,
        A::Elem: 'a,
    {
        const {
            assert_rank_compatible::<A, D>();
            assert_rank_compatible::<A, G>();
        };
        if self.rank() != shape.rank() {
            return None;
        }

        let axes = self.as_slice();
        let extents = shape.as_slice();

        Some(GridAxes::new(G::from_fn(self.rank(), |dim| {
            axes[dim].grid(extents[dim])
        })?))
    }
}

/// Grid of a spectrum or FID.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct GridAxes<G>(G);

impl<G> GridAxes<G>
where
    G: Dimension,
{
    /// Creates a new collection of grid axes.
    pub fn new(axes: G) -> Self {
        Self(axes)
    }

    /// Returns the equivalent grid axes with a rank determined at runtime.
    pub fn to_dyn(self) -> Axes<DynDim<G::Elem>> {
        self.to_dimension()
            .expect("DynDim can represent any rank")
    }

    /// Returns the equivalent grid axes of rank `N`.
    ///
    /// Returns `None` if `self` does not have rank `N`.
    pub fn try_to_static<const N: usize>(self) -> Option<Axes<StaticDim<G::Elem, N>>> {
        self.to_dimension()
    }

    /// Returns the equivalent grid axis collection over `G2`.
    ///
    /// Returns `None` if `G2` cannot represent the rank of `self`.
    pub fn to_dimension<G2>(&self) -> Option<Axes<G2>>
    where
        G2: Dimension<Elem = G::Elem>,
    {
        const { assert_rank_compatible::<G, G2>() };

        Some(Axes(G2::from_dimension(&self.0)?))
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns a reference to the grid axis at `index`.
    pub fn get(&self, index: DimIndex) -> Option<&G::Elem> {
        self.0.get(index)
    }

    /// Returns a slice containing all grid axes.
    pub fn as_slice(&self) -> &[G::Elem] {
        self.0.as_slice()
    }
}
