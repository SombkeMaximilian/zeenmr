use crate::axis::{Axes, Axis, DiscreteAxes, DiscreteFrequencyAxis, FrequencyAxis};
use crate::dimension::{DimIndex, Dimension, DynDim, StaticDim, assert_rank_compatible};
use crate::intensity_array::{Access, Array, ArrayView, CowAccess, RawStorage, Shape};
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(feature = "serde")]
use crate::intensity_array::StorageOwned;
#[cfg(feature = "serde")]
use num_traits::Float;
#[cfg(feature = "serde")]
use serde::de::Error;
#[cfg(feature = "serde")]
use serde::ser::SerializeStruct;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 1D Spectrum.
pub type Spectrum1D<T, S> = Spectrum<StaticDim<FrequencyAxis<T>, 1>, S, StaticDim<usize, 1>>;

/// 2D Spectrum.
pub type Spectrum2D<T, S> = Spectrum<StaticDim<FrequencyAxis<T>, 2>, S, StaticDim<usize, 2>>;

/// 3D Spectrum.
pub type Spectrum3D<T, S> = Spectrum<StaticDim<FrequencyAxis<T>, 3>, S, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime.
pub type SpectrumDyn<T, S> = Spectrum<DynDim<FrequencyAxis<T>>, S, DynDim<usize>>;

/// Spectrum borrowing its intensities.
pub type SpectrumView<'s, A, T, D> = Spectrum<A, Access<'s, T>, D>;

/// 1D Spectrum borrowing its intensities.
pub type SpectrumView1D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, Access<'s, T>, StaticDim<usize, 1>>;

/// 2D Spectrum borrowing its intensities.
pub type SpectrumView2D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, Access<'s, T>, StaticDim<usize, 2>>;

/// 3D Spectrum borrowing its intensities.
pub type SpectrumView3D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, Access<'s, T>, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime borrowing its intensities.
pub type SpectrumViewDyn<'s, F, T> =
    Spectrum<DynDim<FrequencyAxis<F>>, Access<'s, T>, DynDim<usize>>;

/// Spectrum borrowing its intensities as a slice.
pub type SpectrumSliceView<'s, A, T, D> = Spectrum<A, &'s [T], D>;

/// 1D Spectrum borrowing its intensities as a slice.
pub type SpectrumSliceView1D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, &'s [T], StaticDim<usize, 1>>;

/// 2D Spectrum borrowing its intensities as a slice.
pub type SpectrumSliceView2D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, &'s [T], StaticDim<usize, 2>>;

/// 3D Spectrum borrowing its intensities as a slice.
pub type SpectrumSliceView3D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, &'s [T], StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime borrowing its intensities as a
/// slice.
pub type SpectrumSliceViewDyn<'s, F, T> =
    Spectrum<DynDim<FrequencyAxis<F>>, &'s [T], DynDim<usize>>;

/// Spectrum owning the storage of its intensities.
pub type SpectrumOwned<A, T, D> = Spectrum<A, Box<[T]>, D>;

/// 1D Spectrum owning the storage of its intensities.
pub type SpectrumOwned1D<F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, Box<[T]>, StaticDim<usize, 1>>;

/// 2D Spectrum owning the storage of its intensities.
pub type SpectrumOwned2D<F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, Box<[T]>, StaticDim<usize, 2>>;

/// 3D Spectrum owning the storage of its intensities.
pub type SpectrumOwned3D<F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, Box<[T]>, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime owning the storage of its
/// intensities.
pub type SpectrumOwnedDyn<F, T> = Spectrum<DynDim<FrequencyAxis<F>>, Box<[T]>, DynDim<usize>>;

/// Spectrum using clone-on-write storage for its intensities.
pub type SpectrumCow<'s, A, T, D> = Spectrum<A, CowAccess<'s, T>, D>;

/// 1D Spectrum using clone-on-write storage for its intensities.
pub type SpectrumCow1D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, CowAccess<'s, T>, StaticDim<usize, 1>>;

/// 2D Spectrum using clone-on-write storage for its intensities.
pub type SpectrumCow2D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, CowAccess<'s, T>, StaticDim<usize, 2>>;

/// 3D Spectrum using clone-on-write storage for its intensities.
pub type SpectrumCow3D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, CowAccess<'s, T>, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime using clone-on-write storage for
/// its intensities.
pub type SpectrumCowDyn<'s, F, T> =
    Spectrum<DynDim<FrequencyAxis<F>>, CowAccess<'s, T>, DynDim<usize>>;

/// Spectrum using the standard library's clone-on-write pointer as storage for
/// its intensities.
pub type SpectrumCowSlice<'s, A, T, D> = Spectrum<A, Cow<'s, [T]>, D>;

/// 1D Spectrum using the standard library's clone-on-write pointer as storage
/// for its intensities.
pub type SpectrumCowSlice1D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, Cow<'s, [T]>, StaticDim<usize, 1>>;

/// 2D Spectrum using the standard library's clone-on-write pointer as storage
/// for its intensities.
pub type SpectrumCowSlice2D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, Cow<'s, [T]>, StaticDim<usize, 2>>;

/// 3D Spectrum using the standard library's clone-on-write pointer as storage
/// for its intensities.
pub type SpectrumCowSlice3D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, Cow<'s, [T]>, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime using the standard library's
/// clone-on-write pointer as storage for its intensities.
pub type SpectrumCowSliceDyn<'s, F, T> =
    Spectrum<DynDim<FrequencyAxis<F>>, Cow<'s, [T]>, DynDim<usize>>;

/// Spectrum using reference counted storage for its intensities.
pub type SpectrumRc<A, T, D> = Spectrum<A, Rc<[T]>, D>;

/// 1D Spectrum using reference counted storage for its intensities.
pub type SpectrumRc1D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, Rc<[T]>, StaticDim<usize, 1>>;

/// 2D Spectrum using reference counted storage for its intensities.
pub type SpectrumRc2D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, Rc<[T]>, StaticDim<usize, 2>>;

/// 3D Spectrum using reference counted storage for its intensities.
pub type SpectrumRc3D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, Rc<[T]>, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime using reference counted storage for
/// its intensities.
pub type SpectrumRcDyn<'s, F, T> = Spectrum<DynDim<FrequencyAxis<F>>, Rc<[T]>, DynDim<usize>>;

/// Spectrum using reference atomic counted storage for its intensities.
pub type SpectrumArc<A, T, D> = Spectrum<A, Arc<[T]>, D>;

/// 1D Spectrum using reference atomic counted storage for its intensities.
pub type SpectrumArc1D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 1>, Arc<[T]>, StaticDim<usize, 1>>;

/// 2D Spectrum using reference atomic counted storage for its intensities.
pub type SpectrumArc2D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 2>, Arc<[T]>, StaticDim<usize, 2>>;

/// 3D Spectrum using reference atomic counted storage for its intensities.
pub type SpectrumArc3D<'s, F, T> =
    Spectrum<StaticDim<FrequencyAxis<F>, 3>, Arc<[T]>, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime using reference atomic counted
/// storage for its intensities.
pub type SpectrumArcDyn<'s, F, T> = Spectrum<DynDim<FrequencyAxis<F>>, Arc<[T]>, DynDim<usize>>;

/// Spectrum type with arbitrary dimensions.
#[derive(Clone, Debug)]
pub struct Spectrum<A, S, D> {
    /// Frequency axes.
    axes: Axes<A>,
    /// Spectral intensities.
    ///
    /// Must not contain any invalid values (e.g., `NaN` or either infinity for
    /// floating point intensities).
    intensities: Array<S, D>,
}

impl<A, S, D> Spectrum<A, S, D>
where
    D: Dimension<Elem = usize>,
{
    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        self.intensities.rank()
    }

    /// Returns a reference to the shape of the spectrum.
    pub fn shape(&self) -> &Shape<D> {
        self.intensities.shape()
    }

    /// Returns the number of intensity values that the spectrum contains.
    pub fn len(&self) -> usize {
        self.intensities.len()
    }

    /// Returns the number of intensity values that the spectrum contains.
    ///
    /// This always returns `false`, since no empty arrays cannot be
    /// constructed.
    pub fn is_empty(&self) -> bool {
        self.intensities.is_empty()
    }
}

impl<A, S, D> Spectrum<A, S, D>
where
    S: RawStorage,
    D: Dimension<Elem = usize>,
{
    /// Returns a view of the intensities.
    pub fn intensities(&self) -> ArrayView<'_, S::Elem, D> {
        self.intensities.view()
    }
}

impl<F, A, S, D> Spectrum<A, S, D>
where
    A: Dimension<Elem = FrequencyAxis<F>>,
    S: RawStorage,
    D: Dimension<Elem = usize>,
{
    /// Creates a new spectrum.
    ///
    /// Returns `None` if `axes` and `intensities` have different ranks.
    pub fn new(axes: Axes<A>, intensities: Array<S, D>) -> Option<Self> {
        const { assert_rank_compatible::<A, D>() }
        if axes.rank() != intensities.rank() {
            return None;
        }

        Some(Self { axes, intensities })
    }

    /// Returns a view of the spectrum.
    pub fn view(&self) -> SpectrumView<'_, A, S::Elem, D> {
        SpectrumView {
            axes: self.axes.clone(),
            intensities: self.intensities.view(),
        }
    }

    /// Returns the parts of `self`.
    ///
    /// The axes and array returned by this method are guaranteed to be valid
    /// for [`Spectrum::new`].
    pub fn into_parts(self) -> (Axes<A>, Array<S, D>) {
        (self.axes, self.intensities)
    }

    /// Returns a reference to the frequency axis of the spectrum at `index`.
    ///
    /// Use [`Spectrum::discrete_axis`] to get the axis with its extent.
    pub fn axis(&self, index: DimIndex) -> Option<&FrequencyAxis<F>> {
        self.axes.get(index)
    }

    /// Returns a discrete frequency axis of the spectrum at `index`.
    pub fn discrete_axis(&self, index: DimIndex) -> Option<DiscreteFrequencyAxis<'_, F>> {
        let extent = self.intensities.shape().get(index)?;

        self.axes
            .get(index)
            .map(|axis| axis.discrete(extent))
    }

    /// Returns the discrete axes of the spectrum with their respective extents.
    pub fn discrete_axes<'a, G>(&'a self) -> DiscreteAxes<G>
    where
        F: 'a,
        G: Dimension<Elem = DiscreteFrequencyAxis<'a, F>>,
    {
        self.axes
            .discrete(self.intensities.shape().clone())
            .expect("type invariant guarantees that this is valid")
    }
}

#[cfg(feature = "serde")]
impl<A, S, D> Serialize for Spectrum<A, S, D>
where
    A: Dimension + Serialize,
    S: RawStorage,
    S::Elem: Serialize,
    D: Dimension<Elem = usize> + Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut spectrum = serializer.serialize_struct("Spectrum", 2)?;
        spectrum.serialize_field("axes", &self.axes)?;
        spectrum.serialize_field("intensities", &self.intensities)?;

        spectrum.end()
    }
}

/// Raw spectrum without invariants as an intermediate for deserialization.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(bound(
    deserialize = "A: Deserialize<'de>, S::Elem: Deserialize<'de>, D: Deserialize<'de>"
))]
struct RawSpectrum<A, S, D>
where
    A: Dimension,
    S: StorageOwned,
    D: Dimension<Elem = usize>,
{
    /// Frequency axes.
    axes: Axes<A>,
    /// Spectral intensities.
    intensities: Array<S, D>,
}

#[cfg(feature = "serde")]
impl<'de, T, A, S, D> Deserialize<'de> for Spectrum<A, S, D>
where
    T: Float + Deserialize<'de>,
    A: Dimension<Elem = FrequencyAxis<T>> + Deserialize<'de>,
    S: StorageOwned,
    S::Elem: Deserialize<'de>,
    D: Dimension<Elem = usize> + Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let RawSpectrum { axes, intensities } = RawSpectrum::<A, S, D>::deserialize(deserializer)?;

        if axes.rank() != intensities.rank() {
            return Err(De::Error::custom("rank mismatch between axes and order"));
        }

        Ok(Spectrum::new(axes, intensities).expect("failure modes should be exhausted"))
    }
}
