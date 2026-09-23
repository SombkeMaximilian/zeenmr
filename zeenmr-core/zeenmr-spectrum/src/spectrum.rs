use crate::axis::{Axes, FrequencyAxis, FrequencyGrid, GridAxes};
use crate::dimension::{Dimension, DynDim, StaticDim, assert_rank_compatible};
use crate::intensity_array::{Access, Array, ArrayView, RawStorage, StorageOwned};
use num_traits::Float;
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::de::Error;
#[cfg(feature = "serde")]
use serde::ser::SerializeStruct;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Spectrum borrowing its intensities.
pub type SpectrumView<'s, A, T, D> = Spectrum<A, Access<'s, T>, D>;

/// Spectrum borrowing its intensities as a slice.
pub type SpectrumSliceView<'s, A, T, D> = Spectrum<A, &'s [T], D>;

/// Spectrum owning the storage of its intensities.
pub type SpectrumOwned<A, T, D> = Spectrum<A, Box<[T]>, D>;

/// Spectrum using clone-on-write storage for its intensities.
pub type SpectrumCow<'s, A, T, D> = Spectrum<A, Cow<'s, [T]>, D>;

/// Spectrum using reference counted storage for its intensities.
pub type SpectrumRc<A, T, D> = Spectrum<A, Rc<[T]>, D>;

/// Spectrum using reference atomic counted storage for its intensities.
pub type SpectrumArc<A, T, D> = Spectrum<A, Arc<[T]>, D>;

/// 1D Spectrum.
pub type Spectrum1D<F, S> = Spectrum<StaticDim<FrequencyAxis<F>, 1>, S, StaticDim<usize, 1>>;

/// 2D Spectrum.
pub type Spectrum2D<F, S> = Spectrum<StaticDim<FrequencyAxis<F>, 2>, S, StaticDim<usize, 2>>;

/// 3D Spectrum.
pub type Spectrum3D<F, S> = Spectrum<StaticDim<FrequencyAxis<F>, 3>, S, StaticDim<usize, 3>>;

/// Spectrum of a rank determined at runtime.
pub type SpectrumDyn<F, S> = Spectrum<DynDim<FrequencyAxis<F>>, S, DynDim<usize>>;

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

    /// Returns the parts of `self`.
    ///
    /// The axes and array returned by this method are guaranteed to be valid
    /// for [`Spectrum::new`].
    pub fn into_parts(self) -> (Axes<A>, Array<S, D>) {
        (self.axes, self.intensities)
    }

    /// Returns the rank of `self`.
    pub fn rank(&self) -> usize {
        debug_assert_eq!(self.axes.rank(), self.intensities.rank());

        self.axes.rank()
    }

    /// Returns the grid axes of the spectrum, with the respective extent
    /// attached.
    pub fn axes<'a, G>(&'a self) -> GridAxes<G>
    where
        F: 'a,
        G: Dimension<Elem = FrequencyGrid<'a, F>>,
    {
        debug_assert_eq!(self.axes.rank(), self.intensities.rank());

        self.axes
            .grid(self.intensities.shape().clone())
            .expect("type invariant guarantees that this is valid")
    }

    /// Returns a view of the intensities.
    pub fn intensities(&self) -> ArrayView<'_, S::Elem, D> {
        self.intensities.view()
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
