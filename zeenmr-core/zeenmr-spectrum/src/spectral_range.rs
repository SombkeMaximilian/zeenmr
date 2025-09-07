use crate::Spectrum;
use crate::error::{Error, Result};
use std::ops::{Bound, Range, RangeBounds, RangeInclusive};
use uom::si::f64::{Frequency, Ratio};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// [`SpectralRange`] type alias for index ranges.
///
/// Note that for this type, `end` is exclusive.
pub type IndexRange = SpectralRange<usize>;

/// [`SpectralRange`] type alias for frequency ranges.
pub type FrequencyRange = SpectralRange<Frequency>;

/// [`SpectralRange`] type alias for chemical shift ranges.
pub type ChemicalShiftRange = SpectralRange<Ratio>;

/// [`SpectralRange`] type alias for relative ranges.
pub type RelativeRange = SpectralRange<f64>;

/// A generic spectral range with a start and end value.
///
/// # Range Types
///
/// Using the following type aliases is recommended for clarity and convenience:
/// - [`IndexRange`]: A range of `usize` indices into the intensities,
///   expressed as `[i, j)`, where `i` is inclusive and `j` is exclusive.
/// - [`FrequencyRange`]: A range of [`Frequency`] values, expressed as
///   `[f1, f2]`. Both bounds are inclusive and may appear in any order.
/// - [`ChemicalShiftRange`]: A range of chemical shifts as [`Ratio`]s,
///   expressed as `[s1, s2]`. Both bounds are inclusive and may appear in any
///   order.
/// - [`RelativeRange`]: A range of relative positions along the spectral
///   axis, expressed as `[r1, r2]`. Both bounds are inclusive, with values
///   in the normalized range `[0_f64, 1_f64]` and `r1 < r2`.
///
/// These types each implement [`From<Range<T>>`] or
/// [`From<RangeInclusive<T>>`] respectively, and [`From<(T, T)>`].
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs
///
/// When the `serde` feature is enabled, `SpectralRange` implements the
/// [`Serialize`] and [`Deserialize`] traits provided the type `T` also
/// implements them. For the type aliases mentioned above, this is the
/// case.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpectralRange<T> {
    /// Start of the range (inclusive).
    pub start: T,
    /// End of the range (inclusive, except for index ranges).
    pub end: T,
}

/// Macro to implement `RangeBounds` for a given range type.
macro_rules! range_bounds_impls {
    ($R:ty, $T:ty, $start:ident, $end:ident) => {
        impl RangeBounds<$T> for $R {
            fn start_bound(&self) -> Bound<&$T> {
                Bound::$start(&self.start)
            }

            fn end_bound(&self) -> Bound<&$T> {
                Bound::$end(&self.end)
            }
        }
    };
}

range_bounds_impls!(IndexRange, usize, Included, Excluded);
range_bounds_impls!(FrequencyRange, Frequency, Included, Included);
range_bounds_impls!(ChemicalShiftRange, Ratio, Included, Included);
range_bounds_impls!(RelativeRange, f64, Included, Included);

/// Macro to implement `From` for tuples and range types.
macro_rules! from_impls {
    ($($R:ident<$T:ty>: $start:ident, $end:ident), + $(,)?) => {
        $(
            impl From<($T, $T)> for SpectralRange<$T> {
                fn from(value: ($T, $T)) -> Self {
                    Self {
                        start: value.0,
                        end: value.1,
                    }
                }
            }

            impl From<$R<$T>> for SpectralRange<$T> {
                fn from(value: $R<$T>) -> Self {
                    Self {
                        start: *value.$start(),
                        end: *value.$end(),
                    }
                }
            }
        )+
    };
}

impl From<(usize, usize)> for SpectralRange<usize> {
    fn from(value: (usize, usize)) -> Self {
        Self {
            start: value.0,
            end: value.1,
        }
    }
}

impl From<Range<usize>> for SpectralRange<usize> {
    fn from(value: Range<usize>) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

from_impls!(
    RangeInclusive<Frequency>: start, end,
    RangeInclusive<Ratio>: start, end,
    RangeInclusive<f64>: start, end
);

/// Trait for converting spectral ranges to ranges of indices within a
/// [`Spectrum`].
pub trait TryIntoIndexRange {
    /// Converts the [`SpectralRange`] to an [`IndexRange`].
    ///
    /// An [`IndexRange`] returned by this is ordered such that `start` is less
    /// than or equal to `end`, and
    ///
    /// # Errors
    ///
    /// Returns an error if either range bound is out of bounds of the spectral
    /// axis or non-finite. Even a conversion from [`IndexRange`] to
    /// [`IndexRange`] can fail if the indices are out of bounds.
    fn try_into_index_range(self, spectrum: &Spectrum) -> Result<IndexRange>;
}

/// Trait for converting spectral index ranges within a [`Spectrum`] to ranges
/// of a given type.
pub trait TryFromIndexRange {
    /// Converts the [`IndexRange`] to a [`SpectralRange`] of the implementor
    /// type.
    ///
    /// # Errors
    ///
    /// Returns an error if either range bound is out of bounds of the spectral
    /// axis.
    fn try_from_index_range(range: IndexRange, spectrum: &Spectrum) -> Result<Self>
    where
        Self: Sized;
}

/// Macro to implement [`TryIntoIndexRange`] and [`TryFromIndexRange`] for a
/// given type, using the specified methods of [`SpectralLinspace`] for
/// conversion.
///
/// [`SpectralLinspace`]: crate::SpectralLinspace
macro_rules! try_index_range_impls {
    ($T:ty, $to_fractional:ident, $from_index:ident) => {
        impl TryIntoIndexRange for $T {
            fn try_into_index_range(self, spectrum: &Spectrum) -> Result<IndexRange> {
                let linspace = &spectrum.spectral_linspace;
                let start = linspace.$to_fractional(self.start)?;
                let end = linspace.$to_fractional(self.end)?;

                Ok((
                    start.min(end).ceil() as usize,
                    start.max(end).floor() as usize + 1,
                )
                    .into())
            }
        }

        impl TryFromIndexRange for $T {
            fn try_from_index_range(range: IndexRange, spectrum: &Spectrum) -> Result<Self> {
                let linspace = &spectrum.spectral_linspace;
                let start = linspace.$from_index(range.start)?;
                let end = linspace.$from_index(range.end - 1)?;

                Ok((start, end).into())
            }
        }
    };
}

impl TryIntoIndexRange for IndexRange {
    fn try_into_index_range(self, spectrum: &Spectrum) -> Result<IndexRange> {
        let linspace_range = 0..=spectrum.len();

        match linspace_range.contains(&self.start) && linspace_range.contains(&self.end) {
            true => Ok(self),
            false => Err(Error::out_of_bounds()),
        }
    }
}

impl TryFromIndexRange for IndexRange {
    fn try_from_index_range(range: IndexRange, spectrum: &Spectrum) -> Result<Self> {
        range.try_into_index_range(spectrum)
    }
}

try_index_range_impls!(FrequencyRange, freq_to_fractional, index_to_freq);
try_index_range_impls!(ChemicalShiftRange, shift_to_fractional, index_to_shift);
try_index_range_impls!(RelativeRange, relative_to_fractional, index_to_relative);

impl<T> SpectralRange<T>
where
    T: Copy + PartialOrd,
{
    /// Returns an ordered version of the range, where `start` is less than or
    /// equal to `end`.
    pub fn ordered(&self) -> Self {
        if self.start <= self.end {
            *self
        } else {
            Self {
                start: self.end,
                end: self.start,
            }
        }
    }
}
