//! Types for representing ranges in spectral axes.

use num_traits::Zero;
use std::ops::{Add, Mul, Sub};
use uom::si::f64::{Frequency, Ratio};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Index range on a spectral axis.
///
/// This is an alias of [`core::ops::Range`] for compatibility with index
/// slicing, which uses `[start, end)` unlike the other ranges in this module.
pub type IndexRange = core::ops::Range<usize>;

/// Frequency range on a spectral axis.
///
/// # Invariants
///
/// The bounds are non-negative, finite, and not `NaN`.
pub type FrequencyRange = Range<Frequency>;

impl FrequencyRange {
    /// Creates a new `FrequencyRange`.
    ///
    /// Returns `None` if either `start` or `end` is negative or non-finite.
    pub fn new(start: Frequency, end: Frequency) -> Option<Self> {
        if start.is_finite()
            && start >= Frequency::zero()
            && end.is_finite()
            && end >= Frequency::zero()
        {
            Some(Range { start, end })
        } else {
            None
        }
    }
}

/// Chemical shift range on a spectral axis.
///
/// # Invariants
///
/// The bounds are finite and not `NaN`.
pub type ShiftRange = Range<Ratio>;

impl ShiftRange {
    /// Creates a new `ShiftRange`.
    ///
    /// Returns `None` if either `start` or `end` is non-finite.
    pub fn new(start: Ratio, end: Ratio) -> Option<Self> {
        if start.is_finite() && end.is_finite() {
            Some(Range { start, end })
        } else {
            None
        }
    }
}

/// Relative range on a spectral axis.
///
/// The range bounds represent the offset from the start of a frequency axis in
/// terms of its total width.
///
/// # Invariants
///
/// The bounds are within `[0, 1]`.
pub type RelativeRange = Range<f64>;

impl RelativeRange {
    /// Creates a new `RelativeRange`.
    ///
    /// Returns `None` if `start, end ∉ [0, 1]`.
    pub fn new(start: f64, end: f64) -> Option<Self> {
        if (0.0..=1.0).contains(&start) && (0.0..=1.0).contains(&end) {
            Some(RelativeRange { start, end })
        } else {
            None
        }
    }
}

/// A generic range with a start and end value.
///
/// This range type is intended to be used for chemical shift, frequency, or
/// relative ranges. For index ranges (e.g., into slices), [`core::range`]
/// should be used.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs
///
/// When the `serde` feature is enabled, `Range` implements the [`Serialize`]
/// and [`Deserialize`] traits provided the type `T` also implements them. For
/// the type aliases mentioned above, this is the case.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Range<T> {
    /// Start of the range (inclusive).
    start: T,
    /// End of the range (inclusive).
    end: T,
}

impl<T> Range<T> {
    /// Returns a reference to the `start` bound.
    ///
    /// Prefer using [`Range::start`] if `T` is [`Copy`].
    pub fn start_ref(&self) -> &T {
        &self.start
    }

    /// Returns a reference to the `end` bound.
    ///
    /// Prefer using [`Range::end`] if `T` is [`Copy`].
    pub fn end_ref(&self) -> &T {
        &self.end
    }
}

impl<T> Range<T>
where
    T: Copy,
{
    /// Returns the `start` bound.
    pub fn start(&self) -> T {
        self.start
    }

    /// Returns the `end` bound.
    pub fn end(&self) -> T {
        self.end
    }
}

impl<T> Range<T>
where
    T: PartialOrd,
{
    /// Returns `true` if `start < end`.
    pub fn is_ascending(&self) -> bool {
        self.start < self.end
    }

    /// Returns `true` if `start > end`.
    pub fn is_descending(&self) -> bool {
        self.start > self.end
    }
}

impl<T> Range<T>
where
    T: Copy + PartialOrd,
{
    /// Returns an equivalent range with `start <= end`, swapping bounds if
    /// necessary.
    pub fn normalized(&self) -> Self {
        if self.start <= self.end {
            *self
        } else {
            Self {
                start: self.end,
                end: self.start,
            }
        }
    }

    /// Returns the greater of `start` and `end`.
    pub fn max(&self) -> T {
        if self.is_descending() {
            self.end
        } else {
            self.start
        }
    }

    /// Returns the lesser of `start` and `end`.
    pub fn min(&self) -> T {
        if self.is_ascending() {
            self.start
        } else {
            self.end
        }
    }

    /// Returns `true` if the value lies within the range.
    ///
    /// Unlike the standard library's range type, it does not check for
    /// `v ∈ [start, end]` but `v ∈ [min(start, end), max(start, end)]`.
    pub fn contains(&self, value: T) -> bool {
        self.min() <= value && value <= self.max()
    }
}

impl<T> Range<T>
where
    T: Copy + PartialOrd + Sub<Output = T>,
{
    /// Returns the width of the range.
    ///
    /// The result is always a positive value.
    pub fn width(&self) -> T {
        self.max() - self.min()
    }

    /// Returns the signed width of the range by computing `end - start`.
    pub fn signed_width(&self) -> T {
        self.end - self.start
    }
}

impl<T> Range<T>
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    /// Returns the center of the range.
    pub fn center(&self) -> T {
        (self.start + self.end) * 0.5
    }
}
