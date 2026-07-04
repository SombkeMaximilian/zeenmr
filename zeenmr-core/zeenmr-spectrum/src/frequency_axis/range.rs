//! Types for representing ranges in spectral axes.

use num_traits::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for ranges that enforce finite bounds at construction.
pub trait FiniteBounds<T> {
    /// Returns the `start` bound.
    fn start(&self) -> T;

    /// Returns the `end` bound.
    fn end(&self) -> T;
}

/// Trait of ranges of values in spectral axes.
pub trait SpectralRange<T> {
    /// Returns `true` if `start < end`.
    fn is_ascending(&self) -> bool;

    /// Returns `true` if `start > end`.
    fn is_descending(&self) -> bool;

    /// Returns the greater of `start` and `end`.
    fn max(&self) -> T;

    /// Returns the lesser of `start` and `end`.
    fn min(&self) -> T;

    /// Returns `true` if the value lies within the range.
    ///
    /// Unlike the standard library's range types, it does not check for
    /// `v ∈ [start, end]` but `v ∈ [min(start, end), max(start, end)]`.
    fn contains(&self, value: T) -> bool;

    /// Returns the width of the range.
    ///
    /// The result is always a positive value.
    fn width(&self) -> T;

    /// Returns the signed width of the range.
    ///
    /// The result is negative if `end < start`, and positive otherwise.
    fn signed_width(&self) -> T;

    /// Returns the center of the range.
    fn center(&self) -> T;
}

impl<T, R> SpectralRange<T> for R
where
    T: Float,
    R: FiniteBounds<T>,
{
    fn is_ascending(&self) -> bool {
        self.start() < self.end()
    }

    fn is_descending(&self) -> bool {
        self.start() > self.end()
    }

    fn max(&self) -> T {
        self.start().max(self.end())
    }

    fn min(&self) -> T {
        self.start().min(self.end())
    }

    fn contains(&self, value: T) -> bool {
        self.min() <= value && value <= self.max()
    }

    fn width(&self) -> T {
        self.signed_width().abs()
    }

    fn signed_width(&self) -> T {
        self.end() - self.start()
    }

    fn center(&self) -> T {
        (self.start() + self.end()) / (T::one() + T::one())
    }
}

/// Frequency range on a spectral axis.
///
/// # Invariants
///
/// The bounds are non-negative, finite, and not `NaN`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct FrequencyRange<T> {
    /// Start of the range (inclusive).
    start: T,
    /// End of the range (inclusive).
    end: T,
}

impl<T> FiniteBounds<T> for FrequencyRange<T>
where
    T: Copy,
{
    fn start(&self) -> T {
        self.start
    }

    fn end(&self) -> T {
        self.end
    }
}

impl<T> FrequencyRange<T>
where
    T: Float,
{
    /// Creates a new `FrequencyRange`.
    ///
    /// Returns `None` if either `start` or `end` is negative or non-finite.
    pub fn new(start: T, end: T) -> Option<Self> {
        if start.is_finite() && start >= T::zero() && end.is_finite() && end >= T::zero() {
            Some(Self { start, end })
        } else {
            None
        }
    }

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
}

/// Chemical shift range on a spectral axis.
///
/// # Invariants
///
/// The bounds are finite and not `NaN`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct ShiftRange<T> {
    /// Start of the range (inclusive).
    start: T,
    /// End of the range (inclusive).
    end: T,
}

impl<T> FiniteBounds<T> for ShiftRange<T>
where
    T: Copy,
{
    fn start(&self) -> T {
        self.start
    }

    fn end(&self) -> T {
        self.end
    }
}

impl<T> ShiftRange<T>
where
    T: Float,
{
    /// Creates a new `ShiftRange`.
    ///
    /// Returns `None` if either `start` or `end` is non-finite.
    pub fn new(start: T, end: T) -> Option<Self> {
        if start.is_finite() && end.is_finite() {
            Some(Self { start, end })
        } else {
            None
        }
    }

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
}

/// Relative range on a spectral axis.
///
/// The range bounds represent the offset from the start of a frequency axis in
/// terms of its total width.
///
/// # Invariants
///
/// The bounds are within `[0, 1]`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct RelativeRange<T> {
    /// Start of the range (inclusive).
    start: T,
    /// End of the range (inclusive).
    end: T,
}

impl<T> FiniteBounds<T> for RelativeRange<T>
where
    T: Copy,
{
    fn start(&self) -> T {
        self.start
    }

    fn end(&self) -> T {
        self.end
    }
}

impl<T> RelativeRange<T>
where
    T: Float,
{
    /// Creates a new `RelativeRange`.
    ///
    /// Returns `None` if `start, end ∉ [0, 1]`.
    pub fn new(start: T, end: T) -> Option<Self> {
        if (T::zero()..=T::one()).contains(&start) && (T::zero()..=T::one()).contains(&end) {
            Some(Self { start, end })
        } else {
            None
        }
    }

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
}
