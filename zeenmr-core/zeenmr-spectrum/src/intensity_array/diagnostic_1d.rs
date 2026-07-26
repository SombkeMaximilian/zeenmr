//! Intensity diagnostics for detecting 1D spectral properties.

use crate::error::{Error, Result};
use crate::intensity_array::Storage;
use num_complex::Complex;
use num_traits::Float;
use std::ops::Range;

/// Intensities are magnitude transformed, i.e., `s = sqrt(r^2 + i^2)`.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Magnitude;

/// Intensities are the real or imaginary channel.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct SingleChannel;

/// Intensities are the real and imaginary channel as complex numbers.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct DualChannel;

/// Trait for validating intensities.
pub trait ValidateIntensities<T> {
    /// Validates the intensity array.
    ///
    /// # Errors
    ///
    /// At minimum, must return an error if the array is empty or contains any
    /// non-finite values in the case of a float type.
    fn validate<A>(array: A) -> Result<()>
    where
        A: Storage<Elem = T>;
}

impl<T> ValidateIntensities<T> for Magnitude
where
    T: Float,
{
    fn validate<A>(array: A) -> Result<()>
    where
        A: Storage<Elem = T>,
    {
        let array = array.as_slice();

        if array.is_empty() {
            Err(Error::empty_array())
        } else if array.iter().any(|x| !x.is_finite()) {
            Err(Error::non_finite_float())
        } else if array.iter().any(|x| x.is_sign_negative()) {
            Err(Error::incorrect_intensity())
        } else {
            Ok(())
        }
    }
}

impl<T> ValidateIntensities<T> for SingleChannel
where
    T: Float,
{
    fn validate<A>(array: A) -> Result<()>
    where
        A: Storage<Elem = T>,
    {
        let array = array.as_slice();

        if array.is_empty() {
            Err(Error::empty_array())
        } else if array.iter().any(|x| !x.is_finite()) {
            Err(Error::non_finite_float())
        } else {
            Ok(())
        }
    }
}

impl<T> ValidateIntensities<Complex<T>> for DualChannel
where
    T: Float,
{
    fn validate<A>(array: A) -> Result<()>
    where
        A: Storage<Elem = Complex<T>>,
    {
        let array = array.as_slice();

        if array.is_empty() {
            Err(Error::empty_array())
        } else if array
            .iter()
            .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            Err(Error::non_finite_float())
        } else {
            Ok(())
        }
    }
}

/// Trait for finding the range within which signals are found.
pub trait FindSignalRange<T, K> {
    /// Computes the signal range.
    ///
    /// May perform expensive computations.
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Storage<Elem = T>;
}

impl<T, K> FindSignalRange<T, K> for Range<usize> {
    /// Returns a copy of `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if either of the range bounds is not within bounds of
    /// the array, or if `self` is empty.
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Storage<Elem = T>,
    {
        if self.start > array.as_slice().len() || self.end > array.as_slice().len() {
            Err(Error::out_of_bounds())
        } else if self.is_empty() {
            Err(Error::no_signal())
        } else {
            Ok(self.clone())
        }
    }
}

/// Cumulative sum test.
///
/// This test statistic relies on there being noise at the edges of the array.
/// If there is no or extremely little noise, division by zero will occur.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct CumulativeSum<T> {
    /// Flagging limit above which to mark a position as signal start or end.
    limit: T,
    /// Penalty for each term.
    penalty: T,
    /// Edges to use for mean and standard deviation estimate.
    edges: f64,
    /// Padding around the start and end points to avoid cutting off signals.
    padding: usize,
}

impl<T> FindSignalRange<T, Magnitude> for CumulativeSum<T>
where
    T: Float,
{
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Storage<Elem = T>,
    {
        let array = array
            .as_slice()
            .iter()
            .copied()
            .map(T::sqrt)
            .collect::<Vec<T>>();
        let (mean, std) = self.edge_stats(&array)?;
        if std.abs()
            <= T::from(100_u8).expect("conversion from u8 to T must never fail") * T::epsilon()
        {
            return Err(Error::divide_by_zero());
        }
        let signal_range = self.one_sided_scan(&array, mean, std);

        if signal_range.start <= signal_range.end {
            Ok(signal_range)
        } else {
            Err(Error::no_signal())
        }
    }
}

impl<T> FindSignalRange<T, SingleChannel> for CumulativeSum<T>
where
    T: Float,
{
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Storage<Elem = T>,
    {
        let array = array.as_slice();
        let (mean, std) = self.edge_stats(array)?;
        if std.abs()
            <= T::from(100_u8).expect("conversion from u8 to T must never fail") * T::epsilon()
        {
            return Err(Error::divide_by_zero());
        }
        let signal_range = self.two_sided_scan(array, mean, std);

        if signal_range.start <= signal_range.end {
            Ok(signal_range)
        } else {
            Err(Error::no_signal())
        }
    }
}

impl<T> FindSignalRange<Complex<T>, DualChannel> for CumulativeSum<T>
where
    T: Float,
{
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Storage<Elem = Complex<T>>,
    {
        let array = array
            .as_slice()
            .iter()
            .map(|c| c.norm())
            .collect::<Vec<T>>();

        FindSignalRange::<T, Magnitude>::find_signal_range(self, array)
    }
}

impl<T> CumulativeSum<T>
where
    T: Float,
{
    /// Creates a new `CumulativeSum`.
    pub fn new(limit: T, penalty: T) -> Self {
        Self {
            limit,
            penalty,
            edges: 0.1,
            padding: 20,
        }
    }

    /// Sets the edge width as a fraction of total width.
    ///
    /// Returns `None` if the width is not in `[0.05, 0.30]`. Shorter edges
    /// result in very poor estimates, and we assume that at least 40% of an
    /// array contains some kind of signal.
    ///
    /// Note that the default value was chosen empirically based on performance
    /// in a variety of spectra and modifying this will usually not change much.
    pub fn edge_width(mut self, width: f64) -> Option<Self> {
        if !(0.05..0.3).contains(&width) {
            return None;
        }

        self.edges = width;

        Some(self)
    }

    /// Sets the padding from the detected position.
    ///
    /// Setting this to be too large may deteriorate downstream performance. It
    /// is recommended to choose values in `[10, 100]` but never more than 1% of
    /// total length.
    ///
    /// Note that the default value was chosen empirically based on performance
    /// in a variety of spectra and modifying this will usually not change much.
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;

        self
    }

    /// Performs a one-sided scan which only checks for positive deviance.
    fn one_sided_scan(&self, array: &[T], mean: T, std: T) -> Range<usize> {
        let mut start = 0;
        let mut flagging = T::zero();
        for (pos, std_int) in array
            .iter()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging = (flagging + std_int - self.penalty).max(T::zero());

            if flagging >= self.limit {
                start = pos.saturating_sub(self.padding);
                break;
            }
        }

        let mut end = array.len();
        let mut flagging = T::zero();
        for (pos, std_int) in array
            .iter()
            .rev()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging = (flagging + std_int - self.penalty).max(T::zero());

            if flagging >= self.limit {
                end -= pos.saturating_sub(self.padding);
                break;
            }
        }

        start..end
    }

    /// Performs a two-sided scan which checks for positive and negative
    /// deviance.
    fn two_sided_scan(&self, array: &[T], mean: T, std: T) -> Range<usize> {
        let mut start = 0;
        let mut flagging_p = T::zero();
        let mut flagging_n = T::zero();
        for (pos, std_int) in array
            .iter()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging_p = (flagging_p + std_int - self.penalty).max(T::zero());
            flagging_n = (flagging_n - std_int - self.penalty).max(T::zero());

            if flagging_p >= self.limit || flagging_n >= self.limit {
                start = pos.saturating_sub(self.padding);
                break;
            }
        }

        let mut end = array.len();
        let mut flagging_p = T::zero();
        let mut flagging_n = T::zero();
        for (pos, std_int) in array
            .iter()
            .rev()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging_p = (flagging_p + std_int - self.penalty).max(T::zero());
            flagging_n = (flagging_n - std_int - self.penalty).max(T::zero());

            if flagging_p >= self.limit || flagging_n >= self.limit {
                end -= pos.saturating_sub(self.padding);
                break;
            }
        }

        start..end
    }

    /// Returns combined mean and std of the edges.
    ///
    /// # Error
    ///
    /// Returns an error if the length of the edges is zero.
    fn edge_stats(&self, array: &[T]) -> Result<(T, T)> {
        let edge_width = (array.len() as f64 * self.edges) as usize;

        if edge_width == 0 {
            return Err(Error::divide_by_zero());
        }

        let right_edge = array.len() - edge_width;
        let num = T::from(2 * edge_width).expect("conversion from usize to T must never fail");

        let left_sum = array[..edge_width]
            .iter()
            .fold(T::zero(), |acc, x| acc + *x);
        let right_sum = array[right_edge..]
            .iter()
            .fold(T::zero(), |acc, x| acc + *x);
        let edge_mean = (left_sum + right_sum) / num;

        let left_dev = array[..edge_width]
            .iter()
            .copied()
            .map(|x| (x - edge_mean).powi(2))
            .fold(T::zero(), |acc, x| acc + x);
        let right_dev = array[right_edge..]
            .iter()
            .copied()
            .map(|x| (x - edge_mean).powi(2))
            .fold(T::zero(), |acc, x| acc + x);
        let edge_std = ((left_dev + right_dev) / (num - T::one())).sqrt();

        Ok((edge_mean, edge_std))
    }
}
