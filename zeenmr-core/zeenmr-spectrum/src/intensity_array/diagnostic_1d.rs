//! Intensity diagnostics for detecting 1D spectral properties.

use crate::intensity_array::Array1D;
use crate::intensity_array::error::{Error, Result};
use num_traits::{Float, FromPrimitive};
use std::iter;
use std::marker::PhantomData;
use std::ops::Range;

/// Intensities are magnitude transformed, i.e., `s = sqrt(r^2 + i^2)`.
#[derive(Copy, Clone, Debug)]
pub struct Magnitude;

/// Intensities are the phase-corrected real channel.
#[derive(Copy, Clone, Debug)]
pub struct RealChannel;

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
        A: Array1D<Elem = T>;
}

impl<F> ValidateIntensities<F> for Magnitude
where
    F: Float,
{
    fn validate<A>(array: A) -> Result<()>
    where
        A: Array1D<Elem=F>,
    {
        let array = array.as_ref();

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

impl<F> ValidateIntensities<F> for RealChannel
where
    F: Float,
{
    fn validate<A>(array: A) -> Result<()>
    where
        A: Array1D<Elem=F>,
    {
        let array = array.as_ref();

        if array.is_empty() {
            Err(Error::empty_array())
        } else if array.iter().any(|x| !x.is_finite()) {
            Err(Error::non_finite_float())
        } else {
            Ok(())
        }
    }
}

/// Trait for finding the range within which signals are found.
pub trait FindSignalRange<T> {
    /// Computes the signal range.
    ///
    /// May perform expensive computations.
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Array1D<Elem = T>;
}

impl<T> FindSignalRange<T> for Range<usize> {
    /// Returns a copy of `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if either of the range bounds is not within bounds of
    /// the array, or if `self` is empty.
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Array1D<Elem = T>,
    {
        if self.start >= array.as_ref().len() || self.end >= array.as_ref().len() {
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
pub struct CumulativeSum<F, K> {
    /// Flagging limit above which to mark a position as signal start or end.
    limit: F,
    /// Penalty for each term.
    penalty: F,
    /// Edges to use for mean and standard deviation estimate.
    edges: f64,
    /// Padding around the start and end points to avoid cutting off signals.
    padding: usize,
    /// Real channel or magnitude.
    intensity_kind: PhantomData<K>,
}

impl<F> FindSignalRange<F> for CumulativeSum<F, Magnitude>
where
    F: Copy + iter::Sum<F> + Float + FromPrimitive,
{
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Array1D<Elem = F>,
    {
        let array = array
            .as_ref()
            .iter()
            .copied()
            .map(F::sqrt)
            .collect::<Vec<F>>();
        let (mean, std) = self.edge_stats(&array)?;
        if std.abs()
            <= F::from_u8(100).expect("conversion from u8 to F must never fail") * F::epsilon()
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

impl<F> FindSignalRange<F> for CumulativeSum<F, RealChannel>
where
    F: Copy + iter::Sum<F> + Float + FromPrimitive,
{
    fn find_signal_range<A>(&self, array: A) -> Result<Range<usize>>
    where
        A: Array1D<Elem = F>,
    {
        let array = array.as_ref();
        let (mean, std) = self.edge_stats(array)?;
        if std.abs()
            <= F::from_u8(100).expect("conversion from u8 to F must never fail") * F::epsilon()
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

impl<F, K> CumulativeSum<F, K>
where
    F: Copy + iter::Sum<F> + Float + FromPrimitive,
{
    /// Creates a new `CumulativeSum`.
    pub fn new(limit: F, penalty: F) -> Self {
        Self {
            limit,
            penalty,
            edges: 0.1,
            padding: 20,
            intensity_kind: PhantomData,
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

    /// Sets the intensity kind from `M` to `N`.
    pub fn intensity_kind<N>(self) -> CumulativeSum<F, N> {
        CumulativeSum::<F, N> {
            limit: self.limit,
            penalty: self.penalty,
            edges: self.edges,
            padding: self.padding,
            intensity_kind: PhantomData,
        }
    }

    /// Performs a one-sided scan which only checks for positive deviance.
    fn one_sided_scan(&self, array: &[F], mean: F, std: F) -> Range<usize> {
        let mut start = 0;
        let mut flagging = F::zero();
        for (pos, std_int) in array
            .iter()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging = (flagging + std_int - self.penalty).max(F::zero());

            if flagging >= self.limit {
                start = pos.saturating_sub(self.padding);
                break;
            }
        }

        let mut end = array.len();
        let mut flagging = F::zero();
        for (pos, std_int) in array
            .iter()
            .rev()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging = (flagging + std_int - self.penalty).max(F::zero());

            if flagging >= self.limit {
                end = end - pos.saturating_sub(self.padding);
                break;
            }
        }

        start..end
    }

    /// Performs a two-sided scan which checks for positive and negative
    /// deviance.
    fn two_sided_scan(&self, array: &[F], mean: F, std: F) -> Range<usize> {
        let mut start = 0;
        let mut flagging_p = F::zero();
        let mut flagging_n = F::zero();
        for (pos, std_int) in array
            .iter()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging_p = (flagging_p + std_int - self.penalty).max(F::zero());
            flagging_n = (flagging_n - std_int - self.penalty).max(F::zero());

            if flagging_p >= self.limit || flagging_n >= self.limit {
                start = pos.saturating_sub(self.padding);
                break;
            }
        }

        let mut end = array.len();
        let mut flagging_p = F::zero();
        let mut flagging_n = F::zero();
        for (pos, std_int) in array
            .iter()
            .rev()
            .map(|x| (*x - mean) / std)
            .enumerate()
        {
            flagging_p = (flagging_p + std_int - self.penalty).max(F::zero());
            flagging_n = (flagging_n - std_int - self.penalty).max(F::zero());

            if flagging_p >= self.limit || flagging_n >= self.limit {
                end = end - pos.saturating_sub(self.padding);
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
    fn edge_stats(&self, array: &[F]) -> Result<(F, F)> {
        let edge_width = (array.len() as f64 * self.edges) as usize;
        let right_edge = array.len() - edge_width;
        let num =
            F::from_usize(2 * edge_width).expect("conversion from usize to F must never fail");
        if num <= F::from_u8(100).expect("conversion from u8 to F must never fail") * F::epsilon() {
            return Err(Error::divide_by_zero());
        }

        let left_sum = array[..edge_width].iter().copied().sum::<F>();
        let right_sum = array[right_edge..].iter().copied().sum::<F>();
        let edge_mean = (left_sum + right_sum) / num;

        let left_dev = array[..edge_width]
            .iter()
            .copied()
            .map(|x| (x - edge_mean).powi(2))
            .sum::<F>();
        let right_dev = array[right_edge..]
            .iter()
            .copied()
            .map(|x| (x - edge_mean).powi(2))
            .sum::<F>();
        let edge_std = ((left_dev + right_dev) / (num - F::one())).sqrt();

        Ok((edge_mean, edge_std))
    }
}
