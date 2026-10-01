//! Types and traits for finding ranges of data points with real signals.

use num_traits::Float;
use std::ops::Range;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::axis::range::{
    FiniteBounds, FrequencyRange, RelativeRange, ShiftRange, SpectralRange,
};
use zeenmr_spectrum::dimension::{DimIndex, StaticDim};
use zeenmr_spectrum::intensity_array::ArrayView;

/// Trait for finding the range within which signals are found.
pub trait SignalBoundaries<T> {
    /// Returns an estimate for the index range within which real signals are
    /// contained.
    fn signal_boundaries(&self, spectrum: SpectrumView1D<T, T>) -> Range<usize>;
}

/// Fallback helper if detecting the signal boundaries fails.
///
/// Returns the equivalent of the relative range `[0.2, 0.8]`.
pub fn fallback<T>(array: ArrayView<T, StaticDim<usize, 1>>) -> Range<usize> {
    let len = array.len() as f64;

    ((0.2 * len) as usize)..((0.8 * len) as usize)
}

impl<T> SignalBoundaries<T> for Range<usize> {
    fn signal_boundaries(&self, spectrum: SpectrumView1D<T, T>) -> Range<usize> {
        let upper = spectrum.intensities().len();
        let start = self
            .start
            .min(self.end)
            .min(upper.saturating_sub(1));
        let end = self.start.max(self.end).min(upper);

        debug_assert!(start <= end);

        start..end
    }
}

impl<R, T> SignalBoundaries<T> for RelativeRange<R>
where
    R: Float,
{
    fn signal_boundaries(&self, spectrum: SpectrumView1D<T, T>) -> Range<usize> {
        let len = R::from(spectrum.intensities().len())
            .expect("conversion from usize to T must never fail");
        let start = (self.lower() * len)
            .ceil()
            .to_usize()
            .expect("multiplying by relative bounds never increases the value");
        let end = (self.upper() * len)
            .floor()
            .to_usize()
            .expect("multiplying by relative bounds never increases the value");

        debug_assert!(start <= end);

        start..end
    }
}

impl<T> SignalBoundaries<T> for FrequencyRange<T>
where
    T: Float,
{
    fn signal_boundaries(&self, spectrum: SpectrumView1D<T, T>) -> Range<usize> {
        let len = T::from(spectrum.intensities().len())
            .expect("conversion from usize to T must never fail");
        let axis = spectrum
            .axis(DimIndex(0))
            .expect("1D spectrum always has a first dimension");
        let range = axis.freq_range().normalized();
        let start = axis
            .freq_to_rel(self.lower().clamp(range.start(), range.end()))
            .and_then(|rel| (rel * len).ceil().to_usize())
            .expect("clamped inside the axis range");
        let end = axis
            .freq_to_rel(self.upper().clamp(range.start(), range.end()))
            .and_then(|rel| (rel * len).floor().to_usize())
            .expect("clamped inside the axis range");

        debug_assert!(start <= end);

        start..end
    }
}

impl<T> SignalBoundaries<T> for ShiftRange<T>
where
    T: Float,
{
    fn signal_boundaries(&self, spectrum: SpectrumView1D<T, T>) -> Range<usize> {
        let len = T::from(spectrum.intensities().len())
            .expect("conversion from usize to T must never fail");
        let axis = spectrum
            .axis(DimIndex(0))
            .expect("1D spectrum always has a first dimension");
        let range = axis.shift_range().normalized();
        let start = axis
            .shift_to_rel(self.lower().clamp(range.start(), range.end()))
            .and_then(|rel| (rel * len).ceil().to_usize())
            .expect("clamped inside the axis range");
        let end = axis
            .shift_to_rel(self.upper().clamp(range.start(), range.end()))
            .and_then(|rel| (rel * len).floor().to_usize())
            .expect("clamped inside the axis range");

        debug_assert!(start <= end);

        start..end
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

impl<T> SignalBoundaries<T> for CumulativeSum<T>
where
    T: Float,
{
    fn signal_boundaries(&self, spectrum: SpectrumView1D<T, T>) -> Range<usize> {
        let intensities = spectrum.intensities();
        let Some((mean, std)) = self.edge_stats(intensities.clone()) else {
            return fallback(spectrum.intensities());
        };
        let signal_range = self.two_sided_scan(intensities, mean, std);

        if !signal_range.is_empty() {
            signal_range
        } else {
            fallback(spectrum.intensities())
        }
    }
}

impl<T> CumulativeSum<T>
where
    T: Float,
{
    /// Creates a new cumulative sum test.
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
    pub fn with_edge_width(mut self, width: f64) -> Option<Self> {
        if !(0.05..0.30).contains(&width) {
            return None;
        }

        self.edges = width;

        Some(self)
    }

    /// Sets the padding from the detected position.
    ///
    /// Setting this to be too large may deteriorate downstream performance. It
    /// is recommended to choose values in `[10, 100]` but never more than 1% of
    /// expected total length.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;

        self
    }

    /// Performs a two-sided scan which checks for positive and negative
    /// deviance.
    fn two_sided_scan(
        &self,
        array: ArrayView<T, StaticDim<usize, 1>>,
        mean: T,
        std: T,
    ) -> Range<usize> {
        let mut start = 0;
        let mut flagging_p = T::zero();
        let mut flagging_n = T::zero();
        for (pos, std_int) in array
            .elem()
            .map(|&x| (x - mean) / std)
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
            .elem()
            .rev()
            .map(|&x| (x - mean) / std)
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

    /// Returns mean and standard deviation of the edges, or `None` if the
    /// length of the edges is zero.
    fn edge_stats(&self, array: ArrayView<T, StaticDim<usize, 1>>) -> Option<(T, T)> {
        let edge_width = (array.len() as f64 * self.edges) as usize;

        if edge_width == 0 {
            return None;
        }

        let right_edge = array.len() - edge_width;
        let num = T::from(2 * edge_width).expect("conversion from usize to T must never fail");

        let left_sum = array
            .elem()
            .take(edge_width)
            .fold(T::zero(), |acc, x| acc + *x);
        let right_sum = array
            .elem()
            .skip(right_edge)
            .fold(T::zero(), |acc, x| acc + *x);
        let edge_mean = (left_sum + right_sum) / num;

        let left_dev = array
            .elem()
            .take(edge_width)
            .fold(T::zero(), |acc, &x| acc + (x - edge_mean).powi(2));
        let right_dev = array
            .elem()
            .skip(right_edge)
            .fold(T::zero(), |acc, &x| acc + (x - edge_mean).powi(2));
        let edge_std = ((left_dev + right_dev) / (num - T::one())).sqrt();

        Some((edge_mean, edge_std))
    }
}
