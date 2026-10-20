//! Functions for computing various metrics to evaluate deconvolution quality.

use num_traits::Float;
use std::ops::Range;
use std::sync::OnceLock;
use zeenmr_peakshape::PeakShape;
use zeenmr_peakshape::batch_superposition::SuperpositionKernel;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::dimension::DimIndex;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use zeenmr_peakshape::batch_superposition::ParSuperpositionKernel;

/// Bundled fitting context.
pub struct FitContext<'s, 'p, 'i, T, P, K> {
    /// Base spectrum.
    pub spectrum: SpectrumView1D<'s, T, T>,
    /// Fitted peak shapes.
    pub peak_shapes: &'p [P],
    /// Offset of the grid used to fit the peak shapes.
    ///
    /// Normally, this is the start bound of the chemical shift range of the
    /// spectrum. However, some models use a different grid to perform the
    /// fitting.
    pub grid_offset: T,
    /// Offset of the grid used to fit the peak shapes.
    ///
    /// Normally, this is the chemical shift step size of the spectrum. However,
    /// some models use a different grid to perform the fitting.
    pub grid_step: T,
    /// Superposition kernel.
    pub kernel: K,
    /// Regions that were ignored during deconvolution.
    pub ignore: &'i [Range<usize>],
}

/// Useful intermediate results of metrics.
///
/// A lot of metrics will need roughly the same few building blocks, like the
/// superposition or residuals.
// Feel free to add pretty much anything here.
#[derive(Clone, PartialEq, Debug)]
pub struct IntermediateCache<T> {
    /// Ranges used for computations.
    included_bounds: OnceLock<Vec<Range<usize>>>,
    /// Concatenated intensities within the included bounds.
    intensities: OnceLock<Vec<T>>,
    /// Concatenated superposition within the included bounds.
    superposition: OnceLock<Vec<T>>,
    /// Concatenated residuals within the included bounds.
    residuals: OnceLock<Vec<T>>,
}

impl<T> Default for IntermediateCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> IntermediateCache<T> {
    /// Creates a new intermediate results cache.
    pub fn new() -> Self {
        Self {
            included_bounds: OnceLock::new(),
            intensities: OnceLock::new(),
            superposition: OnceLock::new(),
            residuals: OnceLock::new(),
        }
    }
}

impl<T> IntermediateCache<T>
where
    T: Float,
{
    /// Returns the ranges of indices used for metric computation.
    pub fn included_bounds<P, K>(&self, context: &FitContext<T, P, K>) -> &[Range<usize>]
    where
        P: PeakShape<T>,
    {
        self.included_bounds.get_or_init(|| {
            let len = context.spectrum.len();
            let (min, max) = context
                .peak_shapes
                .iter()
                .map(|p| (p.center(), p.full_width()))
                .fold(
                    (T::infinity(), T::neg_infinity()),
                    |acc, (center, width)| {
                        ((center - width).min(acc.0), (center + width).max(acc.1))
                    },
                );
            let min_frac_index = (min - context.grid_offset) / context.grid_step;
            let max_frac_index = (max - context.grid_offset) / context.grid_step;
            let start = min_frac_index
                .min(max_frac_index)
                .floor()
                .to_usize()
                .unwrap_or(0);
            let end = min_frac_index
                .max(max_frac_index)
                .ceil()
                .to_usize()
                .unwrap_or(len)
                .min(len);
            let signal = start..end;
            let iter = std::iter::once(signal.start)
                .chain(
                    context
                        .ignore
                        .iter()
                        .map(|r| r.start.max(signal.start)..r.end.min(signal.end))
                        .filter(|r| r.start < r.end)
                        .flat_map(|r| [r.start, r.end]),
                )
                .chain(std::iter::once(signal.end));

            iter.clone()
                .step_by(2)
                .zip(iter.skip(1).step_by(2))
                .map(|(start, end)| start..end)
                .collect()
        })
    }

    /// Returns the concatenated intensities within the included bounds.
    pub fn intensities<P, K>(&self, context: &FitContext<T, P, K>) -> &[T]
    where
        P: PeakShape<T>,
    {
        self.intensities.get_or_init(|| {
            let bounds = self.included_bounds(context);
            let len = bounds.iter().map(|range| range.len()).sum();
            let mut intensities = Vec::<T>::with_capacity(len);
            for range in bounds {
                intensities.extend(
                    context
                        .spectrum
                        .intensities()
                        .cropped(DimIndex(0), range.clone())
                        .expect("1D spectrum always has a first dimension")
                        .elem()
                        .copied(),
                );
            }

            intensities
        })
    }

    /// Returns the concatenated superposition within the included bounds.
    pub fn superposition<P, K>(&self, context: &FitContext<T, P, K>) -> &[T]
    where
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        self.superposition.get_or_init(|| {
            let bounds = self.included_bounds(context);
            let len = bounds.iter().map(|range| range.len()).sum();
            let mut grid = Vec::<T>::with_capacity(
                bounds
                    .iter()
                    .map(|range| range.len())
                    .max()
                    .unwrap_or(0),
            );
            let mut superposition = vec![T::zero(); len];
            let mut filled = 0;
            let index_to_grid = |index| {
                context.grid_offset
                    + context.grid_step
                        * T::from(index).expect("conversion from usize to T must never fail")
            };
            for range in bounds {
                let end = filled + range.len();
                grid.extend(range.clone().map(index_to_grid));
                context.kernel.accumulate(
                    context.peak_shapes,
                    &grid,
                    &mut superposition[filled..end],
                );
                filled = end;
                grid.clear();
            }

            superposition
        })
    }

    /// Returns the concatenated residuals within the included bounds.
    pub fn residuals<P, K>(&self, context: &FitContext<T, P, K>) -> &[T]
    where
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        self.residuals.get_or_init(|| {
            let bounds = self.included_bounds(context);
            let superposition = self.superposition(context);
            let mut residuals = self
                .intensities(context)
                .iter()
                .map(|raw| -(*raw))
                .collect::<Vec<T>>();
            let mut filled = 0;
            for range in bounds {
                let end = filled + range.len();
                residuals
                    .iter_mut()
                    .zip(superposition[filled..end].iter())
                    .for_each(|(raw, &sup)| {
                        *raw = *raw - sup;
                    });
                filled = residuals.len();
            }

            residuals
        })
    }
}

#[cfg(feature = "rayon")]
impl<T> IntermediateCache<T>
where
    T: Float + Send + Sync,
{
    /// Returns the concatenated superposition within the included bounds.
    pub fn par_superposition<P, K>(&self, context: &FitContext<T, P, K>) -> &[T]
    where
        P: PeakShape<T>,
        K: ParSuperpositionKernel<T, P>,
    {
        self.superposition.get_or_init(|| {
            let bounds = self.included_bounds(context);
            let index_to_grid = |index| {
                context.grid_offset
                    + context.grid_step
                        * T::from(index).expect("conversion from usize to T must never fail")
            };
            let grid = bounds
                .iter()
                .flat_map(|range| range.clone().map(index_to_grid))
                .collect::<Vec<T>>();

            context
                .kernel
                .par_superposition(context.peak_shapes, &grid)
        })
    }

    /// Returns the concatenated residuals within the included bounds.
    pub fn par_residuals<P, K>(&self, context: &FitContext<T, P, K>) -> &[T]
    where
        P: PeakShape<T>,
        K: ParSuperpositionKernel<T, P>,
    {
        self.residuals.get_or_init(|| {
            let intensities = self.intensities(context);
            let superposition = self.par_superposition(context);

            intensities
                .par_iter()
                .zip(superposition.par_iter())
                .map(|(&raw, &sup)| raw - sup)
                .collect()
        })
    }
}

/// Evaluation metric for deconvolution quality.
pub trait Metric {
    /// Returns the value of the metric for fitted spectrum.
    fn evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float,
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>;
}

/// Evaluation metric for deconvolution quality that can be computed in
/// parallel.
#[cfg(feature = "rayon")]
pub trait ParMetric {
    /// Returns the value of the metric for fitted spectrum.
    fn par_evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float + Send + Sync,
        P: PeakShape<T> + Send + Sync,
        K: ParSuperpositionKernel<T, P> + Send + Sync;
}

/// Mean squared error.
#[derive(Copy, Clone, Debug)]
pub struct MeanSquaredError;

impl Metric for MeanSquaredError {
    fn evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float,
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        let residuals = cache.residuals(context);
        let len = T::from(residuals.len()).expect("conversion from usize to T must never fail");

        residuals
            .iter()
            .fold(T::zero(), |acc, res| acc + res.powi(2))
            / len
    }
}

#[cfg(feature = "rayon")]
impl ParMetric for MeanSquaredError {
    fn par_evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float + Send + Sync,
        P: PeakShape<T> + Send + Sync,
        K: ParSuperpositionKernel<T, P> + Send + Sync,
    {
        let residuals = cache.par_residuals(context);
        let len = T::from(residuals.len()).expect("conversion from usize to T must never fail");

        residuals
            .par_iter()
            .fold(|| T::zero(), |acc, res| acc + res.powi(2))
            .reduce(|| T::zero(), |a, b| a + b)
            / len
    }
}

/// Root mean squared error.
#[derive(Copy, Clone, Debug)]
pub struct RootMeanSquaredError;

impl Metric for RootMeanSquaredError {
    fn evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float,
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        MeanSquaredError
            .evaluate_fit(context, cache)
            .sqrt()
    }
}

#[cfg(feature = "rayon")]
impl ParMetric for RootMeanSquaredError {
    fn par_evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float + Send + Sync,
        P: PeakShape<T> + Send + Sync,
        K: ParSuperpositionKernel<T, P> + Send + Sync,
    {
        MeanSquaredError
            .par_evaluate_fit(context, cache)
            .sqrt()
    }
}

/// Mean absolute error.
#[derive(Copy, Clone, Debug)]
pub struct MeanAbsoluteError;

impl Metric for MeanAbsoluteError {
    fn evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float,
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        let residuals = cache.residuals(context);
        let len = T::from(residuals.len()).expect("conversion from usize to T must never fail");

        residuals
            .iter()
            .fold(T::zero(), |acc, res| acc + res.abs())
            / len
    }
}

#[cfg(feature = "rayon")]
impl ParMetric for MeanAbsoluteError {
    fn par_evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float + Send + Sync,
        P: PeakShape<T> + Send + Sync,
        K: ParSuperpositionKernel<T, P> + Send + Sync,
    {
        let residuals = cache.par_residuals(context);
        let len = T::from(residuals.len()).expect("conversion from usize to T must never fail");

        residuals
            .par_iter()
            .fold(|| T::zero(), |acc, res| acc + res.abs())
            .reduce(|| T::zero(), |a, b| a + b)
            / len
    }
}

/// Coefficient of determination.
pub struct CoefficientOfDetermination;

impl Metric for CoefficientOfDetermination {
    fn evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float,
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        let intensities = cache.intensities(context);
        let mean_intensity = intensities
            .iter()
            .fold(T::zero(), |acc, &raw| acc + raw)
            / T::from(intensities.len()).expect("conversion from usize to T must never fail");
        let total_sum_of_squares = intensities
            .iter()
            .fold(T::zero(), |acc, &raw| acc + (raw - mean_intensity).powi(2));
        let sum_of_residual_squares = cache
            .residuals(context)
            .iter()
            .fold(T::zero(), |acc, res| acc + res.powi(2));

        T::one() - sum_of_residual_squares / total_sum_of_squares
    }
}

#[cfg(feature = "rayon")]
impl ParMetric for CoefficientOfDetermination {
    fn par_evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float + Send + Sync,
        P: PeakShape<T> + Send + Sync,
        K: ParSuperpositionKernel<T, P> + Send + Sync,
    {
        let intensities = cache.intensities(context);
        let mean = intensities
            .par_iter()
            .fold(|| T::zero(), |acc, &raw| acc + raw)
            .reduce(|| T::zero(), |a, b| a + b)
            / T::from(intensities.len()).expect("conversion from usize to T must never fail");
        let total_sum_of_squares = intensities
            .par_iter()
            .fold(|| T::zero(), |acc, &raw| acc + (raw - mean).powi(2))
            .reduce(|| T::zero(), |a, b| a + b);
        let sum_of_residual_squares = cache
            .par_residuals(context)
            .par_iter()
            .fold(|| T::zero(), |acc, res| acc + res.powi(2))
            .reduce(|| T::zero(), |a, b| a + b);

        T::one() - sum_of_residual_squares / total_sum_of_squares
    }
}
