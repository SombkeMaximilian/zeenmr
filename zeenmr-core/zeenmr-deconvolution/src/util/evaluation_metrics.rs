//! Functions for computing various metrics to evaluate deconvolution quality.

use num_traits::Float;
use std::ops::Range;
use std::sync::OnceLock;
use zeenmr_peakshape::PeakShape;
use zeenmr_peakshape::batch_superposition::SuperpositionKernel;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::dimension::DimIndex;

/// Bundled fitting context.
pub struct FitContext<'s, 'p, T, P, K> {
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
    pub ignore: Vec<Range<usize>>,
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
            let &FitContext {
                ref spectrum,
                peak_shapes,
                grid_offset,
                grid_step,
                ref ignore,
                ..
            } = context;
            let len = spectrum.len();
            let (min, max) = peak_shapes
                .iter()
                .map(|p| (p.center(), p.full_width()))
                .fold(
                    (T::infinity(), T::neg_infinity()),
                    |acc, (center, width)| ((center - width).min(acc.0), (center + width).max(acc.1)),
                );
            let min_frac_index = (min - grid_offset) / grid_step;
            let max_frac_index = (max - grid_offset) / grid_step;
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
                    ignore
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

    /// Returns the concatenated superposition within the included bounds.
    pub fn superposition<P, K>(&self, context: &FitContext<T, P, K>) -> &[T]
    where
        P: PeakShape<T>,
        K: SuperpositionKernel<T, P>,
    {
        self.superposition.get_or_init(|| {
            let &FitContext {
                peak_shapes,
                grid_offset,
                grid_step,
                ref kernel,
                ..
            } = context;
            let bounds = self.included_bounds(context);
            let len = bounds
                .iter()
                .map(|range| range.len())
                .sum();
            let mut grid = Vec::<T>::with_capacity(bounds.iter().map(|range| range.len()).max().unwrap_or(0));
            let mut superposition = vec![T::zero(); len];
            let mut filled = 0;
            let index_to_grid = |index| {
                grid_offset + grid_step
                    * T::from(index).expect("conversion from usize to T must never fail")
            };
            for range in bounds {
                let end = filled + range.len();
                grid.extend(range.clone().map(index_to_grid));
                kernel.accumulate(peak_shapes, &grid, &mut superposition[filled..end]);
                filled = end;
                grid.truncate(0);
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
            let &FitContext {
                ref spectrum,
                ..
            } = context;
            let superposition = self.superposition(context);
            let mut residuals = Vec::<T>::with_capacity(superposition.len());
            let mut filled = 0;
            let bounds = self.included_bounds(context);
            for range in bounds {
                let end = filled + range.len();
                residuals.extend(
                    spectrum.intensities()
                        .cropped(DimIndex(0), range.clone())
                        .expect("1D spectrum always has a first dimension")
                        .elem()
                        .zip(superposition[filled..end].iter())
                        .map(|(&raw, &sup)| raw - sup)
                );
                filled = residuals.len();
            }

            residuals
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
        K: SuperpositionKernel<T, P>,;
}

/// Evaluation metric for deconvolution quality that can be computed in
/// parallel.
#[cfg(feature = "rayon")]
pub trait ParMetric {
    /// Returns the value of the metric for fitted spectrum.
    fn evaluate_fit<T, P, K>(
        &self,
        context: &FitContext<T, P, K>,
        cache: &IntermediateCache<T>,
    ) -> T
    where
        T: Float + Send + Sync,
        P: PeakShape<T> + Send + Sync,
        K: SuperpositionKernel<T, P> + Send + Sync,;
}

/// Mean squared error.
#[derive(Copy, Clone, Debug)]
pub struct Mse;

impl Metric for Mse {
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
        let len = T::from(residuals.len())
            .expect("conversion from usize to T must never fail");

        residuals.iter().fold(T::zero(), |acc, res| acc + res.powi(2)) / len
    }
}
