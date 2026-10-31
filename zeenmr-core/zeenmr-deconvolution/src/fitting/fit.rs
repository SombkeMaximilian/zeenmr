use crate::peak_finding::Peak;
use crate::util::evaluation_metrics::{IntermediateCache, Metric};
use num_traits::Float;
use std::ops::Range;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::axis::FrequencyAxis;
use zeenmr_spectrum::axis::range::{FiniteBounds, SpectralRange};
use zeenmr_spectrum::intensity_array::ArrayView1D;
use zeenmr_util::StableGrid;

#[cfg(feature = "rayon")]
use crate::util::evaluation_metrics::ParMetric;

/// Fitting algorithm output.
#[derive(Clone, PartialEq, Debug)]
pub struct FittedComponents<T, P> {
    /// Fitted peak shapes.
    peak_shapes: Vec<P>,
    /// Grid used to fit them.
    grid: StableGrid<T>,
}

impl<T, P> FittedComponents<T, P>
where
    T: Float,
    P: PeakShape<T>,
{
    /// Creates a new collection of fitted components.
    pub fn new(peak_shapes: Vec<P>, grid: StableGrid<T>) -> Self {
        Self { peak_shapes, grid }
    }

    /// Returns the peak shapes in fitting grid space.
    pub fn peak_shapes(&self) -> &[P] {
        &self.peak_shapes
    }

    /// Returns the fitting grid.
    pub fn grid(&self) -> StableGrid<T> {
        self.grid
    }

    /// Returns the peak shapes in frequency space.
    pub fn into_frequency_space(mut self, axis: &FrequencyAxis<T>) -> Vec<P> {
        let mut peak_shapes = std::mem::take(&mut self.peak_shapes);
        self.map_into_range(&mut peak_shapes, axis.freq_range());

        peak_shapes
    }

    /// Returns the peak shapes in shift space.
    pub fn into_shift_space(mut self, axis: &FrequencyAxis<T>) -> Vec<P> {
        let mut peak_shapes = std::mem::take(&mut self.peak_shapes);
        self.map_into_range(&mut peak_shapes, axis.shift_range());

        peak_shapes
    }

    /// Returns the peak shapes in frequency space.
    ///
    /// Note that this clones the data. Prefer
    /// [`FittedComponents::into_frequency_space`] to consume `self` instead.
    pub fn frequency_space(&self, axis: &FrequencyAxis<T>) -> Vec<P>
    where
        P: Clone,
    {
        let mut peak_shapes = self.peak_shapes.clone();
        self.map_into_range(&mut peak_shapes, axis.freq_range());

        peak_shapes
    }

    /// Returns the peak shapes in shift space.
    ///
    /// Note that this clones the data. Prefer
    /// [`FittedComponents::into_shift_space`] to consume `self` instead.
    pub fn shift_space(&self, axis: &FrequencyAxis<T>) -> Vec<P>
    where
        P: Clone,
    {
        let mut peak_shapes = self.peak_shapes.clone();
        self.map_into_range(&mut peak_shapes, axis.shift_range());

        peak_shapes
    }

    /// Maps the peak shapes into the provided range.
    fn map_into_range<R>(&self, peak_shapes: &mut [P], range: R)
    where
        R: FiniteBounds<T>,
    {
        let len = self.grid.len();
        let offset = range.start();
        let scale = if len > 1 {
            let last = T::from(len - 1).expect("conversion from usize to T must never fail");

            range.signed_width() / (last * self.grid.step())
        } else {
            T::one()
        };

        peak_shapes
            .iter_mut()
            .for_each(|p| p.grid_transform(offset, scale));
    }
}

/// Trait for fitting peak shapes to a spectrum, given a set of detected peaks.
///
/// # Notes for Implementors
///
/// It can be assumed that the `spectrum` contains no floating point `NaN` or
/// either infinity, and that `left < center < right` holds for all `peaks`.
///
/// It can be useful for type inferences to include a `PhantomData<fn() -> P>`
/// or similar in the type implementing this trait.
pub trait Fit<T, P>
where
    P: PeakShape<T>,
{
    /// Error type when an error occurs during fitting.
    type Error;

    /// Fits peak shapes to the spectrum using the provided peaks.
    fn fit(
        &self,
        spectrum: SpectrumView1D<T, T>,
        smoothed: ArrayView1D<T>,
        peaks: &[Peak],
        ignore: &[Range<usize>],
    ) -> Result<FittedComponents<T, P>, Self::Error>;
}

/// Fitting algorithms that iteratively improve the peak shape parameters.
pub trait IterativeFit<T, P>: Fit<T, P>
where
    P: PeakShape<T>,
{
    /// Fitting model.
    type Model<'s>: FitModel<T, P, Error = Self::Error>
    where
        T: 's;

    /// Returns a model for iterative fitting.
    fn model<'s>(
        &self,
        spectrum: SpectrumView1D<'s, T, T>,
        smoothed: ArrayView1D<T>,
        peaks: &[Peak],
        ignore: &[Range<usize>],
    ) -> Result<Self::Model<'s>, Self::Error>;
}

/// Fitting model.
pub trait FitModel<T, P>
where
    P: PeakShape<T>,
{
    /// Error type when an error occurs during fitting.
    type Error;

    /// Performs one iteration of the algorithm.
    fn step(&mut self) -> Result<(), Self::Error>;

    /// Performs `n` iterations of the algorithm.
    fn step_n(&mut self, n: usize) -> Result<(), Self::Error> {
        for _ in 0..n {
            self.step()?;
        }

        Ok(())
    }

    /// Applies the algorithm until the predicate returns `false` or `max`
    /// iterations have been performed.
    fn step_while<F>(&mut self, mut predicate: F, max: usize) -> Result<usize, Self::Error>
    where
        F: FnMut(&Self) -> bool,
    {
        let mut iterations = 0;
        while predicate(self) && iterations < max {
            self.step()?;
            iterations += 1;
        }

        Ok(iterations)
    }

    /// Extracts the peak shapes from the model.
    ///
    /// This undoes any transformations.
    fn extract(self) -> FittedComponents<T, P>
    where
        Self: Sized;

    /// Returns the current peak shapes of the model.
    ///
    /// Note that positions and widths may have been transformed. Use
    /// [`FitModel::fitting_grid`] to generate points with the correct scaling.
    fn peak_shapes(&self) -> &[P];

    /// Returns `(offset, step)` for generating points on the fitting grid.
    fn fitting_grid(&self) -> (T, T);

    /// Applies an evaluation metric to the current peak shapes.
    fn apply_metric<M>(&self, metric: M, cache: &IntermediateCache<T>) -> T
    where
        M: Metric;
}

/// Trait for fitting peak shapes to a spectrum in parallel, given a set of
/// detected peaks.
///
/// Some fitting algorithms can be efficiently parallelized for certain input
/// sizes.
#[cfg(feature = "rayon")]
pub trait ParFit<T, P>
where
    P: PeakShape<T> + Send + Sync,
{
    /// Error type when an error occurs during fitting.
    type Error;

    /// Fits peak shapes to the spectrum in parallel using the provided peaks.
    fn par_fit(
        &self,
        spectrum: SpectrumView1D<T, T>,
        smoothed: ArrayView1D<T>,
        peaks: &[Peak],
        ignore: &[Range<usize>],
    ) -> Result<FittedComponents<T, P>, Self::Error>;
}

/// Parallelized fitting algorithms that iteratively improve the peak shape
/// parameters.
#[cfg(feature = "rayon")]
pub trait ParIterativeFit<T, P>: ParFit<T, P>
where
    P: PeakShape<T> + Send + Sync,
{
    /// Parallelized fitting model.
    type ParModel<'s>: ParFitModel<T, P, Error = Self::Error>
    where
        T: 's;

    /// Returns a model for parallelized, iterative fitting.
    fn par_model<'s>(
        &self,
        spectrum: SpectrumView1D<'s, T, T>,
        smoothed: ArrayView1D<T>,
        peaks: &[Peak],
        ignore: &[Range<usize>],
    ) -> Result<Self::ParModel<'s>, Self::Error>;
}

/// Parallelized fitting model.
#[cfg(feature = "rayon")]
pub trait ParFitModel<T, P>: FitModel<T, P>
where
    P: PeakShape<T> + Send + Sync,
{
    /// Performs one iteration of the algorithm in parallel.
    fn par_step(&mut self) -> Result<(), Self::Error>;

    /// Performs `n` iterations of the algorithm in parallel.
    fn par_step_n(&mut self, n: usize) -> Result<(), Self::Error> {
        for _ in 0..n {
            self.par_step()?;
        }

        Ok(())
    }

    /// Applies the algorithm in parallel until the predicate returns `false`
    /// or `max` iterations have been performed.
    fn par_step_while<F>(&mut self, mut predicate: F, max: usize) -> Result<usize, Self::Error>
    where
        F: FnMut(&Self) -> bool,
    {
        let mut iterations = 0;
        while predicate(self) && iterations < max {
            self.par_step()?;
            iterations += 1;
        }

        Ok(iterations)
    }

    /// Applies an evaluation metric to the current peak shapes in parallel.
    fn par_apply_metric<M>(&self, metric: M, cache: &IntermediateCache<T>) -> T
    where
        M: ParMetric;
}
