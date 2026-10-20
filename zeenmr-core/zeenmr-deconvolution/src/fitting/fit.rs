use crate::peak_finding::Peak;
use crate::util::evaluation_metrics::{IntermediateCache, Metric};
use std::ops::Range;
use zeenmr_peakshape::PeakShape;
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::intensity_array::ArrayView1D;

#[cfg(feature = "rayon")]
use crate::util::evaluation_metrics::ParMetric;

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
    ) -> Result<Vec<P>, Self::Error>;
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
    fn extract(self) -> Vec<P>
    where
        Self: Sized;

    /// Returns the current peak shapes of the model.
    ///
    /// Note that positions and widths may have been transformed.
    /// [`FitModel::inverse_transform`] provides a way to transform them back
    /// to chemical shift space.
    fn peak_shapes(&self) -> &[P];

    /// Returns a function to undo the transformation applied to peak shapes.
    fn inverse_transform(&self) -> impl Fn(&mut P);

    /// Returns an owned copy of the peak shapes of the model, undoing any
    /// transformations first.
    fn restored_peak_shapes(&self) -> Vec<P>
    where
        P: Clone,
    {
        let mut peak_shapes = self.peak_shapes().to_vec();
        let inverse = self.inverse_transform();
        peak_shapes.iter_mut().for_each(inverse);

        peak_shapes
    }

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
    ) -> Result<Vec<P>, Self::Error>;
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
