use crate::Deconvolution;
use crate::error::Result;
use crate::fitting::Fit;
use crate::peak_finding::Find;
use crate::smoothing::Smooth;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::Range;
use uom::si::ratio::part_per_million as ppm;
use zeenmr_peakshape::PeakShape;
use zeenmr_peakshape::iter::SuperpositionMap;
use zeenmr_spectrum::Spectrum1D;
use zeenmr_spectrum::frequency_axis::range::ShiftRange;
use zeenmr_spectrum::intensity_array::Array1D;

#[cfg(feature = "rayon")]
use crate::fitting::ParFit;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use zeenmr_peakshape::iter::ParSuperpositionMap;

/// Trait for deconvoluting a spectrum into its constituent signals.
///
/// Implementors of this trait are called 'deconvoluters'.
pub trait Deconvolute<T, P> {
    /// Deconvolutes the provided `Spectrum` into its constituent signals.
    ///
    /// Each signal is modeled as a peak shape.
    fn deconvolute<S>(&self, spectrum: &Spectrum1D<S>) -> Result<Deconvolution<S::Elem, P>>
    where
        S: Array1D<Elem = T>;
}

/// Trait for deconvoluting a spectrum into its constituent signals in
/// parallel.
///
/// Implementors of this trait are called 'parallelized deconvoluters'.
#[cfg(feature = "rayon")]
pub trait ParDeconvolute<T, P> {
    /// Deconvolutes the provided `Spectrum` into its constituent signals in
    /// parallel.
    ///
    /// Each signal is modeled as a peak shape.
    fn par_deconvolute<S>(&self, spectrum: &Spectrum1D<S>) -> Result<Deconvolution<S::Elem, P>>
    where
        S: Array1D<Elem = T>;
}

/// Extension trait for iterators of [`AsRef<Spectrum>`] types to deconvolute
/// each item using a provided 'deconvoluter'.
pub trait DeconvoluteMap<T, P>: Iterator {
    /// Applies the provided deconvoluter to each item in the iterator.
    fn deconvolute<D>(self, deconvoluter: &D) -> impl Iterator<Item = Result<Deconvolution<T, P>>>
    where
        D: Deconvolute<T, P>;
}

impl<'s, S, I, P> DeconvoluteMap<S::Elem, P> for I
where
    S: Array1D + 's,
    I: Iterator<Item = &'s Spectrum1D<S>>,
    P: PeakShape<S::Elem>,
{
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl Iterator<Item = Result<Deconvolution<S::Elem, P>>>
    where
        D: Deconvolute<S::Elem, P>,
    {
        self.map(move |spectrum| deconvoluter.deconvolute(spectrum))
    }
}

/// Extension trait for parallel iterators of [`AsRef<Spectrum>`] types to
/// deconvolute each item using a provided parallelized deconvoluter.
#[cfg(feature = "rayon")]
pub trait ParDeconvoluteMap<T, P>: IndexedParallelIterator {
    /// Applies the provided parallelized deconvoluter to each item in the
    /// iterator.
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl IndexedParallelIterator<Item = Result<Deconvolution<T, P>>>
    where
        D: ParDeconvolute<T, P> + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<'s, S, I, P> ParDeconvoluteMap<S::Elem, P> for I
where
    S: Array1D + 's,
    S::Elem: Send + Sync,
    I: IndexedParallelIterator<Item = &'s Spectrum1D<S>>,
    P: PeakShape<S::Elem> + Send + Sync,
{
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl IndexedParallelIterator<Item = Result<Deconvolution<S::Elem, P>>>
    where
        D: ParDeconvolute<S::Elem, P> + Send + Sync,
    {
        self.map(move |spectrum| deconvoluter.par_deconvolute(spectrum))
    }
}

/// Initialization marker for the smoothing algorithm.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct NeedsSmoother;

/// Initialization marker for the peak finding algorithm.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct NeedsFinder;

/// Initialization marker for the peak fitting algorithm.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct NeedsFitter;

/// Deconvolution config for processing spectra into their constituent signals.
#[derive(Clone, Debug)]
pub struct Deconvoluter<T, SM, PF, FT> {
    /// Smoothing algorithm.
    ///
    /// Must implement [`Smooth`], or be [`NeedsSmoother`].
    smoother: SM,
    /// Peak finding algorithm.
    ///
    /// Must implement [`Find`], or be [`NeedsFinder`].
    finder: PF,
    /// Peak fitting algorithm.
    ///
    /// Must implement [`Fit`] or [`ParFit`], or be [`NeedsFitter`].
    fitter: FT,
    /// Chemical shift ranges to ignore during deconvolution.
    ignore: Vec<ShiftRange>,
    /// Numeric type this deconvoluter can process.
    numeric: PhantomData<T>,
}

impl<T, P, SM, PF, FT> Deconvolute<T, P> for Deconvoluter<T, SM, PF, FT>
where
    T: Float,
    P: PeakShape<T>,
    SM: Smooth<T>,
    PF: Find<T>,
    FT: Fit<T, P>,
{
    fn deconvolute<S>(&self, spectrum: &Spectrum1D<S>) -> Result<Deconvolution<T, P>>
    where
        S: Array1D<Elem = T>,
    {
        let len = spectrum.intensities().len();
        let axis = spectrum.axis();
        let smoothed = self.smoother.smooth(spectrum.intensities());
        let ignore = self
            .ignore
            .iter()
            .filter_map(|range| {
                let start = axis.shift_to_rel(range.start())? * len as f64;
                let end = axis.shift_to_rel(range.end())? * len as f64;

                Some(start.min(end).ceil() as usize..start.max(end).floor() as usize + 1)
            })
            .collect::<Vec<Range<usize>>>();
        let peaks = self
            .finder
            .find(&smoothed, spectrum.signal_range(), &ignore)?;
        let peak_shapes = self.fitter.fit(spectrum, &peaks);
        let superpositions = spectrum
            .axis()
            .shifts(len)
            .map(|shift| {
                T::from(shift.get::<ppm>()).expect("conversion from f64 to T must not fail")
            })
            .superposition(&peak_shapes)
            .collect::<Vec<T>>();

        Ok(Deconvolution::new(
            peak_shapes,
            mse(spectrum, &superpositions, &ignore),
        ))
    }
}

#[cfg(feature = "rayon")]
impl<T, P, SM, PF, FT> ParDeconvolute<T, P> for Deconvoluter<T, SM, PF, FT>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + Send + Sync,
    SM: Smooth<T>,
    PF: Find<T>,
    FT: ParFit<T, P>,
{
    fn par_deconvolute<S>(&self, spectrum: &Spectrum1D<S>) -> Result<Deconvolution<T, P>>
    where
        S: Array1D<Elem = T>,
    {
        let len = spectrum.intensities().len();
        let axis = spectrum.axis();
        let intensities = self.smoother.smooth(spectrum.intensities());
        let ignore = self
            .ignore
            .iter()
            .filter_map(|range| {
                let start = axis.shift_to_rel(range.start())? * len as f64;
                let end = axis.shift_to_rel(range.end())? * len as f64;

                Some(start.min(end).ceil() as usize..start.max(end).floor() as usize + 1)
            })
            .collect::<Vec<Range<usize>>>();
        let peaks = self
            .finder
            .find(&intensities, spectrum.signal_range(), &ignore)?;
        let peak_shapes = self.fitter.par_fit(spectrum, &peaks);
        let superpositions = spectrum
            .axis()
            .par_shifts(len)
            .map(|shift| {
                T::from(shift.get::<ppm>()).expect("conversion from f64 to T must not fail")
            })
            .superposition(&peak_shapes)
            .collect::<Vec<T>>();

        Ok(Deconvolution::new(
            peak_shapes,
            mse(spectrum, &superpositions, &ignore),
        ))
    }
}

impl Deconvoluter<(), NeedsSmoother, NeedsFinder, NeedsFitter> {
    /// Creates a new, unconfigured deconvoluter that can be configured using
    /// the builder pattern.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_deconvolution::Deconvoluter;
    /// use zeenmr_deconvolution::fitting::IterativeRefinement;
    /// use zeenmr_deconvolution::peak_finding::CurvatureAnalysis;
    /// use zeenmr_deconvolution::smoothing::MovingAverage;
    /// use zeenmr_peakshape::Lorentzian;
    ///
    /// // setting the numeric type first helps with type inference
    /// let deconvoluter = Deconvoluter::new::<f64>()
    ///     .with_smoother(MovingAverage::default())
    ///     .with_finder(CurvatureAnalysis::default())
    ///     .with_fitter(IterativeRefinement::<Lorentzian<_>>::default());
    ///
    /// // alternatively, set the fitter first for the same effect
    /// let deconvoluter = Deconvoluter::new()
    ///     .with_fitter(IterativeRefinement::<Lorentzian<f64>>::default())
    ///     .with_finder(CurvatureAnalysis::default())
    ///     .with_smoother(MovingAverage::default());
    /// ```
    pub fn new<T>() -> Deconvoluter<T, NeedsSmoother, NeedsFinder, NeedsFitter> {
        Deconvoluter {
            smoother: NeedsSmoother,
            finder: NeedsFinder,
            fitter: NeedsFitter,
            ignore: Vec::new(),
            numeric: PhantomData,
        }
    }
}

impl<T, PF, FT> Deconvoluter<T, NeedsSmoother, PF, FT>
where
    T: Clone,
{
    /// Sets the smoothing algorithm for the deconvoluter.
    pub fn with_smoother<SM>(self, smoother: SM) -> Deconvoluter<T, SM, PF, FT>
    where
        SM: Smooth<T>,
    {
        Deconvoluter {
            smoother,
            finder: self.finder,
            fitter: self.fitter,
            ignore: self.ignore,
            numeric: self.numeric,
        }
    }
}

impl<T, SM, FT> Deconvoluter<T, SM, NeedsFinder, FT> {
    /// Sets the peak finding algorithm for the deconvoluter.
    pub fn with_finder<PF>(self, peak_finder: PF) -> Deconvoluter<T, SM, PF, FT>
    where
        PF: Find<T>,
    {
        Deconvoluter {
            smoother: self.smoother,
            finder: peak_finder,
            fitter: self.fitter,
            ignore: self.ignore,
            numeric: self.numeric,
        }
    }
}

impl<T, SM, PF> Deconvoluter<T, SM, PF, NeedsFitter> {
    /// Sets the peak fitting algorithm for the deconvoluter.
    ///
    /// If the fitter also implements [`ParFit`], the deconvoluter additionally
    /// becomes a parallelized deconvoluter.
    pub fn with_fitter<P, FT>(self, fitter: FT) -> Deconvoluter<T, SM, PF, FT>
    where
        P: PeakShape<T> + Send + Sync,
        FT: Fit<T, P>,
    {
        Deconvoluter {
            smoother: self.smoother,
            finder: self.finder,
            fitter,
            ignore: self.ignore,
            numeric: self.numeric,
        }
    }
}

impl<T, SM, PF, FT> Deconvoluter<T, SM, PF, FT> {
    /// Adds a chemical shift range to ignore during deconvolution.
    ///
    /// Some samples contain signals that cannot be fitted with the intended
    /// peak shape, such as a signal from water or stabilizing agents. Regions
    /// where these signals are expected can be excluded from peak finding and
    /// fitting. In general, this feature should only be used if including the
    /// regions causes the deconvolution performance to degrade.
    ///
    /// Overlapping ranges are automatically merged, so the input is not
    /// necessarily recoverable. For example, adding 4.7–4.9 ppm and then
    /// 4.8–5.0 ppm results in a single 4.7–5.0 ppm range.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided chemical shift range contains a bound
    /// that is `INF`, `NEG_INF` or `NaN`.
    pub fn ignore(mut self, range: ShiftRange) -> Result<Self> {
        self.ignore.push(range.normalized());
        self.ignore.sort_unstable_by(|a, b| {
            a.start()
                .partial_cmp(&b.start())
                .expect("shift range should validate at construction")
        });
        while let Some(overlap) = self
            .ignore
            .windows(2)
            .position(|w| w[1].start() <= w[0].end())
        {
            self.ignore[overlap] = ShiftRange::new(
                self.ignore[overlap].start(),
                self.ignore[overlap]
                    .end()
                    .max(self.ignore[overlap + 1].end()),
            )
            .expect("other shift ranges should be validated at construction");
            self.ignore.remove(overlap + 1);
        }

        Ok(self)
    }
}

/// Computes the mean squared error between the observed intensities and
/// the superposition of the fitted peak shapes.
///
/// Ignored regions and signal-free region are excluded.
fn mse<T, S>(
    spectrum: &Spectrum1D<S>,
    superpositions: &[S::Elem],
    ignore: &[Range<usize>],
) -> S::Elem
where
    T: Float,
    S: Array1D<Elem = T>,
{
    let signal = spectrum.signal_range();
    let iter = std::iter::once(signal.start)
        .chain(
            ignore
                .iter()
                .flat_map(|range| [range.start, range.end]),
        )
        .chain(std::iter::once(signal.end));
    let (residual, count) = iter
        .clone()
        .step_by(2)
        .zip(iter.skip(1).step_by(2))
        .fold((T::zero(), 0), |acc, (start, end)| {
            let residual = superpositions[start..end]
                .iter()
                .zip(spectrum.intensities()[start..end].iter())
                .map(|(&sup, &obs)| (sup - obs).powi(2))
                .fold(S::Elem::zero(), |acc, x| acc + x);

            (acc.0 + residual, acc.1 + end - start)
        });

    residual / T::from(count).expect("conversion from usize to T must never fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitting::IterativeRefinement;
    use crate::peak_finding::CurvatureAnalysis;
    use crate::smoothing::MovingAverage;
    use zeenmr_peakshape::Lorentzian;

    #[test]
    fn type_inference() {
        let _ = Deconvoluter::new::<f64>()
            .with_smoother(MovingAverage::default())
            .with_finder(CurvatureAnalysis::default())
            .with_fitter(IterativeRefinement::<Lorentzian<_>>::default());
    }
}
