use crate::Deconvolution;
use crate::error::Error;
use crate::fitting::Fit;
use crate::peak_finding::Find;
use crate::smoothing::Smooth;
use num_traits::Float;
use std::ops::Range;
use zeenmr_peakshape::PeakShape;
use zeenmr_peakshape::batch_superposition::{Standard, SuperpositionKernel};
use zeenmr_spectrum::Spectrum1D;
use zeenmr_spectrum::axis::range::{FiniteBounds, ShiftRange};
use zeenmr_spectrum::intensity_array::Storage;

#[cfg(feature = "rayon")]
use crate::fitting::ParFit;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Temporary way to make the new API work.
const SUPERPOSITION: Standard = Standard::new();

/// Trait for deconvoluting a spectrum into its constituent signals.
///
/// Implementors of this trait are called deconvoluters.
pub trait Deconvolute<T, P> {
    /// Error type when an error occurs during deconvolution.
    type Error;

    /// Deconvolutes the provided `Spectrum` into its constituent signals.
    ///
    /// Each signal is modeled as a peak shape.
    fn deconvolute<S>(
        &self,
        spectrum: &Spectrum1D<T, S>,
    ) -> Result<Deconvolution<T, P>, Self::Error>
    where
        S: Storage<Elem = T>;
}

/// Trait for deconvoluting a spectrum into its constituent signals in
/// parallel.
///
/// Implementors of this trait are called parallelized deconvoluters.
#[cfg(feature = "rayon")]
pub trait ParDeconvolute<T, P> {
    /// Error type when an error occurs during deconvolution.
    type Error;

    /// Deconvolutes the provided `Spectrum` into its constituent signals in
    /// parallel.
    ///
    /// Each signal is modeled as a peak shape.
    fn par_deconvolute<S>(
        &self,
        spectrum: &Spectrum1D<T, S>,
    ) -> Result<Deconvolution<T, P>, Self::Error>
    where
        S: Storage<Elem = T>;
}

/// Extension trait for iterators of spectrum types to deconvolute each item
/// using a provided deconvoluter.
pub trait DeconvoluteMap<T, P>: Iterator {
    /// Applies the provided deconvoluter to each item in the iterator.
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl Iterator<Item = Result<Deconvolution<T, P>, D::Error>>
    where
        D: Deconvolute<T, P>;
}

impl<'s, S, I, P> DeconvoluteMap<S::Elem, P> for I
where
    S: Storage + 's,
    I: Iterator<Item = &'s Spectrum1D<S::Elem, S>>,
    P: PeakShape<S::Elem>,
{
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl Iterator<Item = Result<Deconvolution<S::Elem, P>, D::Error>>
    where
        D: Deconvolute<S::Elem, P>,
    {
        self.map(move |spectrum| deconvoluter.deconvolute(spectrum))
    }
}

/// Extension trait for iterators of spectrum types to deconvolute each item
/// using a provided parallelized deconvoluter.
#[cfg(feature = "rayon")]
pub trait ParDeconvoluteMap<T, P>: IndexedParallelIterator {
    /// Applies the provided parallelized deconvoluter to each item in the
    /// iterator.
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl IndexedParallelIterator<Item = Result<Deconvolution<T, P>, D::Error>>
    where
        D: ParDeconvolute<T, P> + Send + Sync,
        D::Error: Send;
}

#[cfg(feature = "rayon")]
impl<'s, S, I, P> ParDeconvoluteMap<S::Elem, P> for I
where
    S: Storage + 's,
    S::Elem: Send + Sync,
    I: IndexedParallelIterator<Item = &'s Spectrum1D<S::Elem, S>>,
    P: PeakShape<S::Elem> + Send + Sync,
{
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl IndexedParallelIterator<Item = Result<Deconvolution<S::Elem, P>, D::Error>>
    where
        D: ParDeconvolute<S::Elem, P> + Send + Sync,
        D::Error: Send,
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
#[derive(Clone, PartialEq, Debug)]
pub struct Deconvoluter<T, C1, C2, C3> {
    /// Smoothing algorithm.
    ///
    /// Must implement [`Smooth`], or be [`NeedsSmoother`].
    smoother: C1,
    /// Peak finding algorithm.
    ///
    /// Must implement [`Find`], or be [`NeedsFinder`].
    finder: C2,
    /// Peak fitting algorithm.
    ///
    /// Must implement [`Fit`] or [`ParFit`], or be [`NeedsFitter`].
    fitter: C3,
    /// Chemical shift ranges to ignore during deconvolution.
    ignore: Vec<ShiftRange<T>>,
}

impl<T, P, C1, C2, C3> Deconvolute<T, P> for Deconvoluter<T, C1, C2, C3>
where
    T: Float,
    P: PeakShape<T>,
    C1: Smooth<T>,
    C2: Find<T>,
    C3: Fit<T, P>,
{
    type Error = Error<C1::Error, C2::Error, C3::Error>;

    fn deconvolute<S>(
        &self,
        spectrum: &Spectrum1D<T, S>,
    ) -> Result<Deconvolution<T, P>, Self::Error>
    where
        S: Storage<Elem = T>,
    {
        let len = spectrum.intensities().len();
        let len_as_t = T::from(len).expect("conversion from usize to T must never fail");
        let axis = spectrum.axis();
        let smoothed = self
            .smoother
            .smooth(spectrum.intensities())
            .map_err(Error::smoothing)?;
        let ignore = self
            .ignore
            .iter()
            .filter_map(|range| {
                let start = axis.shift_to_rel(range.start())? * len_as_t;
                let end = axis.shift_to_rel(range.end())? * len_as_t;
                let min = start.min(end).ceil().to_usize()?;
                let max = start.max(end).floor().to_usize()? + 1;

                // no need to check against len because relative coordinates
                // returned by `FrequencyAxis` are always in `[0, 1]`
                Some(min..max)
            })
            .collect::<Vec<Range<usize>>>();
        let peaks = self
            .finder
            .find(&smoothed, spectrum.signal_range(), &ignore)
            .map_err(Error::finding)?;
        let peak_shapes = self
            .fitter
            .fit(spectrum.view(), &peaks)
            .map_err(Error::fitting)?;
        let mse = mse(spectrum, &peak_shapes, &ignore);

        Ok(Deconvolution::new(peak_shapes, mse))
    }
}

#[cfg(feature = "rayon")]
impl<T, P, C1, C2, C3> ParDeconvolute<T, P> for Deconvoluter<T, C1, C2, C3>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + Send + Sync,
    C1: Smooth<T>,
    C2: Find<T>,
    C3: ParFit<T, P>,
{
    type Error = Error<C1::Error, C2::Error, C3::Error>;

    fn par_deconvolute<S>(
        &self,
        spectrum: &Spectrum1D<T, S>,
    ) -> Result<Deconvolution<T, P>, Self::Error>
    where
        S: Storage<Elem = T>,
    {
        let len = spectrum.intensities().len();
        let len_as_t = T::from(len).expect("conversion from usize to T must never fail");
        let axis = spectrum.axis();
        let intensities = self
            .smoother
            .smooth(spectrum.intensities())
            .map_err(Error::smoothing)?;
        let ignore = self
            .ignore
            .iter()
            .filter_map(|range| {
                let start = axis.shift_to_rel(range.start())? * len_as_t;
                let end = axis.shift_to_rel(range.end())? * len_as_t;
                let min = start.min(end).ceil().to_usize()?;
                let max = start.max(end).floor().to_usize()? + 1;

                // no need to check against len because relative coordinates
                // returned by `FrequencyAxis` are always in `[0, 1]`
                Some(min..max)
            })
            .collect::<Vec<Range<usize>>>();
        let peaks = self
            .finder
            .find(&intensities, spectrum.signal_range(), &ignore)
            .map_err(Error::finding)?;
        let peak_shapes = self
            .fitter
            .par_fit(spectrum.view(), &peaks)
            .map_err(Error::fitting)?;
        let mse = mse(spectrum, &peak_shapes, &ignore);

        Ok(Deconvolution::new(peak_shapes, mse))
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
    /// use zeenmr_deconvolution::fitting::ThreePoint;
    /// use zeenmr_deconvolution::peak_finding::CurvatureAnalysis;
    /// use zeenmr_deconvolution::smoothing::MovingAverage;
    /// use zeenmr_peakshape::Lorentzian;
    ///
    /// // setting the numeric type first helps with type inference
    /// let deconvoluter = Deconvoluter::new::<f64>()
    ///     .with_smoother(MovingAverage::default())
    ///     .with_finder(CurvatureAnalysis::default())
    ///     .with_fitter(ThreePoint::<Lorentzian<_>>::default());
    ///
    /// // alternatively, set the fitter first for the same effect
    /// let deconvoluter = Deconvoluter::new()
    ///     .with_fitter(ThreePoint::<Lorentzian<f64>>::default())
    ///     .with_finder(CurvatureAnalysis::default())
    ///     .with_smoother(MovingAverage::default());
    /// ```
    pub fn new<T>() -> Deconvoluter<T, NeedsSmoother, NeedsFinder, NeedsFitter> {
        Deconvoluter {
            smoother: NeedsSmoother,
            finder: NeedsFinder,
            fitter: NeedsFitter,
            ignore: Vec::new(),
        }
    }
}

impl<T, C2, C3> Deconvoluter<T, NeedsSmoother, C2, C3>
where
    T: Clone,
{
    /// Sets the smoothing algorithm for the deconvoluter.
    pub fn with_smoother<C1>(self, smoother: C1) -> Deconvoluter<T, C1, C2, C3>
    where
        C1: Smooth<T>,
    {
        Deconvoluter {
            smoother,
            finder: self.finder,
            fitter: self.fitter,
            ignore: self.ignore,
        }
    }
}

impl<T, C1, C3> Deconvoluter<T, C1, NeedsFinder, C3> {
    /// Sets the peak finding algorithm for the deconvoluter.
    pub fn with_finder<C2>(self, peak_finder: C2) -> Deconvoluter<T, C1, C2, C3>
    where
        C2: Find<T>,
    {
        Deconvoluter {
            smoother: self.smoother,
            finder: peak_finder,
            fitter: self.fitter,
            ignore: self.ignore,
        }
    }
}

impl<T, C1, C2> Deconvoluter<T, C1, C2, NeedsFitter> {
    /// Sets the peak fitting algorithm for the deconvoluter.
    ///
    /// If the fitter also implements [`ParFit`], the deconvoluter additionally
    /// becomes a parallelized deconvoluter.
    pub fn with_fitter<P, C3>(self, fitter: C3) -> Deconvoluter<T, C1, C2, C3>
    where
        P: PeakShape<T> + Send + Sync,
        C3: Fit<T, P>,
    {
        Deconvoluter {
            smoother: self.smoother,
            finder: self.finder,
            fitter,
            ignore: self.ignore,
        }
    }

    /// Sets the peak fitting algorithm for the deconvoluter.
    ///
    /// Use this method only if `C3` is not also [`Fit`]. Prefer [`with_fitter`]
    /// for better type inference otherwise.
    ///
    /// [`with_fitter`]: Self::with_fitter
    #[cfg(feature = "rayon")]
    pub fn with_par_fitter<P, C3>(self, fitter: C3) -> Deconvoluter<T, C1, C2, C3>
    where
        P: PeakShape<T> + Send + Sync,
        C3: ParFit<T, P>,
    {
        Deconvoluter {
            smoother: self.smoother,
            finder: self.finder,
            fitter,
            ignore: self.ignore,
        }
    }
}

impl<T, C1, C2, C3> Deconvoluter<T, C1, C2, C3>
where
    T: Float,
{
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
    pub fn ignore(mut self, range: ShiftRange<T>) -> Self {
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

        self
    }
}

/// Computes the mean squared error between the observed intensities and
/// the superposition of the fitted peak shapes.
///
/// Ignored regions and signal-free region are excluded.
fn mse<T, P, S>(spectrum: &Spectrum1D<T, S>, peak_shapes: &[P], ignore: &[Range<usize>]) -> T
where
    T: Float,
    P: PeakShape<T>,
    S: Storage<Elem = T>,
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
            let shifts = spectrum
                .axis()
                .shifts(spectrum.intensities().len())
                .skip(start)
                .take(end - start)
                .collect::<Vec<T>>();
            let superposition = SUPERPOSITION.superposition(peak_shapes, &shifts);

            let residual = superposition
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
    use crate::fitting::ThreePoint;
    use crate::peak_finding::CurvatureAnalysis;
    use crate::smoothing::MovingAverage;
    use zeenmr_peakshape::Lorentzian;

    #[test]
    fn type_inference() {
        let _ = Deconvoluter::new::<f64>()
            .with_smoother(MovingAverage::default())
            .with_finder(CurvatureAnalysis::default())
            .with_fitter(ThreePoint::<Lorentzian<_>>::default());
    }
}
