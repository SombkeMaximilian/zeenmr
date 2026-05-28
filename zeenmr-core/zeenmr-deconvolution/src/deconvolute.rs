use crate::Deconvolution;
use crate::error::{Error, Result};
use crate::fitting::Fit;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use uom::si::ratio::part_per_million as ppm;
use zeenmr_peakshape::PeakShape;
use zeenmr_peakshape::iter::SuperpositionMap;
use zeenmr_spectrum::{ChemicalShiftRange, IndexRange, Spectrum, TryIntoIndexRange};

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use zeenmr_peakshape::iter::ParSuperpositionMap;

/// Trait for deconvoluting a [`Spectrum`] into its constituent signals.
///
/// Implementors of this trait are called 'deconvoluters'.
pub trait Deconvolute<P> {
    /// Deconvolutes the provided `Spectrum` into its constituent signals.
    ///
    /// Each signal is modeled as a peak shape.
    fn deconvolute(&self, spectrum: &Spectrum) -> Result<Deconvolution<P>>;
}

/// Trait for deconvoluting a [`Spectrum`] into its constituent signals in
/// parallel.
///
/// Implementors of this trait are called 'parallelized deconvoluters'.
#[cfg(feature = "rayon")]
pub trait ParDeconvolute<P> {
    /// Deconvolutes the provided `Spectrum` into its constituent signals in
    /// parallel.
    ///
    /// Each signal is modeled as a peak shape.
    fn par_deconvolute(&self, spectrum: &Spectrum) -> Result<Deconvolution<P>>;
}

/// Extension trait for iterators of [`AsRef<Spectrum>`] types to deconvolute
/// each item using a provided 'deconvoluter'.
pub trait DeconvoluteMap<P>: Iterator {
    /// Applies the provided deconvoluter to each item in the iterator.
    fn deconvolute<D>(self, deconvoluter: &D) -> impl Iterator<Item = Result<Deconvolution<P>>>
    where
        D: Deconvolute<P>;
}

impl<S, I, P> DeconvoluteMap<P> for I
where
    S: AsRef<Spectrum>,
    I: Iterator<Item = S>,
    P: PeakShape,
{
    fn deconvolute<D>(self, deconvoluter: &D) -> impl Iterator<Item = Result<Deconvolution<P>>>
    where
        D: Deconvolute<P>,
    {
        self.map(move |spectrum| deconvoluter.deconvolute(spectrum.as_ref()))
    }
}

/// Extension trait for parallel iterators of [`AsRef<Spectrum>`] types to
/// deconvolute each item using a provided parallelized deconvoluter.
#[cfg(feature = "rayon")]
pub trait ParDeconvoluteMap<P>: IndexedParallelIterator {
    /// Applies the provided parallelized deconvoluter to each item in the
    /// iterator.
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl IndexedParallelIterator<Item = Result<Deconvolution<P>>>
    where
        D: ParDeconvolute<P> + Send + Sync;
}

#[cfg(feature = "rayon")]
impl<S, I, P> ParDeconvoluteMap<P> for I
where
    S: AsRef<Spectrum>,
    I: IndexedParallelIterator<Item = S>,
    P: PeakShape + Send + Sync,
{
    fn deconvolute<D>(
        self,
        deconvoluter: &D,
    ) -> impl IndexedParallelIterator<Item = Result<Deconvolution<P>>>
    where
        D: ParDeconvolute<P> + Send + Sync,
    {
        self.map(move |spectrum| deconvoluter.par_deconvolute(spectrum.as_ref()))
    }
}

/// Used to indicate that the deconvoluter is missing a smoother.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct MissingSmoother;

/// Used to indicate that the deconvoluter is missing a peak finder.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct MissingPeakFinder;

/// Used to indicate that the deconvoluter is missing a fitter.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct MissingFitter;

/// Deconvolution config for processing spectra into their constituent signals.
#[derive(Clone, Debug, Default)]
pub struct Deconvoluter<SM, PF, FT> {
    /// Smoothing algorithm.
    ///
    /// Must implement [`Smooth`], or be [`MissingSmoother`].
    smoother: SM,
    /// Peak finding algorithm.
    ///
    /// Must implement [`FindPeaks`], or be [`MissingPeakFinder`].
    peak_finder: PF,
    /// Fitting algorithm.
    ///
    /// Must implement [`Fit`], or be [`MissingFitter`].
    fitter: FT,
    /// Chemical shift ranges to ignore during deconvolution.
    ignore: Vec<ChemicalShiftRange>,
}

impl<P, SM, PF, FT> Deconvolute<P> for Deconvoluter<SM, PF, FT>
where
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: Fit<P>,
{
    fn deconvolute(&self, spectrum: &Spectrum) -> Result<Deconvolution<P>> {
        let smoothed = self.smoother.smooth(spectrum.intensities());
        let ignore = self
            .ignore
            .iter()
            .filter_map(|range| range.try_into_index_range(spectrum).ok())
            .collect::<Vec<IndexRange>>();
        let peaks =
            self.peak_finder
                .find_peaks(&smoothed, spectrum.signal_boundaries(), &ignore)?;
        let peak_shapes = self.fitter.par_fit(spectrum, &peaks);
        let superpositions = spectrum
            .shifts()
            .map(|shift| shift.get::<ppm>())
            .superposition(&peak_shapes)
            .collect::<Vec<f64>>();

        Ok(Deconvolution::new(
            peak_shapes,
            mse(spectrum, &superpositions, &ignore),
        ))
    }
}

#[cfg(feature = "rayon")]
impl<P, SM, PF, FT> ParDeconvolute<P> for Deconvoluter<SM, PF, FT>
where
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: Fit<P>,
{
    fn par_deconvolute(&self, spectrum: &Spectrum) -> Result<Deconvolution<P>> {
        let intensities = self.smoother.smooth(spectrum.intensities());
        let ignore = self
            .ignore
            .iter()
            .filter_map(|range| range.try_into_index_range(spectrum).ok())
            .collect::<Vec<IndexRange>>();
        let peaks =
            self.peak_finder
                .find_peaks(&intensities, spectrum.signal_boundaries(), &ignore)?;
        let peak_shapes = self.fitter.par_fit(spectrum, &peaks);
        let superpositions = spectrum
            .par_shifts()
            .map(|shift| shift.get::<ppm>())
            .superposition(&peak_shapes)
            .collect::<Vec<f64>>();

        Ok(Deconvolution::new(
            peak_shapes,
            mse(spectrum, &superpositions, &ignore),
        ))
    }
}

impl Deconvoluter<MissingSmoother, MissingPeakFinder, MissingFitter> {
    /// Creates a new, unconfigured deconvoluter that can be configured using
    /// the builder pattern.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<PF, FT> Deconvoluter<MissingSmoother, PF, FT> {
    /// Sets the smoothing algorithm for the deconvoluter.
    pub fn with_smoother<SM>(self, smoother: SM) -> Deconvoluter<SM, PF, FT>
    where
        SM: Smooth,
    {
        Deconvoluter {
            smoother,
            peak_finder: self.peak_finder,
            fitter: self.fitter,
            ignore: self.ignore,
        }
    }
}

impl<SM, FT> Deconvoluter<SM, MissingPeakFinder, FT> {
    /// Sets the peak finding algorithm for the deconvoluter.
    pub fn with_peak_finder<PF>(self, peak_finder: PF) -> Deconvoluter<SM, PF, FT>
    where
        PF: FindPeaks,
    {
        Deconvoluter {
            smoother: self.smoother,
            peak_finder,
            fitter: self.fitter,
            ignore: self.ignore,
        }
    }
}

impl<SM, PF> Deconvoluter<SM, PF, MissingFitter> {
    /// Sets the peak shape fitting algorithm for the deconvoluter.
    pub fn with_fitter<P, FT>(self, fitter: FT) -> Deconvoluter<SM, PF, FT>
    where
        P: PeakShape + Send + Sync,
        FT: Fit<P>,
    {
        Deconvoluter {
            smoother: self.smoother,
            peak_finder: self.peak_finder,
            fitter,
            ignore: self.ignore,
        }
    }
}

impl<SM, PF, FT> Deconvoluter<SM, PF, FT> {
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
    pub fn ignore(mut self, range: ChemicalShiftRange) -> Result<Self> {
        let range = range.ordered();
        if !range.start.is_finite() || !range.end.is_finite() {
            return Err(Error::invalid_ignore_region());
        }

        self.ignore.push(range);
        self.ignore
            .sort_unstable_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
        while let Some(overlap) = self
            .ignore
            .windows(2)
            .position(|w| w[1].start <= w[0].end)
        {
            let combined = ChemicalShiftRange {
                start: self.ignore[overlap].start,
                end: self.ignore[overlap]
                    .end
                    .max(self.ignore[overlap + 1].end),
            };
            self.ignore[overlap] = combined;
            self.ignore.remove(overlap + 1);
        }

        Ok(self)
    }
}

impl<SM, PF, FT> Deconvoluter<SM, PF, FT> {}

/// Computes the mean squared error between the observed intensities and
/// the superposition of the fitted peak shapes.
///
/// Ignored regions and signal-free region are excluded.
fn mse(spectrum: &Spectrum, superpositions: &[f64], ignore: &[IndexRange]) -> f64 {
    let signal = spectrum.signal_boundaries::<IndexRange>();
    let iter = std::iter::once(signal.start)
        .chain(
            ignore
                .iter()
                .flat_map(|range| [range.start, range.end]),
        )
        .chain(std::iter::once(signal.end));
    let (residual, count) = iter.clone()
        .step_by(2)
        .zip(iter.skip(1).step_by(2))
        .fold((0.0, 0), |acc, (start, end)| {
            let residual = superpositions[start..end]
                .iter()
                .zip(spectrum.intensities()[start..end].iter())
                .map(|(sup, obs)| (sup - obs).powi(2))
                .sum::<f64>();

            (acc.0 + residual, acc.1 + end - start)
        });

    residual / (count as f64)
}
