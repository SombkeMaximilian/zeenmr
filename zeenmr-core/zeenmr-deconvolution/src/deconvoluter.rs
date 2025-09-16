use crate::Deconvolution;
use crate::error::{Error, Result};
use crate::fitting::FitPeakShapes;
use crate::peak_finding::FindPeaks;
use crate::smoothing::Smooth;
use std::marker::PhantomData;
use uom::si::ratio::part_per_million as ppm;
use zeenmr_peakshape::PeakShape;
use zeenmr_peakshape::iter::SuperpositionMap;
use zeenmr_spectrum::{ChemicalShiftRange, IndexRange, Spectrum, TryIntoIndexRange};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
use zeenmr_peakshape::iter::ParSuperpositionMap;

/// Deconvolution pipeline that combines smoothing, peak finding, and peak shape
/// fitting to deconvolute a [`Spectrum`] into its constituent peak shapes.
#[derive(Clone, Debug, Default)]
pub struct Deconvoluter<P, SM, PF, FT> {
    /// Smoothing algorithm.
    smoother: SM,
    /// Peak finding algorithm.
    peak_finder: PF,
    /// Peak shape fitting algorithm.
    fitter: FT,
    /// Chemical shift ranges to ignore during deconvolution.
    ignore: Option<Vec<ChemicalShiftRange>>,
    /// Marker for the peak shape type.
    peak_shape: PhantomData<P>,
}

/// Extension trait for iterators of [`AsRef<Spectrum>`] types to deconvolute
/// each item using a provided [`Deconvoluter`].
pub trait DeconvoluteMap<P, SM, PF, FT>: Iterator
where
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: FitPeakShapes<P>,
{
    /// Apply the provided `Deconvoluter` to each item in the iterator.
    fn deconvolute(
        self,
        deconvoluter: &Deconvoluter<P, SM, PF, FT>,
    ) -> impl Iterator<Item = Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>>>;
}

impl<S, I, P, SM, PF, FT> DeconvoluteMap<P, SM, PF, FT> for I
where
    S: AsRef<Spectrum>,
    I: Iterator<Item = S>,
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: FitPeakShapes<P>,
{
    fn deconvolute(
        self,
        deconvoluter: &Deconvoluter<P, SM, PF, FT>,
    ) -> impl Iterator<Item = Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>>>
    {
        self.map(move |spectrum| deconvoluter.deconvolute(spectrum.as_ref()))
    }
}

/// Extension trait for parallel iterators of [`AsRef<Spectrum>`] types to
/// deconvolute each item using a provided [`Deconvoluter`].
#[cfg(feature = "rayon")]
pub trait ParDeconvoluteMap<P, SM, PF, FT>: IndexedParallelIterator
where
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: FitPeakShapes<P>,
{
    /// Apply the provided `Deconvoluter` to each item in the parallel iterator.
    fn deconvolute(
        self,
        deconvoluter: &Deconvoluter<P, SM, PF, FT>,
    ) -> impl IndexedParallelIterator<
        Item = Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>>,
    >;
}

impl<S, I, P, SM, PF, FT> ParDeconvoluteMap<P, SM, PF, FT> for I
where
    S: AsRef<Spectrum>,
    I: IndexedParallelIterator<Item = S>,
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: FitPeakShapes<P>,
{
    fn deconvolute(
        self,
        deconvoluter: &Deconvoluter<P, SM, PF, FT>,
    ) -> impl IndexedParallelIterator<
        Item = Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>>,
    > {
        self.map(move |spectrum| deconvoluter.par_deconvolute(spectrum.as_ref()))
    }
}

impl<P, SM, PF, FT> Deconvoluter<P, SM, PF, FT>
where
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: FitPeakShapes<P>,
{
    /// Creates a new `Deconvoluter`.
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
    /// let deconvoluter = Deconvoluter::new(
    ///     MovingAverage::default(),
    ///     CurvatureAnalysis::default(),
    ///     IterativeRefinement::<Lorentzian>::default(),
    /// );
    /// ```
    pub fn new(smoother: SM, peak_finder: PF, fitter: FT) -> Self {
        Self {
            smoother,
            peak_finder,
            fitter,
            ignore: None,
            peak_shape: PhantomData::<P>,
        }
    }

    /// Returns the smoothing settings used in the deconvoluter.
    pub fn smoothing_settings(&self) -> SM::Settings {
        self.smoother.settings()
    }

    /// Returns the peak finding settings used in the deconvoluter.
    pub fn peak_finding_settings(&self) -> PF::Settings {
        self.peak_finder.settings()
    }

    /// Returns the fitting settings used in the deconvoluter.
    pub fn fitting_settings(&self) -> FT::Settings {
        self.fitter.settings()
    }

    /// Returns the ignored chemical shift ranges.
    pub fn ignored_ranges(&self) -> Option<&[ChemicalShiftRange]> {
        self.ignore.as_deref()
    }

    /// Adds a [`ChemicalShiftRange`] to ignore during deconvolution.
    ///
    /// Some samples contain compounds that are not of interest, such as a water
    /// signal or stabilizing agents. Regions where these compounds are expected
    /// can be ignored during teh deconvolution.
    ///
    /// Overlapping regions are internally combined, so the input is not
    /// necessarily recoverable. For example, if the ranges 4.7-4.9 ppm and
    /// 4.8-5.0 ppm are added, [`Deconvoluter::ignored_ranges`] will return
    /// a slice containing only a 4.7-5.0 ppm range.
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_deconvolution::Deconvoluter;
    /// use zeenmr_deconvolution::fitting::IterativeRefinement;
    /// use zeenmr_deconvolution::peak_finding::CurvatureAnalysis;
    /// use zeenmr_deconvolution::smoothing::MovingAverage;
    /// use zeenmr_peakshape::Lorentzian;
    ///
    /// # fn main() -> zeenmr_deconvolution::error::Result<()> {
    /// let mut deconvoluter = Deconvoluter::new(
    ///     MovingAverage::default(),
    ///     CurvatureAnalysis::default(),
    ///     IterativeRefinement::<Lorentzian>::default(),
    /// );
    ///
    /// deconvoluter.add_ignore_range(Ratio::new::<ppm>(4.7)..Ratio::new::<ppm>(4.9))?;
    /// deconvoluter.add_ignore_range(Ratio::new::<ppm>(5.2)..Ratio::new::<ppm>(5.6))?;
    /// assert_eq!(deconvoluter.ignored_ranges().unwrap().len(), 2);
    ///
    /// // overlapping ranges are combined
    /// deconvoluter.add_ignore_range(Ratio::new::<ppm>(4.8)..Ratio::new::<ppm>(5.4))?;
    /// assert_eq!(deconvoluter.ignored_ranges().unwrap().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_ignore_range<R>(&mut self, range: R) -> Result<()>
    where
        R: Into<ChemicalShiftRange>,
    {
        let range = range.into().ordered();
        if !range.start.is_finite() || !range.end.is_finite() {
            return Err(Error::invalid_ignore_region());
        }

        if let Some(ignore) = self.ignore.as_mut() {
            ignore.push(range);
            ignore.sort_unstable_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
            while let Some(overlap) = ignore
                .windows(2)
                .position(|w| w[1].start <= w[0].start)
            {
                let combined = ChemicalShiftRange {
                    start: ignore[overlap].start,
                    end: ignore[overlap + 1].end,
                };
                ignore.remove(overlap);
                ignore.remove(overlap);
                ignore.insert(overlap, combined);
            }
        } else {
            self.ignore = Some(vec![range]);
        }

        Ok(())
    }

    /// Clears the ignored chemical shift ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_deconvolution::Deconvoluter;
    /// use zeenmr_deconvolution::fitting::IterativeRefinement;
    /// use zeenmr_deconvolution::peak_finding::CurvatureAnalysis;
    /// use zeenmr_deconvolution::smoothing::MovingAverage;
    /// use zeenmr_peakshape::Lorentzian;
    ///
    /// # fn main() -> zeenmr_deconvolution::error::Result<()> {
    /// let mut deconvoluter = Deconvoluter::new(
    ///     MovingAverage::default(),
    ///     CurvatureAnalysis::default(),
    ///     IterativeRefinement::<Lorentzian>::default(),
    /// );
    ///
    /// deconvoluter.add_ignore_range(Ratio::new::<ppm>(4.7)..Ratio::new::<ppm>(4.9))?;
    /// deconvoluter.clear_ignore_ranges();
    /// assert!(deconvoluter.ignored_ranges().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_ignore_ranges(&mut self) {
        self.ignore = None;
    }

    /// Deconvolutes the provided [`Spectrum`] into its constituent peak shapes.
    ///
    /// # Errors
    ///
    /// Any errors that may occur are outlined in the documentation of the
    /// individual component algorithms of the [`Deconvoluter`].
    pub fn deconvolute<S>(
        &self,
        spectrum: S,
    ) -> Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>>
    where
        S: AsRef<Spectrum>,
    {
        let spectrum = spectrum.as_ref();
        let intensities = self.smoother.smooth(spectrum.intensities());
        let peaks = self.peak_finder.find_peaks(
            intensities,
            spectrum.signal_boundaries(),
            self.ignore_index_ranges(spectrum),
        )?;
        let peak_shapes = self
            .fitter
            .fit_peak_shapes(spectrum, peaks)
            .collect::<Vec<P>>();
        let superpositions = spectrum
            .shifts()
            .map(|shift| shift.get::<ppm>())
            .superposition(&peak_shapes)
            .collect::<Vec<f64>>();

        Ok(Deconvolution::new(
            peak_shapes,
            self,
            self.compute_mse(spectrum, superpositions),
        ))
    }

    /// Deconvolutes the provided [`Spectrum`] into its constituent peak shapes
    /// in parallel.
    ///
    /// # Errors
    ///
    /// Any errors that may occur are outlined in the documentation of the
    /// individual component algorithms of the [`Deconvoluter`].
    #[cfg(feature = "rayon")]
    pub fn par_deconvolute<S>(
        &self,
        spectrum: S,
    ) -> Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>>
    where
        S: AsRef<Spectrum>,
    {
        let spectrum = spectrum.as_ref();
        let intensities = self.smoother.smooth(spectrum.intensities());
        let peaks = self.peak_finder.find_peaks(
            intensities,
            spectrum.signal_boundaries(),
            self.ignore_index_ranges(spectrum),
        )?;
        let peak_shapes = self
            .fitter
            .fit_peak_shapes(spectrum, peaks)
            .collect::<Vec<P>>();
        let superpositions = spectrum
            .par_shifts()
            .map(|shift| shift.get::<ppm>())
            .superposition(&peak_shapes)
            .collect::<Vec<f64>>();

        Ok(Deconvolution::new(
            peak_shapes,
            self,
            self.compute_mse(spectrum, superpositions),
        ))
    }

    /// Converts the ignored chemical shift ranges to index ranges for the
    /// provided spectrum.
    fn ignore_index_ranges(&self, spectrum: &Spectrum) -> Option<Vec<IndexRange>> {
        self.ignore.as_ref().map(|ignore| {
            ignore
                .iter()
                .map(|range| range.try_into_index_range(spectrum))
                .filter_map(|r| r.ok())
                .collect()
        })
    }

    /// Computes the mean squared error between the observed intensities and
    /// the superposition of the fitted peak shapes.
    ///
    /// Ignored regions and signal-free region are excluded.
    fn compute_mse(&self, spectrum: &Spectrum, superpositions: Vec<f64>) -> f64 {
        let signal = spectrum.signal_boundaries::<IndexRange>();
        let included = if let Some(ignore) = self.ignore_index_ranges(spectrum) {
            let iter = std::iter::once(signal.start)
                .chain(
                    ignore
                        .iter()
                        .flat_map(|range| [range.start, range.end]),
                )
                .chain(std::iter::once(signal.end));

            iter.clone()
                .step_by(2)
                .zip(iter.skip(1).step_by(2))
                .map(|(start, end)| IndexRange { start, end })
                .collect::<Vec<IndexRange>>()
        } else {
            vec![signal]
        };
        let residual = included
            .iter()
            .map(|range| {
                superpositions[range.start..range.end]
                    .iter()
                    .zip(spectrum.intensities()[range.start..range.end].iter())
                    .map(|(sup, obs)| (sup - obs).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let count = included
            .into_iter()
            .map(|range| range.end - range.start)
            .sum::<usize>();

        residual / (count as f64)
    }
}
