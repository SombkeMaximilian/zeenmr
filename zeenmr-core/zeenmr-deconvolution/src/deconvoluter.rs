use crate::Deconvolution;
use crate::error::Result;
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

#[derive(Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct Deconvoluter<P, SM, PF, FT> {
    smoother: SM,
    peak_finder: PF,
    fitter: FT,
    ignore: Option<Vec<ChemicalShiftRange>>,
    peak_shape: PhantomData<P>,
}

pub trait DeconvoluteMap<P, SM, PF, FT>: Iterator
where
    P: PeakShape + Send + Sync,
    SM: Smooth,
    PF: FindPeaks,
    FT: FitPeakShapes<P>,
{
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

#[cfg(feature = "rayon")]
pub trait ParDeconvoluteMap<P, SM, PF, FT>: IndexedParallelIterator
where
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
    pub fn new(smoother: SM, peak_finder: PF, fitter: FT) -> Self {
        Self {
            smoother,
            peak_finder,
            fitter,
            ignore: None,
            peak_shape: PhantomData::<P>,
        }
    }

    pub fn smoothing_settings(&self) -> SM::Settings {
        self.smoother.settings()
    }

    pub fn peak_finding_settings(&self) -> PF::Settings {
        self.peak_finder.settings()
    }

    pub fn fitting_settings(&self) -> FT::Settings {
        self.fitter.settings()
    }

    pub fn deconvolute(
        &self,
        spectrum: &Spectrum,
    ) -> Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>> {
        let intensities = self.smoother.smooth(spectrum.intensities());
        let peaks = self.peak_finder.find_peaks(
            intensities,
            spectrum.signal_boundaries(),
            self.ignore_ranges(spectrum),
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

    #[cfg(feature = "rayon")]
    pub fn par_deconvolute(
        &self,
        spectrum: &Spectrum,
    ) -> Result<Deconvolution<P, SM::Settings, PF::Settings, FT::Settings>> {
        let intensities = self.smoother.smooth(spectrum.intensities());
        let peaks = self.peak_finder.find_peaks(
            intensities,
            spectrum.signal_boundaries(),
            self.ignore_ranges(spectrum),
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

    fn ignore_ranges(&self, spectrum: &Spectrum) -> Option<Vec<IndexRange>> {
        self.ignore.as_ref().map(|ignore| {
            ignore
                .iter()
                .map(|range| range.try_into_index_range(spectrum))
                .filter_map(|r| r.ok())
                .collect()
        })
    }

    fn compute_mse(&self, spectrum: &Spectrum, superpositions: Vec<f64>) -> f64 {
        let signal = spectrum.signal_boundaries::<IndexRange>();
        let included = if let Some(ignore) = self.ignore_ranges(spectrum) {
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
