use crate::fitting::Fit;
use crate::peak_finding::Peak;
use num_traits::Float;
use std::marker::PhantomData;
use uom::si::ratio::part_per_million as ppm;
use zeenmr_peakshape::iter::SuperpositionMap;
use zeenmr_peakshape::{Lorentzian, PeakShape};
use zeenmr_spectrum::Spectrum1D;
use zeenmr_spectrum::intensity_array::Array1D;

#[cfg(feature = "rayon")]
use crate::fitting::ParFit;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use zeenmr_peakshape::iter::ParSuperpositionMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for estimating parameters of a peak shape from three data points.
pub trait ThreePointStencil<T> {
    /// Estimate the parameters of the peak shape from three data points.
    fn estimate_parameters(x: (T, T, T), y: (T, T, T)) -> Self;
}

impl<T> ThreePointStencil<T> for Lorentzian<T>
where
    T: Float,
{
    fn estimate_parameters(x: (T, T, T), y: (T, T, T)) -> Self {
        let half = T::one() / (T::one() + T::one());

        let numerator = x.0.powi(2) * y.0 * (y.1 - y.2)
            + x.1.powi(2) * y.1 * (y.2 - y.0)
            + x.2.powi(2) * y.2 * (y.0 - y.1);
        let denominator =
            y.0 * y.1 * (x.0 - x.1) + y.1 * y.2 * (x.1 - x.2) + y.2 * y.0 * (x.2 - x.0);
        let center = half * numerator / denominator;

        let left = (y.0 * (x.0 - center).powi(2) - y.1 * (x.1 - center).powi(2)) / (y.1 - y.0);
        let right = (y.1 * (x.1 - center).powi(2) - y.2 * (x.2 - center).powi(2)) / (y.2 - y.1);
        let scale2 = (half * (left + right)).max(T::epsilon());

        let amp_scale = y.1 * (scale2 + (x.1 - center).powi(2));

        Lorentzian::new(amp_scale, scale2, center)
    }
}

/// A reduced representation of a spectrum that only contains the data points
/// that are part of peaks.
#[derive(Clone, Debug)]
struct ReducedSpectrum<T> {
    /// Chemical shifts that are part of the peaks, in ppm.
    shifts: Vec<T>,
    /// Intensity values that are part of the peaks.
    intensities: Vec<T>,
}

impl<T> ReducedSpectrum<T>
where
    T: Float,
{
    /// Extracts the positions and intensities of the peaks from the spectrum
    /// and constructs a `ReducedSpectrum` from them.
    fn new<S>(spectrum: &Spectrum1D<S>, peaks: &[Peak]) -> Self
    where
        S: Array1D<Elem = T>,
    {
        let len = spectrum.intensities().len();
        let axis = spectrum.axis();
        let (shifts, intensities) = peaks
            .iter()
            .flat_map(|peak| [peak.left, peak.center, peak.right])
            .filter_map(|index| {
                if index < len {
                    Some((
                        T::from(
                            axis.rel_to_shift((index as f64) / (len as f64))
                                .unwrap()
                                .get::<ppm>(),
                        )
                        .expect("conversion from f64 to T must never fail"),
                        spectrum.intensities()[index],
                    ))
                } else {
                    None
                }
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        Self {
            shifts,
            intensities,
        }
    }

    /// Returns an iterator over the peak stencils in the reduced spectrum.
    fn stencils(&self) -> impl Iterator<Item = PeakStencil<T>> {
        self.shifts
            .chunks(3)
            .zip(self.intensities.chunks(3))
            .map(|(shifts, intensities)| {
                let mut stencil = PeakStencil {
                    shifts: (shifts[0], shifts[1], shifts[2]),
                    intensities: (intensities[0], intensities[1], intensities[2]),
                };
                stencil.mirror_shoulder();

                stencil
            })
    }
}

#[derive(Copy, Clone, Debug)]
struct PeakStencil<T> {
    /// Chemical shifts of the three points in ppm.
    shifts: (T, T, T),
    /// Intensity values of the three points.
    intensities: (T, T, T),
}

impl<T> PeakStencil<T>
where
    T: Float,
{
    /// Mirrors the left/right data points onto the right/left data point if the
    /// intensities are ascending/descending from left to center to right.
    ///
    /// For cases where the peak is a shoulder of another, larger peak, it is
    /// required to make an assumption about the shape of the peak. This method
    /// assumes that the peak is symmetric about the center data point and
    /// mirrors the data point for which the intensity is lower than the center
    /// data point onto the other side. This is done to ensure that the 3-point
    /// stencil is working with data that has a peak-like shape.
    fn mirror_shoulder(&mut self) {
        let two = T::one() + T::one();

        let increasing =
            self.intensities.0 <= self.intensities.1 && self.intensities.1 <= self.intensities.2;
        let decreasing =
            self.intensities.0 >= self.intensities.1 && self.intensities.1 >= self.intensities.2;
        match (increasing, decreasing) {
            (true, _) => {
                self.intensities.2 = self.intensities.0;
                self.shifts.2 = two * self.shifts.1 - self.shifts.0;
            }
            (_, true) => {
                self.intensities.0 = self.intensities.2;
                self.shifts.0 = two * self.shifts.1 - self.shifts.2;
            }
            _ => {}
        };
    }
}

/// Fitting algorithm based on the analytical solution of a system of equations
/// using a 3-point peak stencil.
#[derive(Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IterativeRefinement<P> {
    /// Number of iterations to refine the peak parameters.
    pub iterations: usize,
    /// Marker for the peak shape type.
    #[cfg_attr(feature = "serde", serde(skip))]
    peak_shape: PhantomData<fn() -> P>,
}

// manual impls to avoid `P: Copy`, which isn't necessary with PhantomData.
impl<P> Copy for IterativeRefinement<P> {}

impl<P> Clone for IterativeRefinement<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, P> Fit<T, P> for IterativeRefinement<P>
where
    T: Float,
    P: PeakShape<T> + ThreePointStencil<T>,
{
    fn fit<S>(&self, spectrum: &Spectrum1D<S>, peaks: &[Peak]) -> Vec<P>
    where
        S: Array1D<Elem = T>,
    {
        let reduced_spectrum = ReducedSpectrum::new(spectrum, peaks);
        let mut stencils = reduced_spectrum.stencils().collect::<Vec<_>>();
        let mut peak_shapes = stencils
            .iter()
            .map(|stencil| P::estimate_parameters(stencil.shifts, stencil.intensities))
            .collect::<Vec<_>>();
        for _ in 0..self.iterations {
            let superpositions = reduced_spectrum
                .shifts
                .iter()
                .copied()
                .superposition(&peak_shapes);
            let ratios = reduced_spectrum
                .intensities
                .iter()
                .zip(superpositions)
                .map(|(&intensity, superposition)| intensity / superposition)
                .collect::<Vec<_>>();
            for (stencil, ratios) in stencils.iter_mut().zip(ratios.chunks(3)) {
                stencil.intensities.0 = stencil.intensities.0 * ratios[0];
                stencil.intensities.1 = stencil.intensities.1 * ratios[1];
                stencil.intensities.2 = stencil.intensities.2 * ratios[2];
                stencil.mirror_shoulder();
            }
            for (peak_shape, stencil) in peak_shapes.iter_mut().zip(stencils.iter()) {
                *peak_shape = P::estimate_parameters(stencil.shifts, stencil.intensities);
            }
        }
        peak_shapes.retain(|peak_shape| {
            peak_shape.is_valid() && peak_shape.is_significant(crate::precision())
        });

        peak_shapes
    }
}

#[cfg(feature = "rayon")]
impl<T, P> ParFit<T, P> for IterativeRefinement<P>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + ThreePointStencil<T> + Send + Sync,
{
    fn par_fit<S>(&self, spectrum: &Spectrum1D<S>, peaks: &[Peak]) -> Vec<P>
    where
        S: Array1D<Elem = T>,
    {
        let reduced_spectrum = ReducedSpectrum::new(spectrum, peaks);
        let mut stencils = reduced_spectrum.stencils().collect::<Vec<_>>();
        let mut peak_shapes = stencils
            .iter()
            .map(|stencil| P::estimate_parameters(stencil.shifts, stencil.intensities))
            .collect::<Vec<_>>();
        for _ in 0..self.iterations {
            let superpositions = reduced_spectrum
                .shifts
                .par_iter()
                .copied()
                .superposition(&peak_shapes)
                .collect::<Vec<_>>();
            let ratios = reduced_spectrum
                .intensities
                .iter()
                .zip(superpositions)
                .map(|(&intensity, superposition)| intensity / superposition)
                .collect::<Vec<_>>();
            for (stencil, ratios) in stencils.iter_mut().zip(ratios.chunks(3)) {
                stencil.intensities.0 = stencil.intensities.0 * ratios[0];
                stencil.intensities.1 = stencil.intensities.1 * ratios[1];
                stencil.intensities.2 = stencil.intensities.2 * ratios[2];
                stencil.mirror_shoulder();
            }
            peak_shapes
                .par_iter_mut()
                .zip(stencils.par_iter())
                .for_each(|(peak_shape, stencil)| {
                    *peak_shape = P::estimate_parameters(stencil.shifts, stencil.intensities);
                });
        }
        peak_shapes.retain(|peak_shape| {
            peak_shape.is_valid() && peak_shape.is_significant(crate::precision())
        });

        peak_shapes
    }
}

impl<P> Default for IterativeRefinement<P> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<P> IterativeRefinement<P> {
    /// Creates a new `IterativeRefinement` fitter with the specified number of
    /// iterations.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            peak_shape: PhantomData,
        }
    }
}
