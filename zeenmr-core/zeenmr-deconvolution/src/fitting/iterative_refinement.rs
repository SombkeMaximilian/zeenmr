use crate::fitting::Fit;
use crate::peak_finding::Peak;
use num_traits::Float;
use std::marker::PhantomData;
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
    fn estimate_parameters(x: [T; 3], y: [T; 3]) -> Self;
}

impl<T> ThreePointStencil<T> for Lorentzian<T>
where
    T: Float,
{
    fn estimate_parameters(x: [T; 3], y: [T; 3]) -> Self {
        let half = T::one() / (T::one() + T::one());

        let numerator = x[0].powi(2) * y[0] * (y[1] - y[2])
            + x[1].powi(2) * y[1] * (y[2] - y[0])
            + x[2].powi(2) * y[2] * (y[0] - y[1]);
        let denominator =
            y[0] * y[1] * (x[0] - x[1]) + y[1] * y[2] * (x[1] - x[2]) + y[2] * y[0] * (x[2] - x[0]);
        let center = half * numerator / denominator;

        let left =
            (y[0] * (x[0] - center).powi(2) - y[1] * (x[1] - center).powi(2)) / (y[1] - y[0]);
        let right =
            (y[1] * (x[1] - center).powi(2) - y[2] * (x[2] - center).powi(2)) / (y[2] - y[1]);
        let scale2 = half * (left + right);

        let amp_scale = y[1] * (scale2 + (x[1] - center).powi(2));

        Lorentzian::new(amp_scale, scale2, center)
    }
}

/// A reduced representation of a spectrum that only contains the data points
/// that are part of peaks.
#[derive(Clone, Debug)]
struct ReducedSpectrum<T> {
    /// Chemical shifts that are part of the peaks, in ppm.
    shifts: Vec<[T; 3]>,
    /// Intensity values that are part of the peaks.
    intensities: Vec<[T; 3]>,
}

impl<T> ReducedSpectrum<T>
where
    T: Float,
{
    /// Extracts the positions and intensities of the peaks from the spectrum
    /// and constructs a `ReducedSpectrum` from them.
    fn new<S>(spectrum: &Spectrum1D<T, S>, peaks: &[Peak]) -> Self
    where
        S: Array1D<Elem = T>,
    {
        let len = spectrum.intensities().len();
        let len_as_t = T::from(len).expect("conversion from usize to T must never fail");
        let axis = spectrum.axis();
        let index_to_shift = move |index: usize| {
            let index_as_t = T::from(index).expect("conversion from usize to T must never fail");

            axis.rel_to_shift(index_as_t / len_as_t)
                .expect("index should be less than len")
        };
        let (shifts, intensities) = peaks
            .iter()
            .filter(|peak| peak.right < len)
            .map(|peak| {
                (
                    [
                        index_to_shift(peak.left),
                        index_to_shift(peak.center),
                        index_to_shift(peak.right),
                    ],
                    [
                        spectrum.intensities()[peak.left],
                        spectrum.intensities()[peak.center],
                        spectrum.intensities()[peak.right],
                    ],
                )
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
            .iter()
            .zip(self.intensities.iter())
            .map(|(shifts, intensities)| {
                let mut stencil = PeakStencil {
                    shifts: [shifts[0], shifts[1], shifts[2]],
                    intensities: [intensities[0], intensities[1], intensities[2]],
                };
                stencil.mirror_shoulder();

                stencil
            })
    }
}

#[derive(Copy, Clone, Debug)]
struct PeakStencil<T> {
    /// Chemical shifts of the three points in ppm.
    shifts: [T; 3],
    /// Intensity values of the three points.
    intensities: [T; 3],
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
        let increasing = self.intensities[0] <= self.intensities[1]
            && self.intensities[1] <= self.intensities[2];
        let decreasing = self.intensities[0] >= self.intensities[1]
            && self.intensities[1] >= self.intensities[2];
        match (increasing, decreasing) {
            (true, _) => {
                self.intensities[2] = self.intensities[0];
                self.shifts[2] = two * self.shifts[1] - self.shifts[0];
            }
            (_, true) => {
                self.intensities[0] = self.intensities[2];
                self.shifts[0] = two * self.shifts[1] - self.shifts[2];
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
    fn fit<S>(&self, spectrum: &Spectrum1D<T, S>, peaks: &[Peak]) -> Vec<P>
    where
        S: Array1D<Elem = T>,
    {
        let mut reduced = ReducedSpectrum::new(spectrum, peaks);
        let mut stencils = reduced.stencils().collect::<Vec<_>>();
        let mut peak_shapes = stencils
            .iter()
            .map(|stencil| P::estimate_parameters(stencil.shifts, stencil.intensities))
            .collect::<Vec<_>>();
        for _ in 0..self.iterations {
            prune(
                &mut reduced.shifts,
                &mut reduced.intensities,
                &mut stencils,
                &mut peak_shapes,
            );
            if peak_shapes.is_empty() {
                break;
            }
            let ratios = reduced
                .shifts
                .as_flattened()
                .iter()
                .copied()
                .superposition(&peak_shapes)
                .zip(reduced.intensities.as_flattened().iter())
                .map(|(superposition, &intensity)| intensity / superposition)
                .collect::<Vec<_>>();
            for (stencil, ratios) in stencils.iter_mut().zip(ratios.chunks(3)) {
                stencil.intensities[0] = stencil.intensities[0] * ratios[0];
                stencil.intensities[1] = stencil.intensities[1] * ratios[1];
                stencil.intensities[2] = stencil.intensities[2] * ratios[2];
                stencil.mirror_shoulder();
            }
            peak_shapes
                .iter_mut()
                .zip(stencils.iter())
                .for_each(|(p, stencil)| {
                    *p = P::estimate_parameters(stencil.shifts, stencil.intensities);
                });
        }
        peak_shapes.retain(|p| p.is_valid() && p.is_significant(crate::precision()));

        peak_shapes
    }
}

#[cfg(feature = "rayon")]
impl<T, P> ParFit<T, P> for IterativeRefinement<P>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + ThreePointStencil<T> + Send + Sync,
{
    fn par_fit<S>(&self, spectrum: &Spectrum1D<T, S>, peaks: &[Peak]) -> Vec<P>
    where
        S: Array1D<Elem = T>,
    {
        let mut reduced = ReducedSpectrum::new(spectrum, peaks);
        let mut stencils = reduced.stencils().collect::<Vec<_>>();
        let mut peak_shapes = stencils
            .iter()
            .map(|stencil| P::estimate_parameters(stencil.shifts, stencil.intensities))
            .collect::<Vec<_>>();
        for _ in 0..self.iterations {
            prune(
                &mut reduced.shifts,
                &mut reduced.intensities,
                &mut stencils,
                &mut peak_shapes,
            );
            if peak_shapes.is_empty() {
                break;
            }
            let ratios = reduced
                .shifts
                .as_flattened()
                .par_iter()
                .copied()
                .superposition(&peak_shapes)
                .zip(reduced.intensities.as_flattened().par_iter())
                .map(|(superposition, &intensity)| intensity / superposition)
                .collect::<Vec<_>>();
            for (stencil, ratios) in stencils.iter_mut().zip(ratios.chunks(3)) {
                stencil.intensities[0] = stencil.intensities[0] * ratios[0];
                stencil.intensities[1] = stencil.intensities[1] * ratios[1];
                stencil.intensities[2] = stencil.intensities[2] * ratios[2];
                stencil.mirror_shoulder();
            }
            peak_shapes
                .par_iter_mut()
                .zip(stencils.par_iter())
                .for_each(|(p, stencil)| {
                    *p = P::estimate_parameters(stencil.shifts, stencil.intensities);
                });
        }
        peak_shapes.retain(|p| p.is_valid() && p.is_significant(crate::precision()));

        peak_shapes
    }
}

impl<P> Default for IterativeRefinement<P> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<P> IterativeRefinement<P> {
    /// Creates a new `IterativeRefinement` fitter.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            peak_shape: PhantomData,
        }
    }
}

/// Prunes the loop components by whether the associated peak shape is
/// degenerate.
///
/// # Panics
///
/// Panics in debug mode if the loop components have differing lengths.
fn prune<T, P>(
    shifts: &mut Vec<[T; 3]>,
    intensities: &mut Vec<[T; 3]>,
    stencils: &mut Vec<PeakStencil<T>>,
    peak_shapes: &mut Vec<P>,
) where
    T: Float,
    P: PeakShape<T>,
{
    debug_assert_eq!(shifts.len(), intensities.len());
    debug_assert_eq!(intensities.len(), stencils.len());
    debug_assert_eq!(stencils.len(), peak_shapes.len());
    let mut curr = 0;
    while curr < peak_shapes.len() {
        if peak_shapes[curr].is_valid() && peak_shapes[curr].is_significant(crate::precision()) {
            curr += 1;
        } else {
            shifts.swap_remove(curr);
            intensities.swap_remove(curr);
            stencils.swap_remove(curr);
            peak_shapes.swap_remove(curr);
        }
    }
}
