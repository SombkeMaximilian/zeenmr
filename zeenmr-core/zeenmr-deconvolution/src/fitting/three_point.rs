use crate::fitting::Fit;
use crate::peak_finding::Peak;
use num_traits::Float;
use std::marker::PhantomData;
use zeenmr_peakshape::{BatchSuperposition, Gaussian, Lorentzian, PeakShape};
use zeenmr_spectrum::SpectrumView1D;

#[cfg(feature = "rayon")]
use crate::fitting::ParFit;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use zeenmr_peakshape::ParBatchSuperposition;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for estimating parameters of a peak shape from three data points.
///
/// The three points are expected to bracket a peak, with `x[1]` the apex sample
/// and `x[0]`, `x[2]` placed at approximately equal distances on either side.
/// The implementations are conditioned for this near-symmetric spacing. It
/// cannot be assumed that the flanking samples fall at any particular height on
/// the profile since overlapping peaks distort the lineshape and displace
/// features such as the inflection points.
pub trait ThreePointStencil<T> {
    /// Estimate the parameters of the peak shape from three data points.
    fn estimate_parameters(x: [T; 3], y: [T; 3]) -> Self;
}

impl<T> ThreePointStencil<T> for Gaussian<T>
where
    T: Float,
{
    fn estimate_parameters(x: [T; 3], y: [T; 3]) -> Self {
        let two = T::one() + T::one();

        let left_spacing = x[1] - x[0];
        let right_spacing = x[2] - x[1];
        let total_spacing = left_spacing + right_spacing;

        let left_log_drop = (y[0] / y[1]).log2();
        let right_log_drop = (y[2] / y[1]).log2();

        let left_slope = left_log_drop / left_spacing;
        let right_slope = right_log_drop / right_spacing;

        let curvature = (left_slope + right_slope) / total_spacing;
        let apex_slope = (left_spacing * right_slope - right_spacing * left_slope) / total_spacing;

        let apex_offset = apex_slope / (two * curvature);
        let center = x[1] - apex_offset;
        let amp = y[1] * (-curvature * apex_offset * apex_offset).exp2();

        Gaussian::new(amp, curvature, center)
    }
}

impl<T> ThreePointStencil<T> for Lorentzian<T>
where
    T: Float,
{
    fn estimate_parameters(x: [T; 3], y: [T; 3]) -> Self {
        let half = T::one() / (T::one() + T::one());
        let two = T::one() + T::one();

        let left_spacing = x[1] - x[0];
        let right_spacing = x[2] - x[1];
        let total_spacing = left_spacing + right_spacing;
        let left_rise = (y[1] / y[0] - T::one()) / y[1];
        let right_rise = (y[1] / y[2] - T::one()) / y[1];
        let left_slope = left_rise / left_spacing;
        let right_slope = right_rise / right_spacing;
        let curvature = (left_slope + right_slope) / total_spacing;
        let apex_slope = (left_spacing * right_slope - right_spacing * left_slope) / total_spacing;
        let apex_offset = apex_slope / (two * curvature);
        let center = x[1] - apex_offset;

        let apex_dist2 = apex_offset * apex_offset;
        let left_scale2 = (y[0] * (x[0] - center).powi(2) - y[1] * apex_dist2) / (y[1] - y[0]);
        let right_scale2 = (y[1] * apex_dist2 - y[2] * (x[2] - center).powi(2)) / (y[2] - y[1]);
        let scale2 = half * (left_scale2 + right_scale2);

        let amp_scale = y[1] * (scale2 + apex_dist2);

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
    fn new(spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Self {
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
    /// While this might result in the chemical shift of the stencil getting
    /// desynced from the reduced spectrum by 1~2 data points, the spectrum is
    /// assumed to be continuous enough to where this does not cause an issue.
    /// Empirically, this seems to hold.
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
#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreePoint<P> {
    /// Number of iterations to refine the peak parameters.
    pub iterations: usize,
    /// Marker for the peak shape type.
    #[cfg_attr(feature = "serde", serde(skip))]
    peak_shape: PhantomData<fn() -> P>,
}

// manual impls to avoid `P: Copy`, which isn't necessary with PhantomData.
impl<P> Copy for ThreePoint<P> {}

impl<P> Clone for ThreePoint<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, P> Fit<T, P> for ThreePoint<P>
where
    T: Float,
    P: PeakShape<T> + ThreePointStencil<T>,
{
    type Error = std::convert::Infallible;

    fn fit(&self, spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Result<Vec<P>, Self::Error> {
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
            let mut ratios = peak_shapes.superposition(reduced.shifts.as_flattened());
            ratios
                .iter_mut()
                .zip(reduced.intensities.as_flattened().iter())
                .for_each(|(sup, &raw)| *sup = raw / *sup);
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

        Ok(peak_shapes)
    }
}

#[cfg(feature = "rayon")]
impl<T, P> ParFit<T, P> for ThreePoint<P>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + ThreePointStencil<T> + Send + Sync,
{
    type Error = std::convert::Infallible;

    fn par_fit(
        &self,
        spectrum: SpectrumView1D<T, T>,
        peaks: &[Peak],
    ) -> Result<Vec<P>, Self::Error> {
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
            let mut ratios = peak_shapes.par_superposition(reduced.shifts.as_flattened());
            ratios
                .iter_mut()
                .zip(reduced.intensities.as_flattened().iter())
                .for_each(|(superposition, &intensity)| {
                    *superposition = intensity / *superposition
                });
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

        Ok(peak_shapes)
    }
}

impl<P> Default for ThreePoint<P> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<P> ThreePoint<P> {
    /// Creates a new `ThreePoint` fitter.
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
/// Panics in debug builds if the loop components have differing lengths.
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

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::{ApproxEq, assert_approx_eq};

    fn recover_peak_shape<T, P>(peak_shape: P)
    where
        T: Float + ApproxEq + std::fmt::Debug,
        P: PeakShape<T> + ThreePointStencil<T> + std::fmt::Debug,
    {
        let stencil_points = [
            peak_shape.center() - peak_shape.half_width(),
            peak_shape.center(),
            peak_shape.center() + peak_shape.half_width(),
        ];
        let stencil_values = stencil_points.map(|x| peak_shape.evaluate(x));
        let recovered = P::estimate_parameters(stencil_points, stencil_values);

        assert_approx_eq!(T, peak_shape.center(), recovered.center());
        assert_approx_eq!(T, peak_shape.half_width(), recovered.half_width());
        assert_approx_eq!(T, peak_shape.maximum(), recovered.maximum());
        assert_approx_eq!(T, peak_shape.area(), recovered.area());
    }

    #[test]
    fn lorentzian_stencil() {
        recover_peak_shape(Lorentzian::new(1_f32, 1_f32, 0_f32));
        recover_peak_shape(Lorentzian::new(1_f64, 1_f64, 0_f64));
    }

    #[test]
    fn gaussian_stencil() {
        recover_peak_shape(Gaussian::new(1_f32, -1_f32, 0_f32));
        recover_peak_shape(Gaussian::new(1_f64, -1_f64, 0_f64));
    }
}
