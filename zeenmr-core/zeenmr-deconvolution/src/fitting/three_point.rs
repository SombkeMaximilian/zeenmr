use crate::fitting::Fit;
use crate::peak_finding::Peak;
use crate::util::noise::{GaussianNoise, NoiseLevel};
use num_traits::Float;
use std::marker::PhantomData;
use zeenmr_peakshape::batch_superposition::{Standard, SuperpositionKernel};
use zeenmr_peakshape::{DefaultSupport, Gaussian, Lorentzian, PeakShape};
use zeenmr_spectrum::SpectrumView1D;
use zeenmr_spectrum::axis::range::{FiniteBounds, RelativeRange};
use zeenmr_spectrum::dimension::DimIndex;
use zeenmr_spectrum::intensity_array::index;

#[cfg(feature = "rayon")]
use crate::fitting::ParFit;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use zeenmr_peakshape::batch_superposition::ParSuperpositionKernel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Temporary way to make the new API work.
const SUPERPOSITION: Standard = Standard::new();

/// Minimum number of peak shapes to parallelize the update loop.
const PAR_UPDATE_THRESHOLD: usize = 1024;

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
    /// Positions that are part of the peaks, usually in scaled indices.
    positions: Vec<[T; 3]>,
    /// Intensity values that are part of the peaks.
    intensities: Vec<[T; 3]>,
    /// Grid spacing between adjacent data points.
    grid_step: T,
    /// Scaling from grid coordinates to chemical shifts.
    shift_scale: T,
    /// Shift from grid coordinates to chemical shifts.
    shift_offset: T,
}

impl<T> ReducedSpectrum<T>
where
    T: Float + Send + Sync,
{
    /// Extracts the positions and intensities of the peaks from the spectrum
    /// and constructs a `ReducedSpectrum` from them.
    fn new(spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Self {
        let intensities = spectrum.intensities();
        let intensities = intensities
            .lane_at(DimIndex(0), &index([0]))
            .expect("1D spectrum always has a first dimension");
        let len = intensities.len();
        let shift_range = spectrum
            .axis(DimIndex(0))
            .map(|axis| axis.shift_range())
            .expect("1D spectrum always has a first dimension");
        let grid_step = grid_step::<T>(len);
        let shift_offset = shift_range.start();
        let shift_scale = if len > 1 {
            let last = T::from(len - 1).expect("conversion from usize to T must never fail");

            (shift_range.end() - shift_offset) / (last * grid_step)
        } else {
            T::one()
        };
        let index_to_grid = |index: usize| {
            T::from(index).expect("conversion from usize to T must never fail") * grid_step
        };
        let (positions, intensities) = peaks
            .iter()
            .filter(|peak| peak.right < len)
            .map(|peak| {
                (
                    [
                        index_to_grid(peak.left),
                        index_to_grid(peak.center),
                        index_to_grid(peak.right),
                    ],
                    // SAFETY: every index is less than len and therefore within
                    // bounds of the intensities.
                    unsafe {
                        [
                            *(intensities.get_unchecked(peak.left)),
                            *(intensities.get_unchecked(peak.center)),
                            *(intensities.get_unchecked(peak.right)),
                        ]
                    },
                )
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        Self {
            positions,
            intensities,
            grid_step,
            shift_scale,
            shift_offset,
        }
    }

    /// Returns an iterator over the peak stencils in the reduced spectrum.
    fn stencils(&self) -> impl Iterator<Item = PeakStencil<T>> {
        self.positions
            .iter()
            .zip(self.intensities.iter())
            .map(|(shifts, intensities)| {
                let mut stencil = PeakStencil {
                    positions: [shifts[0], shifts[1], shifts[2]],
                    intensities: [intensities[0], intensities[1], intensities[2]],
                };
                stencil.mirror_shoulder();

                stencil
            })
    }

    /// Undoes the grid transformation.
    fn restore<P>(&self, peak_shapes: &mut [P])
    where
        P: PeakShape<T>,
    {
        let map = |p: &mut P| p.affine_transform(self.shift_offset, self.shift_scale);
        peak_shapes.iter_mut().for_each(map);
    }
}

#[derive(Copy, Clone, Debug)]
struct PeakStencil<T> {
    /// Positions of the three points in ppm, usually in scaled indices.
    positions: [T; 3],
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
    /// While this might result in the position of the stencil getting desynced
    /// from the reduced spectrum by 1~2 data points, the spectrum is assumed
    /// to be continuous enough to where this does not cause an issue.
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
                self.positions[2] = two * self.positions[1] - self.positions[0];
            }
            (_, true) => {
                self.intensities[0] = self.intensities[2];
                self.positions[0] = two * self.positions[1] - self.positions[2];
            }
            _ => {}
        };
    }
}

/// Fitting algorithm based on the analytical solution of a system of equations
/// using a 3-point peak stencil.
#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreePoint<P, N> {
    /// Number of iterations to refine the peak parameters.
    pub iterations: usize,
    /// Type determining the noise level of the input spectrum.
    pub noise_finder: N,
    /// Marker for the peak shape type.
    #[cfg_attr(feature = "serde", serde(skip))]
    peak_shape: PhantomData<fn() -> P>,
}

// manual impls to avoid `P: Copy`, which isn't necessary with PhantomData.
impl<P, N> Copy for ThreePoint<P, N> where N: Copy {}

impl<P, N> Clone for ThreePoint<P, N>
where
    N: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, P, N> Fit<T, P> for ThreePoint<P, N>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + ThreePointStencil<T>,
    N: NoiseLevel<T>,
{
    type Error = std::convert::Infallible;

    fn fit(&self, spectrum: SpectrumView1D<T, T>, peaks: &[Peak]) -> Result<Vec<P>, Self::Error> {
        let mut reduced = ReducedSpectrum::new(spectrum.view(), peaks);
        let support = P::Support::from(DefaultSupport {
            width: reduced.grid_step,
            intensity: self.noise_finder.noise_level(spectrum.view()).0,
        });
        let mut stencils = reduced.stencils().collect::<Vec<_>>();
        let mut peak_shapes = stencils
            .iter()
            .map(|stencil| P::estimate_parameters(stencil.positions, stencil.intensities))
            .collect::<Vec<_>>();
        let mut superposition = vec![[T::zero(); 3]; peak_shapes.len()];
        for _ in 0..self.iterations {
            prune(
                &mut reduced.positions,
                &mut reduced.intensities,
                &mut stencils,
                &mut peak_shapes,
                &support,
            );
            if peak_shapes.is_empty() {
                break;
            }
            superposition.truncate(peak_shapes.len());
            SUPERPOSITION.accumulate(
                &peak_shapes,
                reduced.positions.as_flattened(),
                superposition.as_flattened_mut(),
            );
            for (((stencil, peak_shape), sup), raw) in stencils
                .iter_mut()
                .zip(peak_shapes.iter_mut())
                .zip(superposition.iter())
                .zip(reduced.intensities.iter())
            {
                stencil.intensities[0] = stencil.intensities[0] * raw[0] / sup[0];
                stencil.intensities[1] = stencil.intensities[1] * raw[1] / sup[1];
                stencil.intensities[2] = stencil.intensities[2] * raw[2] / sup[2];
                stencil.mirror_shoulder();
                *peak_shape = P::estimate_parameters(stencil.positions, stencil.intensities);
            }
            superposition.fill([T::zero(); 3]);
        }
        peak_shapes.retain(|p| p.is_valid() && p.is_significant(&support));
        reduced.restore(&mut peak_shapes);

        Ok(peak_shapes)
    }
}

#[cfg(feature = "rayon")]
impl<T, P, N> ParFit<T, P> for ThreePoint<P, N>
where
    T: Float + Send + Sync,
    P: PeakShape<T> + ThreePointStencil<T> + Send + Sync,
    N: NoiseLevel<T>,
{
    type Error = std::convert::Infallible;

    fn par_fit(
        &self,
        spectrum: SpectrumView1D<T, T>,
        peaks: &[Peak],
    ) -> Result<Vec<P>, Self::Error> {
        let mut reduced = ReducedSpectrum::new(spectrum.view(), peaks);
        let support = P::Support::from(DefaultSupport {
            width: reduced.grid_step,
            intensity: self.noise_finder.noise_level(spectrum.view()).0,
        });
        let mut stencils = reduced.stencils().collect::<Vec<_>>();
        let mut peak_shapes = stencils
            .iter()
            .map(|stencil| P::estimate_parameters(stencil.positions, stencil.intensities))
            .collect::<Vec<_>>();
        let mut superposition = vec![[T::zero(); 3]; peak_shapes.len()];
        for _ in 0..self.iterations {
            prune(
                &mut reduced.positions,
                &mut reduced.intensities,
                &mut stencils,
                &mut peak_shapes,
                &support,
            );
            if peak_shapes.is_empty() {
                break;
            }
            superposition.truncate(peak_shapes.len());
            SUPERPOSITION.par_accumulate(
                &peak_shapes,
                reduced.positions.as_flattened(),
                superposition.as_flattened_mut(),
            );
            stencils
                .par_iter_mut()
                .zip(peak_shapes.par_iter_mut())
                .zip(superposition.par_iter())
                .zip(reduced.intensities.par_iter())
                .with_min_len(PAR_UPDATE_THRESHOLD)
                .for_each(|(((stencil, peak_shape), sup), raw)| {
                    stencil.intensities[0] = stencil.intensities[0] * raw[0] / sup[0];
                    stencil.intensities[1] = stencil.intensities[1] * raw[1] / sup[1];
                    stencil.intensities[2] = stencil.intensities[2] * raw[2] / sup[2];
                    stencil.mirror_shoulder();
                    *peak_shape = P::estimate_parameters(stencil.positions, stencil.intensities);
                });
            superposition.fill([T::zero(); 3]);
        }
        peak_shapes.retain(|p| p.is_valid() && p.is_significant(&support));
        reduced.restore(&mut peak_shapes);

        Ok(peak_shapes)
    }
}

impl<T, P> Default for ThreePoint<P, GaussianNoise<T, RelativeRange<T>>>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            iterations: 10,
            noise_finder: GaussianNoise::default(),
            peak_shape: PhantomData,
        }
    }
}

impl<P, N> ThreePoint<P, N> {
    /// Creates a new `ThreePoint` fitter.
    pub fn new(iterations: usize, noise_level: N) -> Self {
        Self {
            iterations,
            noise_finder: noise_level,
            peak_shape: PhantomData,
        }
    }
}

/// Largest power of 2 step such that `(len - 1) * step` does not exceed
/// `clamp(len, 2^15, 2^17)`.
fn grid_step<T>(len: usize) -> T
where
    T: Float,
{
    let span = len.saturating_sub(1).max(1);
    let target = len.clamp(1 << 15, 1 << 17);
    let exp = if span <= target {
        (target / span).ilog2() as i32
    } else {
        -(span.div_ceil(target).next_power_of_two().ilog2() as i32)
    };

    (T::one() + T::one()).powi(exp)
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
    support: &P::Support,
) where
    T: Float,
    P: PeakShape<T>,
{
    debug_assert_eq!(shifts.len(), intensities.len());
    debug_assert_eq!(intensities.len(), stencils.len());
    debug_assert_eq!(stencils.len(), peak_shapes.len());

    let mut curr = 0;
    while curr < peak_shapes.len() {
        if peak_shapes[curr].is_valid() && peak_shapes[curr].is_significant(support) {
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
