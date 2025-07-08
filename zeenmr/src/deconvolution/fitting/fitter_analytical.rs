use crate::deconvolution::fitting::{Fitter, FittingSettings, PeakStencil, ReducedSpectrum};
use crate::deconvolution::lorentzian::Lorentzian;
use crate::deconvolution::peak_selection::Peak;
use crate::spectrum::Spectrum;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fitting algorithm based on the analytical solution of a system of equations
/// using a 3-point peak stencil.
#[derive(Debug)]
pub(crate) struct FitterAnalytical {
    /// The number of iterations to refine the Lorentzian parameters.
    iterations: usize,
}

impl Fitter for FitterAnalytical {
    /// Fits a set of Lorentzians to the spectrum using the given peaks.
    fn fit_lorentzian(&self, spectrum: &Spectrum, peaks: &[Peak]) -> Vec<Lorentzian> {
        let reduced_spectrum = ReducedSpectrum::new(spectrum, peaks);
        let mut peak_data = peaks
            .iter()
            .map(|peak| {
                let mut stencil = PeakStencil::new(spectrum, peak);
                stencil.mirror_shoulder();
                stencil
            })
            .collect::<Vec<_>>();
        let mut lorentzians = peak_data
            .iter()
            .map(|peak| {
                let maxp = Self::maximum_position(peak);
                let hw2 = Self::half_width2(peak, maxp);
                let sfhw = Self::scale_factor_half_width(peak, maxp, hw2);
                Lorentzian::new(sfhw, hw2, maxp)
            })
            .collect::<Vec<_>>();

        for _ in 0..self.iterations {
            let superpositions =
                Lorentzian::superposition_vec(reduced_spectrum.chemical_shifts(), &lorentzians);
            let ratios = reduced_spectrum
                .intensities()
                .iter()
                .zip(superpositions.iter())
                .map(|(intensity, superposition)| intensity / superposition)
                .collect::<Vec<_>>();
            peak_data
                .iter_mut()
                .zip(ratios.chunks(3))
                .for_each(|(stencil, ratio)| {
                    stencil.set_y_1(stencil.y_1() * ratio[0]);
                    stencil.set_y_2(stencil.y_2() * ratio[1]);
                    stencil.set_y_3(stencil.y_3() * ratio[2]);
                    stencil.mirror_shoulder();
                });
            lorentzians
                .iter_mut()
                .zip(peak_data.iter())
                .for_each(|(lorentzian, stencil)| {
                    let maxp = Self::maximum_position(stencil);
                    let hw2 = Self::half_width2(stencil, maxp);
                    let sfhw = Self::scale_factor_half_width(stencil, maxp, hw2);
                    lorentzian.set_parameters(sfhw, hw2, maxp);
                });
        }
        lorentzians.retain(|lorentzian| {
            lorentzian.sfhw() > crate::CHECK_PRECISION && lorentzian.hw2() > crate::CHECK_PRECISION
        });

        lorentzians
    }

    /// Fits a set of Lorentzians to the spectrum using the given peaks in
    /// parallel.
    #[cfg(feature = "parallel")]
    fn par_fit_lorentzian(&self, spectrum: &Spectrum, peaks: &[Peak]) -> Vec<Lorentzian> {
        let reduced_spectrum = ReducedSpectrum::new(spectrum, peaks);
        let mut peak_data = peaks
            .iter()
            .map(|peak| {
                let mut stencil = PeakStencil::new(spectrum, peak);
                stencil.mirror_shoulder();
                stencil
            })
            .collect::<Vec<_>>();
        let mut lorentzians = peak_data
            .iter()
            .map(|peak| {
                let maxp = Self::maximum_position(peak);
                let hw2 = Self::half_width2(peak, maxp);
                let sfhw = Self::scale_factor_half_width(peak, maxp, hw2);
                Lorentzian::new(sfhw, hw2, maxp)
            })
            .collect::<Vec<_>>();

        for _ in 0..self.iterations {
            let superpositions =
                Lorentzian::par_superposition_vec(reduced_spectrum.chemical_shifts(), &lorentzians);
            let ratios = reduced_spectrum
                .intensities()
                .iter()
                .zip(superpositions.iter())
                .map(|(intensity, superposition)| intensity / superposition)
                .collect::<Vec<_>>();
            peak_data
                .iter_mut()
                .zip(ratios.chunks(3))
                .for_each(|(stencil, ratio)| {
                    stencil.set_y_1(stencil.y_1() * ratio[0]);
                    stencil.set_y_2(stencil.y_2() * ratio[1]);
                    stencil.set_y_3(stencil.y_3() * ratio[2]);
                    stencil.mirror_shoulder();
                });
            lorentzians
                .par_iter_mut()
                .zip(peak_data.par_iter())
                .for_each(|(lorentzian, stencil)| {
                    let maxp = Self::maximum_position(stencil);
                    let hw2 = Self::half_width2(stencil, maxp);
                    let sfhw = Self::scale_factor_half_width(stencil, maxp, hw2);
                    lorentzian.set_parameters(sfhw, hw2, maxp);
                });
        }
        lorentzians.retain(|lorentzian| {
            lorentzian.sfhw() > crate::CHECK_PRECISION && lorentzian.hw2() > crate::CHECK_PRECISION
        });

        lorentzians
    }

    fn settings(&self) -> FittingSettings {
        FittingSettings::Analytical {
            iterations: self.iterations,
        }
    }
}

impl FitterAnalytical {
    /// Constructs a new `FitterAnalytical` with the given number of iterations.
    pub(crate) fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    /// Internal helper function to analytically compute the maximum position of
    /// the peak in ppm by solving the system of 3 equations.
    fn maximum_position(p: &PeakStencil) -> f64 {
        let numerator = p.x_1().powi(2) * p.y_1() * (p.y_2() - p.y_3())
            + p.x_2().powi(2) * p.y_2() * (p.y_3() - p.y_1())
            + p.x_3().powi(2) * p.y_3() * (p.y_1() - p.y_2());
        let divisor = 2.0 * (p.x_1() - p.x_2()) * p.y_1() * p.y_2()
            + 2.0 * (p.x_2() - p.x_3()) * p.y_2() * p.y_3()
            + 2.0 * (p.x_3() - p.x_1()) * p.y_3() * p.y_1();
        numerator / divisor
    }

    /// Internal helper function to analytically compute the half width at half
    /// maximum of the peak in ppm^2 by solving the system of 3 equations.
    fn half_width2(p: &PeakStencil, maxp: f64) -> f64 {
        let left = (p.y_1() * (p.x_1() - maxp).powi(2) - p.y_2() * (p.x_2() - maxp).powi(2))
            / (p.y_2() - p.y_1());
        let right = (p.y_2() * (p.x_2() - maxp).powi(2) - p.y_3() * (p.x_3() - maxp).powi(2))
            / (p.y_3() - p.y_2());
        ((left + right) / 2.0).max(f64::EPSILON)
    }

    /// Internal helper function to analytically compute the scale factor times
    /// the half width at half maximum of the peak in ppm^2 by solving the
    /// system of 3 equations.
    fn scale_factor_half_width(p: &PeakStencil, maxp: f64, hw2: f64) -> f64 {
        p.y_2() * (hw2 + (p.x_2() - maxp).powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};
    use float_cmp::assert_approx_eq;

    #[test]
    fn thread_safety() {
        assert_send!(FitterAnalytical);
        assert_sync!(FitterAnalytical);
    }

    #[test]
    fn approximations() {
        let peak = PeakStencil::from_data(4.0, 8.0, 12.0, 5.0, 10.0, 5.0);
        let maxp = FitterAnalytical::maximum_position(&peak);
        let hw2 = FitterAnalytical::half_width2(&peak, maxp);
        let sfhw = FitterAnalytical::scale_factor_half_width(&peak, maxp, hw2);
        assert_approx_eq!(f64, maxp, 8.0);
        assert_approx_eq!(f64, hw2.sqrt(), 4.0);
        assert_approx_eq!(f64, sfhw / hw2.sqrt(), 40.0);
    }
}
