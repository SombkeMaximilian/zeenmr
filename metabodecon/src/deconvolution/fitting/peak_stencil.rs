use crate::deconvolution::peak_selection::Peak;
use crate::spectrum::Spectrum;

/// Data structure to store the points for approximating a peak with a
/// [`Lorentzian`].
///
/// [`Lorentzian`]: crate::deconvolution::lorentzian::Lorentzian
#[derive(Debug)]
pub(crate) struct PeakStencil {
    /// Left chemical shift data point in ppm.
    left_chemical_shift: f64,
    /// Center chemical shift data point in ppm.
    center_chemical_shift: f64,
    /// Right chemical shift data point in ppm.
    right_chemical_shift: f64,
    /// Left intensity data point.
    left_intensity: f64,
    /// Center intensity data point.
    center_intensity: f64,
    /// Right intensity data point.
    right_intensity: f64,
}

impl PeakStencil {
    /// Extracts the chemical shifts and intensities of the peak from the
    /// spectrum and constructs `PeakStencil`s from them.
    pub(crate) fn new(spectrum: &Spectrum, peak: &Peak) -> Self {
        Self {
            left_chemical_shift: spectrum.chemical_shifts()[peak.left()],
            center_chemical_shift: spectrum.chemical_shifts()[peak.center()],
            right_chemical_shift: spectrum.chemical_shifts()[peak.right()],
            left_intensity: spectrum.intensities()[peak.left()],
            center_intensity: spectrum.intensities()[peak.center()],
            right_intensity: spectrum.intensities()[peak.right()],
        }
    }

    /// Internal helper function to create a `PeakStencil` from the given data
    /// for testing purposes.
    #[cfg(test)]
    pub(crate) fn from_data(
        left_chemical_shift: f64,
        center_chemical_shift: f64,
        right_chemical_shift: f64,
        left_intensity: f64,
        center_intensity: f64,
        right_intensity: f64,
    ) -> Self {
        Self {
            left_chemical_shift,
            center_chemical_shift,
            right_chemical_shift,
            left_intensity,
            center_intensity,
            right_intensity,
        }
    }

    /// Returns the left chemical shift value.
    pub(crate) fn x_1(&self) -> f64 {
        self.left_chemical_shift
    }

    /// Returns the center chemical shift value.
    pub(crate) fn x_2(&self) -> f64 {
        self.center_chemical_shift
    }

    /// Returns the right chemical shift value.
    pub(crate) fn x_3(&self) -> f64 {
        self.right_chemical_shift
    }

    /// Returns the left intensity value.
    pub(crate) fn y_1(&self) -> f64 {
        self.left_intensity
    }

    /// Returns the center intensity value.
    pub(crate) fn y_2(&self) -> f64 {
        self.center_intensity
    }

    /// Returns the right intensity value.
    pub(crate) fn y_3(&self) -> f64 {
        self.right_intensity
    }

    /// Sets the left intensity value.
    pub(crate) fn set_y_1(&mut self, y_1: f64) {
        self.left_intensity = y_1;
    }

    /// Sets the center intensity value.
    pub(crate) fn set_y_2(&mut self, y_2: f64) {
        self.center_intensity = y_2;
    }

    /// Sets the right intensity value.
    pub(crate) fn set_y_3(&mut self, y_3: f64) {
        self.right_intensity = y_3;
    }

    /// Mirrors the left/right data points onto the right/left data point if the
    /// intensities are ascending/descending from left to center to right.
    ///
    /// For cases where the peak is a shoulder of another, larger peak, it is
    /// required to make an assumption about the shape of the peak. This method
    /// assumes that the peak is symmetric about the center data point and
    /// mirrors the data point for which the intensity is lower than the center
    /// data point onto the other side. This is done to ensure that the 3-point
    /// stencil is working with data that has a peak-like shape.
    pub(crate) fn mirror_shoulder(&mut self) {
        let increasing = self.left_intensity <= self.center_intensity
            && self.center_intensity <= self.right_intensity;
        let decreasing = self.left_intensity >= self.center_intensity
            && self.center_intensity >= self.right_intensity;
        match (increasing, decreasing) {
            (true, _) => {
                self.right_intensity = self.left_intensity;
                self.right_chemical_shift =
                    2.0 * self.center_chemical_shift - self.left_chemical_shift;
            }
            (_, true) => {
                self.left_intensity = self.right_intensity;
                self.left_chemical_shift =
                    2.0 * self.center_chemical_shift - self.right_chemical_shift;
            }
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn accessors() {
        let peak = PeakStencil {
            left_chemical_shift: 1.0,
            center_chemical_shift: 2.0,
            right_chemical_shift: 3.0,
            left_intensity: 1.0,
            center_intensity: 2.0,
            right_intensity: 3.0,
        };
        assert_approx_eq!(f64, peak.x_1(), 1.0);
        assert_approx_eq!(f64, peak.x_2(), 2.0);
        assert_approx_eq!(f64, peak.x_3(), 3.0);
        assert_approx_eq!(f64, peak.y_1(), 1.0);
        assert_approx_eq!(f64, peak.y_2(), 2.0);
        assert_approx_eq!(f64, peak.y_3(), 3.0);
    }

    #[test]
    fn mutators() {
        let mut peak = PeakStencil {
            left_chemical_shift: 1.0,
            center_chemical_shift: 2.0,
            right_chemical_shift: 3.0,
            left_intensity: 1.0,
            center_intensity: 2.0,
            right_intensity: 3.0,
        };
        peak.set_y_1(3.0);
        peak.set_y_2(2.0);
        peak.set_y_3(1.0);
        assert_approx_eq!(f64, peak.y_1(), 3.0);
        assert_approx_eq!(f64, peak.y_2(), 2.0);
        assert_approx_eq!(f64, peak.y_3(), 1.0);
    }

    #[test]
    fn mirror_shoulder() {
        let mut peak = PeakStencil {
            left_chemical_shift: 1.0,
            center_chemical_shift: 2.0,
            right_chemical_shift: 3.0,
            left_intensity: 1.0,
            center_intensity: 2.0,
            right_intensity: 3.0,
        };
        peak.mirror_shoulder();
        assert_approx_eq!(f64, peak.x_1(), 1.0);
        assert_approx_eq!(f64, peak.x_2(), 2.0);
        assert_approx_eq!(f64, peak.x_3(), 3.0);
        assert_approx_eq!(f64, peak.y_1(), 1.0);
        assert_approx_eq!(f64, peak.y_2(), 2.0);
        assert_approx_eq!(f64, peak.y_3(), 1.0);

        let mut peak = PeakStencil {
            left_chemical_shift: 1.0,
            center_chemical_shift: 2.0,
            right_chemical_shift: 4.0,
            left_intensity: 3.0,
            center_intensity: 2.0,
            right_intensity: 1.0,
        };
        peak.mirror_shoulder();
        assert_approx_eq!(f64, peak.x_1(), 0.0);
        assert_approx_eq!(f64, peak.x_2(), 2.0);
        assert_approx_eq!(f64, peak.x_3(), 4.0);
        assert_approx_eq!(f64, peak.y_1(), 1.0);
        assert_approx_eq!(f64, peak.y_2(), 2.0);
        assert_approx_eq!(f64, peak.y_3(), 1.0);
    }
}
