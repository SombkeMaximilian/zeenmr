use crate::Evaluate;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lorentzian {
    /// Numerator of the Lorentzian function: A * HWHM.
    scaled_hwhm: f64,
    /// Half-width at half-maximum (HWHM) squared.
    hwhm2: f64,
    /// Center of the Lorentzian function.
    center: f64,
}

impl Evaluate for Lorentzian {
    fn evaluate(&self, at: f64) -> f64 {
        self.scaled_hwhm / (self.hwhm2 + (at - self.center).powi(2))
    }
}

impl Lorentzian {
    pub fn new(scale: f64, hwhm: f64, center: f64) -> Self {
        Self {
            scaled_hwhm: scale * hwhm,
            hwhm2: hwhm.powi(2),
            center,
        }
    }
}
