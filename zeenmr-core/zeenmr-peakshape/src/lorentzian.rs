use crate::Evaluate;
use num_traits::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lorentzian<F: Float> {
    /// Numerator of the Lorentzian function: A * HWHM.
    scaled_hwhm: F,
    /// Half-width at half-maximum (HWHM) squared.
    hwhm2: F,
    /// Center of the Lorentzian function.
    center: F,
}

impl<F: Float> Evaluate for Lorentzian<F> {
    type Scalar = F;

    fn evaluate(&self, at: Self::Scalar) -> Self::Scalar {
        self.scaled_hwhm / (self.hwhm2 + (at - self.center).powi(2))
    }
}

impl<F: Float> Lorentzian<F> {
    pub fn new(scale: F, hwhm: F, center: F) -> Self {
        Self {
            scaled_hwhm: scale * hwhm,
            hwhm2: hwhm.powi(2),
            center,
        }
    }
}
