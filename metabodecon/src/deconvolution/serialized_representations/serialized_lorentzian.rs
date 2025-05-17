use crate::deconvolution::Lorentzian;
use serde::{Deserialize, Serialize};

/// Form of [`Lorentzian`] used for serialization/deserialization.
///
/// [`Lorentzian`] internally uses transformed parameters (`sfhw`, `hw2`) to
/// improve computational efficiency, as this representation reduces redundant
/// calculations. This obscures the geometric meaning of the parameters. The
/// original parameters `(sf, hw)` directly correspond to the peak's shape:
/// - `sf` is the scale factor, controlling peak's height.
/// - `hw` is the half-width at half-maximum, defining peak width.
///
/// Since the transformed representation is specific to internal computations,
/// this serialized form retains the conventional parameters to ensure clarity
/// and compatibility with external applications.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(rename = "Lorentzian")]
pub(crate) struct SerializedLorentzian {
    /// Scale factor.
    sf: f64,
    /// Half-width.
    hw: f64,
    /// Maximum position.
    maxp: f64,
}

impl<L: AsRef<Lorentzian>> From<L> for SerializedLorentzian {
    fn from(value: L) -> Self {
        let lorentzian = value.as_ref();

        Self {
            sf: lorentzian.sf(),
            hw: lorentzian.hw(),
            maxp: lorentzian.maxp(),
        }
    }
}

impl From<SerializedLorentzian> for Lorentzian {
    fn from(value: SerializedLorentzian) -> Self {
        Lorentzian::new(value.sf * value.hw, value.hw.powi(2), value.maxp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn lorentzian_conversion_forward() {
        let initial = SerializedLorentzian {
            sf: 12.5,
            hw: 0.25,
            maxp: 5.0,
        };
        let recovered = SerializedLorentzian::from(Lorentzian::from(initial));
        assert_approx_eq!(f64, initial.sf, recovered.sf);
        assert_approx_eq!(f64, initial.hw, recovered.hw);
        assert_approx_eq!(f64, initial.maxp, recovered.maxp);
    }

    #[test]
    fn lorentzian_conversion_backward() {
        let initial = Lorentzian::new(12.5, 0.25, 5.0);
        let recovered = Lorentzian::from(SerializedLorentzian::from(initial));
        assert_approx_eq!(f64, initial.sfhw(), recovered.sfhw());
        assert_approx_eq!(f64, initial.hw2(), recovered.hw2());
        assert_approx_eq!(f64, initial.maxp(), recovered.maxp());
    }
}
