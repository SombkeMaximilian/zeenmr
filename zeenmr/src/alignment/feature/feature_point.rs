use crate::deconvolution::Lorentzian;

#[derive(Debug)]
pub(crate) struct FeaturePoint {
    sf: f64,
    hw: f64,
    maxp: f64,
}

impl AsRef<FeaturePoint> for FeaturePoint {
    fn as_ref(&self) -> &FeaturePoint {
        self
    }
}

impl<T: AsRef<Lorentzian>> From<T> for FeaturePoint {
    fn from(value: T) -> Self {
        let lorentzian = value.as_ref();

        Self {
            sf: lorentzian.sf(),
            hw: lorentzian.hw(),
            maxp: lorentzian.maxp(),
        }
    }
}

impl<T: AsRef<FeaturePoint>> From<T> for Lorentzian {
    fn from(value: T) -> Self {
        let feature = value.as_ref();

        Lorentzian::new(
            feature.sf() * feature.hw(),
            feature.hw().powi(2),
            feature.maxp(),
        )
    }
}

impl FeaturePoint {
    pub(crate) fn sf(&self) -> f64 {
        self.sf
    }

    pub(crate) fn hw(&self) -> f64 {
        self.hw
    }

    pub(crate) fn maxp(&self) -> f64 {
        self.maxp
    }

    pub(crate) fn set_maxp(&mut self, maxp: f64) {
        self.maxp = maxp;
    }

    pub(crate) fn max(&self) -> f64 {
        self.sf / self.hw
    }
}
