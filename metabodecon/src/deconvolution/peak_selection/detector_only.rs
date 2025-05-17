use crate::Result;
use crate::deconvolution::peak_selection::{
    Detector, Peak, SelectionSettings, Selector, second_derivative,
};

/// Detects peaks in a spectrum without any filtering.
///
/// Peaks are detected using the curvature of the signal through the second
/// derivative.
///
/// Optionally, regions to be ignored can be provided. Peaks within these
/// regions are also filtered out.
#[derive(Debug)]
pub(crate) struct DetectorOnly;

impl Selector for DetectorOnly {
    fn select_peaks(
        &self,
        intensities: &[f64],
        signal_boundaries: (usize, usize),
        ignore_regions: Option<&[(usize, usize)]>,
    ) -> Result<Vec<Peak>> {
        let second_derivative = second_derivative(intensities);
        let detector = Detector::new(&second_derivative);
        let mut peaks = detector.detect_peaks()?;
        peaks.retain(|peak| {
            peak.left() >= signal_boundaries.0 && peak.right() <= signal_boundaries.1
        });
        if let Some(ignore_regions) = ignore_regions {
            peaks.retain(|peak| {
                !ignore_regions.iter().any(|(start, end)| {
                    (peak.left() >= *start && peak.left() < *end)
                        || (peak.right() >= *start && peak.right() < *end)
                })
            });
        }

        Ok(peaks)
    }

    fn settings(&self) -> SelectionSettings {
        SelectionSettings::DetectorOnly
    }
}

impl DetectorOnly {
    /// Creates a new `DetectorOnly`.
    pub(crate) fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};

    #[test]
    fn thread_safety() {
        assert_send!(DetectorOnly);
        assert_sync!(DetectorOnly);
    }
}
