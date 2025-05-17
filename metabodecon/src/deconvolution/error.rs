//! Error types for the deconvolution process.

use crate::deconvolution::{FittingSettings, SelectionSettings, SmoothingSettings};

/// An `Error` that occurred during the deconvolution process.
///
/// This type of error is generally unrecoverable and indicates a problem with
/// the input data. For example, if no peaks are detected in the input data,
/// or there are no signals in the part of the spectrum where they would be
/// expected.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Clone, Debug)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
}

impl Error {
    /// Constructs a new `Error` from the given `Kind`.
    pub fn new(kind: Kind) -> Self {
        kind.into()
    }

    /// Returns the `Kind` of the `Error`.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}

/// The kind of `Error` that can occur during the [`deconvolution`] process.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
///
/// [`deconvolution`]: crate::deconvolution
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Kind {
    /// The provided smoothing settings are invalid.
    ///
    /// Some configurations, such as a `window_size` of 0 for a moving
    /// average filter, are invalid.
    InvalidSmoothingSettings {
        /// The provided smoothing settings.
        settings: SmoothingSettings,
    },
    /// The provided peak selection settings are invalid.
    ///
    /// Some configurations, such as a negative `threshold` for a noise score
    /// filter, are invalid.
    InvalidSelectionSettings {
        /// The provided peak selection settings.
        settings: SelectionSettings,
    },
    /// The provided fitting settings are invalid.
    ///
    /// Some configurations, such as 0 `iterations` for the analytical fitting
    /// algorithm, are invalid.
    InvalidFittingSettings {
        /// The provided fitting settings.
        settings: FittingSettings,
    },
    /// The provided region to be ignored is invalid.
    ///
    /// The region must be a tuple of two finite floating point numbers, with an
    /// absolute difference is greater than a small multiple of the floating
    /// point precision.
    InvalidIgnoreRegion {
        /// The provided ignore region.
        region: (f64, f64),
    },
    /// No peaks were detected in the input data.
    ///
    /// Most of the time this will happen if the intensities of the [`Spectrum`]
    /// were not read correctly or if something went wrong during the smoothing
    /// process.
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    NoPeaksDetected,
    /// No peaks were found in the part of the spectrum where signals would be
    /// expected.
    ///
    /// This can happen if there were no signals in the signal region or if all
    /// the signals were filtered out by the peak selection algorithm if the
    /// threshold was set too high.
    EmptySignalRegion,
    /// No peaks were found in the part of the spectrum where random signals due
    /// to noise would be expected.
    ///
    /// This is an error because the deconvolution process uses the noise peaks
    /// to estimate the noise level in the spectrum and filter out peaks that
    /// are likely to be random noise.
    EmptySignalFreeRegion,
}

impl std::error::Error for Error {}

impl From<Kind> for Error {
    fn from(value: Kind) -> Self {
        Self { kind: value }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let description = match &self.kind {
            Kind::InvalidSmoothingSettings { settings } => match settings {
                SmoothingSettings::Identity => unreachable!("no smoothing is always valid"),
                SmoothingSettings::MovingAverage {
                    iterations,
                    window_size,
                } => match (*iterations == 0, *window_size <= 1) {
                    (true, true) => {
                        "iterations and window size must be greater than 0 and 1 respectively"
                            .to_string()
                    }
                    (true, false) => "iterations must be greater than 0".to_string(),
                    (false, true) => "window size must be greater than 1".to_string(),
                    (false, false) => {
                        unreachable!("valid settings falsely detected as invalid")
                    }
                },
            },
            Kind::InvalidSelectionSettings { settings } => match settings {
                SelectionSettings::DetectorOnly => unreachable!("detector only is always valid"),
                SelectionSettings::NoiseScoreFilter { threshold, .. } => {
                    match (threshold.is_finite(), *threshold <= 0.0) {
                        (false, _) => "threshold must be a finite number".to_string(),
                        (true, true) => "threshold must be greater than 0".to_string(),
                        (true, false) => unreachable!("valid settings falsely detected as invalid"),
                    }
                }
            },
            Kind::InvalidFittingSettings { settings } => match settings {
                FittingSettings::Analytical { iterations } => match *iterations == 0 {
                    true => "iterations must be greater than 0".to_string(),
                    false => unreachable!("valid settings falsely detected as invalid"),
                },
            },
            Kind::InvalidIgnoreRegion { region } => {
                match (
                    region.0.is_finite() && region.1.is_finite(),
                    f64::abs(region.0 - region.1) > crate::CHECK_PRECISION,
                ) {
                    (false, _) => format!(
                        "ignore region boundaries [{}, {}] contain non-finite values",
                        region.0, region.1
                    ),
                    (true, false) => format!(
                        "ignore region boundaries [{}, {}] are (almost) equal",
                        region.0, region.1
                    ),
                    (true, true) => unreachable!("valid ignore region falsely detected as invalid"),
                }
            }
            Kind::NoPeaksDetected => "no peaks detected in the spectrum".to_string(),
            Kind::EmptySignalRegion => {
                "no peaks found in the signal region of the spectrum".to_string()
            }
            Kind::EmptySignalFreeRegion => {
                "no peaks found in the signal-free region of the spectrum".to_string()
            }
        };

        write!(f, "{}", description)
    }
}
