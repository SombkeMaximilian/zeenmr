use crate::deconvolution::Deconvolution;
use crate::deconvolution::error::{Error, Kind};
use crate::deconvolution::fitting::{Fitter, FitterAnalytical, FittingSettings};
use crate::deconvolution::lorentzian::Lorentzian;
use crate::deconvolution::peak_selection::{
    DetectorOnly, NoiseScoreFilter, ScoringMethod, SelectionSettings, Selector,
};
use crate::deconvolution::smoothing::{Identity, MovingAverage, Smoother, SmoothingSettings};
use crate::spectrum::Spectrum;
use crate::{Result, Settings};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Deconvolution pipeline that applies smoothing, peak selection, and fitting
/// to a spectrum to deconvolute it into individual signals.
///
/// The output of the pipeline is a [`Deconvolution`] struct containing the
/// deconvoluted signals, the deconvolution settings, and the [MSE] between the
/// superposition of signals and the intensities within the signal region.
///
/// [MSE]: https://en.wikipedia.org/wiki/Mean_squared_error
///
/// # Example: Deconvoluting a [`Spectrum`]
///
/// ```
/// use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
/// use metabodecon::spectrum::Bruker;
///
/// # fn main() -> metabodecon::Result<()> {
/// // Read a spectrum in Bruker TopSpin format.
/// let path = "path/to/spectrum";
/// # let path = "../data/bruker/blood/blood_01";
/// let spectrum = Bruker::read_spectrum(
///     path,
///     // Experiment number
///     10,
///     // Processing number
///     10,
///     // Signal boundaries
///     (-2.2, 11.8),
/// )?;
///
/// // Deconvolute the spectrum.
/// let deconvoluter = Deconvoluter::default();
/// let deconvolution = deconvoluter.deconvolute_spectrum(&spectrum)?;
/// # Ok(())
/// # }
/// ```
///
/// # Example: Parallelized Deconvolution
///
/// The most expensive parts of the deconvolution process can also be performed
/// in parallel by enabling the `parallel` feature (part of the `default`
/// features). This adds [Rayon] as a dependency.
///
/// [rayon]: https://docs.rs/rayon/
///
/// ```
/// use metabodecon::deconvolution::Deconvoluter;
/// use metabodecon::spectrum::Bruker;
///
/// # fn main() -> metabodecon::Result<()> {
/// // Read a spectrum in Bruker TopSpin format.
/// let path = "path/to/spectrum";
/// # let path = "../data/bruker/blood/blood_01";
/// let spectrum = Bruker::read_spectrum(
///     path,
///     // Experiment number
///     10,
///     // Processing number
///     10,
///     // Signal boundaries
///     (-2.2, 11.8),
/// )?;
///
/// // Deconvolute the spectrum in parallel.
/// let deconvoluter = Deconvoluter::default();
/// let deconvolution = deconvoluter.par_deconvolute_spectrum(&spectrum)?;
/// # Ok(())
/// # }
/// ```
///
/// # Example: Configuring the `Deconvoluter`
///
/// `Deconvoluter` is modular and allows you to configure the smoothing, peak
/// selection, and fitting settings independently, though currently only one
/// method is implemented for each. Additionally, you can specify regions to be
/// ignored during the deconvolution. This may be useful for compounds like
/// stabilizing agents or a water signal.
///
/// ```
/// use metabodecon::deconvolution::{
///     Deconvoluter, FittingSettings, ScoringMethod, SelectionSettings, SmoothingSettings,
/// };
///
/// # fn main() -> metabodecon::Result<()> {
/// // Create a new Deconvoluter with the desired settings.
/// let mut deconvoluter = Deconvoluter::new(
///     SmoothingSettings::MovingAverage {
///         iterations: 3,
///         window_size: 3,
///     },
///     SelectionSettings::NoiseScoreFilter {
///         scoring_method: ScoringMethod::MinimumSum,
///         threshold: 5.0,
///     },
///     FittingSettings::Analytical { iterations: 20 },
/// )?;
///
/// // Add a region to ignore during deconvolution.
/// deconvoluter.add_ignore_region((4.7, 4.9))?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Deconvoluter {
    /// Smoothing settings.
    smoother: Arc<dyn Smoother<f64>>,
    /// Peak selection settings.
    selector: Arc<dyn Selector>,
    /// Fitting settings.
    fitter: Arc<dyn Fitter>,
    /// Regions to ignore during deconvolution.
    ignore_regions: Option<Vec<(f64, f64)>>,
}

impl Default for Deconvoluter {
    fn default() -> Self {
        Self::new(
            SmoothingSettings::default(),
            SelectionSettings::default(),
            FittingSettings::default(),
        )
        .unwrap()
    }
}

impl Deconvoluter {
    /// Constructs a new `Deconvoluter` with the provided settings.
    ///
    /// # Errors
    ///
    /// An error is returned if any of the deconvolution settings are invalid.
    /// For Example:
    /// - A `window_size` of 0 for a moving average filter would mean that no
    ///   smoothing is applied.
    /// - Negative `threshold`s for a noise score filter wouldn't make sense.
    /// - 0 `iterations` for the analytical fitting algorithm would mean that
    ///   the fitting algorithm doesn't do anything.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{
    ///     Deconvoluter, FittingSettings, ScoringMethod, SelectionSettings, SmoothingSettings,
    /// };
    ///
    /// let deconvoluter = Deconvoluter::new(
    ///     SmoothingSettings::MovingAverage {
    ///         iterations: 3,
    ///         window_size: 3,
    ///     },
    ///     SelectionSettings::NoiseScoreFilter {
    ///         scoring_method: ScoringMethod::MinimumSum,
    ///         threshold: 5.0,
    ///     },
    ///     FittingSettings::Analytical { iterations: 20 },
    /// );
    /// ```
    pub fn new(
        smoothing_settings: SmoothingSettings,
        selection_settings: SelectionSettings,
        fitting_settings: FittingSettings,
    ) -> Result<Self> {
        smoothing_settings.validate()?;
        selection_settings.validate()?;
        fitting_settings.validate()?;

        let smoother: Arc<dyn Smoother<f64>> = match smoothing_settings {
            SmoothingSettings::Identity => Arc::new(Identity::new()),
            SmoothingSettings::MovingAverage {
                iterations,
                window_size,
            } => Arc::new(MovingAverage::<f64>::new(iterations, window_size)),
        };
        let selector: Arc<dyn Selector> = match selection_settings {
            SelectionSettings::DetectorOnly => Arc::new(DetectorOnly::new()),
            SelectionSettings::NoiseScoreFilter {
                scoring_method,
                threshold,
            } => Arc::new(NoiseScoreFilter::new(scoring_method, threshold)),
        };
        let fitter: Arc<dyn Fitter> = match fitting_settings {
            FittingSettings::Analytical { iterations } => {
                Arc::new(FitterAnalytical::new(iterations))
            }
        };

        Ok(Self {
            smoother,
            selector,
            fitter,
            ignore_regions: None,
        })
    }

    /// Returns the smoothing settings.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, SmoothingSettings};
    ///
    /// let deconvoluter = Deconvoluter::default();
    ///
    /// match deconvoluter.smoothing_settings() {
    ///     SmoothingSettings::MovingAverage {
    ///         iterations,
    ///         window_size,
    ///     } => {
    ///         assert_eq!(iterations, 3);
    ///         assert_eq!(window_size, 3);
    ///     }
    ///     _ => panic!("Unexpected smoothing settings"),
    /// };
    /// ```
    pub fn smoothing_settings(&self) -> SmoothingSettings {
        self.smoother.settings()
    }

    /// Returns the peak selection settings.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::{Deconvoluter, ScoringMethod, SelectionSettings};
    ///
    /// let deconvoluter = Deconvoluter::default();
    ///
    /// match deconvoluter.selection_settings() {
    ///     SelectionSettings::NoiseScoreFilter {
    ///         scoring_method,
    ///         threshold,
    ///     } => {
    ///         match scoring_method {
    ///             ScoringMethod::MinimumSum => {}
    ///             _ => panic!("Unexpected scoring method"),
    ///         };
    ///         assert_approx_eq!(f64, threshold, 5.0);
    ///     }
    ///     _ => panic!("Unexpected peak selection settings"),
    /// };
    /// ```
    pub fn selection_settings(&self) -> SelectionSettings {
        self.selector.settings()
    }

    /// Returns the fitting settings.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, FittingSettings};
    ///
    /// let deconvoluter = Deconvoluter::default();
    ///
    /// match deconvoluter.fitting_settings() {
    ///     FittingSettings::Analytical { iterations } => {
    ///         assert_eq!(iterations, 10);
    ///     }
    ///     _ => panic!("Unexpected fitting settings"),
    /// };
    /// ```
    pub fn fitting_settings(&self) -> FittingSettings {
        self.fitter.settings()
    }

    /// Returns the regions to ignore during deconvolution.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Deconvoluter;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let mut deconvoluter = Deconvoluter::default();
    ///
    /// assert!(deconvoluter.ignore_regions().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn ignore_regions(&self) -> Option<&[(f64, f64)]> {
        self.ignore_regions.as_deref()
    }

    /// Sets the smoothing settings.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided smoothing settings are invalid. For
    /// example, a `window_size` of 0 for a moving average filter would mean
    /// that no smoothing is applied.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, SmoothingSettings};
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let mut deconvoluter = Deconvoluter::default();
    ///
    /// deconvoluter.set_smoothing_settings(SmoothingSettings::MovingAverage {
    ///     iterations: 3,
    ///     window_size: 3,
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_smoothing_settings(&mut self, smoothing_settings: SmoothingSettings) -> Result<()> {
        smoothing_settings.validate()?;
        self.smoother = match smoothing_settings {
            SmoothingSettings::Identity => Arc::new(Identity::new()),
            SmoothingSettings::MovingAverage {
                iterations,
                window_size,
            } => Arc::new(MovingAverage::<f64>::new(iterations, window_size)),
        };

        Ok(())
    }

    /// Sets the peak selection settings.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided peak selection settings are
    /// invalid. For example, a negative `threshold` for a noise score filter
    /// wouldn't make sense.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, ScoringMethod, SelectionSettings};
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let mut deconvoluter = Deconvoluter::default();
    ///
    /// deconvoluter.set_selection_settings(SelectionSettings::NoiseScoreFilter {
    ///     scoring_method: ScoringMethod::MinimumSum,
    ///     threshold: 5.0,
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_selection_settings(&mut self, selection_settings: SelectionSettings) -> Result<()> {
        selection_settings.validate()?;
        self.selector = match selection_settings {
            SelectionSettings::DetectorOnly => Arc::new(DetectorOnly::new()),
            SelectionSettings::NoiseScoreFilter {
                scoring_method,
                threshold,
            } => Arc::new(NoiseScoreFilter::new(scoring_method, threshold)),
        };

        Ok(())
    }

    /// Sets the fitting settings.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided fitting settings are invalid. For
    /// example, 0 `iterations` would mean that the fitting algorithm doesn't
    /// do anything.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, FittingSettings};
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let mut deconvoluter = Deconvoluter::default();
    ///
    /// deconvoluter.set_fitting_settings(FittingSettings::Analytical { iterations: 20 })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_fitting_settings(&mut self, fitting_settings: FittingSettings) -> Result<()> {
        fitting_settings.validate()?;
        self.fitter = match fitting_settings {
            FittingSettings::Analytical { iterations } => {
                Arc::new(FitterAnalytical::new(iterations))
            }
        };

        Ok(())
    }

    /// Adds a region to ignore during deconvolution.
    ///
    /// Some samples contain compounds that are not of interest, such as a water
    /// signal. Regions where these compounds are expected can be ignored during
    /// the deconvolution.
    ///
    /// # Errors
    ///
    /// An error is returned if the start or end value is not finite or if they
    /// are (nearly) equal.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, SmoothingSettings};
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let mut deconvoluter = Deconvoluter::default();
    ///
    /// // No regions are ignored by default.
    /// assert!(deconvoluter.ignore_regions().is_none());
    ///
    /// // Add regions to ignore during deconvolution.
    /// deconvoluter.add_ignore_region((4.7, 4.9))?;
    /// assert!(deconvoluter.ignore_regions().is_some());
    /// assert_eq!(deconvoluter.ignore_regions().unwrap().len(), 1);
    /// deconvoluter.add_ignore_region((5.2, 5.6))?;
    /// assert_eq!(deconvoluter.ignore_regions().unwrap().len(), 2);
    ///
    /// // Overlapping regions are combined.
    /// deconvoluter.add_ignore_region((4.8, 5.4))?;
    /// assert_eq!(deconvoluter.ignore_regions().unwrap().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_ignore_region(&mut self, new: (f64, f64)) -> Result<()> {
        if !new.0.is_finite()
            || !new.1.is_finite()
            || f64::abs(new.0 - new.1) < crate::CHECK_PRECISION
        {
            return Err(Error::new(Kind::InvalidIgnoreRegion { region: new }).into());
        }

        if let Some(ignore_regions) = self.ignore_regions.as_mut() {
            ignore_regions.push((f64::min(new.0, new.1), f64::max(new.0, new.1)));
            ignore_regions.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            while let Some(overlap_position) = ignore_regions
                .windows(2)
                .position(|w| w[1].0 < w[0].1 || f64::abs(w[0].1 - w[1].0) < crate::CHECK_PRECISION)
            {
                let combined = (
                    f64::min(
                        ignore_regions[overlap_position].0,
                        ignore_regions[overlap_position + 1].0,
                    ),
                    f64::max(
                        ignore_regions[overlap_position].1,
                        ignore_regions[overlap_position + 1].1,
                    ),
                );
                ignore_regions.remove(overlap_position);
                ignore_regions.remove(overlap_position);
                ignore_regions.insert(overlap_position, combined);
            }
        } else {
            self.ignore_regions = Some(vec![(f64::min(new.0, new.1), f64::max(new.0, new.1))]);
        }

        Ok(())
    }

    /// Clears the regions to ignore during deconvolution.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, SmoothingSettings};
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let mut deconvoluter = Deconvoluter::default();
    ///
    /// deconvoluter.add_ignore_region((4.7, 4.9))?;
    /// deconvoluter.clear_ignore_regions();
    /// assert!(deconvoluter.ignore_regions().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_ignore_regions(&mut self) {
        self.ignore_regions = None;
    }

    /// Deconvolutes the provided spectrum into individual signals.
    ///
    /// # Errors
    ///
    /// During the deconvolution process, the algorithm relies on finding peaks
    /// in the `Spectrum`. If no peaks are found, an error is returned. The
    /// peaks outside the signal boundaries of the `Spectrum` are used to filter
    /// out noise within the signal region. If no peaks are found outside or
    /// within the signal region, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// // Read a spectrum in Bruker TopSpin format.
    /// let path = "path/to/spectrum";
    /// # let path = "../data/bruker/blood/blood_01";
    /// let spectrum = Bruker::read_spectrum(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    ///
    /// // Deconvolute the spectrum.
    /// let deconvoluter = Deconvoluter::default();
    /// let deconvolution = deconvoluter.deconvolute_spectrum(&spectrum)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn deconvolute_spectrum(&self, spectrum: &Spectrum) -> Result<Deconvolution> {
        let mut intensities = spectrum.intensities().to_vec();
        self.smoother.smooth_values(&mut intensities);
        let ignore_regions = self.ignore_region_indices(spectrum);
        let peaks = self.selector.select_peaks(
            &intensities,
            spectrum.signal_boundaries_indices(),
            ignore_regions.as_deref(),
        )?;
        let lorentzians = self.fitter.fit_lorentzian(spectrum, &peaks);
        let mse = self.compute_mse(
            spectrum,
            Lorentzian::superposition_vec(spectrum.chemical_shifts(), &lorentzians),
        );

        Ok(Deconvolution::new(
            lorentzians,
            self.smoother.settings(),
            self.selector.settings(),
            self.fitter.settings(),
            mse,
        ))
    }

    /// Deconvolutes the provided spectrum into individual signals in parallel.
    ///
    /// # Errors
    ///
    /// During the deconvolution process, the algorithm relies on finding peaks
    /// in the `Spectrum`. If no peaks are found, an error is returned. The
    /// peaks outside the signal boundaries of the `Spectrum` are used to filter
    /// out noise within the signal region. If no peaks are found outside or
    /// within the signal region, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::Deconvoluter;
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// // Read a spectrum in Bruker TopSpin format.
    /// let path = "path/to/spectrum";
    /// # let path = "../data/bruker/blood/blood_01";
    /// let spectrum = Bruker::read_spectrum(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    ///
    /// // Deconvolute the spectrum in parallel.
    /// let deconvoluter = Deconvoluter::default();
    /// let deconvolution = deconvoluter.par_deconvolute_spectrum(&spectrum)?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "parallel")]
    pub fn par_deconvolute_spectrum(&self, spectrum: &Spectrum) -> Result<Deconvolution> {
        let mut intensities = spectrum.intensities().to_vec();
        self.smoother.smooth_values(&mut intensities);
        let ignore_regions = self.ignore_region_indices(spectrum);
        let peaks = self.selector.select_peaks(
            &intensities,
            spectrum.signal_boundaries_indices(),
            ignore_regions.as_deref(),
        )?;
        let lorentzians = self.fitter.par_fit_lorentzian(spectrum, &peaks);
        let mse = self.compute_mse(
            spectrum,
            Lorentzian::par_superposition_vec(spectrum.chemical_shifts(), &lorentzians),
        );

        Ok(Deconvolution::new(
            lorentzians,
            self.smoother.settings(),
            self.selector.settings(),
            self.fitter.settings(),
            mse,
        ))
    }

    /// Deconvolutes the provided spectra into individual signals.
    ///
    /// # Errors
    ///
    /// During the deconvolution process, the algorithm relies on finding peaks
    /// in the `Spectrum`. If no peaks are found, an error is returned. The
    /// peaks outside the signal boundaries of the `Spectrum` are used to filter
    /// out noise within the signal region. If no peaks are found outside or
    /// within the signal region, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// // Read all spectra from Bruker TopSpin format directories within the root.
    /// let path = "path/to/root";
    /// # let path = "../data/bruker/blood";
    /// let spectra = Bruker::read_spectra(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    ///
    /// // Deconvolute the spectra.
    /// let deconvoluter = Deconvoluter::default();
    /// let deconvolution = deconvoluter.deconvolute_spectra(&spectra)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn deconvolute_spectra<S: AsRef<Spectrum>>(
        &self,
        spectra: &[S],
    ) -> Result<Vec<Deconvolution>> {
        let deconvolutions = spectra
            .iter()
            .map(|spectrum| self.deconvolute_spectrum(spectrum.as_ref()))
            .collect::<Result<Vec<Deconvolution>>>()?;

        Ok(deconvolutions)
    }

    /// Deconvolutes the provided spectra into individual signals in parallel.
    ///
    /// # Errors
    ///
    /// During the deconvolution process, the algorithm relies on finding peaks
    /// in the `Spectrum`. If no peaks are found, an error is returned. The
    /// peaks outside the signal boundaries of the `Spectrum` are used to filter
    /// out noise within the signal region. If no peaks are found outside or
    /// within the signal region, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// // Read all spectra from Bruker TopSpin format directories within the root.
    /// let path = "path/to/root";
    /// # let path = "../data/bruker/blood";
    /// let spectra = Bruker::read_spectra(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    ///
    /// // Deconvolute the spectra.
    /// let deconvoluter = Deconvoluter::default();
    /// let deconvolution = deconvoluter.par_deconvolute_spectra(&spectra)?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "parallel")]
    pub fn par_deconvolute_spectra<S: AsRef<Spectrum> + Send + Sync>(
        &self,
        spectra: &[S],
    ) -> Result<Vec<Deconvolution>> {
        let deconvolutions = spectra
            .par_iter()
            .map(|spectrum| self.par_deconvolute_spectrum(spectrum.as_ref()))
            .collect::<Result<Vec<Deconvolution>>>()?;

        Ok(deconvolutions)
    }

    /// Optimizes the deconvolution settings.
    ///
    /// To determine the optimal deconvolution settings, a reference spectrum is
    /// used to evaluate the mean squared error (MSE) across different
    /// combinations of smoothing, peak selection, and fitting settings. Each
    /// stage of the algorithm is tested using a predefined set of settings,
    /// selected from a broader range that was assessed on spectra with varying
    /// resolutions, noise levels, and peak counts. The combination yielding the
    /// lowest MSE is chosen as the optimal configuration.
    ///
    /// # Errors
    ///
    /// During the deconvolution process, the algorithm relies on finding peaks
    /// in the `Spectrum`. If no peaks are found, an error is returned. The
    /// peaks outside the signal boundaries of the `Spectrum` are used to filter
    /// out noise within the signal region. If no peaks are found outside or
    /// within the signal region, an error is returned. If any parameter
    /// combination returns an error, there is likely an issue with the data,
    /// so the optimization process is aborted and the error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::{Deconvoluter, Deconvolution, Lorentzian};
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// // Read all spectra from Bruker TopSpin format directories within the root.
    /// let path = "path/to/root";
    /// # let path = "../data/bruker/sim";
    /// let spectra = Bruker::read_spectra(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (3.339, 3.553),
    /// )?;
    ///
    /// // Optimize the deconvolution settings using the first spectrum as a reference.
    /// let mut deconvoluter = Deconvoluter::default();
    /// deconvoluter.optimize_settings(&spectra[0])?;
    ///
    /// // Deconvolute the spectra with the optimized settings.
    /// let deconvolution = deconvoluter.par_deconvolute_spectra(&spectra)?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "parallel")]
    pub fn optimize_settings(&mut self, reference: &Spectrum) -> Result<f64> {
        let smoothing_settings = (2..=10)
            .flat_map(|iterations| {
                (3..=7)
                    .step_by(2)
                    .map(move |window_size| SmoothingSettings::MovingAverage {
                        iterations,
                        window_size,
                    })
            })
            .collect::<Vec<SmoothingSettings>>();
        let selection_settings = (0..10)
            .map(|coefficient| SelectionSettings::NoiseScoreFilter {
                scoring_method: ScoringMethod::MinimumSum,
                threshold: 5.0 + (coefficient as f64) * (8.0 - 5.0) / 9.0,
            })
            .collect::<Vec<SelectionSettings>>();
        let fitting_settings = (5..=15)
            .step_by(5)
            .map(|iterations| FittingSettings::Analytical { iterations })
            .collect::<Vec<FittingSettings>>();

        let optimal_settings = smoothing_settings
            .par_iter()
            .map(|smoothing| {
                let mut deconvoluter = self.clone();
                deconvoluter
                    .set_smoothing_settings(*smoothing)
                    .unwrap();

                selection_settings
                    .iter()
                    .map(|selection| {
                        deconvoluter
                            .set_selection_settings(*selection)
                            .unwrap();

                        fitting_settings
                            .iter()
                            .map(|fitting| {
                                deconvoluter
                                    .set_fitting_settings(*fitting)
                                    .unwrap();
                                let deconvolution = deconvoluter.deconvolute_spectrum(reference)?;

                                Ok((deconvolution.mse(), *fitting, *selection, *smoothing))
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .flatten()
            .min_by(|(mse_1, ..), (mse_2, ..)| f64::partial_cmp(mse_1, mse_2).unwrap())
            .unwrap();
        let (mse, fitting, selection, smoothing) = optimal_settings;
        self.set_smoothing_settings(smoothing)?;
        self.set_selection_settings(selection)?;
        self.set_fitting_settings(fitting)?;

        Ok(mse)
    }

    /// Internal helper function to compute the MSE within the signal region.
    fn compute_mse(&self, spectrum: &Spectrum, superpositions: Vec<f64>) -> f64 {
        let regions = match self.ignore_region_indices(spectrum) {
            Some(ignore_regions) => {
                let iter = std::iter::once(spectrum.signal_boundaries_indices().0)
                    .chain(
                        ignore_regions
                            .iter()
                            .flat_map(|(start, end)| vec![*start, *end]),
                    )
                    .chain(std::iter::once(spectrum.signal_boundaries_indices().1));

                iter.clone()
                    .step_by(2)
                    .zip(iter.skip(1).step_by(2))
                    .collect::<Vec<(usize, usize)>>()
            }
            None => vec![spectrum.signal_boundaries_indices()],
        };
        let residuals = regions
            .iter()
            .map(|(start, end)| {
                superpositions[*start..*end]
                    .iter()
                    .zip(spectrum.intensities()[*start..*end].iter())
                    .map(|(superposition, intensity)| (superposition - intensity).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let length = regions
            .iter()
            .map(|(start, end)| end - start)
            .sum::<usize>();

        residuals / (length as f64)
    }

    /// Internal helper function to convert the ignore regions to indices.
    fn ignore_region_indices(&self, spectrum: &Spectrum) -> Option<Vec<(usize, usize)>> {
        if let Some(ignore_regions) = self.ignore_regions.as_ref() {
            let step = spectrum.step();
            let first = spectrum.chemical_shifts()[0];
            let boundaries = spectrum.signal_boundaries();
            let (lower_boundary, upper_boundary) = (
                f64::min(boundaries.0, boundaries.1),
                f64::max(boundaries.0, boundaries.1),
            );
            let boundary_indices = spectrum.signal_boundaries_indices();
            let (lower, upper) = (
                usize::min(boundary_indices.0, boundary_indices.1),
                usize::max(boundary_indices.0, boundary_indices.1),
            );
            let indices = ignore_regions
                .iter()
                .filter(|(start, end)| {
                    !(*start < lower_boundary && *end < lower_boundary
                        || *start > upper_boundary && *end > upper_boundary)
                })
                .filter_map(|(start, end)| {
                    let first_index = usize::max(((*start - first) / step).floor() as usize, lower);
                    let second_index = usize::min(((*end - first) / step).ceil() as usize, upper);
                    let boundaries = (
                        usize::min(first_index, second_index),
                        usize::max(first_index, second_index),
                    );
                    if boundaries.0 < boundaries.1 - 1 {
                        Some(boundaries)
                    } else {
                        None
                    }
                })
                .collect();

            Some(indices)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, assert_send, assert_sync};
    use float_cmp::assert_approx_eq;

    #[test]
    fn thread_safe() {
        assert_send!(Deconvoluter);
        assert_sync!(Deconvoluter);
    }

    #[test]
    fn invalid_smoothing_settings() {
        let mut deconvoluter = Deconvoluter::default();
        let zero_iterations = SmoothingSettings::MovingAverage {
            iterations: 0,
            window_size: 3,
        };
        let zero_window_size = SmoothingSettings::MovingAverage {
            iterations: 2,
            window_size: 0,
        };
        let zero_both = SmoothingSettings::MovingAverage {
            iterations: 0,
            window_size: 0,
        };
        let errors = [
            deconvoluter
                .set_smoothing_settings(zero_iterations)
                .unwrap_err(),
            deconvoluter
                .set_smoothing_settings(zero_window_size)
                .unwrap_err(),
            deconvoluter
                .set_smoothing_settings(zero_both)
                .unwrap_err(),
        ];
        let expected_context = [zero_iterations, zero_window_size, zero_both];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Deconvolution(inner) => match inner.kind() {
                    Kind::InvalidSmoothingSettings { settings } => {
                        assert!(SmoothingSettings::compare(settings, &context));
                    }
                    _ => panic!("unexpected kind: {:?}", inner),
                },
                _ => panic!("unexpected error: {:?}", error),
            });
    }

    #[test]
    fn invalid_selection_settings() {
        let mut deconvoluter = Deconvoluter::default();
        let zero_threshold = SelectionSettings::NoiseScoreFilter {
            scoring_method: ScoringMethod::default(),
            threshold: 0.0,
        };
        let nan_threshold = SelectionSettings::NoiseScoreFilter {
            scoring_method: ScoringMethod::default(),
            threshold: f64::NAN,
        };
        let inf_threshold = SelectionSettings::NoiseScoreFilter {
            scoring_method: ScoringMethod::default(),
            threshold: f64::INFINITY,
        };
        let neg_inf_threshold = SelectionSettings::NoiseScoreFilter {
            scoring_method: ScoringMethod::default(),
            threshold: f64::NEG_INFINITY,
        };
        let errors = [
            deconvoluter
                .set_selection_settings(zero_threshold)
                .unwrap_err(),
            deconvoluter
                .set_selection_settings(nan_threshold)
                .unwrap_err(),
            deconvoluter
                .set_selection_settings(inf_threshold)
                .unwrap_err(),
            deconvoluter
                .set_selection_settings(neg_inf_threshold)
                .unwrap_err(),
        ];
        let expected_context = [
            zero_threshold,
            nan_threshold,
            inf_threshold,
            neg_inf_threshold,
        ];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Deconvolution(inner) => match inner.kind() {
                    Kind::InvalidSelectionSettings { settings } => {
                        assert!(SelectionSettings::compare(settings, &context));
                    }
                    _ => panic!("unexpected kind: {:?}", inner),
                },
                _ => panic!("unexpected error: {:?}", error),
            });
    }

    #[test]
    fn invalid_fitting_settings() {
        let mut deconvoluter = Deconvoluter::default();
        let zero_iterations = FittingSettings::Analytical { iterations: 0 };
        let errors = [deconvoluter
            .set_fitting_settings(zero_iterations)
            .unwrap_err()];
        let expected_context = [zero_iterations];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Deconvolution(inner) => match inner.kind() {
                    Kind::InvalidFittingSettings { settings } => {
                        assert!(FittingSettings::compare(settings, &context));
                    }
                    _ => panic!("unexpected kind: {:?}", inner),
                },
                _ => panic!("unexpected error: {:?}", error),
            });
    }

    #[test]
    fn add_ignore_region() {
        let mut deconvoluter = Deconvoluter::default();
        deconvoluter
            .add_ignore_region((1.0, 2.0))
            .unwrap();
        deconvoluter
            .add_ignore_region((3.0, 4.0))
            .unwrap();
        assert!(deconvoluter.ignore_regions().is_some());
        assert_eq!(deconvoluter.ignore_regions().unwrap().len(), 2);
        deconvoluter
            .add_ignore_region((2.0, 3.0))
            .unwrap();
        assert_eq!(deconvoluter.ignore_regions().unwrap().len(), 1);
        assert_approx_eq!(f64, deconvoluter.ignore_regions().unwrap()[0].0, 1.0);
        assert_approx_eq!(f64, deconvoluter.ignore_regions().unwrap()[0].1, 4.0);
    }

    #[test]
    fn invalid_ignore_region() {
        let mut deconvoluter = Deconvoluter::default();
        let errors = [
            deconvoluter
                .add_ignore_region((f64::NAN, 1.0))
                .unwrap_err(),
            deconvoluter
                .add_ignore_region((1.0, f64::NAN))
                .unwrap_err(),
            deconvoluter
                .add_ignore_region((f64::INFINITY, 1.0))
                .unwrap_err(),
            deconvoluter
                .add_ignore_region((1.0, f64::INFINITY))
                .unwrap_err(),
            deconvoluter
                .add_ignore_region((f64::NEG_INFINITY, 1.0))
                .unwrap_err(),
            deconvoluter
                .add_ignore_region((1.0, f64::NEG_INFINITY))
                .unwrap_err(),
            deconvoluter
                .add_ignore_region((1.0, 1.0))
                .unwrap_err(),
        ];
        let expected_context = [
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (1.0, f64::NEG_INFINITY),
            (1.0, 1.0),
        ];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Deconvolution(inner) => match inner.kind() {
                    Kind::InvalidIgnoreRegion { region } => {
                        assert_approx_eq!(f64, region.0, context.0);
                        assert_approx_eq!(f64, region.1, context.1);
                    }
                    _ => panic!("Unexpected kind: {:?}", inner.kind()),
                },
                _ => panic!("Unexpected error: {:?}", error),
            });
    }

    #[test]
    fn clear_ignore_regions() {
        let mut deconvoluter = Deconvoluter::default();
        deconvoluter
            .add_ignore_region((1.0, 2.0))
            .unwrap();
        deconvoluter
            .add_ignore_region((3.0, 4.0))
            .unwrap();
        deconvoluter.clear_ignore_regions();
        assert!(deconvoluter.ignore_regions().is_none());
    }
}
