#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use crate::deconvolution::SerializedLorentzian;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Data structure that represents a [Lorentzian function].
///
/// # Definition
///
/// The [Lorentzian function] is typically defined as:
///
/// ```text
/// f(x) = 1/π * γ / ((x - x₀)² + γ²)
/// ```
///
/// - `γ` is the half-width at half-maximum (`hw`).
/// - `x₀` is the position of the maximum (`maxp`).
/// - The factor `1/π` ensures the integral of the function equals 1.
///
/// To fit the function to data, a scale factor `sf` replaces `1/π`, resulting
/// in:
///
/// ```text
/// f(x) = sf * hw / (hw² + (x - maxp)²)
/// ```
///
/// This form is computationally challenging due to `hw` appearing in both the
/// numerator and denominator. To simplify, the following transformations are
/// applied:
///
/// ```text
/// sfhw = sf * hw
/// hw2 = hw²
/// ```
///
/// The Lorentzian function then becomes:
///
/// ```text
/// f(x) = sfhw / (hw2 + (x - maxp)²)
/// ```
///
/// This representation is used internally for efficiency, avoiding redundant
/// computations during evaluation and solving.
///
/// [Lorentzian function]: https://en.wikipedia.org/wiki/Cauchy_distribution
///
/// # Negative Transformed Parameters
///
/// For `Lorentzian` to represent a valid peak shape, the transformed parameters
/// `sfhw` and `hw2` must be positive. Additionally, if `hw2` is negative,
/// attempting to recover `hw` by taking the square root will corrupt the data.
/// This is not enforced to avoid unnecessary overhead. Instances created by the
/// library are guaranteed to have valid values, but if you construct a
/// `Lorentzian` manually, you are responsible for ensuring the parameters are
/// valid.
///
/// # Thread Safety
///
/// The `Lorentzian` type is both [`Send`] and [`Sync`], allowing safe sharing
/// and access across threads.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// When the `serde` feature is enabled, `Lorentzian` can be serialized and
/// deserialized using `serde`. During serialization, the transformation is
/// reversed, and the original parameters `(sf, hw, maxp)` are stored.
///
/// **Important:** Since `hw2` must be non-negative, an invalid value can lead
/// to corruption when serializing, as taking the square root is needed to
/// recover `hw`. As mentioned above, the library ensures valid values, but
/// manual construction requires caution.
///
/// # Example
///
/// The following demonstrates basic usage of the `Lorentzian` struct. It is,
/// however, usually not very useful to create these manually. Though, this is a
/// simple method to generate some synthetic data.
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use metabodecon::deconvolution::Lorentzian;
///
/// // Create a Lorentzian peak centered at 5.0 ppm with maximum intensity 2.0 and hw 0.15.
/// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
///
/// // Evaluate the Lorentzian at its maximum position.
/// assert_approx_eq!(f64, lorentzian.evaluate(5.0), 2.0);
///
/// // Retrieve the untransformed parameters.
/// let (sf, hw, maxp) = lorentzian.untransformed_parameters();
/// assert_approx_eq!(f64, sf, 0.3); // sf = sfhw / hw = 0.045 / 0.15
/// assert_approx_eq!(f64, hw, 0.15); // hw = √(hw2) = √(0.0225)
/// assert_approx_eq!(f64, maxp, 5.0); // maxp remains unchanged
///
/// // Generate 100 chemical shifts between 0.0 and 10.0 ppm.
/// let chemical_shifts = (0..100)
///     .map(|x| x as f64 * 10.0 / 99.0)
///     .collect::<Vec<f64>>();
///
/// // Evaluate the Lorentzian at the chemical shifts.
/// let intensities = lorentzian.evaluate_vec(&chemical_shifts);
///
/// // Compute the integral of the Lorentzian.
/// let integral = lorentzian.integral();
/// assert_approx_eq!(f64, lorentzian.sf() * std::f64::consts::PI, integral);
///
/// // Create a peak triplet centered at 5.0 ppm with shorter side peaks.
/// let triplet = [
///     Lorentzian::new(0.03, 0.0009, 4.8),
///     Lorentzian::new(0.02, 0.0004, 5.0),
///     Lorentzian::new(0.03, 0.0009, 5.2),
/// ];
///
/// // Evaluate the superposition of the Lorentzians at the maximum.
/// assert_approx_eq!(
///     f64,
///     Lorentzian::superposition(5.0, &triplet),
///     51.466992,
///     epsilon = 1e-6
/// );
///
/// // Evaluate the superposition of the Lorentzians at the chemical shifts.
/// let sup1 = Lorentzian::superposition_vec(&chemical_shifts, &triplet);
/// // ...or in parallel (less efficient for small datasets).
/// let sup2 = Lorentzian::par_superposition_vec(&chemical_shifts, &triplet);
/// ```
#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(into = "SerializedLorentzian", try_from = "SerializedLorentzian")
)]
pub struct Lorentzian {
    /// Scale factor multiplied by the half-width (`sfhw = sf * hw`).
    sfhw: f64,
    /// Half-width squared (`hw2 = hw²`).
    hw2: f64,
    /// Position of the maximum (`maxp`).
    maxp: f64,
}

impl AsRef<Lorentzian> for Lorentzian {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Lorentzian {
    /// Constructs a new `Lorentzian` from the given parameters.
    ///
    /// This function uses a transformed representation of the [Lorentzian
    /// function] to simplify evaluation and computation. The transformed
    /// form is:
    ///
    /// ```text
    /// f(x) = sfhw / (hw2 + (x - maxp)²)
    /// ```
    ///
    /// - `sfhw`: The scale factor multiplied by the half-width at half-maximum
    ///   (`sf * hw`).
    /// - `hw2`: The square of the half-width at half-maximum (`hw²`).
    /// - `maxp`: The position of the maximum (unchanged).
    ///
    /// This representation avoids redundant computations, making it more
    /// efficient for evaluation and solving systems of equations.
    ///
    /// [Lorentzian function]: https://en.wikipedia.org/wiki/Cauchy_distribution
    ///
    /// # Negative Transformed Parameters
    ///
    /// For `Lorentzian` to represent a valid peak shape, the transformed
    /// parameters `sfhw` and `hw2` must be positive. Additionally, if `hw2` is
    /// negative, attempting to recover `hw` by taking the square root will
    /// corrupt the data. This is not enforced to avoid unnecessary overhead.
    /// Instances created by the library are guaranteed to have valid values,
    /// but if you construct a `Lorentzian` manually, you are responsible for
    /// ensuring the parameters are valid.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// // Signal centered at 5 ppm with maximum intensity 2.0 and a half width
    /// // of 0.15 ppm.
    /// let max = 2.0;
    /// let hw = 0.15;
    /// let hw2 = hw * hw;
    /// let sfhw = max * hw2;
    /// let maxp = 5.0;
    /// let lorentzian = Lorentzian::new(sfhw, hw2, maxp);
    ///
    /// // Signal centered at 0 ppm with maximum intensity 30.0 and a half width
    /// // of 0.5 ppm.
    /// let max = 30.0;
    /// let hw = 0.5;
    /// let hw2 = hw * hw;
    /// let sfhw = max * hw2;
    /// let maxp = 0.0;
    /// let lorentzian = Lorentzian::new(sfhw, hw2, maxp);
    /// ```
    pub fn new(sfhw: f64, hw2: f64, maxp: f64) -> Self {
        Self { sfhw, hw2, maxp }
    }

    /// Returns the scale factor multiplied by the half-width (`sfhw`).
    ///
    /// This is part of the transformed parameter interface. On its own, `sfhw`
    /// is not uniquely interpretable, as multiple combinations of scale factors
    /// (`sf`) and half-widths (`hw`) can produce the same `sfhw`. For the
    /// untransformed parameters, see [`sf`] and [`hw`].
    ///
    /// [`sf`]: Lorentzian::sf
    /// [`hw`]: Lorentzian::hw
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.sfhw(), 0.045);
    /// ```
    pub fn sfhw(&self) -> f64 {
        self.sfhw
    }

    /// Returns the square of the half-width (`hw2`).
    ///
    /// This is part of the transformed parameter interface. For the
    /// untransformed half-width, see [`hw`].
    ///
    /// [`hw`]: Lorentzian::hw
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.hw2(), 0.0225);
    /// ```
    pub fn hw2(&self) -> f64 {
        self.hw2
    }

    /// Returns the position of the maximum (`maxp`).
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.maxp(), 5.0);
    /// ```
    pub fn maxp(&self) -> f64 {
        self.maxp
    }

    /// Returns the transformed parameters as a tuple `(sfhw, hw2, maxp)`.
    ///
    /// This is part of the transformed parameter interface. For the
    /// untransformed parameters, see [`untransformed_parameters`].
    ///
    /// [`untransformed_parameters`]: Lorentzian::untransformed_parameters
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// let (sfhw, hw2, maxp) = lorentzian.parameters();
    ///
    /// assert_approx_eq!(f64, sfhw, 0.045);
    /// assert_approx_eq!(f64, hw2, 0.0225);
    /// assert_approx_eq!(f64, maxp, 5.0);
    /// ```
    pub fn parameters(&self) -> (f64, f64, f64) {
        (self.sfhw, self.hw2, self.maxp)
    }

    /// Sets the scale factor multiplied by the half-width (`sfhw`).
    ///
    /// This is part of the transformed parameter interface. Modifying `sfhw`
    /// indirectly adjusts the scale factor (`sf`). To modify the scale factor
    /// directly, use [`set_sf`].
    ///
    /// [`set_sf`]: Lorentzian::set_sf
    /// [`set_hw`]: Lorentzian::set_hw
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_sfhw(0.0675);
    ///
    /// assert_approx_eq!(f64, lorentzian.sfhw(), 0.0675);
    /// ```
    pub fn set_sfhw(&mut self, sfhw: f64) {
        self.sfhw = sfhw;
    }

    /// Sets the square of the half-width (`hw2`).
    ///
    /// This is part of the transformed parameter interface. Modifying `hw2`
    /// indirectly adjusts the half-width (`hw`) and, as a side effect, the
    /// scale factor (`sf`). To modify the half-width directly without side
    /// effects, use [`set_hw`].
    ///
    /// [`set_hw`]: Lorentzian::set_hw
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_hw2(0.03);
    ///
    /// assert_approx_eq!(f64, lorentzian.hw2(), 0.03);
    /// ```
    pub fn set_hw2(&mut self, hw2: f64) {
        self.hw2 = hw2;
    }

    /// Sets the position of the maximum (`maxp`).
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_maxp(-5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.maxp(), -5.0);
    /// ```
    pub fn set_maxp(&mut self, maxp: f64) {
        self.maxp = maxp;
    }

    /// Sets the transformed parameters `(sfhw, hw2, maxp)`.
    ///
    /// This is part of the transformed parameter interface. To set the
    /// parameters using their untransformed representation, use
    /// [`set_untransformed_parameters`].
    ///
    /// [`set_untransformed_parameters`]: Lorentzian::set_untransformed_parameters
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_parameters(0.12, 0.04, -5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.sfhw(), 0.12);
    /// assert_approx_eq!(f64, lorentzian.hw2(), 0.04);
    /// assert_approx_eq!(f64, lorentzian.maxp(), -5.0);
    /// ```
    pub fn set_parameters(&mut self, sfhw: f64, hw2: f64, maxp: f64) {
        self.sfhw = sfhw;
        self.hw2 = hw2;
        self.maxp = maxp;
    }

    /// Returns the untransformed scale factor (`sf`).
    ///
    /// This undoes the transformation applied to the scale factor. Computing
    /// `sf` has some overhead, as it requires a division and a square root.
    /// For the transformed parameter, see [`sfhw`].
    ///
    /// [`sfhw`]: Lorentzian::sfhw
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.sf(), 0.3);
    /// ```
    pub fn sf(&self) -> f64 {
        self.sfhw / self.hw()
    }

    /// Returns the untransformed half-width (`hw`).
    ///
    /// This undoes the transformation applied to the half-width. Computing
    /// `hw` has some overhead, as it requires a square root. For the
    /// transformed parameter, see [`hw2`].
    ///
    /// [`hw2`]: Lorentzian::hw2
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.hw(), 0.15);
    /// ```
    pub fn hw(&self) -> f64 {
        self.hw2.sqrt()
    }

    /// Returns the untransformed parameters as a tuple `(sf, hw, maxp)`.
    ///
    /// This undoes the transformation applied to the parameters. Computing
    /// `sf` and `hw` has some overhead, as it requires a division and a square
    /// root. For the transformed parameters, see [`parameters`].
    ///
    /// [`parameters`]: Lorentzian::parameters
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// let (sf, hw, maxp) = lorentzian.untransformed_parameters();
    ///
    /// assert_approx_eq!(f64, sf, 0.3);
    /// assert_approx_eq!(f64, hw, 0.15);
    /// assert_approx_eq!(f64, maxp, 5.0);
    /// ```
    pub fn untransformed_parameters(&self) -> (f64, f64, f64) {
        (self.sf(), self.hw(), self.maxp())
    }

    /// Sets the untransformed scale factor (`sf`).
    ///
    /// This is part of the untransformed parameter interface. Modifying `sf`
    /// indirectly adjusts the transformed parameter (`sfhw`). Updating `sf`
    /// requires computing the half-width (`hw`). To modify the transformed
    /// parameter directly, use [`set_sfhw`].
    ///
    /// [`set_sfhw`]: Lorentzian::set_sfhw
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_sf(0.45);
    ///
    /// assert_approx_eq!(f64, lorentzian.sf(), 0.45);
    /// ```
    pub fn set_sf(&mut self, sf: f64) {
        self.sfhw = sf * self.hw();
    }

    /// Sets the untransformed half-width (`hw`).
    ///
    /// This is part of the untransformed parameter interface. Modifying `hw`
    /// indirectly adjusts the transformed parameter (`hw2`). Updating `hw`
    /// requires computing the scale factor (`sf`). To modify the transformed
    /// parameter directly, use [`set_hw2`].
    ///
    /// [`set_hw2`]: Lorentzian::set_hw2
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_hw(0.30);
    ///
    /// assert_approx_eq!(f64, lorentzian.hw(), 0.30);
    /// ```
    pub fn set_hw(&mut self, hw: f64) {
        self.sfhw = self.sf() * hw;
        self.hw2 = hw.powi(2);
    }

    /// Sets the untransformed parameters `(sf, hw, maxp)`.
    ///
    /// This is part of the untransformed parameter interface. To set the
    /// parameters using their transformed representation, use
    /// [`set_parameters`].
    ///
    /// [`set_parameters`]: Lorentzian::set_parameters
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let mut lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// lorentzian.set_untransformed_parameters(0.6, 0.2, -5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.sf(), 0.6);
    /// assert_approx_eq!(f64, lorentzian.hw(), 0.2);
    /// assert_approx_eq!(f64, lorentzian.maxp(), -5.0);
    /// assert_approx_eq!(f64, lorentzian.sf() / lorentzian.hw(), 3.0);
    /// ```
    pub fn set_untransformed_parameters(&mut self, sf: f64, hw: f64, maxp: f64) {
        self.sfhw = sf * hw;
        self.hw2 = hw.powi(2);
        self.maxp = maxp;
    }

    /// Evaluates the `Lorentzian` function at a given position `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    ///
    /// assert_approx_eq!(f64, lorentzian.evaluate(5.0), 2.0);
    /// ```
    pub fn evaluate(&self, x: f64) -> f64 {
        self.sfhw / (self.hw2 + (x - self.maxp).powi(2))
    }

    /// Evaluates the `Lorentzian` function at the given positions `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// let chemical_shifts = (0..100)
    ///     .map(|x| x as f64 * 10.0 / 99.0)
    ///     .collect::<Vec<f64>>();
    /// let intensities = lorentzian.evaluate_vec(&chemical_shifts);
    /// ```
    pub fn evaluate_vec(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Computes the integral of the `Lorentzian` over the entire domain.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let lorentzian = Lorentzian::new(0.045, 0.0225, 5.0);
    /// let integral = lorentzian.integral();
    ///
    /// assert_approx_eq!(f64, lorentzian.sf() * std::f64::consts::PI, integral);
    /// ```
    pub fn integral(&self) -> f64 {
        std::f64::consts::PI * self.sf()
    }

    /// Evaluates the superposition of the given `Lorentzian`s at the given
    /// position `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let triplet = [
    ///     Lorentzian::new(0.03, 0.0009, 4.8),
    ///     Lorentzian::new(0.02, 0.0004, 5.0),
    ///     Lorentzian::new(0.03, 0.0009, 5.2),
    /// ];
    ///
    /// assert_approx_eq!(
    ///     f64,
    ///     Lorentzian::superposition(5.0, &triplet),
    ///     51.466992,
    ///     epsilon = 1e-6
    /// );
    /// ```
    pub fn superposition<L: AsRef<Lorentzian>>(x: f64, lorentzians: &[L]) -> f64 {
        lorentzians
            .iter()
            .map(|lorentzian| lorentzian.as_ref().evaluate(x))
            .sum()
    }

    /// Evaluates the superposition of the given `Lorentzian`s at the given
    /// positions `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let triplet = [
    ///     Lorentzian::new(0.03, 0.0009, 4.8),
    ///     Lorentzian::new(0.02, 0.0004, 5.0),
    ///     Lorentzian::new(0.03, 0.0009, 5.2),
    /// ];
    /// let chemical_shifts = (0..100)
    ///     .map(|x| x as f64 * 10.0 / 99.0)
    ///     .collect::<Vec<f64>>();
    /// let sup = Lorentzian::superposition_vec(&chemical_shifts, &triplet);
    /// ```
    pub fn superposition_vec<L: AsRef<Lorentzian>>(x: &[f64], lorentzians: &[L]) -> Vec<f64> {
        x.iter()
            .map(|&x| Self::superposition(x, lorentzians))
            .collect()
    }

    /// Evaluates the superposition of the given `Lorentzian`s at the given
    /// position `x` in parallel.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::deconvolution::Lorentzian;
    ///
    /// let triplet = [
    ///     Lorentzian::new(0.03, 0.0009, 4.8),
    ///     Lorentzian::new(0.02, 0.0004, 5.0),
    ///     Lorentzian::new(0.03, 0.0009, 5.2),
    /// ];
    /// let chemical_shifts = (0..100)
    ///     .map(|x| x as f64 * 10.0 / 99.0)
    ///     .collect::<Vec<f64>>();
    /// let sup = Lorentzian::par_superposition_vec(&chemical_shifts, &triplet);
    /// ```
    #[cfg(feature = "parallel")]
    pub fn par_superposition_vec<L: AsRef<Lorentzian> + Send + Sync>(
        x: &[f64],
        lorentzians: &[L],
    ) -> Vec<f64> {
        x.par_iter()
            .map(|&x| Self::superposition(x, lorentzians))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};
    use float_cmp::assert_approx_eq;

    #[test]
    fn thread_safety() {
        assert_send!(Lorentzian);
        assert_sync!(Lorentzian);
    }

    #[test]
    fn accessors() {
        let lorentzian = Lorentzian::new(1.0, 0.25, 0.0);
        assert_approx_eq!(f64, lorentzian.sfhw(), 1.0);
        assert_approx_eq!(f64, lorentzian.hw2(), 0.25);
        assert_approx_eq!(f64, lorentzian.maxp(), 0.0);
        assert_approx_eq!(f64, lorentzian.sf(), 2.0);
        assert_approx_eq!(f64, lorentzian.hw(), 0.5);
    }

    #[test]
    fn mutators() {
        let mut lorentzian = Lorentzian::new(1.0, 1.0, 0.0);
        lorentzian.set_sfhw(1.5);
        lorentzian.set_hw2(2.25);
        lorentzian.set_maxp(1.0);
        assert_approx_eq!(f64, lorentzian.sfhw(), 1.5);
        assert_approx_eq!(f64, lorentzian.hw2(), 2.25);
        assert_approx_eq!(f64, lorentzian.maxp(), 1.0);
        assert_approx_eq!(f64, lorentzian.sf(), 1.0);
        assert_approx_eq!(f64, lorentzian.hw(), 1.5);
        lorentzian.set_parameters(1.0, 1.0, 0.0);
        assert_approx_eq!(f64, lorentzian.sfhw(), 1.0);
        assert_approx_eq!(f64, lorentzian.hw2(), 1.0);
        assert_approx_eq!(f64, lorentzian.maxp(), 0.0);
        assert_approx_eq!(f64, lorentzian.sf(), 1.0);
        assert_approx_eq!(f64, lorentzian.hw(), 1.0);
    }

    #[test]
    fn evaluate() {
        let lorentzian = Lorentzian::new(1.0, 1.0, 0.0);
        let chemical_shifts = (0..11)
            .map(|x| -5.0 + x as f64)
            .collect::<Vec<f64>>();
        let expected_intensities = [
            1.0 / 26.0,
            1.0 / 17.0,
            1.0 / 10.0,
            1.0 / 5.0,
            1.0 / 2.0,
            1.0 / 1.0,
            1.0 / 2.0,
            1.0 / 5.0,
            1.0 / 10.0,
            1.0 / 17.0,
            1.0 / 26.0,
        ];
        chemical_shifts
            .iter()
            .zip(expected_intensities.iter())
            .for_each(|(&x, &y)| {
                assert_approx_eq!(f64, lorentzian.evaluate(x), y);
            });
        let computed_intensities = lorentzian.evaluate_vec(&chemical_shifts);
        computed_intensities
            .iter()
            .zip(expected_intensities.iter())
            .for_each(|(&yc, &ye)| {
                assert_approx_eq!(f64, yc, ye);
            });
    }

    #[test]
    fn superposition() {
        let lorentzians = vec![
            Lorentzian::new(1.0, 0.5, -2.0),
            Lorentzian::new(2.0, 0.75, 0.0),
            Lorentzian::new(1.0, 0.5, 2.0),
        ];
        let chemical_shifts = (0..11)
            .map(|x| -5.0 + x as f64)
            .collect::<Vec<f64>>();
        let expected_intensities = [
            1.0 / 9.5 + 2.0 / 25.75 + 1.0 / 49.5,
            1.0 / 4.5 + 2.0 / 16.75 + 1.0 / 36.5,
            1.0 / 1.5 + 2.0 / 9.75 + 1.0 / 25.5,
            1.0 / 0.5 + 2.0 / 4.75 + 1.0 / 16.5,
            1.0 / 1.5 + 2.0 / 1.75 + 1.0 / 9.5,
            1.0 / 4.5 + 2.0 / 0.75 + 1.0 / 4.5,
            1.0 / 9.5 + 2.0 / 1.75 + 1.0 / 1.5,
            1.0 / 16.5 + 2.0 / 4.75 + 1.0 / 0.5,
            1.0 / 25.5 + 2.0 / 9.75 + 1.0 / 1.5,
            1.0 / 36.5 + 2.0 / 16.75 + 1.0 / 4.5,
            1.0 / 49.5 + 2.0 / 25.75 + 1.0 / 9.5,
        ];
        chemical_shifts
            .iter()
            .zip(expected_intensities.iter())
            .for_each(|(&x, &y)| {
                assert_approx_eq!(f64, Lorentzian::superposition(x, &lorentzians), y);
            });
        let computed_intensities = Lorentzian::superposition_vec(&chemical_shifts, &lorentzians);
        computed_intensities
            .iter()
            .zip(expected_intensities.iter())
            .for_each(|(&yc, &ye)| {
                assert_approx_eq!(f64, yc, ye);
            });
        #[cfg(feature = "parallel")]
        {
            let computed_intensities =
                Lorentzian::par_superposition_vec(&chemical_shifts, &lorentzians);
            computed_intensities
                .iter()
                .zip(expected_intensities.iter())
                .for_each(|(&yc, &ye)| {
                    assert_approx_eq!(f64, yc, ye);
                });
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let lorentzians = vec![
            Lorentzian::new(5.5, 0.25, 3.0),
            Lorentzian::new(7.0, 0.16, 5.0),
            Lorentzian::new(5.5, 0.25, 7.0),
        ];
        let serialized = serde_json::to_string(&lorentzians).unwrap();
        let deserialized = serde_json::from_str::<Vec<Lorentzian>>(&serialized).unwrap();
        lorentzians
            .iter()
            .zip(deserialized.iter())
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, init.sfhw(), rec.sfhw());
                assert_approx_eq!(f64, init.hw2(), rec.hw2());
                assert_approx_eq!(f64, init.maxp(), rec.maxp());
            });
    }
}
