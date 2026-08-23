//! Types for representing ranges in spectral axes.

use num_traits::Float;
use std::cmp::Ordering;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Access to the raw bounds of a range.
///
/// The bounds are returned in the order they were given. Ordering-aware
/// functionality lives on [`SpectralRange`], which is blanket-implemented for
/// every implementor of this trait.
///
/// # Invariants
///
/// Implementors must guarantee that neither `start` nor `end` can ever return
/// `NaN` or an infinity, i.e. that an instance violating this cannot be
/// obtained in the first place.
pub trait FiniteBounds<T> {
    /// Returns the `start` bound.
    fn start(&self) -> T;

    /// Returns the `end` bound.
    fn end(&self) -> T;
}

/// Ordering-aware operations on a range of values in a spectral axis.
///
/// This trait is blanket-implemented for every [`FiniteBounds`] implementor and
/// is not implemented directly.
pub trait SpectralRange<T> {
    /// Returns `true` if `start < end`.
    ///
    /// A degenerate range (`start == end`) is neither ascending nor descending.
    fn is_ascending(&self) -> bool;

    /// Returns `true` if `start > end`.
    ///
    /// A degenerate range (`start == end`) is neither ascending nor descending.
    fn is_descending(&self) -> bool;

    /// Returns the greater of the two bounds.
    ///
    /// This is `end` for an ascending range and `start` for a descending one.
    fn upper(&self) -> T;

    /// Returns the lesser of the two bounds.
    ///
    /// This is `start` for an ascending range and `end` for a descending one.
    fn lower(&self) -> T;

    /// Returns `true` if `value` lies within the closed interval spanned by the
    /// bounds.
    ///
    /// Unlike the standard library's range types, the check is against
    /// `[lower(), upper()]` rather than `[start, end]`, so it is unaffected by
    /// the direction of the range. Both bounds are inclusive, so two ranges
    /// sharing a bound both contain it.
    ///
    /// Returns `false` for `NaN`.
    fn contains(&self, value: T) -> bool;

    /// Returns the unsigned width of the range.
    ///
    /// Never negative, and zero for a degenerate range. May be infinite if the
    /// bounds are far enough apart that their difference overflows `T`.
    fn width(&self) -> T;

    /// Returns the width of the range, signed by its direction.
    ///
    /// Negative for a descending range, zero for a degenerate one, positive for
    /// an ascending one. May overflow to an infinity, as for
    /// [`SpectralRange::width`].
    fn signed_width(&self) -> T;

    /// Returns the midpoint of the range.
    ///
    /// Never overflows `T`.
    fn center(&self) -> T;
}

impl<T, R> SpectralRange<T> for R
where
    T: Float,
    R: FiniteBounds<T>,
{
    fn is_ascending(&self) -> bool {
        self.start() < self.end()
    }

    fn is_descending(&self) -> bool {
        self.start() > self.end()
    }

    fn upper(&self) -> T {
        self.start().max(self.end())
    }

    fn lower(&self) -> T {
        self.start().min(self.end())
    }

    fn contains(&self, value: T) -> bool {
        self.lower() <= value && value <= self.upper()
    }

    fn width(&self) -> T {
        self.signed_width().abs()
    }

    fn signed_width(&self) -> T {
        self.end() - self.start()
    }

    fn center(&self) -> T {
        let two = T::one() + T::one();

        (self.start() / two) + (self.end() / two)
    }
}

/// The quantity a [`Range`] measures, and the bounds it admits.
///
/// # Invariants
///
/// `is_valid` must be symmetric, i.e., `is_valid(a, b) == is_valid(b, a)`. It
/// must also reject `NaN` and the infinities, so that [`FiniteBounds`]'s
/// invariant holds for every `Range`.
pub trait Domain {
    /// Human-readable name of the domain, used in debug output and in
    /// deserialization errors.
    const NAME: &'static str;

    /// Returns `true` if the pair of bounds is admissible in this domain.
    ///
    /// The order of the bounds carries no meaning here.
    fn is_valid<T>(start: T, end: T) -> bool
    where
        T: Float;
}

/// Marker for ranges of frequencies in Hz.
///
/// # Invariants
///
/// Bounds must not be `NaN` or either infinity.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Frequency;

impl Domain for Frequency {
    const NAME: &'static str = "frequency";

    fn is_valid<T>(start: T, end: T) -> bool
    where
        T: Float,
    {
        start.is_finite() && end.is_finite()
    }
}

/// Marker for ranges of chemical shifts in ppm.
///
/// # Invariants
///
/// Bounds must not be `NaN` or either infinity.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Shift;

impl Domain for Shift {
    const NAME: &'static str = "chemical shift";

    fn is_valid<T>(start: T, end: T) -> bool
    where
        T: Float,
    {
        start.is_finite() && end.is_finite()
    }
}

/// Marker for ranges of relative units.
///
/// A bound of `0` denotes the start of the axis and `1` its end
///
/// # Invariants
///
/// Bounds must be in `[0, 1]`.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Relative;

impl Domain for Relative {
    const NAME: &'static str = "relative";

    fn is_valid<T>(start: T, end: T) -> bool
    where
        T: Float,
    {
        (T::zero()..=T::one()).contains(&start) && (T::zero()..=T::one()).contains(&end)
    }
}

/// A range of frequencies in Hz.
///
/// See [`Frequency`] for the admitted bounds.
pub type FrequencyRange<T> = Range<T, Frequency>;

/// A range of chemical shifts in ppm.
///
/// See [`Shift`] for the admitted bounds.
pub type ShiftRange<T> = Range<T, Shift>;

/// A range of relative units.
///
/// See [`Relative`] for the admitted bounds.
pub type RelativeRange<T> = Range<T, Relative>;

/// Generic range type over a domain.
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// With the `serde` feature enabled, `Range` can be serialized using `serde`.
/// The two directions have different requirements:
///
/// - [`Serialize`] needs only `T: Serialize`. The domain is irrelevant, since a
///   `Range` that exists is already valid.
/// - [`Deserialize`] needs `T: Float + Deserialize<'de>` and `D: Domain`, so
///   that the bounds can be checked against the domain.
///
/// [`Serialize`]: https://docs.rs/serde/latest/serde/trait.Serialize.html
/// [`Deserialize`]: https://docs.rs/serde/latest/serde/trait.Deserialize.html
///
/// Deserialization goes through [`Range::new`] and fails if the bounds are
/// invalid for `D`.
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(
        try_from = "RawRange<T>",
        bound(deserialize = "T: Float + Deserialize<'de>, D: Domain")
    )
)]
pub struct Range<T, D> {
    /// Start of the range (inclusive).
    start: T,
    /// End of the range (inclusive).
    end: T,
    /// Domain of the range (frequency, chemical shift, relative).
    #[cfg_attr(feature = "serde", serde(skip))]
    domain: PhantomData<fn() -> D>,
}

impl<T, D> Copy for Range<T, D> where T: Copy {}

impl<T, D> Clone for Range<T, D>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            start: self.start.clone(),
            end: self.end.clone(),
            domain: PhantomData,
        }
    }
}

impl<T, D> PartialEq for Range<T, D>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

// we banished NaNs so this is okay as per Rust's Eq docs.
impl<T, D> Eq for Range<T, D>
where
    T: Float,
    D: Domain,
{
}

impl<T, D> PartialOrd for Range<T, D>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.start
                .partial_cmp(&other.start)?
                .then(self.end.partial_cmp(&other.end)?),
        )
    }
}

// we banished NaNs so this is okay as per Rust's Ord docs.
impl<T, D> Ord for Range<T, D>
where
    T: Float,
    D: Domain,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("D must reject NaN, so the ordering is total")
    }
}

impl<T, D> std::fmt::Debug for Range<T, D>
where
    T: std::fmt::Debug,
    D: Domain,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Range")
            .field("start", &self.start)
            .field("end", &self.end)
            .field("domain", &format_args!("{}", D::NAME))
            .finish()
    }
}

impl<T, D> FiniteBounds<T> for Range<T, D>
where
    T: Copy,
{
    fn start(&self) -> T {
        self.start
    }

    fn end(&self) -> T {
        self.end
    }
}

impl<T, D> Range<T, D>
where
    T: Float,
    D: Domain,
{
    /// Creates a new `Range`.
    ///
    /// Returns `None` if the bounds are invalid for the domain.
    pub fn new(start: T, end: T) -> Option<Self> {
        if !D::is_valid(start, end) || !(end - start).is_finite() {
            return None;
        }

        Some(Self {
            start,
            end,
            domain: PhantomData,
        })
    }

    /// Returns an equivalent range with `start <= end`, swapping bounds if
    /// necessary.
    pub fn normalized(self) -> Self {
        if self.start <= self.end {
            self
        } else {
            Self {
                start: self.end,
                end: self.start,
                domain: PhantomData,
            }
        }
    }
}

/// Range type without invariants as an intermediate for deserialization.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct RawRange<T> {
    /// Start of the range (inclusive).
    start: T,
    /// End of the range (inclusive).
    end: T,
}

#[cfg(feature = "serde")]
impl<T, D> TryFrom<RawRange<T>> for Range<T, D>
where
    T: Float,
    D: Domain,
{
    type Error = String;

    fn try_from(value: RawRange<T>) -> Result<Self, Self::Error> {
        Self::new(value.start, value.end)
            .ok_or_else(|| format!("invalid bounds for {} range", D::NAME))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    fn frequency_asc_desc<T>() -> (FrequencyRange<T>, FrequencyRange<T>)
    where
        T: Float + std::fmt::Debug,
    {
        let zero = T::zero();
        let hundred = T::from(100).unwrap();
        let asc = FrequencyRange::new(zero, hundred).unwrap();
        let desc = FrequencyRange::new(hundred, zero).unwrap();

        (asc, desc)
    }

    fn shift_asc_desc<T>() -> (ShiftRange<T>, ShiftRange<T>)
    where
        T: Float + std::fmt::Debug,
    {
        let five = T::from(5).unwrap();
        let fifteen = T::from(15).unwrap();
        let asc = ShiftRange::new(-five, fifteen).unwrap();
        let desc = ShiftRange::new(fifteen, -five).unwrap();

        (asc, desc)
    }

    fn relative_asc_desc<T>() -> (RelativeRange<T>, RelativeRange<T>)
    where
        T: Float + std::fmt::Debug,
    {
        let zero = T::zero();
        let one = T::one();
        let asc = RelativeRange::new(zero, one).unwrap();
        let desc = RelativeRange::new(one, zero).unwrap();

        (asc, desc)
    }

    #[test]
    fn thread_safety() {
        assert_impl_all!(FrequencyRange<f32>: Send, Sync);
        assert_impl_all!(FrequencyRange<f64>: Send, Sync);
        assert_impl_all!(ShiftRange<f32>: Send, Sync);
        assert_impl_all!(ShiftRange<f64>: Send, Sync);
        assert_impl_all!(RelativeRange<f32>: Send, Sync);
        assert_impl_all!(RelativeRange<f64>: Send, Sync);
    }

    #[test]
    fn frequency_invariants() {
        fn invariants_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let zero = T::zero();
            let one = T::one();
            let nan = T::nan();
            let inf = T::infinity();
            let neg_inf = T::neg_infinity();

            assert!(FrequencyRange::new(zero, zero).is_some());
            assert!(FrequencyRange::new(zero, one).is_some());
            assert!(FrequencyRange::new(one, zero).is_some());

            assert!(FrequencyRange::new(nan, zero).is_none());
            assert!(FrequencyRange::new(inf, zero).is_none());
            assert!(FrequencyRange::new(neg_inf, zero).is_none());
            assert!(FrequencyRange::new(zero, nan).is_none());
            assert!(FrequencyRange::new(zero, inf).is_none());
            assert!(FrequencyRange::new(zero, neg_inf).is_none());
            assert!(FrequencyRange::new(nan, nan).is_none());
            assert!(FrequencyRange::new(inf, inf).is_none());
            assert!(FrequencyRange::new(neg_inf, neg_inf).is_none());
        }

        invariants_::<f32>();
        invariants_::<f64>();
    }

    #[test]
    fn shift_invariants() {
        fn invariants_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let zero = T::zero();
            let one = T::one();
            let nan = T::nan();
            let inf = T::infinity();
            let neg_inf = T::neg_infinity();

            assert!(ShiftRange::new(zero, zero).is_some());
            assert!(ShiftRange::new(zero, one).is_some());
            assert!(ShiftRange::new(one, zero).is_some());
            assert!(ShiftRange::new(zero, -one).is_some());
            assert!(ShiftRange::new(-one, zero).is_some());

            assert!(ShiftRange::new(zero, nan).is_none());
            assert!(ShiftRange::new(zero, inf).is_none());
            assert!(ShiftRange::new(zero, neg_inf).is_none());
            assert!(ShiftRange::new(nan, zero).is_none());
            assert!(ShiftRange::new(inf, zero).is_none());
            assert!(ShiftRange::new(neg_inf, zero).is_none());
            assert!(ShiftRange::new(nan, nan).is_none());
            assert!(ShiftRange::new(inf, inf).is_none());
            assert!(ShiftRange::new(neg_inf, neg_inf).is_none());
        }

        invariants_::<f32>();
        invariants_::<f64>();
    }

    #[test]
    fn relative_invariants() {
        fn invariants_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let zero = T::zero();
            let one = T::one();
            let two = one + one;
            let nan = T::nan();
            let inf = T::infinity();
            let neg_inf = T::neg_infinity();

            assert!(RelativeRange::new(zero, zero).is_some());
            assert!(RelativeRange::new(zero, one).is_some());
            assert!(RelativeRange::new(one, zero).is_some());

            assert!(RelativeRange::new(zero, two).is_none());
            assert!(RelativeRange::new(zero, -one).is_none());
            assert!(RelativeRange::new(zero, nan).is_none());
            assert!(RelativeRange::new(zero, inf).is_none());
            assert!(RelativeRange::new(zero, neg_inf).is_none());
            assert!(RelativeRange::new(two, zero).is_none());
            assert!(RelativeRange::new(-one, zero).is_none());
            assert!(RelativeRange::new(nan, zero).is_none());
            assert!(RelativeRange::new(inf, zero).is_none());
            assert!(RelativeRange::new(neg_inf, zero).is_none());
            assert!(RelativeRange::new(two, two).is_none());
            assert!(RelativeRange::new(-one, -one).is_none());
            assert!(RelativeRange::new(nan, nan).is_none());
            assert!(RelativeRange::new(inf, inf).is_none());
            assert!(RelativeRange::new(neg_inf, neg_inf).is_none());
        }

        invariants_::<f32>();
        invariants_::<f64>();
    }

    #[test]
    fn normalized() {
        fn normalized_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let (asc, desc) = frequency_asc_desc::<T>();

            assert_eq!(asc, desc.normalized());

            let (asc, desc) = shift_asc_desc::<T>();

            assert_eq!(asc, desc.normalized());

            let (asc, desc) = relative_asc_desc::<T>();

            assert_eq!(asc, desc.normalized());
        }

        normalized_::<f32>();
        normalized_::<f64>();
    }

    #[test]
    fn ascending_descending() {
        fn ascending_descending_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let (asc, desc) = frequency_asc_desc::<T>();
            let neither = FrequencyRange::new(asc.center(), asc.center()).unwrap();

            assert!(asc.is_ascending());
            assert!(desc.is_descending());
            assert!(!(neither.is_ascending() || neither.is_descending()));

            let (asc, desc) = shift_asc_desc::<T>();
            let neither = ShiftRange::new(asc.center(), asc.center()).unwrap();

            assert!(asc.is_ascending());
            assert!(desc.is_descending());
            assert!(!(neither.is_ascending() || neither.is_descending()));

            let (asc, desc) = relative_asc_desc::<T>();
            let neither = RelativeRange::new(asc.center(), asc.center()).unwrap();

            assert!(asc.is_ascending());
            assert!(desc.is_descending());
            assert!(!(neither.is_ascending() || neither.is_descending()));
        }

        ascending_descending_::<f32>();
        ascending_descending_::<f64>();
    }

    #[test]
    fn upper_lower() {
        fn upper_lower_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let (asc, desc) = frequency_asc_desc::<T>();
            let hundred = T::from(100).unwrap();
            let zero = T::zero();

            assert_eq!(asc.upper(), hundred);
            assert_eq!(desc.upper(), hundred);
            assert_eq!(asc.lower(), zero);
            assert_eq!(desc.lower(), zero);

            let (asc, desc) = shift_asc_desc::<T>();
            let five = T::from(5).unwrap();
            let fifteen = T::from(15).unwrap();

            assert_eq!(asc.upper(), fifteen);
            assert_eq!(desc.upper(), fifteen);
            assert_eq!(asc.lower(), -five);
            assert_eq!(desc.lower(), -five);

            let (asc, desc) = relative_asc_desc::<T>();
            let one = T::one();

            assert_eq!(asc.upper(), one);
            assert_eq!(desc.upper(), one);
            assert_eq!(asc.lower(), zero);
            assert_eq!(desc.lower(), zero);
        }

        upper_lower_::<f32>();
        upper_lower_::<f64>();
    }

    #[test]
    fn contains() {
        fn contains_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let (asc, desc) = frequency_asc_desc::<T>();
            let one = T::one();
            let hundred = T::from(100).unwrap();

            assert!(
                (0..101)
                    .map(|i| T::from(i).unwrap())
                    .all(|x| asc.contains(x))
            );
            assert!(
                (0..101)
                    .map(|i| T::from(i).unwrap())
                    .all(|x| desc.contains(x))
            );
            assert!(!asc.contains(hundred + one));
            assert!(!asc.contains(-one));
            assert!(!asc.contains(T::nan()));
            assert!(!asc.contains(T::infinity()));
            assert!(!asc.contains(T::neg_infinity()));
            assert!(!desc.contains(hundred + one));
            assert!(!desc.contains(-one));
            assert!(!desc.contains(T::nan()));
            assert!(!desc.contains(T::infinity()));
            assert!(!desc.contains(T::neg_infinity()));

            let (asc, desc) = shift_asc_desc::<T>();
            let five = T::from(5).unwrap();
            let fifteen = T::from(15).unwrap();
            let twenty = T::from(20).unwrap();

            assert!(
                (0..101)
                    .map(|i| T::from(i).unwrap())
                    .map(|t| -five + twenty * t / hundred)
                    .all(|x| asc.contains(x))
            );
            assert!(
                (0..101)
                    .map(|i| T::from(i).unwrap())
                    .map(|t| -five + twenty * t / hundred)
                    .all(|x| desc.contains(x))
            );
            assert!(!asc.contains(fifteen + one));
            assert!(!asc.contains(-five - one));
            assert!(!asc.contains(T::nan()));
            assert!(!asc.contains(T::infinity()));
            assert!(!asc.contains(T::neg_infinity()));
            assert!(!desc.contains(fifteen + one));
            assert!(!desc.contains(-five - one));
            assert!(!desc.contains(T::nan()));
            assert!(!desc.contains(T::infinity()));
            assert!(!desc.contains(T::neg_infinity()));

            let (asc, desc) = relative_asc_desc::<T>();
            let two = one + one;

            assert!(
                (0..101)
                    .map(|i| T::from(i).unwrap())
                    .map(|t| t / hundred)
                    .all(|x| asc.contains(x))
            );
            assert!(
                (0..101)
                    .map(|i| T::from(i).unwrap())
                    .map(|t| t / hundred)
                    .all(|x| desc.contains(x))
            );
            assert!(!asc.contains(two));
            assert!(!asc.contains(-one));
            assert!(!asc.contains(T::nan()));
            assert!(!asc.contains(T::infinity()));
            assert!(!asc.contains(T::neg_infinity()));
            assert!(!desc.contains(two));
            assert!(!desc.contains(-one));
            assert!(!desc.contains(T::nan()));
            assert!(!desc.contains(T::infinity()));
            assert!(!desc.contains(T::neg_infinity()));
        }

        contains_::<f32>();
        contains_::<f64>();
    }

    #[test]
    fn widths() {
        fn widths_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let (asc, desc) = frequency_asc_desc::<T>();
            let hundred = T::from(100).unwrap();

            assert_eq!(asc.width(), hundred);
            assert_eq!(desc.width(), hundred);
            assert_eq!(asc.signed_width(), hundred);
            assert_eq!(desc.signed_width(), -hundred);

            let (asc, desc) = shift_asc_desc::<T>();
            let twenty = T::from(20).unwrap();

            assert_eq!(asc.width(), twenty);
            assert_eq!(desc.width(), twenty);
            assert_eq!(asc.signed_width(), twenty);
            assert_eq!(desc.signed_width(), -twenty);

            let (asc, desc) = relative_asc_desc::<T>();

            assert_eq!(asc.width(), T::one());
            assert_eq!(desc.width(), T::one());
            assert_eq!(asc.signed_width(), T::one());
            assert_eq!(desc.signed_width(), -T::one());

            let overflow = ShiftRange::new(T::min_value(), T::max_value());
            assert!(overflow.is_none());
        }

        widths_::<f32>();
        widths_::<f64>();
    }

    #[test]
    fn center() {
        fn center_<T>()
        where
            T: Float + std::fmt::Debug,
        {
            let (asc, desc) = frequency_asc_desc::<T>();
            let fifty = T::from(50).unwrap();

            assert_eq!(asc.center(), fifty);
            assert_eq!(desc.center(), fifty);

            let (asc, desc) = shift_asc_desc::<T>();
            let five = T::from(5).unwrap();

            assert_eq!(asc.center(), five);
            assert_eq!(desc.center(), five);

            let (asc, desc) = relative_asc_desc::<T>();
            let half = T::from(0.5).unwrap();

            assert_eq!(asc.center(), half);
            assert_eq!(desc.center(), half);

            let overflow = ShiftRange::new(T::min_value(), T::max_value());
            assert!(overflow.is_none());
        }

        center_::<f32>();
        center_::<f64>();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        fn round_trip_<T, D>((asc, desc): (Range<T, D>, Range<T, D>))
        where
            T: Float + Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
            D: Domain,
        {
            let asc_ser = serde_json5::to_string(&asc).unwrap();
            let asc_de = serde_json5::from_str::<Range<T, D>>(&asc_ser).unwrap();

            assert_eq!(asc, asc_de);

            let desc_ser = serde_json5::to_string(&desc).unwrap();
            let desc_de = serde_json5::from_str::<Range<T, D>>(&desc_ser).unwrap();

            assert_eq!(desc, desc_de);
        }

        round_trip_::<f32, Frequency>(frequency_asc_desc());
        round_trip_::<f32, Shift>(shift_asc_desc());
        round_trip_::<f32, Relative>(relative_asc_desc());
        round_trip_::<f64, Frequency>(frequency_asc_desc());
        round_trip_::<f64, Shift>(shift_asc_desc());
        round_trip_::<f64, Relative>(relative_asc_desc());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization_invariants() {
        fn deserialization_invariants_<T>()
        where
            T: Float + serde::de::DeserializeOwned + std::fmt::Debug,
        {
            let bad_freq_ranges = [
                "{ \"start\": 0.0, \"end\": NaN }",
                "{ \"start\": 0.0, \"end\": Infinity }",
                "{ \"start\": 0.0, \"end\": -Infinity }",
                "{ \"start\": NaN, \"end\": 0.0 }",
                "{ \"start\": Infinity, \"end\": 0.0 }",
                "{ \"start\": -Infinity, \"end\": 0.0 }",
                "{ \"start\": NaN, \"end\": NaN }",
                "{ \"start\": Infinity, \"end\": Infinity }",
                "{ \"start\": -Infinity, \"end\": -Infinity }",
            ];

            for bad_range in bad_freq_ranges {
                let err = serde_json5::from_str::<FrequencyRange<T>>(bad_range)
                    .map_err(|e| e.to_string())
                    .unwrap_err();

                assert!(err.contains("invalid bounds") && err.contains("frequency"));
            }

            let bad_shift_ranges = [
                "{ \"start\": 0.0, \"end\": NaN }",
                "{ \"start\": 0.0, \"end\": Infinity }",
                "{ \"start\": 0.0, \"end\": -Infinity }",
                "{ \"start\": NaN, \"end\": 0.0 }",
                "{ \"start\": Infinity, \"end\": 0.0 }",
                "{ \"start\": -Infinity, \"end\": 0.0 }",
                "{ \"start\": NaN, \"end\": NaN }",
                "{ \"start\": Infinity, \"end\": Infinity }",
                "{ \"start\": -Infinity, \"end\": -Infinity }",
            ];

            for bad_range in bad_shift_ranges {
                let err = serde_json5::from_str::<ShiftRange<T>>(bad_range)
                    .map_err(|e| e.to_string())
                    .unwrap_err();

                assert!(err.contains("invalid bounds") && err.contains("chemical shift"));
            }

            let bad_relative_ranges = [
                "{ \"start\": 0.0, \"end\": 2.0 }",
                "{ \"start\": 0.0, \"end\": -1.0 }",
                "{ \"start\": 0.0, \"end\": NaN }",
                "{ \"start\": 0.0, \"end\": Infinity }",
                "{ \"start\": 0.0, \"end\": -Infinity }",
                "{ \"start\": 2.0, \"end\": 0.0 }",
                "{ \"start\": -1.0, \"end\": 0.0 }",
                "{ \"start\": NaN, \"end\": 0.0 }",
                "{ \"start\": Infinity, \"end\": 0.0 }",
                "{ \"start\": -Infinity, \"end\": 0.0 }",
                "{ \"start\": 2.0, \"end\": 2.0 }",
                "{ \"start\": -1.0, \"end\": -1.0 }",
                "{ \"start\": NaN, \"end\": NaN }",
                "{ \"start\": Infinity, \"end\": Infinity }",
                "{ \"start\": -Infinity, \"end\": -Infinity }",
            ];

            for bad_range in bad_relative_ranges {
                let err = serde_json5::from_str::<RelativeRange<T>>(bad_range)
                    .map_err(|e| e.to_string())
                    .unwrap_err();

                assert!(err.contains("invalid bounds") && err.contains("relative"));
            }
        }

        deserialization_invariants_::<f32>();
        deserialization_invariants_::<f64>();
    }
}
