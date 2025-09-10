
/// A 1D NMR peak, represented by its left bound, center, and right bound.
///
/// The left and right bounds are the points where the curvature changes from
/// concave down to concave up (inflection points). The central point is where
/// a local maximum occurs. When constructing this type, it must be ensured that
/// `left < center < right`, and that the aforementioned curvature properties
/// hold.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Peak {
    /// Left bound index of the peak.
    pub left: usize,
    /// Center index of the peak (the maximum).
    pub center: usize,
    /// Right bound index of the peak.
    pub right: usize,
}
