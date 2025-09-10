
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Peak {
    /// Left bound index of the peak.
    pub left: usize,
    /// Center index of the peak (the maximum).
    pub center: usize,
    /// Right bound index of the peak.
    pub right: usize,
}
