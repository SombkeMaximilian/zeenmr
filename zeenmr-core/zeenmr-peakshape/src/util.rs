use std::ops::{Add, Mul};

/// Fuses two `num` and `den` pairs into a combined `num` and `den`.
///
/// Let `n1`, `d1`, `n2`, `d2` be two such pairs.
///
/// ```text
/// n = n1 * d2 + n2 * d1
/// d = d1 * d2
/// ```
pub(crate) fn fuse<T>((n1, d1): (T, T), (n2, d2): (T, T)) -> (T, T)
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
{
    (n1 * d2 + n2 * d1, d1 * d2)
}

/// Fuses `parts` repeatedly to produce the final output.
pub(crate) fn fuse_fold<T, const K: usize>(parts: [(T, T); K]) -> (T, T)
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
{
    let mut acc = parts[0];
    for part in parts.iter().skip(1) {
        acc = fuse(acc, *part);
    }

    acc
}
