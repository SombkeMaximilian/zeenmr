use num_traits::Float;

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
    T: Float,
{
    (fma(n1, d2, n2 * d1), d1 * d2)
}

/// Fuses `parts` repeatedly to produce the final output.
pub(crate) fn fuse_fold<T, const K: usize>(parts: [(T, T); K]) -> (T, T)
where
    T: Float,
{
    let mut acc = parts[0];
    for part in parts.iter().skip(1) {
        acc = fuse(acc, *part);
    }

    acc
}

/// Performs the fused multiply-add operation `(a * b) + c`.
///
/// Defers to the regular two-stage operation if there is no hardware fma.
#[inline(always)]
pub(crate) fn fma<T>(a: T, b: T, c: T) -> T
where
    T: Float,
{
    #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
    {
        a * b + c
    }
}
