use num_traits::Float;

/// Performs the fused multiply-add operation `(a * b) + c`.
///
/// Defers to the regular two-stage operation if there is no hardware fma. Note
/// that this changes the result due to using two rounding steps instead of one.
#[inline(always)]
pub fn fma<T>(a: T, b: T, c: T) -> T
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
