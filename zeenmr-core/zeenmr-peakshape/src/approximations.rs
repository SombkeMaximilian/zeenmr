//! Numerical approximations non-elementary functions.

use crate::util::fma;
use num_traits::Float;

/// Approximation for `2^(self)`.
pub trait Exp2: Float {
    /// Returns `2^(self)`.
    #[inline(always)]
    fn exp2_fast(&self) -> Self {
        Float::exp2(*self)
    }
}

// Minimax polynomials for 2^x on [-1/2, 1/2], for a vectorizable exp2 kernel
// using Sollya.
//
// The constant term is pinned to 1 (the [|1, ...|] lattice) so that
// exp2(0) == 1.0 bit-wise, allowing the Horner's method's last iteration to be
// fma(m, f, 1.0)
//
// Chosen:  f32 degree 6  -> 0.060 ulp
//          f64 degree 10 -> 2.221 ulp
//
// Sollya script:
// ```
// prec = 1000!;
//
// f = 2^x;
// d = [-0.5; 0.5];
//
// print("f32 (ulp = 2^-24):");
// for i from 5 to 7 do {
// p = fpminimax(f, i, [|1, SG...|], d, relative);
// print("  degree", i, ":", dirtyinfnorm(p/f - 1, d) / 1b-24, "ulp");
// };
//
// print("f64 (ulp = 2^-53):");
// for i from 9 to 12 do {
// p = fpminimax(f, i, [|1, D...|], d, relative);
// print("  degree", i, ":", dirtyinfnorm(p/f - 1, d) / 1b-53, "ulp");
// };
//
// p32 = fpminimax(f, 6,  [|1, SG...|], d, relative);
// p64 = fpminimax(f, 10, [|1, D...|],  d, relative);
//
// print("");
// print("f32 degree  6:", dirtyinfnorm(p32/f - 1, d) / 1b-24, "ulp");
// print("       error :", supnorm(p32, f, d, relative, 1b-30));
// print("f64 degree 10:", dirtyinfnorm(p64/f - 1, d) / 1b-53, "ulp");
// print("       error :", supnorm(p64, f, d, relative, 1b-60));
//
// print("");
// print("const P32: [u32; 6] = [   // f^6 .. f^1");
// for i from 6 to 1 by -1 do { printsingle(coeff(p32, i)); };
// print("];");
//
// print("");
// print("const P64: [u64; 10] = [  // f^10 .. f^1");
// for i from 10 to 1 by -1 do { printdouble(coeff(p64, i)); };
// print("];");
// ```

impl Exp2 for f32 {
    #[inline(always)]
    fn exp2_fast(&self) -> Self {
        // k + 127 must stay in [1, 254] to avoid subnormal results
        let t = self.clamp(-126_f32, 127_f32);

        // Range reduction: t = k + f, k integral, f in [-0.5, 0.5].
        let k = t.round_ties_even();
        let f = t - k;

        // Sollya fpminimax in [-0.5, 0.5] with 0.060 ulp.
        // Order: f^6 .. f^1.
        const P32: [f32; 6] = [
            f32::from_bits(0x391fc834),
            f32::from_bits(0x3aaf8533),
            f32::from_bits(0x3c1d9858),
            f32::from_bits(0x3d6357bc),
            f32::from_bits(0x3e75fdec),
            f32::from_bits(0x3f317218),
        ];

        // Mantissa https://en.wikipedia.org/wiki/Horner%27s_method
        let mut m = P32[0];
        for c in &P32[1..] {
            m = fma(m, f, *c);
        }
        m = fma(m, f, 1.0);

        // 2^k by direct exponent construction.
        let e = f32::from_bits((((k as i32) + 127) as u32) << 23);

        m * e
    }
}

impl Exp2 for f64 {
    #[inline(always)]
    fn exp2_fast(&self) -> Self {
        // k + 1023 must stay in [1, 2046] to avoid subnormal results
        let t = self.clamp(-1022_f64, 1023_f64);

        // Range reduction: t = k + f, k integral, f in [-0.5, 0.5].
        let k = t.round_ties_even();
        let f = t - k;

        // Sollya fpminimax in [-0.5, 0.5] with 2.221 ulp.
        // Order: f^10 .. f^1.
        const P64: [f64; 10] = [
            f64::from_bits(0x3e3e3fe3260d8ef6),
            f64::from_bits(0x3e7b673da5762029),
            f64::from_bits(0x3eb62c111ae2491f),
            f64::from_bits(0x3eeffcb56287b492),
            f64::from_bits(0x3f243091283ef5fd),
            f64::from_bits(0x3f55d87fe9cab400),
            f64::from_bits(0x3f83b2ab6fbce392),
            f64::from_bits(0x3fac6b08d703d59e),
            f64::from_bits(0x3fcebfbdff82c472),
            f64::from_bits(0x3fe62e42fefa3a17),
        ];

        // Mantissa https://en.wikipedia.org/wiki/Horner%27s_method
        let mut m = P64[0];
        for c in &P64[1..] {
            m = fma(m, f, *c);
        }
        m = fma(m, f, 1.0);

        // 2^k by direct exponent construction.
        let e = f64::from_bits((((k as i64) + 1023) as u64) << 52);

        m * e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_f32_reduced_interval() {
        let mut worst = 0.0f64;
        for i in 0..=1_000_000u32 {
            let f = -0.5 + (i as f32) * (1.0 / 1e6);
            let got = f.exp2_fast() as f64;
            let want = (f as f64).exp2();
            worst = worst.max(((got - want) / want).abs());
        }
        assert!(worst < 0.75 * f32::EPSILON as f64, "max rel err {worst:e}");
    }

    #[test]
    fn exp2_f32_full_range() {
        assert_eq!(0.0f32.exp2_fast(), 1.0);
        assert_eq!(1.0f32.exp2_fast(), 2.0);
        assert_eq!((-1.0f32).exp2_fast(), 0.5);

        let mut worst = 0.0f64;
        for i in 0..=2_000_000u32 {
            let x = -126.0 + (i as f32) * (253.0 / 2e6);
            let got = x.exp2_fast() as f64;
            let want = (x as f64).exp2();
            worst = worst.max(((got - want) / want).abs());
        }
        assert!(worst < 1.25 * f32::EPSILON as f64, "max rel err {worst:e}");
    }

    #[test]
    fn exp2_f32_edges() {
        assert!((-200.0f32).exp2_fast().is_normal());
        assert!(200.0f32.exp2_fast().is_finite());
        assert!(f32::NAN.exp2_fast().is_nan());

        for k in -126..=127i32 {
            let x = k as f32;
            assert_eq!(x.exp2_fast(), (x as f64).exp2() as f32, "k = {k}");
        }
    }

    #[test]
    fn exp2_f64_reduced_interval() {
        let mut worst = 0.0f64;
        for i in 0..=1_000_000u32 {
            let f = -0.5 + (i as f64) * (1.0 / 1e6);
            let got = f.exp2_fast();
            let want = f.exp2();
            worst = worst.max(((got - want) / want).abs());
        }
        assert!(worst < 2.5 * f64::EPSILON, "max rel err {worst:e}");
    }

    #[test]
    fn exp2_f64_full_range() {
        assert_eq!(0.0f64.exp2_fast(), 1.0);
        assert_eq!(1.0f64.exp2_fast(), 2.0);
        assert_eq!((-1.0f64).exp2_fast(), 0.5);

        let mut worst = 0.0f64;
        for i in 0..=2_000_000u32 {
            let x = -1022.0 + (i as f64) * (2045.0 / 2e6);
            let got = x.exp2_fast();
            let want = x.exp2();
            worst = worst.max(((got - want) / want).abs());
        }
        assert!(worst < 2.5 * f64::EPSILON, "max rel err {worst:e}");
    }

    #[test]
    fn exp2_f64_edges() {
        assert!((-2000.0f64).exp2_fast().is_normal());
        assert!(2000.0f64.exp2_fast().is_finite());
        assert!(f64::NAN.exp2_fast().is_nan());

        for k in -1022..=1023i32 {
            let x = k as f64;
            assert_eq!(x.exp2_fast(), x.exp2(), "k = {k}");
        }
    }
}
