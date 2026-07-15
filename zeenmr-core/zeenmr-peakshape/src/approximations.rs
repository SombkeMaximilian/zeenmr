//! Numerical approximations non-elementary functions.

use crate::util::fma;
use num_traits::Float;

/// Approximation for `2^(self)`.
pub trait Exp2: Float {
    /// Returns `2^(self)` for non-positive `self`.
    ///
    /// `NaN` maps to the saturation bound. Positive inputs return an
    /// unspecified but valid, non-UB value.
    #[inline(always)]
    fn exp2_fast_nonpos(self) -> Self {
        Float::exp2(self)
    }
}

// Sollya script:
//
//prec = 1000!;
//
//f = 2^x;
//d = [-0.5; 0.5];
//
//f32deg = 3;
//f64deg = 8;
//
//print("f32 (ulp = 2^-24):");
//for i from 3 to 5 do {
//    p = fpminimax(f, i, [|1, SG...|], d, relative);
//    print("  degree", i, ":", dirtyinfnorm(p/f - 1, d) / 1b-24, "ulp");
//};
//
//print("f64 (ulp = 2^-53):");
//for i from 6 to 10 do {
//    p = fpminimax(f, i, [|1, D...|], d, relative);
//    print("  degree", i, ":", dirtyinfnorm(p/f - 1, d) / 1b-53, "ulp");
//};
//
//p32 = fpminimax(f, f32deg, [|1, SG...|], d, relative);
//p64 = fpminimax(f, f64deg, [|1, D...|],  d, relative);
//
//print("");
//print("f32 degree " @ f32deg @ ":", dirtyinfnorm(p32/f - 1, d) / 1b-24, "ulp");
//print("       error:", supnorm(p32, f, d, relative, 1b-30));
//print("f64 degree " @ f64deg @ ":", dirtyinfnorm(p64/f - 1, d) / 1b-53, "ulp");
//print("       error:", supnorm(p64, f, d, relative, 1b-60));
//
//procedure writesinglebits(coefficient_literal) {
//    var res, old;
//    old = display; display = hexadecimal!;
//    res = bashevaluate(
//        "printf '0x%s' $(echo 'printsingle(" @ coefficient_literal @ ");' | \n"
//            @ "sollya 2>/dev/null | \n"
//            @ "tr 'a-f' 'A-F' | \n"
//            @ "grep -oE '[0-9A-F]{4}' | \n"
//            @ "paste -sd_ -); true"
//    );
//    display = old!;
//    write(res);
//};
//
//print("");
//print("const P32: [f32; " @ f32deg @ "] = [");
//for i from f32deg to 1 by -1 do {
//    write("    f32::from_bits(");
//    writesinglebits(coeff(p32, i));
//    print("),");
//};
//print("];");
//
//procedure writedoublebits(coefficient_literal) {
//    var res, old;
//    old = display; display = hexadecimal!;
//    res = bashevaluate(
//        "printf '0x%s' $(echo 'printdouble(" @ coefficient_literal @ ");' | \n"
//            @ "sollya 2>/dev/null | \n"
//            @ "tr 'a-f' 'A-F' | \n"
//            @ "grep -oE '[0-9A-F]{4}' | \n"
//            @ "paste -sd_ -); true"
//    );
//    display = old!;
//    write(res);
//};
//
//print("");
//print("const P64: [f64; " @ f64deg @ "] = [");
//for i from f64deg to 1 by -1 do {
//    write("    f64::from_bits(");
//    writedoublebits(coeff(p64, i));
//    print("),");
//};
//print("];");

impl Exp2 for f32 {
    #[inline(always)]
    fn exp2_fast_nonpos(self) -> Self {
        // 2^k underflows f32 below k = -126, and the reduction below is only
        // exact for |t| < 2^22.
        const EXP2_MIN: f32 = -127_f32;

        // Adding 1.5 * 2^23 rounds the integral part (ties to even) into the
        // low mantissa bits.
        const EVIL_BITS: u32 = 0x4B40_0000;
        const ROUNDING_MAGIC: f32 = f32::from_bits(EVIL_BITS);

        // Exponent bias.
        const BIAS: u32 = 127u32.wrapping_sub(EVIL_BITS);

        // Range reduction: t = k + f, k integral, f in [-0.5, 0.5].
        let t = if self < EXP2_MIN { EXP2_MIN } else { self };
        let z = t + ROUNDING_MAGIC;
        let k = z - ROUNDING_MAGIC;
        let f = t - k;

        // Sollya fpminimax in [-0.5, 0.5].
        const P32: [f32; 3] = [
            f32::from_bits(0x3D5F_BD1E),
            f32::from_bits(0x3E78_0829),
            f32::from_bits(0x3F31_803B),
        ];

        // Mantissa https://en.wikipedia.org/wiki/Horner%27s_method
        let mut m = P32[0];
        for c in &P32[1..] {
            m = fma(m, f, *c);
        }
        m = fma(m, f, 1.0);

        // k + 127 is in [0, 127] for any non-positive input, so the shift
        // cannot overflow the exponent field.
        let e = f32::from_bits((z.to_bits().wrapping_add(BIAS)) << 23);

        m * e
    }
}

impl Exp2 for f64 {
    #[inline(always)]
    fn exp2_fast_nonpos(self) -> Self {
        // 2^k underflows f64 below k = -1023, and the reduction below is only
        // exact for |t| < 2^51.
        const EXP2_MIN: f64 = -1023_f64;

        // Adding 1.5 * 2^52 rounds the integral part (ties to even) into the
        // low mantissa bits.
        const EVIL_BITS: u64 = 0x4338_0000_0000_0000;
        const ROUNDING_MAGIC: f64 = f64::from_bits(EVIL_BITS);

        // Exponent bias.
        const BIAS: u64 = 1023u64.wrapping_sub(EVIL_BITS);

        // Range reduction: t = k + f, k integral, f in [-0.5, 0.5].
        let t = if self < EXP2_MIN { EXP2_MIN } else { self };
        let z = t + ROUNDING_MAGIC;
        let k = z - ROUNDING_MAGIC;
        let f = t - k;

        // Sollya fpminimax in [-0.5, 0.5].
        const P64: [f64; 8] = [
            f64::from_bits(0x3EB6_220A_0A9F_BEA6),
            f64::from_bits(0x3EF0_0DAC_AD3B_1C89),
            f64::from_bits(0x3F24_30A4_CB63_D0AE),
            f64::from_bits(0x3F55_D874_8456_9656),
            f64::from_bits(0x3F83_B2AB_6261_9C06),
            f64::from_bits(0x3FAC_6B08_DD46_02EC),
            f64::from_bits(0x3FCE_BFBD_FF8C_4607),
            f64::from_bits(0x3FE6_2E42_FEF8_61C6),
        ];

        // Mantissa https://en.wikipedia.org/wiki/Horner%27s_method
        let mut m = P64[0];
        for c in &P64[1..] {
            m = fma(m, f, *c);
        }
        m = fma(m, f, 1.0);

        // k + 1023 is in [0, 1023] for any non-positive input, so the shift
        // cannot overflow the exponent field.
        let e = f64::from_bits((z.to_bits().wrapping_add(BIAS)) << 52);

        m * e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! exp2_fast_nonpos_tests {
        (
            $name:ident, $t:ty,
            min = $min:expr,
            top = $top:expr,
            tol = $tol:expr,
            reference = $reference:expr,
        ) => {
            mod $name {
                use super::*;

                const MIN: $t = $min;
                const TOP: i32 = $top;
                const TOL: f64 = $tol;

                fn relative_error(t: $t) -> f64 {
                    let computed = t.exp2_fast_nonpos() as f64;
                    let expected = ($reference)(t);
                    let rel = ((computed - expected) / expected).abs();
                    assert!(
                        !rel.is_nan(),
                        "NaN at t = {t:e}: computed {computed:e}, expected {expected:e}"
                    );

                    rel
                }

                #[test]
                fn polynomial() {
                    const N: u32 = 1 << 20;

                    let mut worst = 0_f64;
                    for i in 0..=N {
                        let t = -(i as $t) / (N as $t);
                        worst = worst.max(relative_error(t));
                    }

                    assert!(
                        worst < TOL,
                        "worst relative error: {worst:e} | {:e}",
                        worst / TOL
                    );
                }

                #[test]
                fn exponent_is_exact() {
                    for j in 0..256_i32 {
                        let t = -(j as $t) / 512.0;
                        let base = t.exp2_fast_nonpos() as f64;
                        for k in TOP..=0 {
                            let computed = (t + k as $t).exp2_fast_nonpos() as f64;
                            let expected = base * (k as f64).exp2();

                            assert_eq!(
                                computed.to_bits(),
                                expected.to_bits(),
                                "k = {k}, t = {t:e}"
                            );
                        }
                    }
                }

                #[test]
                fn full_range() {
                    const N: u32 = 1 << 20;

                    let mut worst = 0_f64;
                    for i in 0..=N {
                        let t = (TOP as $t) * (i as $t) / (N as $t);
                        worst = worst.max(relative_error(t));
                    }

                    assert!(
                        worst < TOL,
                        "worst relative error: {worst:e} | {:e}",
                        worst / TOL
                    );
                }

                #[test]
                fn integral() {
                    for k in TOP..=0 {
                        let t = k as $t;
                        let computed = t.exp2_fast_nonpos() as f64;

                        assert_eq!(computed, (k as f64).exp2(), "k = {k}");
                    }
                }

                #[test]
                fn tail_is_zero() {
                    let cases: [$t; 5] = [MIN, MIN - 1.0, 2.0 * MIN, -1e6, <$t>::NEG_INFINITY];

                    for t in cases {
                        assert_eq!(t.exp2_fast_nonpos(), 0.0, "t = {t:e}");
                    }
                }

                #[test]
                fn nan_propagates() {
                    assert!(<$t>::NAN.exp2_fast_nonpos().is_nan());
                }

                #[test]
                fn known_values() {
                    assert_eq!((0.0 as $t).exp2_fast_nonpos(), 1.0);
                    assert_eq!((-1.0 as $t).exp2_fast_nonpos(), 0.5);
                    assert_eq!((-2.0 as $t).exp2_fast_nonpos(), 0.25);
                }
            }
        };
    }

    exp2_fast_nonpos_tests!(
        f32,
        f32,
        min = -127_f32,
        top = -125_i32,
        tol = 1e3_f64 * f32::EPSILON as f64,
        reference = |t: f32| (t as f64).exp2(),
    );
    exp2_fast_nonpos_tests!(
        f64,
        f64,
        min = -1023_f64,
        top = -1021_i32,
        tol = 5e3_f64 * f64::EPSILON,
        reference = |t: f64| t.exp2(),
    );
}
