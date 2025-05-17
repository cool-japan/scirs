//! High-precision mathematical constants for special functions.
//!
//! This module provides a collection of mathematical constants with enhanced precision
//! for accurate computation of special functions, particularly in edge cases.

/// Mathematical constants with high precision for floating-point calculations.
#[allow(dead_code)]
pub mod f64 {
    /// π (pi) with high precision
    pub const PI: f64 = std::f64::consts::PI;

    /// π/2 (pi/2) with high precision
    pub const PI_2: f64 = std::f64::consts::FRAC_PI_2;

    /// π/4 (pi/4) with high precision
    pub const PI_4: f64 = std::f64::consts::FRAC_PI_4;

    /// 2π (2*pi) with high precision
    pub const TWO_PI: f64 = std::f64::consts::TAU;

    /// √π (square root of pi) with high precision
    pub const SQRT_PI: f64 = 1.772_453_850_905_516;

    /// √(2π) (square root of 2*pi) with high precision
    pub const SQRT_2PI: f64 = 2.506_628_274_631_000_7;

    /// √2 (square root of 2) with high precision
    pub const SQRT_2: f64 = std::f64::consts::SQRT_2;

    /// 1/√π (reciprocal of square root of pi) with high precision
    pub const ONE_OVER_SQRT_PI: f64 = 0.564_189_583_547_756_3;

    /// 1/√(2π) (reciprocal of square root of 2*pi) with high precision
    pub const ONE_OVER_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

    /// e (base of natural logarithm) with high precision
    pub const E: f64 = std::f64::consts::E;

    /// ln(2) (natural logarithm of 2) with high precision
    pub const LN_2: f64 = std::f64::consts::LN_2;

    /// ln(10) (natural logarithm of 10) with high precision
    pub const LN_10: f64 = std::f64::consts::LN_10;

    /// ln(π) (natural logarithm of pi) with high precision
    pub const LN_PI: f64 = 1.144_729_885_849_400_2;

    /// ln(2π) (natural logarithm of 2*pi) with high precision
    pub const LN_2PI: f64 = 1.837_877_066_409_345_6;

    /// ln(√(2π)) (natural logarithm of square root of 2*pi) with high precision
    pub const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

    /// γ (Euler-Mascheroni constant) with high precision
    pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

    /// ζ(3) (Riemann zeta function at 3) with high precision, a.k.a. Apéry's constant
    pub const ZETA_3: f64 = 1.202_056_903_159_594_2;

    /// Machine epsilon for f64 - the difference between 1.0 and the next representable f64 value
    pub const EPSILON: f64 = 2.220_446_049_250_313e-16;

    /// Minimum positive normal f64 value
    pub const MIN_POSITIVE: f64 = 2.2250738585072014e-308;

    /// Maximum f64 value
    pub const MAX: f64 = 1.7976931348623157e308;

    /// Natural logarithm of the maximum f64 value
    pub const LN_MAX: f64 = 709.782712893384;

    /// Natural logarithm of the minimum positive normal f64 value
    pub const LN_MIN: f64 = -708.3964185322641;
}

/// Mathematical constants with high precision for 32-bit floating-point calculations.
#[allow(dead_code)]
pub mod f32 {
    /// π (pi) with high precision
    pub const PI: f32 = std::f32::consts::PI;

    /// π/2 (pi/2) with high precision
    pub const PI_2: f32 = 1.570_796_4;

    /// π/4 (pi/4) with high precision
    pub const PI_4: f32 = std::f32::consts::FRAC_PI_4;

    /// 2π (2*pi) with high precision
    pub const TWO_PI: f32 = 6.283_185_5;

    /// √π (square root of pi) with high precision
    pub const SQRT_PI: f32 = 1.772_453_9;

    /// √(2π) (square root of 2*pi) with high precision
    pub const SQRT_2PI: f32 = 2.506_628_3;

    /// √2 (square root of 2) with high precision
    pub const SQRT_2: f32 = std::f32::consts::SQRT_2;

    /// 1/√π (reciprocal of square root of pi) with high precision
    pub const ONE_OVER_SQRT_PI: f32 = 0.564_189_6;

    /// 1/√(2π) (reciprocal of square root of 2*pi) with high precision
    pub const ONE_OVER_SQRT_2PI: f32 = 0.398_942_3;

    /// e (base of natural logarithm) with high precision
    pub const E: f32 = 2.718_281_7;

    /// ln(2) (natural logarithm of 2) with high precision
    pub const LN_2: f32 = std::f32::consts::LN_2;

    /// ln(10) (natural logarithm of 10) with high precision
    pub const LN_10: f32 = std::f32::consts::LN_10;

    /// ln(π) (natural logarithm of pi) with high precision
    pub const LN_PI: f32 = 1.144_729_9;

    /// ln(2π) (natural logarithm of 2*pi) with high precision
    pub const LN_2PI: f32 = 1.837_877;

    /// ln(√(2π)) (natural logarithm of square root of 2*pi) with high precision
    pub const LN_SQRT_2PI: f32 = 0.918_938_5;

    /// γ (Euler-Mascheroni constant) with high precision
    pub const EULER_MASCHERONI: f32 = 0.577_215_7;

    /// ζ(3) (Riemann zeta function at 3) with high precision, a.k.a. Apéry's constant
    pub const ZETA_3: f32 = 1.202_056_9;

    /// Machine epsilon for f32 - the difference between 1.0 and the next representable f32 value
    pub const EPSILON: f32 = 1.1920929e-7;

    /// Minimum positive normal f32 value
    pub const MIN_POSITIVE: f32 = 1.175494e-38;

    /// Maximum f32 value
    pub const MAX: f32 = 3.402_823e38;

    /// Natural logarithm of the maximum f32 value
    pub const LN_MAX: f32 = 88.72283;

    /// Natural logarithm of the minimum positive normal f32 value
    pub const LN_MIN: f32 = -87.33655;
}

/// Polynomials and series expansion coefficients for various special functions.
#[allow(dead_code)]
pub mod coeffs {
    /// Chebyshev polynomials for Bessel function J₀(x) approximation for x in [0, 8]
    pub const J0_CHEB_PJS: [f64; 7] = [
        1.0,
        -0.1098628627e-2,
        0.2734510407e-4,
        -0.2073370639e-5,
        0.2093887211e-6,
        -0.1562499995e-7,
        0.1430488765e-8,
    ];

    /// Chebyshev polynomials for Bessel function J₀(x) approximation for x in [8, ∞)
    pub const J0_CHEB_PJL: [f64; 9] = [
        1.0,
        -0.1098628627e-2,
        0.2734510407e-4,
        -0.2073370639e-5,
        0.2093887211e-6,
        -0.1562499995e-7,
        0.1430488765e-8,
        -0.6911147651e-10,
        0.1733986234e-11,
    ];

    /// Chebyshev polynomials for Bessel function J₁(x) approximation for x in [0, 8]
    pub const J1_CHEB_PJS: [f64; 7] = [
        0.5,
        0.7346275251e-2,
        -0.1663109526e-3,
        0.1096884640e-4,
        -0.9466149392e-6,
        0.6018609864e-7,
        -0.3457827613e-8,
    ];

    /// Chebyshev polynomials for Bessel function J₁(x) approximation for x in [8, ∞)
    pub const J1_CHEB_PJL: [f64; 9] = [
        0.5,
        0.7346275251e-2,
        -0.1663109526e-3,
        0.1096884640e-4,
        -0.9466149392e-6,
        0.6018609864e-7,
        -0.3457827613e-8,
        0.1753838446e-9,
        -0.4483191311e-11,
    ];

    /// Chebyshev polynomials for Bessel function Y₀(x) approximation for x in [0, 8]
    pub const Y0_CHEB_PYS: [f64; 9] = [
        -0.180_098_163_397_448_3,
        0.011_141_835_799_623_702,
        -0.0003541888723414853,
        0.0000191373317616336,
        -0.0000013695779783857,
        0.0000001180124770733,
        -0.0000000113860493878,
        0.0000000011514390254,
        -0.0000000001195245870,
    ];

    /// Chebyshev polynomials for Bessel function Y₀(x) approximation for x in [8, ∞)
    pub const Y0_CHEB_PYL: [f64; 9] = [
        -0.180_098_163_397_448_3,
        0.011_141_835_799_623_702,
        -0.0003541888723414853,
        0.0000191373317616336,
        -0.0000013695779783857,
        0.0000001180124770733,
        -0.0000000113860493878,
        0.0000000011514390254,
        -0.0000000001195245870,
    ];

    /// Lanczos approximation coefficients for gamma function (g=7)
    pub const LANCZOS_7_COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    /// Improved Lanczos approximation coefficients for gamma function (g=10.900511)
    pub const LANCZOS_G_10_9_COEFFS: [f64; 13] = [
        0.0,
        57.156_235_665_862_92,
        -59.597_960_355_475_49,
        14.136_097_974_741_746,
        -0.491_913_816_097_620_2,
        3.399_464_998_481_189e-5,
        4.652_362_892_704_858e-5,
        -9.837_447_530_487_956e-5,
        1.580_887_032_249_125e-4,
        -2.102_644_417_241_048_8e-4,
        2.174_396_181_152_126_5e-4,
        -1.643_181_065_367_639e-4,
        8.441_822_398_385_275e-5,
    ];
}

/// Lookup tables for function values at specific points, useful for ensuring
/// consistent results in test cases and improved precision.
#[allow(dead_code)]
pub mod lookup {
    /// Bessel function J₀(x) values at specific points
    pub mod j0 {
        /// J₀(0) = 1.0
        pub const AT_0: f64 = 1.0;

        /// J₀(1) = 0.765198...
        pub const AT_1: f64 = 0.765_197_686_557_966_6;

        /// J₀(2) = 0.223891...
        pub const AT_2: f64 = 0.223_890_779_141_235_7;

        /// J₀(5) = -0.177597...
        pub const AT_5: f64 = -0.177_596_771_314_338_32;

        /// J₀(10) = -0.245936...
        pub const AT_10: f64 = -0.245_935_764_451_348_35;
    }

    /// Bessel function J₁(x) values at specific points
    pub mod j1 {
        /// J₁(0) = 0.0
        pub const AT_0: f64 = 0.0;

        /// J₁(1) = 0.440051...
        pub const AT_1: f64 = 0.440_050_585_744_933_5;

        /// J₁(2) = 0.576725...
        pub const AT_2: f64 = 0.576_724_807_756_873_5;

        /// J₁(5) = -0.327579...
        pub const AT_5: f64 = -0.327_579_137_591_465_23;

        /// J₁(10) = 0.043473...
        pub const AT_10: f64 = 0.043_472_746_168_861_44;
    }

    /// Bessel function Y₀(x) values at specific points
    pub mod y0 {
        /// Y₀(0.1) = -1.534238...
        pub const AT_0_1: f64 = -1.534_238_651_350_367_4;

        /// Y₀(1) = 0.088257...
        pub const AT_1: f64 = 0.088_256_964_215_676_96;

        /// Y₀(2) = 0.510376...
        pub const AT_2: f64 = 0.510_375_672_649_745_1;

        /// Y₀(5) = -0.308517...
        pub const AT_5: f64 = -0.308_517_625_248_643_7;

        /// Y₀(10) = 0.055671...
        pub const AT_10: f64 = 0.055_671_167_283_599_395;
    }

    /// Values of the gamma function at specific points
    pub mod gamma {
        /// Γ(0.1) = 9.513508...
        pub const AT_0_1: f64 = 9.513_507_698_668_732;

        /// Γ(0.5) = √π = 1.772454...
        pub const AT_0_5: f64 = 1.772_453_850_905_516;

        /// Γ(1) = 1
        pub const AT_1: f64 = 1.0;

        /// Γ(1.5) = √π/2 = 0.886227...
        pub const AT_1_5: f64 = 0.886_226_925_452_758;

        /// Γ(2) = 1
        pub const AT_2: f64 = 1.0;

        /// Γ(2.5) = 3√π/4 = 1.329340...
        pub const AT_2_5: f64 = 1.329_340_388_179_137_5;

        /// Γ(3) = 2
        pub const AT_3: f64 = 2.0;

        /// Γ(10) = 9! = 362880
        pub const AT_10: f64 = 362880.0;
    }

    /// Values of digamma (ψ) function at specific points
    pub mod digamma {
        /// ψ(0.1) = -10.423754...
        pub const AT_0_1: f64 = -10.423_754_940_411_08;

        /// ψ(0.5) = -γ - 2ln(2) = -1.963510...
        pub const AT_0_5: f64 = -1.963_510_026_021_423_5;

        /// ψ(1) = -γ = -0.577216...
        pub const AT_1: f64 = -0.577_215_664_901_532_9;

        /// ψ(2) = 1 - γ = 0.422784...
        pub const AT_2: f64 = 0.422_784_335_098_467_1;
    }
}
