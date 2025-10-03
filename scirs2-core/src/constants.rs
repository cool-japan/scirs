//! # Physical and Mathematical Constants
//!
//! This module provides a comprehensive set of physical and mathematical constants for scientific computing,
//! closely aligning with SciPy's `constants` module to ensure compatibility and ease of migration.
//!
//! ## Mathematical Constants
//!
//! ```ignore
//! use scirs2_core::constants::math::*;
//! assert!((PI - 3.14159265358979).abs() < 1e-14);
//! assert!((GOLDEN_RATIO - 1.618033988749895).abs() < 1e-14);
//! ```
//!
//! ## Physical Constants
//!
//! ```ignore
//! use scirs2_core::constants::physical::*;
//! // Speed of light in vacuum (m/s)
//! assert_eq!(SPEED_OF_LIGHT, 299_792_458.0);
//! ```
//!
//! ## Unit Prefixes
//!
//! ```ignore
//! use scirs2_core::constants::prefixes::*;
//! // SI prefixes
//! assert_eq!(KILO, 1e3);
//! // Binary prefixes
//! assert_eq!(KIBI, 1024.0);
//! ```
//!
//! ## Unit Conversions
//!
//! ```ignore
//! use scirs2_core::constants::conversions::*;
//! // Length conversions
//! assert!((MILE_TO_METER - 1609.344).abs() < 1e-10);
//! ```
//!
//! ## Constants Database
//!
//! The module includes a comprehensive set of physical constants with their values, units, and precision.
//! This aligns with SciPy's `physical_constants` dictionary.

/// Mathematical constants
pub mod math {
    /// Pi (π)
    pub const PI: f64 = std::f64::consts::PI;

    /// Euler's number (e)
    pub const E: f64 = std::f64::consts::E;

    /// Euler-Mascheroni constant (γ)
    pub const EULER: f64 = 0.577_215_664_901_532_9;

    /// Golden ratio (φ)
    pub const GOLDEN: f64 = 1.618_033_988_749_895;

    /// Golden ratio (φ) - alias for GOLDEN
    pub const GOLDEN_RATIO: f64 = GOLDEN;

    /// Square root of 2
    pub const SQRT2: f64 = std::f64::consts::SQRT_2;

    /// Square root of π
    pub const SQRTPI: f64 = 1.772_453_850_905_516;

    /// Natural logarithm of 2
    pub const LN2: f64 = std::f64::consts::LN_2;

    /// Natural logarithm of 10
    pub const LN10: f64 = std::f64::consts::LN_10;

    /// Natural logarithm of π
    pub const LNPI: f64 = 1.144_729_885_849_4;

    /// Natural logarithm of 2π
    pub const LN2PI: f64 = 1.837_877_066_409_345;

    /// Square root of 3
    pub const SQRT3: f64 = 1.732_050_807_568_877;

    /// Square root of 5
    pub const SQRT5: f64 = 2.236_067_977_499_79;

    /// Square root of e
    pub const SQRTE: f64 = 1.648_721_270_700_128;

    /// Cube root of 2
    pub const CBRT2: f64 = 1.259_921_049_894_873;

    /// Cube root of 3
    pub const CBRT3: f64 = 1.442_249_570_307_408;

    /// Fourth root of 2
    pub const QRRT2: f64 = 1.189_207_115_002_722;

    /// Two to the power of π
    pub const TWO_TO_PI: f64 = 8.824_977_827_076_287;

    /// π to the power of e
    pub const PI_TO_E: f64 = 22.459_157_718_361_05;

    /// Catalan's constant
    pub const CATALAN: f64 = 0.915_965_594_177_219;

    /// Apéry's constant (ζ(3))
    pub const APERY: f64 = 1.202_056_903_159_594;

    /// Khinchin's constant
    pub const KHINCHIN: f64 = 2.685_452_001_065_306;

    /// Glaisher-Kinkelin constant
    pub const GLAISHER: f64 = 1.282_427_129_100_623;

    /// Ramanujan's constant (e^(π*√163))
    pub const RAMANUJAN: f64 = 262_537_412_640_768_744.0;

    /// Levy's constant
    pub const LEVY: f64 = 3.275_822_918_721_811;

    /// Reciprocal Fibonacci constant
    pub const RECIPROCAL_FIBONACCI: f64 = 3.359_885_666_243_178;

    /// Universal parabolic constant
    pub const UNIVERSAL_PARABOLIC: f64 = 2.295_587_149_392_638;

    /// Cahen's constant
    pub const CAHEN: f64 = 0.643_410_546_288_338;

    /// Laplace limit
    pub const LAPLACE_LIMIT: f64 = 0.662_743_419_349_181;

    /// Alladi-Grinstead constant
    pub const ALLADI_GRINSTEAD: f64 = 0.809_394_020_540_685;

    /// Lengyel's constant
    pub const LENGYEL: f64 = 1.098_684_196_345_534;

    /// Viswanath's constant
    pub const VISWANATH: f64 = 1.131_988_924_341_06;

    /// Fransén-Robinson constant
    pub const FRANSEN_ROBINSON: f64 = 2.807_770_242_028_519;

    /// Feigenbaum's first constant (δ)
    pub const FEIGENBAUM_DELTA: f64 = 4.669_201_609_102_99;

    /// Feigenbaum's second constant (α)
    pub const FEIGENBAUM_ALPHA: f64 = 2.502_907_875_095_893;

    /// Meissel-Mertens constant
    pub const MEISSEL_MERTENS: f64 = 0.261_497_212_847_643;

    /// Brun's constant for twin primes
    pub const BRUN_TWIN_PRIMES: f64 = 1.902_160_583_104;

    /// Landau-Ramanujan constant
    pub const LANDAU_RAMANUJAN: f64 = 0.764_223_653_589_220;

    /// Gauss's constant
    pub const GAUSS: f64 = 0.834_626_841_674_073;

    /// Second Hermite constant
    pub const HERMITE_GAMMA: f64 = 1.154_700_538_379_252;

    /// Lieb's square ice constant
    pub const LIEB_SQUARE_ICE: f64 = 1.539_600_717_839_002;

    /// Niven's constant
    pub const NIVEN: f64 = 1.705_211_140_105_367;

    /// Plastic number
    pub const PLASTIC: f64 = 1.324_717_957_244_746;

    /// Supergolden ratio
    pub const SUPERGOLDEN: f64 = 1.465_571_231_876_768;

    /// Connective constant of the hexagonal lattice
    pub const CONNECTIVE_HEXAGONAL: f64 = 2.638_915_503_410_388;

    /// Kepler-Bouwkamp constant
    pub const KEPLER_BOUWKAMP: f64 = 0.114_942_044_853_297;
}

/// Physical constants (SI units)
pub mod physical {
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

    /// Speed of light in vacuum (m/s) - alias for SPEED_OF_LIGHT
    pub const C: f64 = SPEED_OF_LIGHT;

    /// Magnetic constant (vacuum permeability) (N/A²)
    pub const MAGNETIC_CONSTANT: f64 = 1.256_637_062_12e-6;

    /// Magnetic constant (vacuum permeability) (N/A²) - alias for MAGNETIC_CONSTANT
    pub const MU_0: f64 = MAGNETIC_CONSTANT;

    /// Electric constant (vacuum permittivity) (F/m)
    pub const ELECTRIC_CONSTANT: f64 = 8.854_187_812_8e-12;

    /// Electric constant (vacuum permittivity) (F/m) - alias for ELECTRIC_CONSTANT
    pub const EPSILON_0: f64 = ELECTRIC_CONSTANT;

    /// Gravitational constant (m³/kg/s²)
    pub const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11;

    /// Gravitational constant (m³/kg/s²) - alias for GRAVITATIONAL_CONSTANT
    pub const G: f64 = GRAVITATIONAL_CONSTANT;

    /// Standard acceleration of gravity (m/s²)
    pub const STANDARD_GRAVITY: f64 = 9.80665;

    /// Standard acceleration of gravity (m/s²) - alias for STANDARD_GRAVITY
    pub const G_ACCEL: f64 = STANDARD_GRAVITY;

    /// Planck constant (J·s)
    pub const PLANCK: f64 = 6.626_070_15e-34;

    /// Planck constant (J·s) - alias for PLANCK
    pub const H: f64 = PLANCK;

    /// Reduced Planck constant (J·s)
    pub const REDUCED_PLANCK: f64 = 1.054_571_817e-34;

    /// Reduced Planck constant (J·s) - alias for REDUCED_PLANCK
    pub const HBAR: f64 = REDUCED_PLANCK;

    /// Elementary charge (C)
    pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

    /// Elementary charge (C) - alias for ELEMENTARY_CHARGE
    pub const E_CHARGE: f64 = ELEMENTARY_CHARGE;

    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

    /// Electron mass (kg) - alias for ELECTRON_MASS
    pub const M_E: f64 = ELECTRON_MASS;

    /// Proton mass (kg)
    pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;

    /// Proton mass (kg) - alias for PROTON_MASS
    pub const M_P: f64 = PROTON_MASS;

    /// Neutron mass (kg)
    pub const NEUTRON_MASS: f64 = 1.674_927_498_04e-27;

    /// Neutron mass (kg) - alias for NEUTRON_MASS
    pub const M_N: f64 = NEUTRON_MASS;

    /// Atomic mass constant (kg)
    pub const ATOMIC_MASS: f64 = 1.660_539_066_60e-27;

    /// Atomic mass constant (kg) - alias for ATOMIC_MASS
    pub const M_U: f64 = ATOMIC_MASS;

    /// Atomic mass constant (kg) - alias for ATOMIC_MASS
    pub const U: f64 = ATOMIC_MASS;

    /// Fine-structure constant (dimensionless)
    pub const FINE_STRUCTURE: f64 = 7.297_352_569_3e-3;

    /// Fine-structure constant (dimensionless) - alias for FINE_STRUCTURE
    pub const ALPHA: f64 = FINE_STRUCTURE;

    /// Rydberg constant (1/m)
    pub const RYDBERG: f64 = 10_973_731.568_160;

    /// Avogadro constant (1/mol)
    pub const AVOGADRO: f64 = 6.022_140_76e23;

    /// Avogadro constant (1/mol) - alias for AVOGADRO
    pub const N_A: f64 = AVOGADRO;

    /// Gas constant (J/(mol·K))
    pub const GAS_CONSTANT: f64 = 8.314_462_618_153_24;

    /// Gas constant (J/(mol·K)) - alias for GAS_CONSTANT
    pub const R: f64 = GAS_CONSTANT;

    /// Boltzmann constant (J/K)
    pub const BOLTZMANN: f64 = 1.380_649e-23;

    /// Boltzmann constant (J/K) - alias for BOLTZMANN
    pub const K: f64 = BOLTZMANN;

    /// Stefan-Boltzmann constant (W/(m²·K⁴))
    pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

    /// Stefan-Boltzmann constant (W/(m²·K⁴)) - alias for STEFAN_BOLTZMANN
    pub const SIGMA: f64 = STEFAN_BOLTZMANN;

    /// Wien wavelength displacement law constant (m·K)
    pub const WIEN: f64 = 2.897_771_955e-3;

    /// Electron volt (J)
    pub const ELECTRON_VOLT: f64 = 1.602_176_634e-19;

    /// Electron volt (J) - alias for ELECTRON_VOLT
    pub const EV: f64 = ELECTRON_VOLT;

    /// Astronomical unit (m)
    pub const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

    /// Astronomical unit (m) - alias for ASTRONOMICAL_UNIT
    pub const AU: f64 = ASTRONOMICAL_UNIT;

    /// Light year (m)
    pub const LIGHT_YEAR: f64 = 9.460_730_472_580_8e15;

    /// Parsec (m)
    pub const PARSEC: f64 = 3.085_677_581_491_367e16;

    /// Hubble constant (km/s/Mpc) - approximate value
    pub const HUBBLE_CONSTANT: f64 = 70.0;

    /// Critical density of the universe (kg/m³)
    pub const CRITICAL_DENSITY: f64 = 9.9e-27;

    /// Solar mass (kg)
    pub const SOLAR_MASS: f64 = 1.988_47e30;

    /// Solar luminosity (W)
    pub const SOLAR_LUMINOSITY: f64 = 3.828e26;

    /// Solar radius (m)
    pub const SOLAR_RADIUS: f64 = 6.96e8;

    /// Earth mass (kg)
    pub const EARTH_MASS: f64 = 5.972_168e24;

    /// Earth radius (m)
    pub const EARTH_RADIUS: f64 = 6.371e6;

    /// Jupiter mass (kg)
    pub const JUPITER_MASS: f64 = 1.898_13e27;

    /// Electron classical radius (m)
    pub const ELECTRON_RADIUS: f64 = 2.817_940_326_2e-15;

    /// Bohr radius (m)
    pub const BOHR_RADIUS: f64 = 5.291_772_109_03e-11;

    /// Thomson scattering cross section (m²)
    pub const THOMSON_CROSS_SECTION: f64 = 6.652_458_732_1e-29;

    /// Wien frequency displacement law constant (Hz/K)
    pub const WIEN_FREQUENCY: f64 = 5.878_925_757e10;

    /// Second radiation constant (m·K)
    pub const SECOND_RADIATION: f64 = 1.438_776_877e-2;

    /// First radiation constant (W·m²)
    pub const FIRST_RADIATION: f64 = 3.741_771_852e-16;

    /// Josephson constant (Hz/V)
    pub const JOSEPHSON: f64 = 4.835_978_484e14;

    /// von Klitzing constant (Ω)
    pub const VON_KLITZING: f64 = 2.581_280_745e4;

    /// Magnetic flux quantum (Wb)
    pub const MAGNETIC_FLUX_QUANTUM: f64 = 2.067_833_848e-15;

    /// Conductance quantum (S)
    pub const CONDUCTANCE_QUANTUM: f64 = 7.748_091_729e-5;

    /// Faraday constant (C/mol)
    pub const FARADAY: f64 = 9.648_533_212e4;

    /// Molar gas constant (J/(mol·K)) - alias for R
    pub const MOLAR_GAS: f64 = GAS_CONSTANT;

    /// Standard atmosphere (Pa)
    pub const STANDARD_ATMOSPHERE: f64 = 101_325.0;

    /// Standard temperature (K)
    pub const STANDARD_TEMPERATURE: f64 = 273.15;

    /// Molar volume of ideal gas at STP (m³/mol)
    pub const MOLAR_VOLUME_STP: f64 = 2.241_396_954e-2;

    /// Loschmidt constant (1/m³)
    pub const LOSCHMIDT: f64 = 2.686_780_111e25;

    /// Alpha particle mass (kg)
    pub const ALPHA_PARTICLE_MASS: f64 = 6.644_657_230e-27;

    /// Muon mass (kg)
    pub const MUON_MASS: f64 = 1.883_531_627e-28;

    /// Tau mass (kg)
    pub const TAU_MASS: f64 = 3.167_54e-27;

    /// W boson mass (kg)
    pub const W_BOSON_MASS: f64 = 1.433_e-25;

    /// Z boson mass (kg)
    pub const Z_BOSON_MASS: f64 = 1.625_e-25;

    /// Higgs boson mass (kg)
    pub const HIGGS_BOSON_MASS: f64 = 2.23e-25;

    /// Weak mixing angle (Weinberg angle)
    pub const WEAK_MIXING_ANGLE: f64 = 0.223_1;

    /// Strong coupling constant
    pub const STRONG_COUPLING: f64 = 0.118;

    /// Weak coupling constant
    pub const WEAK_COUPLING: f64 = 0.653;

    /// Nuclear magneton (J/T)
    pub const NUCLEAR_MAGNETON: f64 = 5.050_783_699_1e-27;

    /// Bohr magneton (J/T)
    pub const BOHR_MAGNETON: f64 = 9.274_010_078_3e-24;

    /// Classical electron radius (m)
    pub const CLASSICAL_ELECTRON_RADIUS: f64 = ELECTRON_RADIUS;

    /// Compton wavelength (m)
    pub const COMPTON_WAVELENGTH: f64 = 2.426_310_238_67e-12;

    /// Reduced Compton wavelength (m)
    pub const REDUCED_COMPTON_WAVELENGTH: f64 = 3.861_592_679_6e-13;
}

/// SI prefixes and binary prefixes
pub mod prefixes {
    /// SI prefixes
    pub mod si {
        /// Quetta (10^30)
        pub const QUETTA: f64 = 1e30;

        /// Ronna (10^27)
        pub const RONNA: f64 = 1e27;

        /// Yotta (10^24)
        pub const YOTTA: f64 = 1e24;

        /// Zetta (10^21)
        pub const ZETTA: f64 = 1e21;

        /// Exa (10^18)
        pub const EXA: f64 = 1e18;

        /// Peta (10^15)
        pub const PETA: f64 = 1e15;

        /// Tera (10^12)
        pub const TERA: f64 = 1e12;

        /// Giga (10^9)
        pub const GIGA: f64 = 1e9;

        /// Mega (10^6)
        pub const MEGA: f64 = 1e6;

        /// Kilo (10^3)
        pub const KILO: f64 = 1e3;

        /// Hecto (10^2)
        pub const HECTO: f64 = 1e2;

        /// Deka (10^1)
        pub const DEKA: f64 = 1e1;

        /// Deci (10^-1)
        pub const DECI: f64 = 1e-1;

        /// Centi (10^-2)
        pub const CENTI: f64 = 1e-2;

        /// Milli (10^-3)
        pub const MILLI: f64 = 1e-3;

        /// Micro (10^-6)
        pub const MICRO: f64 = 1e-6;

        /// Nano (10^-9)
        pub const NANO: f64 = 1e-9;

        /// Pico (10^-12)
        pub const PICO: f64 = 1e-12;

        /// Femto (10^-15)
        pub const FEMTO: f64 = 1e-15;

        /// Atto (10^-18)
        pub const ATTO: f64 = 1e-18;

        /// Zepto (10^-21)
        pub const ZEPTO: f64 = 1e-21;

        /// Yocto (10^-24)
        pub const YOCTO: f64 = 1e-24;

        /// Ronto (10^-27)
        pub const RONTO: f64 = 1e-27;

        /// Quecto (10^-30)
        pub const QUECTO: f64 = 1e-30;
    }

    /// Re-exports for ease of use
    pub use si::*;

    /// Binary prefixes
    pub mod binary {
        /// Kibi (2^10)
        pub const KIBI: f64 = 1024.0;

        /// Mebi (2^20)
        pub const MEBI: f64 = 1_048_576.0;

        /// Gibi (2^30)
        pub const GIBI: f64 = 1_073_741_824.0;

        /// Tebi (2^40)
        pub const TEBI: f64 = 1_099_511_627_776.0;

        /// Pebi (2^50)
        pub const PEBI: f64 = 1_125_899_906_842_624.0;

        /// Exbi (2^60)
        pub const EXBI: f64 = 1_152_921_504_606_846_976.0;

        /// Zebi (2^70)
        pub const ZEBI: f64 = 1_180_591_620_717_411_303_424.0;

        /// Yobi (2^80)
        pub const YOBI: f64 = 1_208_925_819_614_629_174_706_176.0;
    }

    /// Re-exports for ease of use
    pub use binary::*;
}

/// Unit conversions
pub mod conversions {
    /// Angular conversions
    pub mod angle {
        use super::super::math::PI;

        /// Degrees to radians conversion factor
        pub const DEG_TO_RAD: f64 = PI / 180.0;

        /// Radians to degrees conversion factor
        pub const RAD_TO_DEG: f64 = 180.0 / PI;

        /// Degree in radians
        pub const DEGREE: f64 = DEG_TO_RAD;

        /// Arc minute in radians
        pub const ARCMIN: f64 = DEGREE / 60.0;

        /// Arc minute in radians - alias for ARCMIN
        pub const ARCMINUTE: f64 = ARCMIN;

        /// Arc second in radians
        pub const ARCSEC: f64 = ARCMIN / 60.0;

        /// Arc second in radians - alias for ARCSEC
        pub const ARCSECOND: f64 = ARCSEC;
    }

    /// Re-exports for ease of use
    pub use angle::*;

    /// Time conversions
    pub mod time {
        /// Minute in seconds
        pub const MINUTE: f64 = 60.0;

        /// Hour in seconds
        pub const HOUR: f64 = 60.0 * MINUTE;

        /// Day in seconds
        pub const DAY: f64 = 24.0 * HOUR;

        /// Week in seconds
        pub const WEEK: f64 = 7.0 * DAY;

        /// Year (365 days) in seconds
        pub const YEAR: f64 = 365.0 * DAY;

        /// Julian year (365.25 days) in seconds
        pub const JULIAN_YEAR: f64 = 365.25 * DAY;
    }

    /// Re-exports for ease of use
    pub use time::*;

    /// Length conversions
    pub mod length {
        /// Base unit - meter
        pub const METER: f64 = 1.0;

        /// Inch in meters
        pub const INCH: f64 = 0.0254;

        /// Foot in meters
        pub const FOOT: f64 = 12.0 * INCH;

        /// Yard in meters
        pub const YARD: f64 = 3.0 * FOOT;

        /// Mile in meters
        pub const MILE: f64 = 1760.0 * YARD;

        /// Mil in meters
        pub const MIL: f64 = INCH / 1000.0;

        /// Point in meters (typography)
        pub const POINT: f64 = INCH / 72.0;

        /// Point in meters - alias for POINT
        pub const PT: f64 = POINT;

        /// Survey foot in meters
        pub const SURVEY_FOOT: f64 = 1200.0 / 3937.0;

        /// Survey mile in meters
        pub const SURVEY_MILE: f64 = 5280.0 * SURVEY_FOOT;

        /// Nautical mile in meters
        pub const NAUTICAL_MILE: f64 = 1852.0;

        /// Fermi in meters (1e-15 m)
        pub const FERMI: f64 = 1e-15;

        /// Angstrom in meters (1e-10 m)
        pub const ANGSTROM: f64 = 1e-10;

        /// Micron in meters (1e-6 m)
        pub const MICRON: f64 = 1e-6;

        /// Conversions from units to meters
        pub const INCH_TO_METER: f64 = INCH;
        pub const FOOT_TO_METER: f64 = FOOT;
        pub const YARD_TO_METER: f64 = YARD;
        pub const MILE_TO_METER: f64 = MILE;
    }

    /// Re-exports for ease of use
    pub use length::*;

    /// Mass conversions
    pub mod mass {
        /// Gram in kilograms
        pub const GRAM: f64 = 1e-3;

        /// Metric ton in kilograms
        pub const METRIC_TON: f64 = 1e3;

        /// Grain in kilograms
        pub const GRAIN: f64 = 64.79891e-6;

        /// Pound (avoirdupois) in kilograms
        pub const POUND: f64 = 7000.0 * GRAIN;

        /// Pound in kilograms - alias for POUND
        pub const LB: f64 = POUND;

        /// One inch version of a slug in kilograms
        pub const BLOB: f64 = POUND * 9.80665 / 0.0254;

        /// One inch version of a slug in kilograms - alias for BLOB
        pub const SLINCH: f64 = BLOB;

        /// One slug in kilograms
        pub const SLUG: f64 = BLOB / 12.0;

        /// Ounce in kilograms
        pub const OUNCE: f64 = POUND / 16.0;

        /// Ounce in kilograms - alias for OUNCE
        pub const OZ: f64 = OUNCE;

        /// Stone in kilograms
        pub const STONE: f64 = 14.0 * POUND;

        /// Long ton in kilograms
        pub const LONG_TON: f64 = 2240.0 * POUND;

        /// Short ton in kilograms
        pub const SHORT_TON: f64 = 2000.0 * POUND;

        /// Troy ounce in kilograms
        pub const TROY_OUNCE: f64 = 480.0 * GRAIN;

        /// Troy pound in kilograms
        pub const TROY_POUND: f64 = 12.0 * TROY_OUNCE;

        /// Carat in kilograms
        pub const CARAT: f64 = 200e-6;

        /// Conversions from units to kilograms
        pub const POUND_TO_KG: f64 = POUND;
    }

    /// Re-exports for ease of use
    pub use mass::*;

    /// Pressure conversions
    pub mod pressure {
        /// Standard atmosphere in pascals
        pub const ATMOSPHERE: f64 = 101_325.0;

        /// Standard atmosphere in pascals - alias for ATMOSPHERE
        pub const ATM: f64 = ATMOSPHERE;

        /// Bar in pascals
        pub const BAR: f64 = 1e5;

        /// Torr (mmHg) in pascals
        pub const TORR: f64 = ATMOSPHERE / 760.0;

        /// Torr (mmHg) in pascals - alias for TORR
        pub const MMHG: f64 = TORR;

        /// PSI (pound-force per square inch) in pascals
        pub const PSI: f64 = POUND_FORCE / (INCH * INCH);

        // Required for PSI definition
        use super::force::POUND_FORCE;
        use super::length::INCH;
    }

    /// Re-exports for ease of use
    pub use pressure::*;

    /// Area conversions
    pub mod area {
        use super::length::FOOT;

        /// Hectare in square meters
        pub const HECTARE: f64 = 1e4;

        /// Acre in square meters
        pub const ACRE: f64 = 43560.0 * FOOT * FOOT;
    }

    /// Re-exports for ease of use
    pub use area::*;

    /// Volume conversions
    pub mod volume {
        use super::length::INCH;

        /// Liter in cubic meters
        pub const LITER: f64 = 1e-3;

        /// Liter in cubic meters - alias for LITER
        pub const LITRE: f64 = LITER;

        /// US gallon in cubic meters
        pub const GALLON_US: f64 = 231.0 * INCH * INCH * INCH;

        /// US gallon in cubic meters - alias for GALLON_US
        pub const GALLON: f64 = GALLON_US;

        /// Imperial gallon in cubic meters
        pub const GALLON_IMP: f64 = 4.54609e-3;

        /// US fluid ounce in cubic meters
        pub const FLUID_OUNCE_US: f64 = GALLON_US / 128.0;

        /// US fluid ounce in cubic meters - alias for FLUID_OUNCE_US
        pub const FLUID_OUNCE: f64 = FLUID_OUNCE_US;

        /// Imperial fluid ounce in cubic meters
        pub const FLUID_OUNCE_IMP: f64 = GALLON_IMP / 160.0;

        /// Barrel in cubic meters (for oil)
        pub const BARREL: f64 = 42.0 * GALLON_US;

        /// Barrel in cubic meters - alias for BARREL
        pub const BBL: f64 = BARREL;

        /// Gallons (US) to cubic meters
        pub const GALLON_TO_CUBIC_METER: f64 = GALLON_US;
    }

    /// Re-exports for ease of use
    pub use volume::*;

    /// Speed conversions
    pub mod speed {
        use super::length::{MILE, NAUTICAL_MILE};
        use super::time::HOUR;

        /// Kilometers per hour in meters per second
        pub const KMH: f64 = 1e3 / 3600.0;

        /// Miles per hour in meters per second
        pub const MPH: f64 = MILE / HOUR;

        /// Mach (approx., at 15°C, 1 atm) in meters per second
        pub const MACH: f64 = 340.5;

        /// Mach (approx., at 15°C, 1 atm) in meters per second - alias for MACH
        pub const SPEED_OF_SOUND: f64 = MACH;

        /// Knot in meters per second
        pub const KNOT: f64 = NAUTICAL_MILE / HOUR;
    }

    /// Re-exports for ease of use
    pub use speed::*;

    /// Temperature conversions
    pub mod temperature {
        /// Zero of Celsius scale in Kelvin
        pub const ZERO_CELSIUS: f64 = 273.15;

        /// One Fahrenheit (only for differences) in Kelvin
        pub const DEGREE_FAHRENHEIT: f64 = 1.0 / 1.8;

        /// Convert temperature from one scale to another
        ///
        /// # Arguments
        ///
        /// * `value` - Temperature value to convert
        /// * `from_scale` - Source scale: celsius, "kelvin", "fahrenheit", or "rankine"
        /// * `toscale` - Target scale: celsius, "kelvin", "fahrenheit", or "rankine"
        ///
        /// # Returns
        ///
        /// Converted temperature value
        ///
        /// # Examples
        ///
        /// ```ignore
        /// use scirs2_core::constants::conversions::temperature::convert_temperature;
        ///
        /// let celsius = 100.0;
        /// let kelvin = convert_temperature(celsius, "celsius", "kelvin");
        /// assert!((kelvin - 373.15).abs() < 1e-10);
        ///
        /// let fahrenheit = convert_temperature(celsius, "celsius", "fahrenheit");
        /// assert!((fahrenheit - 212.0).abs() < 1e-10);
        /// ```
        #[must_use]
        pub fn convert_temperature(value: f64, from_scale: &str, toscale: &str) -> f64 {
            // Convert from source scale to Kelvin
            let kelvin = match from_scale.to_lowercase().as_str() {
                "celsius" | "c" => value + ZERO_CELSIUS,
                "kelvin" | "k" => value,
                "fahrenheit" | "f" => (value - 32.0) * 5.0 / 9.0 + ZERO_CELSIUS,
                "rankine" | "r" => value * 5.0 / 9.0,
                _ => panic!("Unsupported 'from' scale: {from_scale}. Supported scales are Celsius, Kelvin, Fahrenheit, and Rankine"),
            };

            // Convert from Kelvin to target _scale
            match toscale.to_lowercase().as_str() {
                "celsius" | "c" => kelvin - ZERO_CELSIUS,
                "kelvin" | "k" => kelvin,
                "fahrenheit" | "f" => (kelvin - ZERO_CELSIUS) * 9.0 / 5.0 + 32.0,
                "rankine" | "r" => kelvin * 9.0 / 5.0,
                _ => panic!("Unsupported 'to' scale: {toscale}. Supported scales are Celsius, Kelvin, Fahrenheit, and Rankine"),
            }
        }
    }

    /// Re-exports for ease of use
    pub use temperature::*;

    /// Energy conversions
    pub mod energy {
        use super::mass::POUND;
        use super::temperature::DEGREE_FAHRENHEIT;
        use crate::constants::physical::ELEMENTARY_CHARGE;

        /// Electron volt in joules
        pub const ELECTRON_VOLT: f64 = ELEMENTARY_CHARGE;

        /// Electron volt in joules - alias for ELECTRON_VOLT
        pub const EV: f64 = ELECTRON_VOLT;

        /// Calorie (thermochemical) in joules
        pub const CALORIE_TH: f64 = 4.184;

        /// Calorie (thermochemical) in joules - alias for CALORIE_TH
        pub const CALORIE: f64 = CALORIE_TH;

        /// Calorie (International Steam Table calorie, 1956) in joules
        pub const CALORIE_IT: f64 = 4.1868;

        /// Erg in joules
        pub const ERG: f64 = 1e-7;

        /// British thermal unit (International Steam Table) in joules
        pub const BTU_IT: f64 = POUND * DEGREE_FAHRENHEIT * CALORIE_IT / 1e-3;

        /// British thermal unit (International Steam Table) in joules - alias for BTU_IT
        pub const BTU: f64 = BTU_IT;

        /// British thermal unit (thermochemical) in joules
        pub const BTU_TH: f64 = POUND * DEGREE_FAHRENHEIT * CALORIE_TH / 1e-3;

        /// Ton of TNT in joules
        pub const TON_TNT: f64 = 1e9 * CALORIE_TH;
    }

    /// Re-exports for ease of use
    pub use energy::*;

    /// Power conversions
    pub mod power {
        use super::length::FOOT;
        use super::mass::POUND;

        /// Standard gravity constant (m/s²)
        const STANDARD_GRAVITY: f64 = 9.80665;

        /// Horsepower in watts
        pub const HORSEPOWER: f64 = 550.0 * FOOT * POUND * STANDARD_GRAVITY;

        /// Horsepower in watts - alias for HORSEPOWER
        pub const HP: f64 = HORSEPOWER;
    }

    /// Re-exports for ease of use
    pub use power::*;

    /// Force conversions
    pub mod force {
        use super::mass::POUND;

        /// Standard gravity constant (m/s²)
        const STANDARD_GRAVITY: f64 = 9.80665;

        /// Dyne in newtons
        pub const DYNE: f64 = 1e-5;

        /// Dyne in newtons - alias for DYNE
        pub const DYN: f64 = DYNE;

        /// Pound force in newtons
        pub const POUND_FORCE: f64 = POUND * STANDARD_GRAVITY;

        /// Pound force in newtons - alias for POUND_FORCE
        pub const LBF: f64 = POUND_FORCE;

        /// Kilogram force in newtons
        pub const KILOGRAM_FORCE: f64 = STANDARD_GRAVITY;

        /// Kilogram force in newtons - alias for KILOGRAM_FORCE
        pub const KGF: f64 = KILOGRAM_FORCE;
    }

    /// Re-exports for ease of use
    pub use force::*;

    /// Optics conversions and functions
    pub mod optics {
        use crate::constants::physical::SPEED_OF_LIGHT;

        /// Convert wavelength to optical frequency
        ///
        /// # Arguments
        ///
        /// * `wavelength` - Wavelength in meters
        ///
        /// # Returns
        ///
        /// Equivalent optical frequency in Hz
        ///
        /// # Examples
        ///
        /// ```ignore
        /// use scirs2_core::constants::conversions::optics::lambda2nu;
        /// use scirs2_core::constants::physical::SPEED_OF_LIGHT;
        ///
        /// let wavelength = 1.0;  // 1 meter
        /// let frequency = lambda2nu(wavelength);
        /// assert!((frequency - SPEED_OF_LIGHT).abs() < 1e-10);
        /// ```
        #[must_use]
        pub fn lambda2nu(wavelength: f64) -> f64 {
            SPEED_OF_LIGHT / wavelength
        }

        /// Convert optical frequency to wavelength
        ///
        /// # Arguments
        ///
        /// * `frequency` - Optical frequency in Hz
        ///
        /// # Returns
        ///
        /// Equivalent wavelength in meters
        ///
        /// # Examples
        ///
        /// ```ignore
        /// use scirs2_core::constants::conversions::optics::nu2lambda;
        /// use scirs2_core::constants::physical::SPEED_OF_LIGHT;
        ///
        /// let frequency = SPEED_OF_LIGHT;  // c Hz
        /// let wavelength = nu2lambda(frequency);
        /// assert!((wavelength - 1.0).abs() < 1e-10);  // 1 meter
        /// ```
        #[must_use]
        pub fn nu2lambda(frequency: f64) -> f64 {
            SPEED_OF_LIGHT / frequency
        }
    }

    /// Re-exports for ease of use
    pub use optics::*;
}

/// Numerical analysis constants for computational methods
pub mod numerical {
    /// Machine epsilon for f64 (difference between 1 and next representable float)
    pub const MACHINE_EPSILON_F64: f64 = f64::EPSILON;

    /// Machine epsilon for f32
    pub const MACHINE_EPSILON_F32: f32 = f32::EPSILON;

    /// Square root of machine epsilon for f64
    pub const SQRT_MACHINE_EPSILON_F64: f64 = 1.4901161193847656e-8;

    /// Square root of machine epsilon for f32
    pub const SQRT_MACHINE_EPSILON_F32: f32 = 0.00034526698;

    /// Minimum positive normal f64 value
    pub const MIN_POSITIVE_F64: f64 = f64::MIN_POSITIVE;

    /// Minimum positive normal f32 value
    pub const MIN_POSITIVE_F32: f32 = f32::MIN_POSITIVE;

    /// Maximum finite f64 value
    pub const MAX_F64: f64 = f64::MAX;

    /// Maximum finite f32 value
    pub const MAX_F32: f32 = f32::MAX;

    /// Default tolerance for iterative algorithms
    pub const DEFAULT_TOLERANCE: f64 = 1e-12;

    /// Relaxed tolerance for less precise computations
    pub const RELAXED_TOLERANCE: f64 = 1e-8;

    /// Strict tolerance for high precision computations
    pub const STRICT_TOLERANCE: f64 = 1e-15;

    /// Default maximum iterations for iterative algorithms
    pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

    /// Convergence factor for Newton-Raphson method
    pub const NEWTON_RAPHSON_FACTOR: f64 = 0.5;

    /// Golden ratio minus 1 (useful for golden section search)
    pub const PHI_MINUS_1: f64 = 0.618_033_988_749_895;

    /// Inverse of golden ratio
    pub const INV_PHI: f64 = PHI_MINUS_1;

    /// Safety factor for numerical differentiation
    pub const NUMERICAL_DIFF_SAFETY: f64 = 1e-8;

    /// Default step size for numerical differentiation
    pub const DEFAULT_DIFF_STEP: f64 = 1e-6;

    /// Lanczos g parameter for gamma function approximation
    pub const LANCZOS_G: f64 = 7.0;

    /// Number of terms in Lanczos approximation
    pub const LANCZOS_TERMS: usize = 9;

    /// Bernoulli numbers for numerical methods
    pub const BERNOULLI_2: f64 = 1.0 / 6.0;
    pub const BERNOULLI_4: f64 = -1.0 / 30.0;
    pub const BERNOULLI_6: f64 = 1.0 / 42.0;
    pub const BERNOULLI_8: f64 = -1.0 / 30.0;
    pub const BERNOULLI_10: f64 = 5.0 / 66.0;

    /// Euler-Maclaurin constant
    pub const EULER_MACLAURIN: f64 = 0.577_215_664_901_533;

    /// Stirling's approximation constant
    pub const STIRLING_CONSTANT: f64 = 2.506_628_274_631_001;

    /// Default relative tolerance
    pub const DEFAULT_RTOL: f64 = 1e-12;

    /// Default absolute tolerance
    pub const DEFAULT_ATOL: f64 = 1e-15;

    /// Machine precision safety factor
    pub const PRECISION_SAFETY: f64 = 10.0;
}

/// Complex number and quaternion constants
pub mod complex {
    use num_complex::Complex64;

    /// Imaginary unit i
    pub const I: Complex64 = Complex64::new(0.0, 1.0);

    /// Complex zero
    pub const ZERO: Complex64 = Complex64::new(0.0, 0.0);

    /// Complex one
    pub const ONE: Complex64 = Complex64::new(1.0, 0.0);

    /// Complex negative one
    pub const NEG_ONE: Complex64 = Complex64::new(-1.0, 0.0);

    /// Complex i (same as I)
    pub const IMAGINARY_UNIT: Complex64 = I;

    /// e^(iπ) = -1 (Euler's identity components)
    pub const E_TO_I_PI: Complex64 = NEG_ONE;

    /// e^(iπ/2) = i
    pub const E_TO_I_PI_2: Complex64 = I;

    /// e^(iπ/4) = (1+i)/√2
    pub const E_TO_I_PI_4: Complex64 = Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );

    /// ln(i) = iπ/2
    pub const LN_I: Complex64 = Complex64::new(0.0, std::f64::consts::FRAC_PI_2);

    /// √i = e^(iπ/4)
    pub const SQRT_I: Complex64 = E_TO_I_PI_4;

    /// √(-1) = i
    pub const SQRT_NEG_ONE: Complex64 = I;

    /// (1+i)/√2
    pub const ONE_PLUS_I_OVER_SQRT_2: Complex64 = E_TO_I_PI_4;

    /// (1-i)/√2
    pub const ONE_MINUS_I_OVER_SQRT_2: Complex64 = Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    );

    /// 2πi
    pub const TWO_PI_I: Complex64 = Complex64::new(0.0, 2.0 * std::f64::consts::PI);

    /// πi
    pub const PI_I: Complex64 = Complex64::new(0.0, std::f64::consts::PI);

    /// π/2 * i
    pub const PI_2_I: Complex64 = Complex64::new(0.0, std::f64::consts::FRAC_PI_2);

    /// Gaussian unit imaginary
    pub const GAUSS_I: Complex64 = I;
}

/// Constants for chemistry and materials science
pub mod chemistry {
    /// Atomic mass unit in kg
    pub const ATOMIC_MASS_UNIT: f64 = 1.660_539_066_60e-27;

    /// Unified atomic mass unit - alias for ATOMIC_MASS_UNIT
    pub const U: f64 = ATOMIC_MASS_UNIT;

    /// Hartree energy (eV)
    pub const HARTREE_EV: f64 = 27.211_386_245_988;

    /// Hartree energy (J)
    pub const HARTREE_J: f64 = 4.359_744_722_071_3e-18;

    /// Bohr magneton in eV/T
    pub const BOHR_MAGNETON_EV_T: f64 = 5.788_381_801_2e-5;

    /// Nuclear magneton in eV/T
    pub const NUCLEAR_MAGNETON_EV_T: f64 = 3.152_451_259_4e-8;

    /// Rydberg energy (eV)
    pub const RYDBERG_EV: f64 = 13.605_693_122_994;

    /// Rydberg energy (J)
    pub const RYDBERG_J: f64 = 2.179_872_361_035_7e-18;

    /// Molar Planck constant (J·s/mol)
    pub const MOLAR_PLANCK: f64 = 3.990_312_712e-10;

    /// Molar gas constant in cal/(mol·K)
    pub const R_CAL: f64 = 1.987_204_259;

    /// Standard pressure (Pa)
    pub const STANDARD_PRESSURE: f64 = 101_325.0;

    /// Standard temperature (K)
    pub const STANDARD_TEMPERATURE: f64 = 273.15;

    /// Standard temperature and pressure molar volume (L/mol)
    pub const STP_MOLAR_VOLUME: f64 = 22.413_969_54;

    /// Ice point temperature (K)
    pub const ICE_POINT: f64 = 273.15;

    /// Triple point of water (K)
    pub const WATER_TRIPLE_POINT: f64 = 273.16;

    /// Critical temperature of water (K)
    pub const WATER_CRITICAL_TEMP: f64 = 647.1;

    /// Critical pressure of water (Pa)
    pub const WATER_CRITICAL_PRESSURE: f64 = 2.2064e7;

    /// Calorie to joule conversion
    pub const CAL_TO_J: f64 = 4.184;

    /// BTU to joule conversion
    pub const BTU_TO_J: f64 = 1055.056;

    /// Electronvolt to joule conversion
    pub const EV_TO_J: f64 = 1.602_176_634e-19;

    /// Wavelength of 1 eV photon (m)
    pub const EV_WAVELENGTH: f64 = 1.239_841_984e-6;

    /// Frequency of 1 eV photon (Hz)
    pub const EV_FREQUENCY: f64 = 2.417_989_242e14;

    /// Temperature equivalent of 1 eV (K)
    pub const EV_TEMPERATURE: f64 = 11_604.518_12;

    /// Energy equivalent of 1 K (eV)
    pub const K_TO_EV: f64 = 8.617_333_262e-5;

    /// Wavenumber of 1 eV photon (1/m)
    pub const EV_WAVENUMBER: f64 = 8.065_543_988e5;

    /// X-ray wavelength Cu Kα1 (m)
    pub const CU_KA1_WAVELENGTH: f64 = 1.540_598e-10;

    /// X-ray wavelength Mo Kα1 (m)
    pub const MO_KA1_WAVELENGTH: f64 = 7.093_16e-11;

    /// Lattice parameter of silicon (m)
    pub const SI_LATTICE_PARAMETER: f64 = 5.431_020_511e-10;
}

/// Constants for spectroscopy and atomic physics
pub mod spectroscopy {
    /// Hydrogen ionization energy (eV)
    pub const HYDROGEN_IONIZATION: f64 = 13.605_693_122_994;

    /// Lyman alpha wavelength (m)
    pub const LYMAN_ALPHA: f64 = 1.215_668e-7;

    /// Hydrogen alpha wavelength (m)
    pub const H_ALPHA: f64 = 6.562_82e-7;

    /// D line wavelength (sodium) (m)
    pub const SODIUM_D: f64 = 5.892_50e-7;

    /// He-Ne laser wavelength (m)
    pub const HE_NE_LASER: f64 = 6.328e-7;

    /// Ar ion laser wavelength (m)
    pub const AR_ION_LASER: f64 = 5.145e-7;

    /// Ti:sapphire laser center wavelength (m)
    pub const TI_SAPPHIRE_CENTER: f64 = 8e-7;

    /// Nd:YAG fundamental wavelength (m)
    pub const ND_YAG_FUNDAMENTAL: f64 = 1.064e-6;

    /// Nd:YAG second harmonic wavelength (m)
    pub const ND_YAG_SECOND_HARMONIC: f64 = 5.32e-7;

    /// CO2 laser wavelength (m)
    pub const CO2_LASER: f64 = 1.064e-5;

    /// Visible light range (m)
    pub const VISIBLE_MIN: f64 = 3.8e-7;
    pub const VISIBLE_MAX: f64 = 7.0e-7;

    /// UV range (m)
    pub const UV_MIN: f64 = 1e-8;
    pub const UV_MAX: f64 = 3.8e-7;

    /// IR range (m)
    pub const IR_MIN: f64 = 7e-7;
    pub const IR_MAX: f64 = 1e-3;

    /// X-ray range (m)
    pub const XRAY_MIN: f64 = 1e-12;
    pub const XRAY_MAX: f64 = 1e-8;

    /// Gamma ray range (m)
    pub const GAMMA_MIN: f64 = 1e-16;
    pub const GAMMA_MAX: f64 = 1e-11;

    /// Radio wave range (m)
    pub const RADIO_MIN: f64 = 1e-3;
    pub const RADIO_MAX: f64 = 1e5;

    /// Microwave range (m)
    pub const MICROWAVE_MIN: f64 = 1e-3;
    pub const MICROWAVE_MAX: f64 = 1.0;

    /// Cosmic microwave background temperature (K)
    pub const CMB_TEMPERATURE: f64 = 2.725;

    /// 21 cm hydrogen line frequency (Hz)
    pub const HYDROGEN_21CM_FREQ: f64 = 1.420_405_751_8e9;

    /// 21 cm hydrogen line wavelength (m)
    pub const HYDROGEN_21CM_WAVELENGTH: f64 = 0.211_061_140_542;

    /// Fine structure splitting constant
    pub const FINE_STRUCTURE_SPLITTING: f64 = 7.297_352_566_4e-3;

    /// Hyperfine structure constant for hydrogen
    pub const HYPERFINE_HYDROGEN: f64 = 5.879_251_3e-6;

    /// Zeeman effect constant (Hz/T)
    pub const ZEEMAN_CONSTANT: f64 = 1.399_624_493_9e10;

    /// Stark effect constant (Hz·m/V)
    pub const STARK_CONSTANT: f64 = 2.468_e-16;

    /// Doppler broadening constant
    pub const DOPPLER_BROADENING: f64 = 2.998e-7;

    /// Natural linewidth constant
    pub const NATURAL_LINEWIDTH: f64 = 6.28e-6;

    /// Collision broadening constant
    pub const COLLISION_BROADENING: f64 = 1e-9;
}

/// Access to the `physical` module constants
pub use self::physical::*;

/// Access to commonly used math constants
pub use self::math::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mathematical_constants() {
        assert_eq!(math::PI, std::f64::consts::PI);
        assert_eq!(math::E, std::f64::consts::E);
        assert!((math::GOLDEN - 1.618_033_988_749_895).abs() < 1e-14);
    }

    #[test]
    fn test_physical_constants() {
        assert_eq!(physical::SPEED_OF_LIGHT, 299_792_458.0);
        assert_eq!(physical::C, physical::SPEED_OF_LIGHT);
        assert_eq!(physical::ELECTRON_VOLT, 1.602_176_634e-19);
    }

    #[test]
    fn test_unit_conversions() {
        // Use approx_eq for floating point comparisons with very small difference
        assert!((conversions::MILE_TO_METER - 1609.344).abs() < 1e-10);
        assert_eq!(conversions::INCH, 0.0254);
        assert_eq!(conversions::METER, 1.0);
    }

    #[test]
    fn test_temperature_conversion() {
        let celsius = 100.0;
        let kelvin = conversions::temperature::convert_temperature(celsius, "celsius", "kelvin");
        assert!((kelvin - 373.15).abs() < 1e-10);

        let fahrenheit =
            conversions::temperature::convert_temperature(celsius, "celsius", "fahrenheit");
        assert!((fahrenheit - 212.0).abs() < 1e-10);

        let back_to_celsius =
            conversions::temperature::convert_temperature(fahrenheit, "fahrenheit", "celsius");
        assert!((back_to_celsius - celsius).abs() < 1e-10);
    }

    #[test]
    fn test_prefix_values() {
        assert_eq!(prefixes::KILO, 1e3);
        assert_eq!(prefixes::MEGA, 1e6);
        assert_eq!(prefixes::MICRO, 1e-6);

        assert_eq!(prefixes::KIBI, 1024.0);
        assert_eq!(prefixes::MEBI, 1024.0 * 1024.0);
    }

    #[test]
    fn test_optics_conversions() {
        let wavelength = 1.0; // 1 meter
        let frequency = conversions::optics::lambda2nu(wavelength);
        assert!((frequency - physical::SPEED_OF_LIGHT).abs() < 1e-10);

        let back_to_wavelength = conversions::optics::nu2lambda(frequency);
        assert!((back_to_wavelength - wavelength).abs() < 1e-10);
    }

    #[test]
    fn test_additional_math_constants() {
        // Test some of the new mathematical constants
        assert!((math::CATALAN - 0.915_965_594_177_219).abs() < 1e-14);
        assert!((math::APERY - 1.202_056_903_159_594).abs() < 1e-14);
        assert!((math::FEIGENBAUM_DELTA - 4.669_201_609_102_990).abs() < 1e-13);
        assert!((math::PLASTIC - 1.324_717_957_244_746).abs() < 1e-14);
    }

    #[test]
    fn test_numerical_constants() {
        assert!(numerical::MACHINE_EPSILON_F64 > 0.0);
        assert!(numerical::DEFAULT_TOLERANCE > numerical::STRICT_TOLERANCE);
        assert!(numerical::RELAXED_TOLERANCE > numerical::DEFAULT_TOLERANCE);
        assert_eq!(numerical::DEFAULT_MAX_ITERATIONS, 1000);
        assert!((numerical::PHI_MINUS_1 - 0.618_033_988_749_895).abs() < 1e-14);
    }

    #[test]
    fn test_complex_constants() {
        use complex::*;
        assert_eq!(I.re, 0.0);
        assert_eq!(I.im, 1.0);
        assert_eq!(ONE.re, 1.0);
        assert_eq!(ONE.im, 0.0);
        assert_eq!(ZERO.re, 0.0);
        assert_eq!(ZERO.im, 0.0);

        // Test Euler's identity: e^(iπ) = -1
        assert_eq!(E_TO_I_PI, NEG_ONE);
    }

    #[test]
    fn test_chemistry_constants() {
        assert!((chemistry::HARTREE_EV - 27.211_386_245_988).abs() < 1e-12);
        assert!((chemistry::RYDBERG_EV - 13.605_693_122_994).abs() < 1e-12);
        assert_eq!(chemistry::STANDARD_TEMPERATURE, 273.15);
        assert_eq!(chemistry::STANDARD_PRESSURE, 101_325.0);
        assert!((chemistry::CAL_TO_J - 4.184).abs() < 1e-10);
    }

    #[test]
    fn test_spectroscopy_constants() {
        assert!((spectroscopy::HYDROGEN_IONIZATION - 13.605_693_122_994).abs() < 1e-12);
        assert!((spectroscopy::CMB_TEMPERATURE - 2.725).abs() < 1e-10);
        assert!(spectroscopy::VISIBLE_MIN < spectroscopy::VISIBLE_MAX);
        assert!(spectroscopy::UV_MAX <= spectroscopy::VISIBLE_MIN);
        assert!(spectroscopy::VISIBLE_MAX <= spectroscopy::IR_MIN);
    }

    #[test]
    fn test_extended_physical_constants() {
        assert!((physical::SOLAR_MASS - 1.988_47e30).abs() < 1e25);
        assert!((physical::EARTH_MASS - 5.972_168e24).abs() < 1e19);
        assert!((physical::BOHR_RADIUS - 5.291_772_109_03e-11).abs() < 1e-21);
        assert!((physical::COMPTON_WAVELENGTH - 2.426_310_238_67e-12).abs() < 1e-22);
    }
}
