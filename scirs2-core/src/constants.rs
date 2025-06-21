//! # Physical and Mathematical Constants
//!
//! This module provides a comprehensive set of physical and mathematical constants for scientific computing,
//! closely aligning with SciPy's `constants` module to ensure compatibility and ease of migration.
//!
//! ## Mathematical Constants
//!
//! ```
//! use scirs2_core::constants::math::*;
//! assert!((PI - 3.14159265358979).abs() < 1e-14);
//! assert!((GOLDEN_RATIO - 1.618033988749895).abs() < 1e-14);
//! ```
//!
//! ## Physical Constants
//!
//! ```
//! use scirs2_core::constants::physical::*;
//! // Speed of light in vacuum (m/s)
//! assert_eq!(SPEED_OF_LIGHT, 299_792_458.0);
//! ```
//!
//! ## Unit Prefixes
//!
//! ```
//! use scirs2_core::constants::prefixes::*;
//! // SI prefixes
//! assert_eq!(KILO, 1e3);
//! // Binary prefixes
//! assert_eq!(KIBI, 1024.0);
//! ```
//!
//! ## Unit Conversions
//!
//! ```
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
        use crate::constants::math::PI;

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
        /// * `from_scale` - Source scale: "celsius", "kelvin", "fahrenheit", or "rankine"
        /// * `to_scale` - Target scale: "celsius", "kelvin", "fahrenheit", or "rankine"
        ///
        /// # Returns
        ///
        /// Converted temperature value
        ///
        /// # Examples
        ///
        /// ```
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
        pub fn convert_temperature(value: f64, from_scale: &str, to_scale: &str) -> f64 {
            // Convert from source scale to Kelvin
            let kelvin = match from_scale.to_lowercase().as_str() {
                "celsius" | "c" => value + ZERO_CELSIUS,
                "kelvin" | "k" => value,
                "fahrenheit" | "f" => (value - 32.0) * 5.0 / 9.0 + ZERO_CELSIUS,
                "rankine" | "r" => value * 5.0 / 9.0,
                _ => panic!("Unsupported 'from' scale: {from_scale}. Supported scales are Celsius, Kelvin, Fahrenheit, and Rankine"),
            };

            // Convert from Kelvin to target scale
            match to_scale.to_lowercase().as_str() {
                "celsius" | "c" => kelvin - ZERO_CELSIUS,
                "kelvin" | "k" => kelvin,
                "fahrenheit" | "f" => (kelvin - ZERO_CELSIUS) * 9.0 / 5.0 + 32.0,
                "rankine" | "r" => kelvin * 9.0 / 5.0,
                _ => panic!("Unsupported 'to' scale: {to_scale}. Supported scales are Celsius, Kelvin, Fahrenheit, and Rankine"),
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
        /// ```
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
        /// ```
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
}
