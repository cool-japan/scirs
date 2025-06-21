//! # Unit Conversion System
//!
//! This module provides a comprehensive unit conversion system for scientific computing,
//! supporting dimensional analysis, automatic conversions, and unit safety.

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::numeric::ScientificNumber;
use std::collections::HashMap;
use std::fmt;

/// Represents a physical dimension (length, time, mass, etc.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Length,
    Time,
    Mass,
    Temperature,
    Current,
    Amount, // Amount of substance
    LuminousIntensity,
    Angle,
    SolidAngle,
    // Derived dimensions
    Area,          // Length²
    Volume,        // Length³
    Velocity,      // Length/Time
    Acceleration,  // Length/Time²
    Force,         // Mass⋅Length/Time²
    Energy,        // Mass⋅Length²/Time²
    Power,         // Mass⋅Length²/Time³
    Pressure,      // Mass/(Length⋅Time²)
    Frequency,     // 1/Time
    Voltage,       // Mass⋅Length²/(Time³⋅Current)
    Resistance,    // Mass⋅Length²/(Time³⋅Current²)
    Capacitance,   // Time⁴⋅Current²/(Mass⋅Length²)
    Inductance,    // Mass⋅Length²/(Time²⋅Current²)
    MagneticField, // Mass/(Time²⋅Current)
    Dimensionless, // No dimension
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Length => "Length",
            Self::Time => "Time",
            Self::Mass => "Mass",
            Self::Temperature => "Temperature",
            Self::Current => "Current",
            Self::Amount => "Amount",
            Self::LuminousIntensity => "Luminous Intensity",
            Self::Angle => "Angle",
            Self::SolidAngle => "Solid Angle",
            Self::Area => "Area",
            Self::Volume => "Volume",
            Self::Velocity => "Velocity",
            Self::Acceleration => "Acceleration",
            Self::Force => "Force",
            Self::Energy => "Energy",
            Self::Power => "Power",
            Self::Pressure => "Pressure",
            Self::Frequency => "Frequency",
            Self::Voltage => "Voltage",
            Self::Resistance => "Resistance",
            Self::Capacitance => "Capacitance",
            Self::Inductance => "Inductance",
            Self::MagneticField => "Magnetic Field",
            Self::Dimensionless => "Dimensionless",
        };
        write!(f, "{name}")
    }
}

/// Represents a unit system (SI, Imperial, CGS, etc.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnitSystem {
    SI,       // International System of Units
    Imperial, // Imperial/US customary units
    CGS,      // Centimeter-gram-second system
    Natural,  // Natural units (c = ℏ = 1)
    Atomic,   // Atomic units
    Planck,   // Planck units
}

impl fmt::Display for UnitSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SI => "SI",
            Self::Imperial => "Imperial",
            Self::CGS => "CGS",
            Self::Natural => "Natural",
            Self::Atomic => "Atomic",
            Self::Planck => "Planck",
        };
        write!(f, "{name}")
    }
}

/// Represents a specific unit with its properties
#[derive(Debug, Clone)]
pub struct UnitDefinition {
    /// The name of the unit
    pub name: String,
    /// The symbol of the unit
    pub symbol: String,
    /// The dimension this unit measures
    pub dimension: Dimension,
    /// The unit system this unit belongs to
    pub system: UnitSystem,
    /// Conversion factor to the base unit in SI (multiply to convert to SI)
    pub si_factor: f64,
    /// Offset for non-linear conversions (e.g., temperature)
    pub si_offset: f64,
    /// Whether this is a base unit in its system
    pub is_base: bool,
}

impl UnitDefinition {
    /// Create a new unit definition
    pub fn new(
        name: String,
        symbol: String,
        dimension: Dimension,
        system: UnitSystem,
        si_factor: f64,
        si_offset: f64,
        is_base: bool,
    ) -> Self {
        Self {
            name,
            symbol,
            dimension,
            system,
            si_factor,
            si_offset,
            is_base,
        }
    }

    /// Convert a value from this unit to SI units
    pub fn to_si(&self, value: f64) -> f64 {
        value * self.si_factor + self.si_offset
    }

    /// Convert a value from SI units to this unit
    pub fn from_si(&self, value: f64) -> f64 {
        (value - self.si_offset) / self.si_factor
    }
}

impl fmt::Display for UnitDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) - {} [{}]",
            self.name, self.symbol, self.dimension, self.system
        )
    }
}

/// A value with associated units
#[derive(Debug, Clone)]
pub struct UnitValue<T: ScientificNumber> {
    value: T,
    unit: String, // Unit symbol
}

impl<T: ScientificNumber> UnitValue<T> {
    /// Create a new unit value
    pub fn new(value: T, unit: String) -> Self {
        Self { value, unit }
    }

    /// Get the raw value
    pub fn value(&self) -> T {
        self.value
    }

    /// Get the unit symbol
    pub fn unit(&self) -> &str {
        &self.unit
    }
}

impl<T: ScientificNumber + fmt::Display> fmt::Display for UnitValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

/// Registry for unit definitions and conversions
pub struct UnitRegistry {
    units: HashMap<String, UnitDefinition>,
    aliases: HashMap<String, String>, // Map aliases to canonical unit symbols
}

impl UnitRegistry {
    /// Create a new unit registry
    pub fn new() -> Self {
        let mut registry = Self {
            units: HashMap::new(),
            aliases: HashMap::new(),
        };

        registry.register_standard_units();
        registry
    }

    /// Register a unit definition
    pub fn register_unit(&mut self, unit: UnitDefinition) {
        self.units.insert(unit.symbol.clone(), unit);
    }

    /// Register an alias for a unit
    pub fn register_alias(&mut self, alias: String, canonical: String) {
        self.aliases.insert(alias, canonical);
    }

    /// Get a unit definition by symbol or alias
    pub fn get_unit(&self, symbol: &str) -> Option<&UnitDefinition> {
        // First try direct lookup
        if let Some(unit) = self.units.get(symbol) {
            return Some(unit);
        }

        // Then try alias lookup
        if let Some(canonical) = self.aliases.get(symbol) {
            return self.units.get(canonical);
        }

        None
    }

    /// Check if two units are compatible (same dimension)
    pub fn are_compatible(&self, unit1: &str, unit2: &str) -> bool {
        if let (Some(u1), Some(u2)) = (self.get_unit(unit1), self.get_unit(unit2)) {
            u1.dimension == u2.dimension
        } else {
            false
        }
    }

    /// Convert a value from one unit to another
    pub fn convert(&self, value: f64, from_unit: &str, to_unit: &str) -> CoreResult<f64> {
        let from = self.get_unit(from_unit).ok_or_else(|| {
            CoreError::InvalidArgument(ErrorContext::new(format!("Unknown unit: {from_unit}")))
        })?;

        let to = self.get_unit(to_unit).ok_or_else(|| {
            CoreError::InvalidArgument(ErrorContext::new(format!("Unknown unit: {to_unit}")))
        })?;

        if from.dimension != to.dimension {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Cannot convert {} ({}) to {} ({}): incompatible dimensions",
                from_unit, from.dimension, to_unit, to.dimension
            ))));
        }

        // Convert from source unit to SI, then from SI to target unit
        let si_value = from.to_si(value);
        let result = to.from_si(si_value);

        Ok(result)
    }

    /// Convert a UnitValue to a different unit
    pub fn convert_value<T>(&self, value: &UnitValue<T>, to_unit: &str) -> CoreResult<UnitValue<T>>
    where
        T: ScientificNumber + TryFrom<f64>,
        f64: From<T>,
    {
        let converted_f64 = self.convert(value.value.into(), &value.unit, to_unit)?;

        let converted_value = T::try_from(converted_f64).map_err(|_| {
            CoreError::TypeError(ErrorContext::new(
                "Failed to convert back to original numeric type",
            ))
        })?;

        Ok(UnitValue::new(converted_value, to_unit.to_string()))
    }

    /// Get all units for a given dimension
    pub fn get_units_for_dimension(&self, dimension: &Dimension) -> Vec<&UnitDefinition> {
        self.units
            .values()
            .filter(|unit| unit.dimension == *dimension)
            .collect()
    }

    /// Get all units in a given system
    pub fn get_units_for_system(&self, system: &UnitSystem) -> Vec<&UnitDefinition> {
        self.units
            .values()
            .filter(|unit| unit.system == *system)
            .collect()
    }

    /// List all registered units
    pub fn list_units(&self) -> Vec<&UnitDefinition> {
        self.units.values().collect()
    }

    /// Register standard units (SI, Imperial, etc.)
    fn register_standard_units(&mut self) {
        // SI Base Units
        self.register_unit(UnitDefinition::new(
            "meter".to_string(),
            "m".to_string(),
            Dimension::Length,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));
        self.register_unit(UnitDefinition::new(
            "second".to_string(),
            "s".to_string(),
            Dimension::Time,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));
        self.register_unit(UnitDefinition::new(
            "kilogram".to_string(),
            "kg".to_string(),
            Dimension::Mass,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));
        self.register_unit(UnitDefinition::new(
            "kelvin".to_string(),
            "K".to_string(),
            Dimension::Temperature,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));
        self.register_unit(UnitDefinition::new(
            "ampere".to_string(),
            "A".to_string(),
            Dimension::Current,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));
        self.register_unit(UnitDefinition::new(
            "mole".to_string(),
            "mol".to_string(),
            Dimension::Amount,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));
        self.register_unit(UnitDefinition::new(
            "candela".to_string(),
            "cd".to_string(),
            Dimension::LuminousIntensity,
            UnitSystem::SI,
            1.0,
            0.0,
            true,
        ));

        // SI Derived Units
        self.register_unit(UnitDefinition::new(
            "radian".to_string(),
            "rad".to_string(),
            Dimension::Angle,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "hertz".to_string(),
            "Hz".to_string(),
            Dimension::Frequency,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "newton".to_string(),
            "N".to_string(),
            Dimension::Force,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "joule".to_string(),
            "J".to_string(),
            Dimension::Energy,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "watt".to_string(),
            "W".to_string(),
            Dimension::Power,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "pascal".to_string(),
            "Pa".to_string(),
            Dimension::Pressure,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "volt".to_string(),
            "V".to_string(),
            Dimension::Voltage,
            UnitSystem::SI,
            1.0,
            0.0,
            false,
        ));

        // Length units
        self.register_unit(UnitDefinition::new(
            "centimeter".to_string(),
            "cm".to_string(),
            Dimension::Length,
            UnitSystem::CGS,
            0.01,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "millimeter".to_string(),
            "mm".to_string(),
            Dimension::Length,
            UnitSystem::SI,
            0.001,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "kilometer".to_string(),
            "km".to_string(),
            Dimension::Length,
            UnitSystem::SI,
            1000.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "inch".to_string(),
            "in".to_string(),
            Dimension::Length,
            UnitSystem::Imperial,
            0.0254,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "foot".to_string(),
            "ft".to_string(),
            Dimension::Length,
            UnitSystem::Imperial,
            0.3048,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "yard".to_string(),
            "yd".to_string(),
            Dimension::Length,
            UnitSystem::Imperial,
            0.9144,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "mile".to_string(),
            "mi".to_string(),
            Dimension::Length,
            UnitSystem::Imperial,
            1609.344,
            0.0,
            false,
        ));

        // Time units
        self.register_unit(UnitDefinition::new(
            "minute".to_string(),
            "min".to_string(),
            Dimension::Time,
            UnitSystem::SI,
            60.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "hour".to_string(),
            "h".to_string(),
            Dimension::Time,
            UnitSystem::SI,
            3600.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "day".to_string(),
            "d".to_string(),
            Dimension::Time,
            UnitSystem::SI,
            86400.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "year".to_string(),
            "yr".to_string(),
            Dimension::Time,
            UnitSystem::SI,
            31_557_600.0,
            0.0,
            false,
        )); // Julian year

        // Mass units
        self.register_unit(UnitDefinition::new(
            "gram".to_string(),
            "g".to_string(),
            Dimension::Mass,
            UnitSystem::CGS,
            0.001,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "pound".to_string(),
            "lb".to_string(),
            Dimension::Mass,
            UnitSystem::Imperial,
            0.453_592_37,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "ounce".to_string(),
            "oz".to_string(),
            Dimension::Mass,
            UnitSystem::Imperial,
            0.028_349_523_125,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "ton".to_string(),
            "t".to_string(),
            Dimension::Mass,
            UnitSystem::SI,
            1000.0,
            0.0,
            false,
        ));

        // Temperature units
        self.register_unit(UnitDefinition::new(
            "celsius".to_string(),
            "°C".to_string(),
            Dimension::Temperature,
            UnitSystem::SI,
            1.0,
            273.15,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "fahrenheit".to_string(),
            "°F".to_string(),
            Dimension::Temperature,
            UnitSystem::Imperial,
            5.0 / 9.0,
            459.67 * 5.0 / 9.0,
            false,
        ));

        // Angle units
        self.register_unit(UnitDefinition::new(
            "degree".to_string(),
            "°".to_string(),
            Dimension::Angle,
            UnitSystem::SI,
            std::f64::consts::PI / 180.0,
            0.0,
            false,
        ));

        // Energy units
        self.register_unit(UnitDefinition::new(
            "calorie".to_string(),
            "cal".to_string(),
            Dimension::Energy,
            UnitSystem::CGS,
            4.184,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "kilocalorie".to_string(),
            "kcal".to_string(),
            Dimension::Energy,
            UnitSystem::SI,
            4184.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "electron_volt".to_string(),
            "eV".to_string(),
            Dimension::Energy,
            UnitSystem::Atomic,
            1.602_176_634e-19,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "kilowatt_hour".to_string(),
            "kWh".to_string(),
            Dimension::Energy,
            UnitSystem::SI,
            3_600_000.0,
            0.0,
            false,
        ));

        // Power units
        self.register_unit(UnitDefinition::new(
            "horsepower".to_string(),
            "hp".to_string(),
            Dimension::Power,
            UnitSystem::Imperial,
            745.7,
            0.0,
            false,
        ));

        // Pressure units
        self.register_unit(UnitDefinition::new(
            "atmosphere".to_string(),
            "atm".to_string(),
            Dimension::Pressure,
            UnitSystem::SI,
            101_325.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "bar".to_string(),
            "bar".to_string(),
            Dimension::Pressure,
            UnitSystem::SI,
            100_000.0,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "torr".to_string(),
            "Torr".to_string(),
            Dimension::Pressure,
            UnitSystem::SI,
            133.322,
            0.0,
            false,
        ));
        self.register_unit(UnitDefinition::new(
            "pounds_per_square_inch".to_string(),
            "psi".to_string(),
            Dimension::Pressure,
            UnitSystem::Imperial,
            6894.76,
            0.0,
            false,
        ));

        // Register common aliases
        self.register_alias("metre".to_string(), "m".to_string());
        self.register_alias("meters".to_string(), "m".to_string());
        self.register_alias("metres".to_string(), "m".to_string());
        self.register_alias("seconds".to_string(), "s".to_string());
        self.register_alias("minutes".to_string(), "min".to_string());
        self.register_alias("hours".to_string(), "h".to_string());
        self.register_alias("days".to_string(), "d".to_string());
        self.register_alias("years".to_string(), "yr".to_string());
        self.register_alias("degrees".to_string(), "°".to_string());
        self.register_alias("deg".to_string(), "°".to_string());
        self.register_alias("radians".to_string(), "rad".to_string());
        self.register_alias("inches".to_string(), "in".to_string());
        self.register_alias("feet".to_string(), "ft".to_string());
        self.register_alias("pounds".to_string(), "lb".to_string());
        self.register_alias("lbs".to_string(), "lb".to_string());
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global unit registry instance
static GLOBAL_UNIT_REGISTRY: std::sync::LazyLock<std::sync::RwLock<UnitRegistry>> =
    std::sync::LazyLock::new(|| std::sync::RwLock::new(UnitRegistry::new()));

/// Get the global unit registry
pub fn global_unit_registry() -> &'static std::sync::RwLock<UnitRegistry> {
    &GLOBAL_UNIT_REGISTRY
}

/// Convert a value between units using the global registry
pub fn convert(value: f64, from_unit: &str, to_unit: &str) -> CoreResult<f64> {
    let registry = global_unit_registry().read().map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(
            "Failed to acquire read lock on unit registry",
        ))
    })?;
    registry.convert(value, from_unit, to_unit)
}

/// Create a UnitValue
pub fn unit_value<T: ScientificNumber>(value: T, unit: &str) -> UnitValue<T> {
    UnitValue::new(value, unit.to_string())
}

/// Utility functions for common conversions
pub mod conversions {

    /// Temperature conversions
    pub fn celsius_to_fahrenheit(celsius: f64) -> f64 {
        celsius * 9.0 / 5.0 + 32.0
    }

    pub fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
        (fahrenheit - 32.0) * 5.0 / 9.0
    }

    pub fn celsius_to_kelvin(celsius: f64) -> f64 {
        celsius + 273.15
    }

    pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
        kelvin - 273.15
    }

    /// Length conversions
    pub fn meters_to_feet(meters: f64) -> f64 {
        meters / 0.3048
    }

    pub fn feet_to_meters(feet: f64) -> f64 {
        feet * 0.3048
    }

    pub fn inches_to_centimeters(inches: f64) -> f64 {
        inches * 2.54
    }

    pub fn centimeters_to_inches(cm: f64) -> f64 {
        cm / 2.54
    }

    /// Mass conversions
    pub fn kilograms_to_pounds(kg: f64) -> f64 {
        kg / 0.453_592_37
    }

    pub fn pounds_to_kilograms(lbs: f64) -> f64 {
        lbs * 0.453_592_37
    }

    /// Angle conversions
    pub fn degrees_to_radians(degrees: f64) -> f64 {
        degrees * std::f64::consts::PI / 180.0
    }

    pub fn radians_to_degrees(radians: f64) -> f64 {
        radians * 180.0 / std::f64::consts::PI
    }

    /// Energy conversions
    pub fn joules_to_calories(joules: f64) -> f64 {
        joules / 4.184
    }

    pub fn calories_to_joules(calories: f64) -> f64 {
        calories * 4.184
    }

    pub fn ev_to_joules(ev: f64) -> f64 {
        ev * 1.602_176_634e-19
    }

    pub fn joules_to_ev(joules: f64) -> f64 {
        joules / 1.602_176_634e-19
    }
}

/// Macro for convenient unit conversions
#[macro_export]
macro_rules! convert_units {
    ($value:expr, $from:expr => $to:expr) => {
        $crate::units::convert($value, $from, $to)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_registry_creation() {
        let registry = UnitRegistry::new();

        assert!(registry.get_unit("m").is_some());
        assert!(registry.get_unit("kg").is_some());
        assert!(registry.get_unit("s").is_some());
        assert!(registry.get_unit("K").is_some());
    }

    #[test]
    fn test_unit_compatibility() {
        let registry = UnitRegistry::new();

        assert!(registry.are_compatible("m", "km"));
        assert!(registry.are_compatible("m", "ft"));
        assert!(!registry.are_compatible("m", "kg"));
        assert!(!registry.are_compatible("s", "m"));
    }

    #[test]
    fn test_length_conversions() {
        let registry = UnitRegistry::new();

        // Meter to kilometer
        let result = registry.convert(1000.0, "m", "km").unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Meter to foot
        let result = registry.convert(1.0, "m", "ft").unwrap();
        assert!((result - 3.280_839_895).abs() < 1e-6);

        // Inch to centimeter
        let result = registry.convert(1.0, "in", "cm").unwrap();
        assert!((result - 2.54).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_conversions() {
        let registry = UnitRegistry::new();

        // Celsius to Kelvin
        let result = registry.convert(0.0, "°C", "K").unwrap();
        assert!((result - 273.15).abs() < 1e-10);

        // Fahrenheit to Celsius
        let result = registry.convert(32.0, "°F", "°C").unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // Fahrenheit to Kelvin
        let result = registry.convert(32.0, "°F", "K").unwrap();
        assert!((result - 273.15).abs() < 1e-10);
    }

    #[test]
    fn test_angle_conversions() {
        let registry = UnitRegistry::new();

        // Degrees to radians
        let result = registry.convert(180.0, "°", "rad").unwrap();
        assert!((result - std::f64::consts::PI).abs() < 1e-10);

        // Radians to degrees
        let result = registry
            .convert(std::f64::consts::PI / 2.0, "rad", "°")
            .unwrap();
        assert!((result - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_conversions() {
        let registry = UnitRegistry::new();

        // Joules to calories
        let result = registry.convert(4.184, "J", "cal").unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // eV to Joules
        let result = registry.convert(1.0, "eV", "J").unwrap();
        assert!((result - 1.602_176_634e-19).abs() < 1e-25);
    }

    #[test]
    fn test_unit_value() {
        let value = UnitValue::new(5.0, "m".to_string());
        assert_eq!(value.value(), 5.0);
        assert_eq!(value.unit(), "m");

        let formatted = format!("{}", value);
        assert!(formatted.contains("5"));
        assert!(formatted.contains("m"));
    }

    #[test]
    fn test_unit_value_conversion() {
        let registry = UnitRegistry::new();
        let value = UnitValue::new(1000.0, "m".to_string());

        let converted: UnitValue<f64> = registry.convert_value(&value, "km").unwrap();
        assert!((converted.value() - 1.0).abs() < 1e-10);
        assert_eq!(converted.unit(), "km");
    }

    #[test]
    fn test_incompatible_conversion() {
        let registry = UnitRegistry::new();

        let result = registry.convert(1.0, "m", "kg");
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_unit() {
        let registry = UnitRegistry::new();

        let result = registry.convert(1.0, "unknown", "m");
        assert!(result.is_err());
    }

    #[test]
    fn test_aliases() {
        let registry = UnitRegistry::new();

        // Test that aliases work
        assert!(registry.get_unit("meters").is_some());
        assert!(registry.get_unit("degrees").is_some());
        assert!(registry.get_unit("feet").is_some());
    }

    #[test]
    fn test_dimension_filtering() {
        let registry = UnitRegistry::new();

        let length_units = registry.get_units_for_dimension(&Dimension::Length);
        assert!(length_units.len() > 5); // m, cm, mm, km, in, ft, yd, mi

        let si_units = registry.get_units_for_system(&UnitSystem::SI);
        assert!(si_units.len() > 10);
    }

    #[test]
    fn test_global_convert_function() {
        let result = convert(1000.0, "m", "km").unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conversion_utilities() {
        use conversions::*;

        assert!((celsius_to_fahrenheit(0.0) - 32.0).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);

        assert!((degrees_to_radians(180.0) - std::f64::consts::PI).abs() < 1e-10);
        assert!((radians_to_degrees(std::f64::consts::PI) - 180.0).abs() < 1e-10);

        assert!((meters_to_feet(1.0) - 3.280_839_895).abs() < 1e-6);
        assert!((feet_to_meters(1.0) - 0.3048).abs() < 1e-10);
    }
}
