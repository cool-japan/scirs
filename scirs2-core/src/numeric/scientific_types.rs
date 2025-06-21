//! # Specialized Numeric Types for Scientific Domains
//!
//! This module provides strongly-typed numeric values for various scientific domains,
//! ensuring dimensional consistency and providing domain-specific operations.

use crate::numeric::{RealNumber, ScientificNumber};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A trait for physical units
pub trait Unit: Clone + fmt::Debug + 'static {
    /// The name of the unit (e.g., "meter", "second")
    fn name() -> &'static str;

    /// The symbol of the unit (e.g., "m", "s")
    fn symbol() -> &'static str;

    /// Whether this is a base unit or derived unit
    fn is_base_unit() -> bool {
        false
    }
}

/// A strongly-typed quantity with associated units
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Quantity<T: ScientificNumber, U: Unit> {
    value: T,
    _unit: PhantomData<U>,
}

impl<T: ScientificNumber, U: Unit> Quantity<T, U> {
    /// Create a new quantity
    pub fn new(value: T) -> Self {
        Self {
            value,
            _unit: PhantomData,
        }
    }

    /// Get the raw value
    pub fn value(&self) -> T {
        self.value
    }

    /// Get the value with explicit unit information
    pub fn with_unit(&self) -> (T, &'static str) {
        (self.value, U::symbol())
    }
}

impl<T: ScientificNumber, U: Unit> fmt::Display for Quantity<T, U>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, U::symbol())
    }
}

// Arithmetic operations for quantities
impl<T: ScientificNumber, U: Unit> Add for Quantity<T, U> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.value + rhs.value)
    }
}

impl<T: ScientificNumber, U: Unit> Sub for Quantity<T, U> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value - rhs.value)
    }
}

impl<T: ScientificNumber, U: Unit> Mul<T> for Quantity<T, U> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.value * rhs)
    }
}

impl<T: ScientificNumber, U: Unit> Div<T> for Quantity<T, U> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.value / rhs)
    }
}

impl<T: ScientificNumber + Neg<Output = T>, U: Unit> Neg for Quantity<T, U> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.value)
    }
}

// Base units
#[derive(Clone, Debug)]
pub struct Meter;
impl Unit for Meter {
    fn name() -> &'static str {
        "meter"
    }
    fn symbol() -> &'static str {
        "m"
    }
    fn is_base_unit() -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct Second;
impl Unit for Second {
    fn name() -> &'static str {
        "second"
    }
    fn symbol() -> &'static str {
        "s"
    }
    fn is_base_unit() -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct Kilogram;
impl Unit for Kilogram {
    fn name() -> &'static str {
        "kilogram"
    }
    fn symbol() -> &'static str {
        "kg"
    }
    fn is_base_unit() -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct Kelvin;
impl Unit for Kelvin {
    fn name() -> &'static str {
        "kelvin"
    }
    fn symbol() -> &'static str {
        "K"
    }
    fn is_base_unit() -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct Ampere;
impl Unit for Ampere {
    fn name() -> &'static str {
        "ampere"
    }
    fn symbol() -> &'static str {
        "A"
    }
    fn is_base_unit() -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct Mole;
impl Unit for Mole {
    fn name() -> &'static str {
        "mole"
    }
    fn symbol() -> &'static str {
        "mol"
    }
    fn is_base_unit() -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct Candela;
impl Unit for Candela {
    fn name() -> &'static str {
        "candela"
    }
    fn symbol() -> &'static str {
        "cd"
    }
    fn is_base_unit() -> bool {
        true
    }
}

// Derived units
#[derive(Clone, Debug)]
pub struct Radian;
impl Unit for Radian {
    fn name() -> &'static str {
        "radian"
    }
    fn symbol() -> &'static str {
        "rad"
    }
}

#[derive(Clone, Debug)]
pub struct Degree;
impl Unit for Degree {
    fn name() -> &'static str {
        "degree"
    }
    fn symbol() -> &'static str {
        "°"
    }
}

#[derive(Clone, Debug)]
pub struct Hertz;
impl Unit for Hertz {
    fn name() -> &'static str {
        "hertz"
    }
    fn symbol() -> &'static str {
        "Hz"
    }
}

#[derive(Clone, Debug)]
pub struct Newton;
impl Unit for Newton {
    fn name() -> &'static str {
        "newton"
    }
    fn symbol() -> &'static str {
        "N"
    }
}

#[derive(Clone, Debug)]
pub struct Joule;
impl Unit for Joule {
    fn name() -> &'static str {
        "joule"
    }
    fn symbol() -> &'static str {
        "J"
    }
}

#[derive(Clone, Debug)]
pub struct Watt;
impl Unit for Watt {
    fn name() -> &'static str {
        "watt"
    }
    fn symbol() -> &'static str {
        "W"
    }
}

#[derive(Clone, Debug)]
pub struct Pascal;
impl Unit for Pascal {
    fn name() -> &'static str {
        "pascal"
    }
    fn symbol() -> &'static str {
        "Pa"
    }
}

#[derive(Clone, Debug)]
pub struct Volt;
impl Unit for Volt {
    fn name() -> &'static str {
        "volt"
    }
    fn symbol() -> &'static str {
        "V"
    }
}

// Type aliases for common scientific quantities
pub type Length<T> = Quantity<T, Meter>;
pub type Time<T> = Quantity<T, Second>;
pub type Mass<T> = Quantity<T, Kilogram>;
pub type Temperature<T> = Quantity<T, Kelvin>;
pub type Current<T> = Quantity<T, Ampere>;
pub type Amount<T> = Quantity<T, Mole>;
pub type LuminousIntensity<T> = Quantity<T, Candela>;

pub type Angle<T> = Quantity<T, Radian>;
pub type AngleDegrees<T> = Quantity<T, Degree>;
pub type Frequency<T> = Quantity<T, Hertz>;
pub type Force<T> = Quantity<T, Newton>;
pub type Energy<T> = Quantity<T, Joule>;
pub type Power<T> = Quantity<T, Watt>;
pub type Pressure<T> = Quantity<T, Pascal>;
pub type Voltage<T> = Quantity<T, Volt>;

/// Specialized angle type with trigonometric operations
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Angle64 {
    radians: f64,
}

impl Angle64 {
    /// Create an angle from radians
    pub fn from_radians(radians: f64) -> Self {
        Self { radians }
    }

    /// Create an angle from degrees
    pub fn from_degrees(degrees: f64) -> Self {
        Self {
            radians: degrees * std::f64::consts::PI / 180.0,
        }
    }

    /// Get the angle in radians
    pub fn radians(&self) -> f64 {
        self.radians
    }

    /// Get the angle in degrees
    pub fn degrees(&self) -> f64 {
        self.radians * 180.0 / std::f64::consts::PI
    }

    /// Normalize the angle to [0, 2π)
    pub fn normalize(&self) -> Self {
        Self {
            radians: self.radians.rem_euclid(2.0 * std::f64::consts::PI),
        }
    }

    /// Normalize the angle to [-π, π)
    pub fn normalize_symmetric(&self) -> Self {
        let normalized = self.normalize();
        if normalized.radians > std::f64::consts::PI {
            Self {
                radians: normalized.radians - 2.0 * std::f64::consts::PI,
            }
        } else {
            normalized
        }
    }

    /// Sine of the angle
    pub fn sin(&self) -> f64 {
        self.radians.sin()
    }

    /// Cosine of the angle
    pub fn cos(&self) -> f64 {
        self.radians.cos()
    }

    /// Tangent of the angle
    pub fn tan(&self) -> f64 {
        self.radians.tan()
    }
}

impl fmt::Display for Angle64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6} rad ({:.2}°)", self.radians, self.degrees())
    }
}

impl Add for Angle64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_radians(self.radians + rhs.radians)
    }
}

impl Sub for Angle64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_radians(self.radians - rhs.radians)
    }
}

impl Mul<f64> for Angle64 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::from_radians(self.radians * rhs)
    }
}

impl Div<f64> for Angle64 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::from_radians(self.radians / rhs)
    }
}

impl Neg for Angle64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_radians(-self.radians)
    }
}

/// Complex number with scientific extensions
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    /// Create a new complex number
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Create from magnitude and phase
    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }

    /// Get the magnitude (absolute value)
    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Get the phase (argument)
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Get the complex conjugate
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Exponential function
    pub fn exp(&self) -> Self {
        let magnitude = self.re.exp();
        Self::from_polar(magnitude, self.im)
    }

    /// Natural logarithm
    pub fn ln(&self) -> Self {
        Self {
            re: self.magnitude().ln(),
            im: self.phase(),
        }
    }

    /// Power function
    pub fn powc(&self, exp: Self) -> Self {
        (self.ln() * exp).exp()
    }

    /// Power with real exponent
    pub fn powf(&self, exp: f64) -> Self {
        let magnitude = self.magnitude().powf(exp);
        let phase = self.phase() * exp;
        Self::from_polar(magnitude, phase)
    }

    /// Square root
    pub fn sqrt(&self) -> Self {
        self.powf(0.5)
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6} + {:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6} - {:.6}i", self.re, -self.im)
        }
    }
}

impl Add for Complex64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denominator,
            im: (self.im * rhs.re - self.re * rhs.im) / denominator,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

/// Vector in 3D space with scientific operations
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vector3D<T: ScientificNumber> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: ScientificNumber> Vector3D<T> {
    /// Create a new 3D vector
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    /// Unit vector along x-axis
    pub fn unit_x() -> Self {
        Self {
            x: T::one(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    /// Unit vector along y-axis
    pub fn unit_y() -> Self {
        Self {
            x: T::zero(),
            y: T::one(),
            z: T::zero(),
        }
    }

    /// Unit vector along z-axis
    pub fn unit_z() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::one(),
        }
    }

    /// Dot product
    pub fn dot(&self, other: &Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Magnitude squared
    pub fn magnitude_squared(&self) -> T {
        self.dot(self)
    }
}

impl<T: RealNumber> Vector3D<T> {
    /// Magnitude (length) of the vector
    pub fn magnitude(&self) -> T {
        ScientificNumber::sqrt(self.magnitude_squared())
    }

    /// Normalize the vector to unit length
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if ScientificNumber::is_finite(mag) && mag > T::zero() {
            Self {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        } else {
            *self
        }
    }

    /// Distance to another vector
    pub fn distance(&self, other: &Self) -> T {
        (*self - *other).magnitude()
    }

    /// Angle between two vectors (in radians)
    pub fn angle(&self, other: &Self) -> T {
        let dot = self.dot(other);
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product > T::zero() {
            RealNumber::acos(dot / mag_product)
        } else {
            T::zero()
        }
    }
}

impl<T: ScientificNumber> fmt::Display for Vector3D<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T: ScientificNumber> Add for Vector3D<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: ScientificNumber> Sub for Vector3D<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: ScientificNumber> Mul<T> for Vector3D<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: ScientificNumber> Div<T> for Vector3D<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: ScientificNumber + Neg<Output = T>> Neg for Vector3D<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Trait for converting between different unit systems
pub trait UnitConversion<From: Unit, To: Unit> {
    /// Convert from one unit to another
    fn convert(&self) -> Quantity<Self::NumericType, To>
    where
        Self: Sized;

    type NumericType: ScientificNumber;
}

/// Physical constants
pub mod constants {
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

    /// Planck's constant (J⋅s)
    pub const PLANCK: f64 = 6.626_070_15e-34;

    /// Reduced Planck's constant (J⋅s)
    pub const HBAR: f64 = PLANCK / (2.0 * std::f64::consts::PI);

    /// Elementary charge (C)
    pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

    /// Proton mass (kg)
    pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;

    /// Neutron mass (kg)
    pub const NEUTRON_MASS: f64 = 1.674_927_498_04e-27;

    /// Avogadro's number (mol⁻¹)
    pub const AVOGADRO: f64 = 6.022_140_76e23;

    /// Boltzmann constant (J/K)
    pub const BOLTZMANN: f64 = 1.380_649e-23;

    /// Gas constant (J/(mol⋅K))
    pub const GAS_CONSTANT: f64 = AVOGADRO * BOLTZMANN;

    /// Stefan-Boltzmann constant (W/(m²⋅K⁴))
    pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

    /// Gravitational constant (m³/(kg⋅s²))
    pub const GRAVITATIONAL: f64 = 6.674_30e-11;

    /// Standard gravity (m/s²)
    pub const STANDARD_GRAVITY: f64 = 9.806_65;

    /// Standard atmospheric pressure (Pa)
    pub const STANDARD_PRESSURE: f64 = 101_325.0;

    /// Absolute zero (°C)
    pub const ABSOLUTE_ZERO_CELSIUS: f64 = -273.15;
}

/// Utility functions for creating scientific quantities
pub mod utils {
    use super::*;

    /// Create a length from meters
    pub fn meters<T: ScientificNumber>(value: T) -> Length<T> {
        Length::new(value)
    }

    /// Create a time from seconds
    pub fn seconds<T: ScientificNumber>(value: T) -> Time<T> {
        Time::new(value)
    }

    /// Create a mass from kilograms
    pub fn kilograms<T: ScientificNumber>(value: T) -> Mass<T> {
        Mass::new(value)
    }

    /// Create a temperature from kelvin
    pub fn kelvin<T: ScientificNumber>(value: T) -> Temperature<T> {
        Temperature::new(value)
    }

    /// Create an angle from radians
    pub fn radians<T: ScientificNumber>(value: T) -> Angle<T> {
        Angle::new(value)
    }

    /// Create an angle from degrees
    pub fn degrees<T: ScientificNumber>(value: T) -> AngleDegrees<T> {
        AngleDegrees::new(value)
    }

    /// Create a frequency from hertz
    pub fn hertz<T: ScientificNumber>(value: T) -> Frequency<T> {
        Frequency::new(value)
    }

    /// Create a force from newtons
    pub fn newtons<T: ScientificNumber>(value: T) -> Force<T> {
        Force::new(value)
    }

    /// Create an energy from joules
    pub fn joules<T: ScientificNumber>(value: T) -> Energy<T> {
        Energy::new(value)
    }

    /// Create a power from watts
    pub fn watts<T: ScientificNumber>(value: T) -> Power<T> {
        Power::new(value)
    }

    /// Create a pressure from pascals
    pub fn pascals<T: ScientificNumber>(value: T) -> Pressure<T> {
        Pressure::new(value)
    }

    /// Create a voltage from volts
    pub fn volts<T: ScientificNumber>(value: T) -> Voltage<T> {
        Voltage::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantity_operations() {
        let length1 = Length::new(5.0);
        let length2 = Length::new(3.0);

        let sum = length1.clone() + length2.clone();
        assert_eq!(sum.value(), 8.0);

        let diff = length1.clone() - length2.clone();
        assert_eq!(diff.value(), 2.0);

        let scaled = length1.clone() * 2.0;
        assert_eq!(scaled.value(), 10.0);

        let divided = length1.clone() / 2.0;
        assert_eq!(divided.value(), 2.5);
    }

    #[test]
    fn test_angle64_operations() {
        let angle1 = Angle64::from_degrees(90.0);
        let angle2 = Angle64::from_radians(std::f64::consts::PI / 4.0);

        assert!((angle1.radians() - std::f64::consts::PI / 2.0).abs() < 1e-10);
        assert!((angle2.degrees() - 45.0).abs() < 1e-10);

        let sum = angle1 + angle2;
        assert!((sum.degrees() - 135.0).abs() < 1e-10);

        assert!((angle1.sin() - 1.0).abs() < 1e-10);
        assert!((angle1.cos()).abs() < 1e-10);

        let normalized = Angle64::from_radians(3.0 * std::f64::consts::PI).normalize();
        assert!((normalized.radians() - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_complex64_operations() {
        let z1 = Complex64::new(3.0, 4.0);
        let z2 = Complex64::new(1.0, 2.0);

        assert!((z1.magnitude() - 5.0).abs() < 1e-10);
        assert!((z1.phase() - (4.0_f64 / 3.0).atan()).abs() < 1e-10);

        let sum = z1 + z2;
        assert_eq!(sum.re, 4.0);
        assert_eq!(sum.im, 6.0);

        let product = z1 * z2;
        assert_eq!(product.re, -5.0); // 3*1 - 4*2
        assert_eq!(product.im, 10.0); // 3*2 + 4*1

        let conj = z1.conj();
        assert_eq!(conj.re, 3.0);
        assert_eq!(conj.im, -4.0);

        let polar = Complex64::from_polar(5.0, std::f64::consts::PI / 2.0);
        assert!((polar.re).abs() < 1e-10);
        assert!((polar.im - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3d_operations() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);

        let sum = v1 + v2;
        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);

        let dot = v1.dot(&v2);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6

        let cross = v1.cross(&v2);
        assert_eq!(cross.x, -3.0); // 2*6 - 3*5
        assert_eq!(cross.y, 6.0); // 3*4 - 1*6
        assert_eq!(cross.z, -3.0); // 1*5 - 2*4

        let magnitude = v1.magnitude();
        assert!((magnitude - (14.0_f64).sqrt()).abs() < 1e-10);

        let unit_x = Vector3D::<f64>::unit_x();
        let unit_y = Vector3D::<f64>::unit_y();
        let angle = unit_x.angle(&unit_y);
        assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_display() {
        assert_eq!(Meter::name(), "meter");
        assert_eq!(Meter::symbol(), "m");
        assert!(Meter::is_base_unit());

        assert_eq!(Hertz::name(), "hertz");
        assert_eq!(Hertz::symbol(), "Hz");
        assert!(!Hertz::is_base_unit());
    }

    #[test]
    fn test_quantity_display() {
        let length = Length::new(5.5);
        let formatted = format!("{}", length);
        assert!(formatted.contains("5.5"));
        assert!(formatted.contains("m"));
    }

    #[test]
    fn test_utils_functions() {
        let length = utils::meters(10.0);
        assert_eq!(length.value(), 10.0);

        let time = utils::seconds(5.0);
        assert_eq!(time.value(), 5.0);

        let angle = utils::degrees(90.0);
        assert_eq!(angle.value(), 90.0);
    }

    #[test]
    fn test_physical_constants() {
        // Test that constants are accessible and have expected types
        assert_eq!(
            constants::GAS_CONSTANT,
            constants::AVOGADRO * constants::BOLTZMANN
        );
    }
}
