//! `Dimension` — 7-tuple of SI base unit exponents.
//!
//! See [`super`] for module-level overview.

/// SI base unit identifiers.
///
/// Used to look up positions in the [`Dimension::exp`] tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SiBase {
    /// Length (meter, "m")
    Length,
    /// Mass (kilogram, "kg")
    Mass,
    /// Time (second, "s")
    Time,
    /// Electric current (ampere, "A")
    Current,
    /// Thermodynamic temperature (kelvin, "K")
    Temperature,
    /// Amount of substance (mole, "mol")
    Amount,
    /// Luminous intensity (candela, "cd")
    Luminosity,
}

impl SiBase {
    /// Index in the dimension tuple (0..7).
    pub fn index(&self) -> usize {
        match self {
            SiBase::Length => 0,
            SiBase::Mass => 1,
            SiBase::Time => 2,
            SiBase::Current => 3,
            SiBase::Temperature => 4,
            SiBase::Amount => 5,
            SiBase::Luminosity => 6,
        }
    }

    /// SI symbol (e.g. `"m"`, `"kg"`).
    pub fn symbol(&self) -> &'static str {
        match self {
            SiBase::Length => "m",
            SiBase::Mass => "kg",
            SiBase::Time => "s",
            SiBase::Current => "A",
            SiBase::Temperature => "K",
            SiBase::Amount => "mol",
            SiBase::Luminosity => "cd",
        }
    }

    /// All seven base units in canonical order.
    pub fn all() -> [SiBase; 7] {
        [
            SiBase::Length,
            SiBase::Mass,
            SiBase::Time,
            SiBase::Current,
            SiBase::Temperature,
            SiBase::Amount,
            SiBase::Luminosity,
        ]
    }
}

/// 7-tuple of SI base unit exponents (rational; we use `i32` for simplicity).
///
/// `Dimension::new([1, 0, -1, 0, 0, 0, 0])` represents `[length]·[time]^-1`
/// (i.e. velocity).
///
/// The order of exponents in `exp` matches [`SiBase::index`]:
/// `[Length, Mass, Time, Current, Temperature, Amount, Luminosity]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dimension {
    /// Per-base-unit exponents.
    pub exp: [i32; 7],
}

impl Dimension {
    /// New dimension from raw exponents.
    pub fn new(exp: [i32; 7]) -> Self {
        Self { exp }
    }

    /// Dimensionless (all exponents 0).
    pub fn dimensionless() -> Self {
        Self { exp: [0; 7] }
    }

    /// Length only (`[m]`).
    pub fn length() -> Self {
        let mut e = [0; 7];
        e[SiBase::Length.index()] = 1;
        Self { exp: e }
    }

    /// Mass only (`[kg]`).
    pub fn mass() -> Self {
        let mut e = [0; 7];
        e[SiBase::Mass.index()] = 1;
        Self { exp: e }
    }

    /// Time only (`[s]`).
    pub fn time() -> Self {
        let mut e = [0; 7];
        e[SiBase::Time.index()] = 1;
        Self { exp: e }
    }

    /// Electric current only (`[A]`).
    pub fn current() -> Self {
        let mut e = [0; 7];
        e[SiBase::Current.index()] = 1;
        Self { exp: e }
    }

    /// Temperature only (`[K]`).
    pub fn temperature() -> Self {
        let mut e = [0; 7];
        e[SiBase::Temperature.index()] = 1;
        Self { exp: e }
    }

    /// Amount of substance only (`[mol]`).
    pub fn amount() -> Self {
        let mut e = [0; 7];
        e[SiBase::Amount.index()] = 1;
        Self { exp: e }
    }

    /// Luminous intensity only (`[cd]`).
    pub fn luminosity() -> Self {
        let mut e = [0; 7];
        e[SiBase::Luminosity.index()] = 1;
        Self { exp: e }
    }

    /// Velocity (`[m·s^-1]`).
    pub fn velocity() -> Self {
        let mut e = [0; 7];
        e[SiBase::Length.index()] = 1;
        e[SiBase::Time.index()] = -1;
        Self { exp: e }
    }

    /// Acceleration (`[m·s^-2]`).
    pub fn acceleration() -> Self {
        let mut e = [0; 7];
        e[SiBase::Length.index()] = 1;
        e[SiBase::Time.index()] = -2;
        Self { exp: e }
    }

    /// Force (`[kg·m·s^-2]`, i.e. newton).
    pub fn force() -> Self {
        let mut e = [0; 7];
        e[SiBase::Mass.index()] = 1;
        e[SiBase::Length.index()] = 1;
        e[SiBase::Time.index()] = -2;
        Self { exp: e }
    }

    /// Energy (`[kg·m^2·s^-2]`, i.e. joule).
    pub fn energy() -> Self {
        let mut e = [0; 7];
        e[SiBase::Mass.index()] = 1;
        e[SiBase::Length.index()] = 2;
        e[SiBase::Time.index()] = -2;
        Self { exp: e }
    }

    /// Power (`[kg·m^2·s^-3]`, i.e. watt).
    pub fn power() -> Self {
        let mut e = [0; 7];
        e[SiBase::Mass.index()] = 1;
        e[SiBase::Length.index()] = 2;
        e[SiBase::Time.index()] = -3;
        Self { exp: e }
    }

    /// True if all exponents are 0.
    pub fn is_dimensionless(&self) -> bool {
        self.exp.iter().all(|&e| e == 0)
    }

    /// Compute the square-root dimension: returns `Some(d)` if all exponents
    /// are even (each halved), or `None` if any exponent is odd (which would
    /// require a fractional exponent, not representable in `i32`).
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_symbolic::units::Dimension;
    /// let area = Dimension::length().power_int(2); // m^2
    /// let length = area.sqrt_dim().unwrap();
    /// assert_eq!(length, Dimension::length());
    /// ```
    pub fn sqrt_dim(&self) -> Option<Self> {
        if self.exp.iter().any(|&e| e % 2 != 0) {
            return None;
        }
        let mut exp = [0; 7];
        for (i, e) in exp.iter_mut().enumerate() {
            *e = self.exp[i] / 2;
        }
        Some(Self { exp })
    }

    /// Dimensional product: `[a]·[b]` adds exponents.
    pub fn product(&self, other: &Self) -> Self {
        let mut exp = [0; 7];
        for (i, e) in exp.iter_mut().enumerate() {
            *e = self.exp[i] + other.exp[i];
        }
        Self { exp }
    }

    /// Dimensional quotient: `[a]/[b]` subtracts exponents.
    pub fn quotient(&self, other: &Self) -> Self {
        let mut exp = [0; 7];
        for (i, e) in exp.iter_mut().enumerate() {
            *e = self.exp[i] - other.exp[i];
        }
        Self { exp }
    }

    /// Dimensional power: `[a]^n` multiplies exponents.
    pub fn power_int(&self, n: i32) -> Self {
        let mut exp = [0; 7];
        for (i, e) in exp.iter_mut().enumerate() {
            *e = self.exp[i] * n;
        }
        Self { exp }
    }
}

impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "[dimensionless]");
        }
        let mut parts: Vec<String> = Vec::new();
        for base in SiBase::all() {
            let e = self.exp[base.index()];
            if e == 0 {
                continue;
            }
            if e == 1 {
                parts.push(base.symbol().to_string());
            } else {
                parts.push(format!("{}^{}", base.symbol(), e));
            }
        }
        write!(f, "[{}]", parts.join("·"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensionless_constructs() {
        let d = Dimension::dimensionless();
        assert!(d.is_dimensionless());
    }

    #[test]
    fn velocity_is_length_per_time() {
        let v = Dimension::velocity();
        let l = Dimension::length();
        let t = Dimension::time();
        assert_eq!(v, l.quotient(&t));
    }

    #[test]
    fn force_equals_mass_acceleration() {
        let f = Dimension::force();
        let m = Dimension::mass();
        let a = Dimension::acceleration();
        assert_eq!(f, m.product(&a));
    }

    #[test]
    fn energy_equals_force_length() {
        let e = Dimension::energy();
        let f = Dimension::force();
        let l = Dimension::length();
        assert_eq!(e, f.product(&l));
    }

    #[test]
    fn power_int_doubles_exponents() {
        let v = Dimension::velocity();
        let v_sq = v.power_int(2);
        // v^2 has units m^2·s^-2
        assert_eq!(v_sq.exp[SiBase::Length.index()], 2);
        assert_eq!(v_sq.exp[SiBase::Time.index()], -2);
    }

    #[test]
    fn power_int_zero_yields_dimensionless() {
        let v = Dimension::velocity();
        assert!(v.power_int(0).is_dimensionless());
    }

    #[test]
    fn power_int_negative_inverts() {
        let l = Dimension::length();
        let inv = l.power_int(-1);
        assert_eq!(inv.exp[SiBase::Length.index()], -1);
    }

    #[test]
    fn product_then_quotient_inverts() {
        let l = Dimension::length();
        let t = Dimension::time();
        let combined = l.product(&t).quotient(&t);
        assert_eq!(combined, l);
    }

    #[test]
    fn power_equals_energy_per_time() {
        let p = Dimension::power();
        let e = Dimension::energy();
        let t = Dimension::time();
        assert_eq!(p, e.quotient(&t));
    }

    #[test]
    fn display_velocity() {
        let v = Dimension::velocity();
        assert_eq!(format!("{}", v), "[m·s^-1]");
    }

    #[test]
    fn display_dimensionless() {
        assert_eq!(format!("{}", Dimension::dimensionless()), "[dimensionless]");
    }

    #[test]
    fn display_force() {
        // Order in display matches SiBase::all(): Length, Mass, Time, ...
        // Force = kg·m·s^-2; printed as m·kg·s^-2
        assert_eq!(format!("{}", Dimension::force()), "[m·kg·s^-2]");
    }

    #[test]
    fn sibase_all_covers_seven() {
        assert_eq!(SiBase::all().len(), 7);
    }

    #[test]
    fn sibase_index_unique() {
        let mut seen = [false; 7];
        for b in SiBase::all() {
            seen[b.index()] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
