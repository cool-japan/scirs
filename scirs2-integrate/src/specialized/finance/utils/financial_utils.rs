//! Common financial utilities
//!
//! This module provides essential financial infrastructure including:
//! - **Day count conventions**: Actual/360, Actual/365, 30/360, Actual/Actual
//! - **Business day conventions**: Following, Modified Following, Preceding
//! - **Calendar management**: Holiday calendars, business day calculations
//! - **Compounding conventions**: Simple, Compound, Continuous
//!
//! # Example
//! ```
//! use scirs2_integrate::specialized::finance::utils::financial_utils::{
//!     DayCountConvention, CompoundingConvention,
//! };
//!
//! // Calculate year fraction
//! let start = (2024, 1, 15);
//! let end = (2024, 7, 15);
//! let yf = DayCountConvention::Actual360.year_fraction(start, end);
//! println!("Year fraction: {:.6}", yf);
//!
//! // Convert rates between conventions
//! let simple_rate = 0.05;
//! let continuous = CompoundingConvention::continuous_from_simple(simple_rate, 1.0);
//! ```

use crate::error::{IntegrateError, IntegrateResult as Result};

/// Day count convention for calculating time fractions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayCountConvention {
    /// Actual/360 - Actual days over 360 day year (Money market)
    Actual360,
    /// Actual/365 - Actual days over 365 day year (UK markets)
    Actual365,
    /// Actual/Actual - Actual days over actual year days (Treasury bonds)
    ActualActual,
    /// 30/360 - 30 days per month, 360 days per year (Corporate bonds)
    Thirty360,
    /// 30E/360 - European 30/360 (Eurobonds)
    ThirtyE360,
}

impl DayCountConvention {
    /// Calculate year fraction between two dates
    ///
    /// Dates are represented as (year, month, day) tuples
    pub fn year_fraction(&self, start: (i32, u32, u32), end: (i32, u32, u32)) -> f64 {
        match self {
            Self::Actual360 => {
                let days = days_between(start, end);
                days / 360.0
            }
            Self::Actual365 => {
                let days = days_between(start, end);
                days / 365.0
            }
            Self::ActualActual => {
                let days = days_between(start, end);
                let year_days = if is_leap_year(start.0) { 366.0 } else { 365.0 };
                days / year_days
            }
            Self::Thirty360 => {
                let (y1, m1, d1) = start;
                let (y2, m2, d2) = end;

                let d1_adj = if d1 == 31 { 30 } else { d1 };
                let d2_adj = if d2 == 31 && d1_adj == 30 { 30 } else { d2 };

                let days = 360.0 * (y2 - y1) as f64
                    + 30.0 * (m2 as i32 - m1 as i32) as f64
                    + (d2_adj as i32 - d1_adj as i32) as f64;
                days / 360.0
            }
            Self::ThirtyE360 => {
                let (y1, m1, d1) = start;
                let (y2, m2, d2) = end;

                let d1_adj = if d1 == 31 { 30 } else { d1 };
                let d2_adj = if d2 == 31 { 30 } else { d2 };

                let days = 360.0 * (y2 - y1) as f64
                    + 30.0 * (m2 as i32 - m1 as i32) as f64
                    + (d2_adj as i32 - d1_adj as i32) as f64;
                days / 360.0
            }
        }
    }

    /// Get the name of the convention
    pub fn name(&self) -> &'static str {
        match self {
            Self::Actual360 => "Actual/360",
            Self::Actual365 => "Actual/365",
            Self::ActualActual => "Actual/Actual",
            Self::Thirty360 => "30/360",
            Self::ThirtyE360 => "30E/360",
        }
    }
}

/// Business day convention for date adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusinessDayConvention {
    /// Following - Move to next business day
    Following,
    /// Modified Following - Following unless crosses month, then preceding
    ModifiedFollowing,
    /// Preceding - Move to previous business day
    Preceding,
    /// Modified Preceding - Preceding unless crosses month, then following
    ModifiedPreceding,
    /// Unadjusted - No adjustment
    Unadjusted,
}

impl BusinessDayConvention {
    /// Adjust a date according to the convention
    pub fn adjust<C: Calendar>(&self, date: (i32, u32, u32), calendar: &C) -> (i32, u32, u32) {
        match self {
            Self::Following => {
                let mut d = date;
                while !calendar.is_business_day(d) {
                    d = add_days(d, 1);
                }
                d
            }
            Self::ModifiedFollowing => {
                let mut d = date;
                let original_month = date.1;
                while !calendar.is_business_day(d) {
                    d = add_days(d, 1);
                }
                // If we crossed into next month, go back
                if d.1 != original_month {
                    d = date;
                    while !calendar.is_business_day(d) {
                        d = add_days(d, -1);
                    }
                }
                d
            }
            Self::Preceding => {
                let mut d = date;
                while !calendar.is_business_day(d) {
                    d = add_days(d, -1);
                }
                d
            }
            Self::ModifiedPreceding => {
                let mut d = date;
                let original_month = date.1;
                while !calendar.is_business_day(d) {
                    d = add_days(d, -1);
                }
                // If we crossed into previous month, go forward
                if d.1 != original_month {
                    d = date;
                    while !calendar.is_business_day(d) {
                        d = add_days(d, 1);
                    }
                }
                d
            }
            Self::Unadjusted => date,
        }
    }
}

/// Calendar trait for holiday and business day checking
pub trait Calendar {
    /// Check if a date is a business day
    fn is_business_day(&self, date: (i32, u32, u32)) -> bool;

    /// Check if a date is a holiday
    fn is_holiday(&self, date: (i32, u32, u32)) -> bool {
        !self.is_business_day(date)
    }

    /// Add business days to a date
    fn add_business_days(&self, date: (i32, u32, u32), n: i32) -> (i32, u32, u32) {
        let mut current = date;
        let direction = if n > 0 { 1 } else { -1 };
        let mut remaining = n.abs();

        while remaining > 0 {
            current = add_days(current, direction);
            if self.is_business_day(current) {
                remaining -= 1;
            }
        }

        current
    }
}

/// Simple weekend-only calendar (no holidays)
#[derive(Debug, Clone, Copy)]
pub struct WeekendCalendar;

impl Calendar for WeekendCalendar {
    fn is_business_day(&self, date: (i32, u32, u32)) -> bool {
        let weekday = day_of_week(date);
        // Monday = 1, ..., Friday = 5, Saturday = 6, Sunday = 0
        weekday != 0 && weekday != 6
    }
}

/// US Federal Reserve holiday calendar
#[derive(Debug, Clone)]
pub struct USFederalCalendar {
    /// Additional holidays beyond fixed dates
    additional_holidays: Vec<(i32, u32, u32)>,
}

impl USFederalCalendar {
    /// Create a new US Federal calendar
    pub fn new() -> Self {
        Self {
            additional_holidays: Vec::new(),
        }
    }

    /// Add a custom holiday
    pub fn add_holiday(&mut self, date: (i32, u32, u32)) {
        self.additional_holidays.push(date);
    }

    /// Check if date is a US federal holiday
    fn is_federal_holiday(&self, date: (i32, u32, u32)) -> bool {
        let (year, month, day) = date;

        // Fixed date holidays
        match (month, day) {
            (1, 1) => return true,   // New Year's Day
            (7, 4) => return true,   // Independence Day
            (11, 11) => return true, // Veterans Day
            (12, 25) => return true, // Christmas
            _ => {}
        }

        // Floating holidays (simplified - actual rules more complex)
        // Martin Luther King Day - 3rd Monday of January
        if month == 1 && (15..=21).contains(&day) && day_of_week(date) == 1 {
            return true;
        }

        // Presidents Day - 3rd Monday of February
        if month == 2 && (15..=21).contains(&day) && day_of_week(date) == 1 {
            return true;
        }

        // Memorial Day - Last Monday of May
        if month == 5 && (25..=31).contains(&day) && day_of_week(date) == 1 {
            return true;
        }

        // Labor Day - 1st Monday of September
        if month == 9 && day <= 7 && day_of_week(date) == 1 {
            return true;
        }

        // Thanksgiving - 4th Thursday of November
        if month == 11 && (22..=28).contains(&day) && day_of_week(date) == 4 {
            return true;
        }

        // Check additional holidays
        self.additional_holidays.contains(&date)
    }
}

impl Default for USFederalCalendar {
    fn default() -> Self {
        Self::new()
    }
}

impl Calendar for USFederalCalendar {
    fn is_business_day(&self, date: (i32, u32, u32)) -> bool {
        let weekday = day_of_week(date);
        // Not a weekend and not a holiday
        (weekday != 0 && weekday != 6) && !self.is_federal_holiday(date)
    }
}

/// Interest rate compounding convention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundingConvention {
    /// Simple interest: FV = PV(1 + rt)
    Simple,
    /// Compound interest: FV = PV(1 + r)^t
    Compound,
    /// Continuous compounding: FV = PV * e^(rt)
    Continuous,
}

impl CompoundingConvention {
    /// Calculate discount factor
    pub fn discount_factor(&self, rate: f64, time: f64) -> f64 {
        match self {
            Self::Simple => 1.0 / (1.0 + rate * time),
            Self::Compound => (1.0 + rate).powf(-time),
            Self::Continuous => (-rate * time).exp(),
        }
    }

    /// Calculate future value
    pub fn future_value(&self, present_value: f64, rate: f64, time: f64) -> f64 {
        match self {
            Self::Simple => present_value * (1.0 + rate * time),
            Self::Compound => present_value * (1.0 + rate).powf(time),
            Self::Continuous => present_value * (rate * time).exp(),
        }
    }

    /// Convert simple rate to continuous rate
    pub fn continuous_from_simple(simple_rate: f64, time: f64) -> f64 {
        ((1.0 + simple_rate * time).ln()) / time
    }

    /// Convert continuous rate to simple rate
    pub fn simple_from_continuous(continuous_rate: f64, time: f64) -> f64 {
        ((continuous_rate * time).exp() - 1.0) / time
    }

    /// Convert compound rate to continuous rate
    pub fn continuous_from_compound(compound_rate: f64) -> f64 {
        (1.0 + compound_rate).ln()
    }

    /// Convert continuous rate to compound rate
    pub fn compound_from_continuous(continuous_rate: f64) -> f64 {
        continuous_rate.exp() - 1.0
    }
}

// Helper functions for date calculations

/// Calculate days between two dates (end - start)
fn days_between(start: (i32, u32, u32), end: (i32, u32, u32)) -> f64 {
    let start_days = date_to_days(start);
    let end_days = date_to_days(end);
    (end_days - start_days) as f64
}

/// Convert date to days since epoch (simplified Julian day)
fn date_to_days(date: (i32, u32, u32)) -> i32 {
    let (y, m, d) = date;
    let a = (14 - m) / 12;
    let y_adj = y + 4800 - a as i32;
    let m_adj = m + 12 * a - 3;

    d as i32 + (153 * m_adj + 2) as i32 / 5 + 365 * y_adj + y_adj / 4 - y_adj / 100 + y_adj / 400
        - 32045
}

/// Add days to a date
fn add_days(date: (i32, u32, u32), days: i32) -> (i32, u32, u32) {
    let julian = date_to_days(date) + days;
    days_to_date(julian)
}

/// Convert days since epoch to date
fn days_to_date(julian: i32) -> (i32, u32, u32) {
    let a = julian + 32044;
    let b = (4 * a + 3) / 146097;
    let c = a - (146097 * b) / 4;
    let d = (4 * c + 3) / 1461;
    let e = c - (1461 * d) / 4;
    let m = (5 * e + 2) / 153;

    let day = (e - (153 * m + 2) / 5 + 1) as u32;
    let month = (m + 3 - 12 * (m / 10)) as u32;
    let year = (100 * b + d - 4800 + m / 10);

    (year, month, day)
}

/// Check if year is a leap year
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Calculate day of week (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
fn day_of_week(date: (i32, u32, u32)) -> u32 {
    let julian = date_to_days(date);
    ((julian + 1) % 7) as u32
}

/// Validate date format
pub fn validate_date(date: (i32, u32, u32)) -> Result<()> {
    let (year, month, day) = date;

    if !(1..=12).contains(&month) {
        return Err(IntegrateError::ValueError(format!(
            "Invalid month: {}",
            month
        )));
    }

    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => unreachable!(),
    };

    if day < 1 || day > days_in_month {
        return Err(IntegrateError::ValueError(format!(
            "Invalid day {} for month {}",
            day, month
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_day_count_actual360() {
        let start = (2024, 1, 1);
        let end = (2024, 7, 1);
        let yf = DayCountConvention::Actual360.year_fraction(start, end);
        assert!((yf - 0.5055).abs() < 0.001); // ~182 days / 360
    }

    #[test]
    fn test_day_count_actual365() {
        let start = (2024, 1, 1);
        let end = (2025, 1, 1);
        let yf = DayCountConvention::Actual365.year_fraction(start, end);
        // 2024 is leap year, 366 days
        assert!((yf - 1.0027).abs() < 0.001); // 366 / 365
    }

    #[test]
    fn test_day_count_thirty360() {
        let start = (2024, 1, 15);
        let end = (2024, 7, 15);
        let yf = DayCountConvention::Thirty360.year_fraction(start, end);
        assert!((yf - 0.5).abs() < 0.001); // Exactly 6 months
    }

    #[test]
    fn test_weekend_calendar() {
        let cal = WeekendCalendar;

        // Monday, January 1, 2024
        assert!(cal.is_business_day((2024, 1, 1)));

        // Saturday, January 6, 2024
        assert!(!cal.is_business_day((2024, 1, 6)));

        // Sunday, January 7, 2024
        assert!(!cal.is_business_day((2024, 1, 7)));
    }

    #[test]
    fn test_business_day_convention_following() {
        let cal = WeekendCalendar;

        // Saturday -> Monday
        let saturday = (2024, 1, 6);
        let adjusted = BusinessDayConvention::Following.adjust(saturday, &cal);
        assert_eq!(adjusted, (2024, 1, 8)); // Monday
    }

    #[test]
    fn test_business_day_convention_preceding() {
        let cal = WeekendCalendar;

        // Sunday -> Friday
        let sunday = (2024, 1, 7);
        let adjusted = BusinessDayConvention::Preceding.adjust(sunday, &cal);
        assert_eq!(adjusted, (2024, 1, 5)); // Friday
    }

    #[test]
    fn test_add_business_days() {
        let cal = WeekendCalendar;
        let start = (2024, 1, 1); // Monday

        // Add 5 business days: Mon -> Mon (next week)
        let result = cal.add_business_days(start, 5);
        assert_eq!(result, (2024, 1, 8));
    }

    #[test]
    fn test_compounding_discount_factor() {
        let rate = 0.05;
        let time = 1.0;

        let df_simple = CompoundingConvention::Simple.discount_factor(rate, time);
        assert!((df_simple - 0.9524).abs() < 0.001);

        let df_continuous = CompoundingConvention::Continuous.discount_factor(rate, time);
        assert!((df_continuous - 0.9512).abs() < 0.001);
    }

    #[test]
    fn test_compounding_future_value() {
        let pv = 100.0;
        let rate = 0.05;
        let time = 1.0;

        let fv_simple = CompoundingConvention::Simple.future_value(pv, rate, time);
        assert!((fv_simple - 105.0).abs() < 0.01);

        let fv_compound = CompoundingConvention::Compound.future_value(pv, rate, time);
        assert!((fv_compound - 105.0).abs() < 0.01);

        let fv_continuous = CompoundingConvention::Continuous.future_value(pv, rate, time);
        assert!((fv_continuous - 105.127).abs() < 0.01);
    }

    #[test]
    fn test_rate_conversions() {
        let simple = 0.05;
        let time = 1.0;

        let continuous = CompoundingConvention::continuous_from_simple(simple, time);
        let back_to_simple = CompoundingConvention::simple_from_continuous(continuous, time);

        assert!((back_to_simple - simple).abs() < 1e-10);
    }

    #[test]
    fn test_us_federal_calendar_new_years() {
        let cal = USFederalCalendar::new();
        let new_years = (2024, 1, 1);
        assert!(!cal.is_business_day(new_years));
    }

    #[test]
    fn test_us_federal_calendar_july_4() {
        let cal = USFederalCalendar::new();
        let july_4 = (2024, 7, 4);
        assert!(!cal.is_business_day(july_4));
    }

    #[test]
    fn test_us_federal_calendar_christmas() {
        let cal = USFederalCalendar::new();
        let christmas = (2024, 12, 25);
        assert!(!cal.is_business_day(christmas));
    }

    #[test]
    fn test_leap_year() {
        assert!(is_leap_year(2024)); // Divisible by 4
        assert!(!is_leap_year(2023)); // Not divisible by 4
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(!is_leap_year(1900)); // Divisible by 100 but not 400
    }

    #[test]
    fn test_date_validation() {
        assert!(validate_date((2024, 1, 31)).is_ok());
        assert!(validate_date((2024, 2, 29)).is_ok()); // Leap year
        assert!(validate_date((2023, 2, 29)).is_err()); // Not leap year
        assert!(validate_date((2024, 13, 1)).is_err()); // Invalid month
        assert!(validate_date((2024, 4, 31)).is_err()); // April has 30 days
    }

    #[test]
    fn test_day_of_week() {
        // January 1, 2024 is a Monday
        assert_eq!(day_of_week((2024, 1, 1)), 1);

        // January 6, 2024 is a Saturday
        assert_eq!(day_of_week((2024, 1, 6)), 6);

        // January 7, 2024 is a Sunday
        assert_eq!(day_of_week((2024, 1, 7)), 0);
    }

    #[test]
    fn test_add_days_forward() {
        let start = (2024, 1, 1);
        let result = add_days(start, 31);
        assert_eq!(result, (2024, 2, 1));
    }

    #[test]
    fn test_add_days_backward() {
        let start = (2024, 2, 1);
        let result = add_days(start, -31);
        assert_eq!(result, (2024, 1, 1));
    }

    #[test]
    fn test_modified_following_month_end() {
        let cal = WeekendCalendar;
        // Last day of January 2024 is Wednesday (business day)
        let jan_31 = (2024, 1, 31);
        assert!(cal.is_business_day(jan_31));

        // If this were a weekend, Modified Following would go back
        // Let's test with a date that would push into next month
        let saturday_month_end = (2024, 3, 30); // March 30, 2024 is Saturday
        let adjusted = BusinessDayConvention::ModifiedFollowing.adjust(saturday_month_end, &cal);
        // Should go back to Friday March 29 instead of forward to April 1
        assert_eq!(adjusted, (2024, 3, 29));
    }

    #[test]
    fn test_days_between_same_day() {
        let date = (2024, 1, 1);
        assert_eq!(days_between(date, date), 0.0);
    }

    #[test]
    fn test_days_between_one_year() {
        let start = (2024, 1, 1);
        let end = (2025, 1, 1);
        let days = days_between(start, end);
        assert_eq!(days, 366.0); // 2024 is leap year
    }
}
