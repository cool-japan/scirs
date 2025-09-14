//! # Semantic Versioning Implementation
//!
//! Comprehensive semantic versioning implementation with `SciRS2`-specific
//! extensions for scientific computing environments. Provides `SemVer` 2.0.0
//! compliance with additional features for research and enterprise use.

use crate::error::CoreError;
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Semantic version representation following `SemVer` 2.0.0
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Version {
    /// Major version (breaking changes)
    major: u64,
    /// Minor version (backward compatible features)
    minor: u64,
    /// Patch version (backward compatible bug fixes)
    patch: u64,
    /// Pre-release identifier (alpha, beta, rc, etc.)
    prerelease: Option<String>,
    /// Build metadata
    buildmetadata: Option<String>,
}

impl Version {
    /// Create a new version
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
            buildmetadata: None,
        }
    }

    /// Parse a simple version string (major.minor.patch format only)
    pub fn parse_simple(versionstr: &str) -> Result<Self, String> {
        let parts: Vec<&str> = versionstr.split('.').collect();
        if parts.len() != 3 {
            return Err(format!("Invalid version format: {versionstr}"));
        }

        Ok(Self {
            major: parts[0]
                .parse()
                .map_err(|e| format!("Invalid major version: {e}"))?,
            minor: parts[1]
                .parse()
                .map_err(|e| format!("Invalid minor version: {e}"))?,
            patch: parts[2]
                .parse()
                .map_err(|e| format!("Invalid patch version: {e}"))?,
            prerelease: None,
            buildmetadata: None,
        })
    }

    /// Create a version with pre-release
    pub fn release(major: u64, minor: u64, patch: u64, prerelease: String) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: Some(prerelease),
            buildmetadata: None,
        }
    }

    /// Parse a version string (supports pre-release and build metadata)
    pub fn parse(versionstr: &str) -> Result<Self, CoreError> {
        let version = versionstr.trim();

        // Remove 'v' prefix if present
        let version = if version.starts_with('v') || version.starts_with('V') {
            &version[1..]
        } else {
            version
        };

        // Split on '+' to separate build metadata
        let (version_part, buildmetadata) = if let Some(plus_pos) = version.find('+') {
            (
                &version[..plus_pos],
                Some(version[plus_pos + 1..].to_string()),
            )
        } else {
            (version, None)
        };

        // Split on '-' to separate pre-release
        let (core_version, prerelease) = if let Some(dash_pos) = version_part.find('-') {
            (
                &version_part[..dash_pos],
                Some(version_part[dash_pos + 1..].to_string()),
            )
        } else {
            (version_part, None)
        };

        // Parse major.minor.patch
        let parts: Vec<&str> = core_version.split('.').collect();
        if parts.len() != 3 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(format!("Invalid version format: {version}")),
            ));
        }

        let major = parts[0].parse::<u64>().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Invalid major version: {}",
                parts[0]
            )))
        })?;
        let minor = parts[1].parse::<u64>().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Invalid minor version: {}",
                parts[1]
            )))
        })?;
        let patch = parts[2].parse::<u64>().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Invalid patch version: {}",
                parts[2]
            )))
        })?;

        Ok(Self {
            major,
            minor,
            patch,
            prerelease,
            buildmetadata,
        })
    }

    /// Get major version
    pub fn major(&self) -> u64 {
        self.major
    }

    /// Get minor version
    pub fn minor(&self) -> u64 {
        self.minor
    }

    /// Get patch version
    pub fn patch(&self) -> u64 {
        self.patch
    }

    /// Get pre-release identifier
    pub fn prerelease(&self) -> Option<&str> {
        self.prerelease.as_deref()
    }

    /// Get build metadata
    pub fn buildmetadata(&self) -> Option<&str> {
        self.buildmetadata.as_deref()
    }

    /// Check if this is a pre-release version
    pub fn is_prerelease(&self) -> bool {
        self.prerelease.is_some()
    }

    /// Check if this is a stable release
    pub fn is_stable(&self) -> bool {
        !self.is_prerelease()
    }

    /// Increment major version (resets minor and patch to 0)
    pub fn increment_major(&mut self) {
        self.major += 1;
        self.minor = 0;
        self.patch = 0;
        self.prerelease = None;
        self.buildmetadata = None;
    }

    /// Increment minor version (resets patch to 0)
    pub fn increment_minor(&mut self) {
        self.minor += 1;
        self.patch = 0;
        self.prerelease = None;
        self.buildmetadata = None;
    }

    /// Increment patch version
    pub fn increment_patch(&mut self) {
        self.patch += 1;
        self.prerelease = None;
        self.buildmetadata = None;
    }

    /// Set pre-release identifier
    pub fn release_2(&mut self, prerelease: Option<String>) {
        self.prerelease = prerelease;
    }

    /// Set build metadata
    pub fn metadata(&mut self, buildmetadata: Option<String>) {
        self.buildmetadata = buildmetadata;
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Same major version means compatible (assuming proper `SemVer`)
        self.major == other.major && self.major > 0
    }

    /// Check if this version has breaking changes compared to another
    pub fn has_breakingchanges_from(&self, other: &Self) -> bool {
        self.major > other.major
    }

    /// Get the core version without pre-release or build metadata
    pub fn core_version(&self) -> Self {
        Self::new(self.major, self.minor, self.patch)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;

        if let Some(ref prerelease) = self.prerelease {
            write!(f, "-{prerelease}")?;
        }

        if let Some(ref buildmetadata) = self.buildmetadata {
            write!(f, "+{buildmetadata}")?;
        }

        Ok(())
    }
}

impl FromStr for Version {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare major, minor, patch
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            other => return other,
        }

        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            other => return other,
        }

        match self.patch.cmp(&other.patch) {
            Ordering::Equal => {}
            other => return other,
        }

        // Compare pre-release
        match (&self.prerelease, &other.prerelease) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less, // Pre-release < release
            (None, Some(_)) => Ordering::Greater, // Release > pre-release
            (Some(a), Some(b)) => compare_prerelease(a, b),
        }
    }
}

/// Compare pre-release versions according to `SemVer` rules
#[allow(dead_code)]
fn compare_prerelease(a: &str, b: &str) -> Ordering {
    let a_parts: Vec<&str> = a.split('.').collect();
    let b_parts: Vec<&str> = b.split('.').collect();

    for (a_part, b_part) in a_parts.iter().zip(b_parts.iter()) {
        // Try to parse as numbers first
        let a_num = a_part.parse::<u64>();
        let b_num = b_part.parse::<u64>();

        match (a_num, b_num) {
            (Ok(a_n), Ok(b_n)) => match a_n.cmp(&b_n) {
                Ordering::Equal => {}
                other => return other,
            },
            (Ok(_), Err(_)) => return Ordering::Less, // Numeric < alphanumeric
            (Err(_), Ok(_)) => return Ordering::Greater, // Alphanumeric > numeric
            (Err(_), Err(_)) => match a_part.cmp(b_part) {
                Ordering::Equal => {}
                other => return other,
            },
        }
    }

    // If all compared parts are equal, the longer one is greater
    a_parts.len().cmp(&b_parts.len())
}

/// Version constraint for specifying version requirements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionConstraint {
    /// Exact version match
    Exact(Version),
    /// Greater than
    GreaterThan(Version),
    /// Greater than or equal
    GreaterThanOrEqual(Version),
    /// Less than
    LessThan(Version),
    /// Less than or equal
    LessThanOrEqual(Version),
    /// Compatible with (same major version, >= minor.patch)
    Compatible(Version),
    /// Tilde range (~1.2.3 means >=1.2.3 and <1.3.0)
    Tilde(Version),
    /// Caret range (^1.2.3 means >=1.2.3 and <2.0.0)
    Caret(Version),
    /// Wildcard (*) - any version
    Any,
    /// Combined constraints (AND)
    And(Vec<VersionConstraint>),
    /// Alternative constraints (OR)
    Or(Vec<VersionConstraint>),
}

impl VersionConstraint {
    /// Parse a version constraint string
    pub fn constraint(constraintstr: &str) -> Result<Self, CoreError> {
        let constraint = constraintstr.trim();

        if constraint == "*" {
            return Ok(Self::Any);
        }

        if let Some(stripped) = constraint.strip_prefix(">=") {
            let version = Version::parse(stripped)?;
            return Ok(Self::GreaterThanOrEqual(version));
        }

        if let Some(stripped) = constraint.strip_prefix("<=") {
            let version = Version::parse(stripped)?;
            return Ok(Self::LessThanOrEqual(version));
        }

        if let Some(stripped) = constraint.strip_prefix('>') {
            let version = Version::parse(stripped)?;
            return Ok(Self::GreaterThan(version));
        }

        if let Some(stripped) = constraint.strip_prefix('<') {
            let version = Version::parse(stripped)?;
            return Ok(Self::LessThan(version));
        }

        if let Some(stripped) = constraint.strip_prefix('~') {
            let version = Version::parse(stripped)?;
            return Ok(Self::Tilde(version));
        }

        if let Some(stripped) = constraint.strip_prefix('^') {
            let version = Version::parse(stripped)?;
            return Ok(Self::Caret(version));
        }

        if let Some(stripped) = constraint.strip_prefix('=') {
            let version = Version::parse(stripped)?;
            return Ok(Self::Exact(version));
        }

        // Default to exact match
        let version = Version::parse(constraint)?;
        Ok(Self::Exact(version))
    }

    /// Check if a version matches this constraint
    pub fn matches(&self, version: &Version) -> bool {
        match self {
            Self::Exact(v) => version == v,
            Self::GreaterThan(v) => version > v,
            Self::GreaterThanOrEqual(v) => version >= v,
            Self::LessThan(v) => version < v,
            Self::LessThanOrEqual(v) => version <= v,
            Self::Compatible(v) => version.major == v.major && version >= v,
            Self::Tilde(v) => version.major == v.major && version.minor == v.minor && version >= v,
            Self::Caret(v) => {
                if v.major() > 0 {
                    version.major == v.major && version >= v
                } else if v.minor() > 0 {
                    version.major == 0 && version.minor == v.minor && version >= v
                } else {
                    version.major == 0
                        && version.minor == 0
                        && version.patch == v.patch
                        && version >= v
                }
            }
            Self::Any => true,
            Self::And(constraints) => constraints.iter().all(|c| c.matches(version)),
            Self::Or(constraints) => constraints.iter().any(|c| c.matches(version)),
        }
    }
}

impl fmt::Display for VersionConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(v) => write!(f, "={v}"),
            Self::GreaterThan(v) => write!(f, ">{v}"),
            Self::GreaterThanOrEqual(v) => write!(f, ">={v}"),
            Self::LessThan(v) => write!(f, "<{v}"),
            Self::LessThanOrEqual(v) => write!(f, "<={v}"),
            Self::Compatible(v) => write!(f, "~{v}"),
            Self::Tilde(v) => write!(f, "~{v}"),
            Self::Caret(v) => write!(f, "^{v}"),
            Self::Any => write!(f, "*"),
            Self::And(constraints) => {
                let constraintstrs: Vec<String> =
                    constraints.iter().map(|c| c.to_string()).collect();
                let joined = constraintstrs.join(" && ");
                write!(f, "{joined}")
            }
            Self::Or(constraints) => {
                let constraintstrs: Vec<String> =
                    constraints.iter().map(|c| c.to_string()).collect();
                let joined = constraintstrs.join(" || ");
                write!(f, "{joined}")
            }
        }
    }
}

/// Version range specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionRange {
    /// Minimum version (inclusive)
    pub min: Option<Version>,
    /// Maximum version (exclusive)
    pub max: Option<Version>,
    /// Include pre-release versions
    pub include_prerelease: bool,
}

impl VersionRange {
    /// Create a new version range
    pub fn new(min: Option<Version>, max: Option<Version>) -> Self {
        Self {
            min,
            max,
            include_prerelease: false,
        }
    }

    /// Create a range that includes pre-release versions
    pub fn with_prerelease(mut self) -> Self {
        self.include_prerelease = true;
        self
    }

    /// Check if a version is within this range
    pub fn contains(&self, version: &Version) -> bool {
        // Check pre-release inclusion
        if version.is_prerelease() && !self.include_prerelease {
            return false;
        }

        // Check minimum bound
        if let Some(ref min) = self.min {
            if version < min {
                return false;
            }
        }

        // Check maximum bound
        if let Some(ref max) = self.max {
            if version >= max {
                return false;
            }
        }

        true
    }

    /// Get all versions in a list that fall within this range
    pub fn filterversions<'a>(&self, versions: &'a [Version]) -> Vec<&'a Version> {
        versions.iter().filter(|v| self.contains(v)).collect()
    }
}

/// Version builder for convenient version construction
pub struct VersionBuilder {
    major: u64,
    minor: u64,
    patch: u64,
    prerelease: Option<String>,
    buildmetadata: Option<String>,
}

impl VersionBuilder {
    /// Create a new version builder
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
            buildmetadata: None,
        }
    }

    /// Set pre-release identifier
    pub fn release(mut self, prerelease: &str) -> Self {
        self.prerelease = Some(prerelease.to_string());
        self
    }

    /// Set build metadata
    pub fn metadata(mut self, buildmetadata: &str) -> Self {
        self.buildmetadata = Some(buildmetadata.to_string());
        self
    }

    /// Build the version
    pub fn build(self) -> Version {
        Version {
            major: self.major,
            minor: self.minor,
            patch: self.patch,
            prerelease: self.prerelease,
            buildmetadata: self.buildmetadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let version = Version::new(1, 2, 3);
        assert_eq!(version.major(), 1);
        assert_eq!(version.minor(), 2);
        assert_eq!(version.patch(), 3);
        assert!(!version.is_prerelease());
        assert!(version.is_stable());
    }

    #[test]
    fn test_version_parsing() {
        let version = Version::parse("1.2.3").unwrap();
        assert_eq!(version.to_string(), "1.2.3");

        let version = Version::parse("1.2.3-alpha.1").unwrap();
        assert_eq!(version.to_string(), "1.2.3-alpha.1");
        assert_eq!(version.prerelease(), Some("alpha.1"));
        assert!(version.is_prerelease());

        let version = Version::parse("1.2.3+build.123").unwrap();
        assert_eq!(version.buildmetadata(), Some("build.123"));

        let version = Version::parse("v1.2.3").unwrap();
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_comparison() {
        let v1_0_0 = Version::parse("1.0.0").unwrap();
        let v1_0_1 = Version::parse("1.0.1").unwrap();
        let v1_1_0 = Version::parse("1.1.0").unwrap();
        let v2_0_0 = Version::parse("2.0.0").unwrap();
        let v1_0_0_alpha = Version::parse("1.0.0-alpha").unwrap();

        assert!(v1_0_0 < v1_0_1);
        assert!(v1_0_1 < v1_1_0);
        assert!(v1_1_0 < v2_0_0);
        assert!(v1_0_0_alpha < v1_0_0);
    }

    #[test]
    fn test_version_increments() {
        let mut version = Version::parse("1.2.3-alpha+build").unwrap();

        version.increment_patch();
        assert_eq!(version.to_string(), "1.2.4");

        version.increment_minor();
        assert_eq!(version.to_string(), "1.3.0");

        version.increment_major();
        assert_eq!(version.to_string(), "2.0.0");
    }

    #[test]
    fn test_version_constraints() {
        let constraint = VersionConstraint::constraint(">=1.2.0").unwrap();
        let version = Version::parse("1.2.3").unwrap();
        assert!(constraint.matches(&version));

        let constraint = VersionConstraint::constraint("^1.2.0").unwrap();
        let version = Version::parse("1.5.0").unwrap();
        assert!(constraint.matches(&version));
        let version = Version::parse("2.0.0").unwrap();
        assert!(!constraint.matches(&version));

        let constraint = VersionConstraint::constraint("~1.2.0").unwrap();
        let version = Version::parse("1.2.5").unwrap();
        assert!(constraint.matches(&version));
        let version = Version::parse("1.3.0").unwrap();
        assert!(!constraint.matches(&version));
    }

    #[test]
    fn test_version_range() {
        let min = Version::parse("1.0.0").unwrap();
        let max = Version::parse("2.0.0").unwrap();
        let range = VersionRange::new(Some(min), Some(max));

        let version = Version::parse("1.5.0").unwrap();
        assert!(range.contains(&version));

        let version = Version::parse("0.9.0").unwrap();
        assert!(!range.contains(&version));

        let version = Version::parse("2.0.0").unwrap();
        assert!(!range.contains(&version));

        let version = Version::parse("1.5.0-alpha").unwrap();
        assert!(!range.contains(&version));

        let range = range.with_prerelease();
        assert!(range.contains(&version));
    }

    #[test]
    fn test_version_builder() {
        let version = VersionBuilder::new(1, 2, 3)
            .release("alpha.1")
            .metadata("build.123")
            .build();

        assert_eq!(version.to_string(), "1.2.3-alpha.1+build.123");
    }

    #[test]
    fn test_compatibility() {
        let v1_0_0 = Version::parse("1.0.0").unwrap();
        let v1_2_0 = Version::parse("1.2.0").unwrap();
        let v2_0_0 = Version::parse("2.0.0").unwrap();

        assert!(v1_0_0.is_compatible_with(&v1_2_0));
        assert!(v1_2_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v2_0_0));
        assert!(v2_0_0.has_breakingchanges_from(&v1_0_0)); // Major version change indicates breaking changes
    }

    #[test]
    fn test_prerelease_comparison() {
        let versions = vec![
            Version::parse("1.0.0-alpha").unwrap(),
            Version::parse("1.0.0-alpha.1").unwrap(),
            Version::parse("1.0.0-alpha.beta").unwrap(),
            Version::parse("1.0.0-beta").unwrap(),
            Version::parse("1.0.0-beta.2").unwrap(),
            Version::parse("1.0.0-beta.11").unwrap(),
            Version::parse("1.0.0-rc.1").unwrap(),
            Version::parse("1.0.0").unwrap(),
        ];

        let mut sortedversions = versions.clone();
        sortedversions.sort();

        assert_eq!(sortedversions, versions);
    }
}
