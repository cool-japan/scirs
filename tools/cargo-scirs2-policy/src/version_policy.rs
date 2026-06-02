//! Semantic versioning policy and deprecation configuration for SciRS2.
//!
//! This module defines the version policy structures that govern how the
//! SciRS2 project handles backward compatibility, deprecation timelines,
//! and long-term support branches.
//!
//! # SemVer Commitment
//!
//! - **Pre-stable (0.x)**: Breaking changes are allowed per minor version bump.
//! - **Stable (1.x)**: Full backward compatibility is required within the major version.
//!
//! # Deprecation Policy
//!
//! Items must carry `#[deprecated(since = "X.Y.Z", note = "...")]` for at least
//! 2 minor releases before removal (configurable via [`VersionPolicy::deprecation_window`]).

use std::fmt;

// ---------------------------------------------------------------------------
// SemVer commitment
// ---------------------------------------------------------------------------

/// The project's semantic versioning commitment level.
///
/// Determines what kind of breaking changes are permitted.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemVerCommitment {
    /// Pre-stable (0.x): breaking changes are allowed per minor version bump.
    ///
    /// Public API removals still require deprecation warnings, but the
    /// guarantee window is shorter (per [`VersionPolicy`]).
    PreStable,

    /// Stable (1.x+): full backward compatibility is required within the
    /// major version series.
    ///
    /// Public API items may only be removed after following the full
    /// deprecation timeline.
    Stable1x,
}

impl fmt::Display for SemVerCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            SemVerCommitment::PreStable => write!(f, "Pre-stable (0.x)"),
            SemVerCommitment::Stable1x => write!(f, "Stable (1.x)"),
            _ => write!(f, "Unknown"),
        }
    }
}

/// Determine the current SemVer commitment from a version string.
///
/// Returns [`SemVerCommitment::PreStable`] for `0.x.y` versions and
/// [`SemVerCommitment::Stable1x`] for `1.x.y` and above.
///
/// # Errors
///
/// Returns `None` if the version string cannot be parsed (no major component).
pub fn current_commitment(version: &str) -> Option<SemVerCommitment> {
    let major: u32 = version.split('.').next()?.parse().ok()?;
    if major == 0 {
        Some(SemVerCommitment::PreStable)
    } else {
        Some(SemVerCommitment::Stable1x)
    }
}

// ---------------------------------------------------------------------------
// Version policy
// ---------------------------------------------------------------------------

/// Configuration for the deprecation policy.
///
/// Controls how long deprecated items must remain before they can be removed,
/// and what metadata is required on `#[deprecated]` attributes.
#[derive(Debug, Clone)]
pub struct VersionPolicy {
    /// Number of minor versions a deprecated item must survive before removal.
    ///
    /// Default: `2` (e.g., deprecated in 0.4.0, removable in 0.6.0).
    pub deprecation_window: u32,

    /// Whether `#[deprecated]` attributes must include a `since` field.
    ///
    /// Default: `true`.
    pub require_since: bool,

    /// Whether `#[deprecated]` attributes must include a `note` field
    /// explaining the replacement or migration path.
    ///
    /// Default: `true`.
    pub require_note: bool,
}

impl Default for VersionPolicy {
    fn default() -> Self {
        Self {
            deprecation_window: 2,
            require_since: true,
            require_note: true,
        }
    }
}

// ---------------------------------------------------------------------------
// LTS policy
// ---------------------------------------------------------------------------

/// Long-term support (LTS) branch policy.
///
/// Defines which branches receive security patches and for how long.
#[derive(Debug, Clone)]
pub struct LtsPolicy {
    /// Branch names that are designated as LTS (e.g., `["0.4.x"]`).
    pub branches: Vec<String>,

    /// Number of months an LTS branch receives security patches after
    /// the next major/minor release.
    ///
    /// Default: `12`.
    pub security_patch_window_months: u32,
}

impl Default for LtsPolicy {
    fn default() -> Self {
        Self {
            branches: Vec::new(),
            security_patch_window_months: 12,
        }
    }
}

// ---------------------------------------------------------------------------
// Version parsing helpers
// ---------------------------------------------------------------------------

/// A parsed semantic version triple (major, minor, patch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemVer {
    /// Parse a version string like `"0.4.0"` into a [`SemVer`].
    ///
    /// Returns `None` if parsing fails.
    pub fn parse(version: &str) -> Option<Self> {
        // Strip any leading 'v' (e.g., "v0.4.0")
        let version = version.strip_prefix('v').unwrap_or(version);
        let mut parts = version.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts
            .next()
            .and_then(|p| {
                // Handle pre-release suffixes like "0.4.0-rc1"
                p.split('-').next().and_then(|s| s.parse().ok())
            })
            .unwrap_or(0);
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Returns the number of minor versions between `self` and `other`.
    ///
    /// If `self` and `other` have different major versions, returns `u32::MAX`
    /// to indicate a major version gap.
    pub fn minor_distance(&self, other: &SemVer) -> u32 {
        if self.major != other.major {
            return u32::MAX;
        }
        self.minor.abs_diff(other.minor)
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_policy_default() {
        let policy = VersionPolicy::default();
        assert_eq!(policy.deprecation_window, 2);
        assert!(policy.require_since);
        assert!(policy.require_note);
    }

    #[test]
    fn test_lts_policy_default() {
        let policy = LtsPolicy::default();
        assert!(policy.branches.is_empty());
        assert_eq!(policy.security_patch_window_months, 12);
    }

    #[test]
    fn test_semver_commitment_pre_stable() {
        assert_eq!(
            current_commitment("0.4.0"),
            Some(SemVerCommitment::PreStable)
        );
        assert_eq!(
            current_commitment("0.1.0"),
            Some(SemVerCommitment::PreStable)
        );
    }

    #[test]
    fn test_semver_commitment_stable() {
        assert_eq!(
            current_commitment("1.0.0"),
            Some(SemVerCommitment::Stable1x)
        );
        assert_eq!(
            current_commitment("2.3.1"),
            Some(SemVerCommitment::Stable1x)
        );
    }

    #[test]
    fn test_semver_commitment_invalid() {
        assert_eq!(current_commitment(""), None);
        assert_eq!(current_commitment("abc"), None);
    }

    #[test]
    fn test_semver_parse() {
        let v = SemVer::parse("0.4.0");
        assert_eq!(
            v,
            Some(SemVer {
                major: 0,
                minor: 4,
                patch: 0
            })
        );

        let v = SemVer::parse("1.2.3");
        assert_eq!(
            v,
            Some(SemVer {
                major: 1,
                minor: 2,
                patch: 3
            })
        );
    }

    #[test]
    fn test_semver_parse_with_prefix() {
        let v = SemVer::parse("v0.4.0");
        assert_eq!(
            v,
            Some(SemVer {
                major: 0,
                minor: 4,
                patch: 0
            })
        );
    }

    #[test]
    fn test_semver_parse_with_prerelease() {
        let v = SemVer::parse("0.4.0-rc1");
        assert_eq!(
            v,
            Some(SemVer {
                major: 0,
                minor: 4,
                patch: 0
            })
        );
    }

    #[test]
    fn test_semver_parse_two_parts() {
        let v = SemVer::parse("1.2");
        assert_eq!(
            v,
            Some(SemVer {
                major: 1,
                minor: 2,
                patch: 0
            })
        );
    }

    #[test]
    fn test_semver_parse_invalid() {
        assert_eq!(SemVer::parse(""), None);
        assert_eq!(SemVer::parse("abc"), None);
    }

    #[test]
    fn test_minor_distance_same_major() {
        let a = SemVer {
            major: 0,
            minor: 2,
            patch: 0,
        };
        let b = SemVer {
            major: 0,
            minor: 4,
            patch: 0,
        };
        assert_eq!(a.minor_distance(&b), 2);
        assert_eq!(b.minor_distance(&a), 2);
    }

    #[test]
    fn test_minor_distance_different_major() {
        let a = SemVer {
            major: 0,
            minor: 4,
            patch: 0,
        };
        let b = SemVer {
            major: 1,
            minor: 0,
            patch: 0,
        };
        assert_eq!(a.minor_distance(&b), u32::MAX);
    }

    #[test]
    fn test_semver_display() {
        let v = SemVer {
            major: 0,
            minor: 4,
            patch: 0,
        };
        assert_eq!(v.to_string(), "0.4.0");
    }

    #[test]
    fn test_semver_commitment_display() {
        assert_eq!(SemVerCommitment::PreStable.to_string(), "Pre-stable (0.x)");
        assert_eq!(SemVerCommitment::Stable1x.to_string(), "Stable (1.x)");
    }
}
