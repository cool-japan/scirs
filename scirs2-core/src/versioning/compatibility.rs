//! # Backward Compatibility Checking
//!
//! Comprehensive backward compatibility checking and enforcement system
//! for API evolution management in production environments.

use super::{ApiVersion, Version};
use crate::error::CoreError;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Compatibility levels between API versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    /// Fully backward compatible
    BackwardCompatible,
    /// Compatible with minor changes
    MostlyCompatible,
    /// Some breaking changes but migration possible
    PartiallyCompatible,
    /// Major breaking changes requiring significant migration
    BreakingChanges,
    /// Incompatible - no migration path
    Incompatible,
}

impl CompatibilityLevel {
    /// Get the string representation
    pub const fn as_str(&self) -> &'static str {
        match self {
            CompatibilityLevel::BackwardCompatible => "backward_compatible",
            CompatibilityLevel::MostlyCompatible => "mostly_compatible",
            CompatibilityLevel::PartiallyCompatible => "partially_compatible",
            CompatibilityLevel::BreakingChanges => "breakingchanges",
            CompatibilityLevel::Incompatible => "incompatible",
        }
    }

    /// Check if migration is recommended
    pub fn requires_migration(&self) -> bool {
        matches!(
            self,
            CompatibilityLevel::PartiallyCompatible
                | CompatibilityLevel::BreakingChanges
                | CompatibilityLevel::Incompatible
        )
    }

    /// Check if automatic migration is possible
    pub fn supports_auto_migration(&self) -> bool {
        matches!(
            self,
            CompatibilityLevel::BackwardCompatible
                | CompatibilityLevel::MostlyCompatible
                | CompatibilityLevel::PartiallyCompatible
        )
    }
}

/// Detailed compatibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    /// Source version
    pub from_version: Version,
    /// Target version
    pub toversion: Version,
    /// Overall compatibility level
    pub compatibility_level: CompatibilityLevel,
    /// Detailed compatibility issues
    pub issues: Vec<CompatibilityIssue>,
    /// Breaking changes detected
    pub breakingchanges: Vec<BreakingChange>,
    /// Deprecated features that will be removed
    pub deprecated_features: Vec<String>,
    /// New features added
    pub new_features: Vec<String>,
    /// Migration recommendations
    pub migration_recommendations: Vec<String>,
    /// Estimated migration effort (in developer hours)
    pub estimated_migration_effort: Option<u32>,
}

/// Specific compatibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Component affected
    pub component: String,
    /// Issue description
    pub description: String,
    /// Suggested resolution
    pub resolution: Option<String>,
    /// Impact assessment
    pub impact: ImpactLevel,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational - no action required
    Info,
    /// Warning - action recommended
    Warning,
    /// Error - action required
    Error,
    /// Critical - blocking issue
    Critical,
}

/// Impact level of compatibility issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// No user impact
    None,
    /// Minor impact - easily resolved
    Low,
    /// Moderate impact - requires changes
    Medium,
    /// High impact - significant changes required
    High,
    /// Critical impact - may block migration
    Critical,
}

/// Breaking change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingChange {
    /// Change type
    pub change_type: ChangeType,
    /// Component affected
    pub component: String,
    /// Description of the change
    pub description: String,
    /// Migration path
    pub migration_path: Option<String>,
    /// Version where change was introduced
    pub introduced_in: Version,
}

/// Types of breaking changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// API signature changed
    ApiSignatureChange,
    /// Behavior change
    BehaviorChange,
    /// Removed feature
    FeatureRemoval,
    /// Configuration change
    ConfigurationChange,
    /// Dependency change
    DependencyChange,
    /// Data format change
    DataFormatChange,
}

/// Compatibility checker implementation
pub struct CompatibilityChecker {
    /// Registered versions and their metadata
    versions: HashMap<Version, ApiVersion>,
    /// Compatibility rules
    rules: Vec<CompatibilityRule>,
}

impl CompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            rules: Self::default_rules(),
        }
    }

    /// Register a version for compatibility checking
    pub fn register_version(&mut self, apiversion: &ApiVersion) -> Result<(), CoreError> {
        self.versions
            .insert(apiversion.version.clone(), apiversion.clone());
        Ok(())
    }

    /// Check compatibility between two versions
    pub fn check_compatibility(
        &self,
        from_version: &Version,
        toversion: &Version,
    ) -> Result<CompatibilityLevel, CoreError> {
        let report = self.get_compatibility_report(from_version, toversion)?;
        Ok(report.compatibility_level)
    }

    /// Get detailed compatibility report
    pub fn get_compatibility_report(
        &self,
        from_version: &Version,
        toversion: &Version,
    ) -> Result<CompatibilityReport, CoreError> {
        let from_api = self.versions.get(from_version).ok_or_else(|| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Version {from_version} not registered"
            )))
        })?;
        let to_api = self.versions.get(toversion).ok_or_else(|| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Version {toversion} not registered"
            )))
        })?;

        let mut report = CompatibilityReport {
            from_version: from_version.clone(),
            toversion: toversion.clone(),
            compatibility_level: CompatibilityLevel::BackwardCompatible,
            issues: Vec::new(),
            breakingchanges: Vec::new(),
            deprecated_features: to_api.deprecated_features.clone(),
            new_features: to_api.new_features.clone(),
            migration_recommendations: Vec::new(),
            estimated_migration_effort: None,
        };

        // Apply compatibility rules
        for rule in &self.rules {
            rule.apply(from_api, to_api, &mut report)?;
        }

        // Determine overall compatibility level
        report.compatibility_level = self.determine_compatibility_level(&report);

        // Generate migration recommendations
        self.generate_migration_recommendations(&mut report);

        // Estimate migration effort
        report.estimated_migration_effort = self.estimate_migration_effort(&report);

        Ok(report)
    }

    /// Add a custom compatibility rule
    pub fn add_rule(&mut self, rule: CompatibilityRule) {
        self.rules.push(rule);
    }

    /// Create default compatibility rules
    fn default_rules() -> Vec<CompatibilityRule> {
        vec![
            CompatibilityRule::SemVerRule,
            CompatibilityRule::BreakingChangeRule,
            CompatibilityRule::FeatureRemovalRule,
            CompatibilityRule::ApiSignatureRule,
            CompatibilityRule::BehaviorChangeRule,
        ]
    }

    /// Determine overall compatibility level based on issues
    fn determine_compatibility_level(&self, report: &CompatibilityReport) -> CompatibilityLevel {
        let has_critical = report
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Critical);
        let haserrors = report
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error);
        let has_warnings = report
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning);
        let has_breakingchanges = !report.breakingchanges.is_empty();

        if has_critical {
            CompatibilityLevel::Incompatible
        } else if has_breakingchanges || haserrors {
            if report.from_version.major() != report.toversion.major() {
                CompatibilityLevel::BreakingChanges
            } else {
                CompatibilityLevel::PartiallyCompatible
            }
        } else if has_warnings {
            CompatibilityLevel::MostlyCompatible
        } else {
            CompatibilityLevel::BackwardCompatible
        }
    }

    /// Generate migration recommendations
    fn generate_migration_recommendations(&self, report: &mut CompatibilityReport) {
        for issue in &report.issues {
            if let Some(ref resolution) = issue.resolution {
                report
                    .migration_recommendations
                    .push(format!("{}: {}", issue.component, resolution));
            }
        }

        for breaking_change in &report.breakingchanges {
            if let Some(ref migration_path) = breaking_change.migration_path {
                report
                    .migration_recommendations
                    .push(format!("{}, {}", breaking_change.component, migration_path));
            }
        }

        // Add version-specific recommendations
        if report.from_version.major() != report.toversion.major() {
            report
                .migration_recommendations
                .push("Major version upgrade - review all API usage".to_string());
        }

        if !report.deprecated_features.is_empty() {
            report
                .migration_recommendations
                .push("Update code to avoid deprecated features".to_string());
        }
    }

    /// Estimate migration effort in developer hours
    fn estimate_migration_effort(&self, report: &CompatibilityReport) -> Option<u32> {
        let mut effort_hours = 0u32;

        // Base effort for version differences
        let major_diff = report
            .toversion
            .major()
            .saturating_sub(report.from_version.major());
        let minor_diff = if major_diff == 0 {
            report
                .toversion
                .minor()
                .saturating_sub(report.from_version.minor())
        } else {
            0
        };

        effort_hours += (major_diff * 40) as u32; // 40 hours per major version
        effort_hours += (minor_diff * 8) as u32; // 8 hours per minor version

        // Add effort for specific issues
        for issue in &report.issues {
            effort_hours += match issue.impact {
                ImpactLevel::None => 0,
                ImpactLevel::Low => 2,
                ImpactLevel::Medium => 8,
                ImpactLevel::High => 24,
                ImpactLevel::Critical => 80,
            };
        }

        // Add effort for breaking changes
        for breaking_change in &report.breakingchanges {
            effort_hours += match breaking_change.change_type {
                ChangeType::ApiSignatureChange => 16,
                ChangeType::BehaviorChange => 24,
                ChangeType::FeatureRemoval => 32,
                ChangeType::ConfigurationChange => 8,
                ChangeType::DependencyChange => 16,
                ChangeType::DataFormatChange => 40,
            };
        }

        if effort_hours > 0 {
            Some(effort_hours)
        } else {
            None
        }
    }
}

impl Default for CompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility rule trait
pub trait CompatibilityRuleTrait {
    /// Apply the rule to generate compatibility issues
    fn apply(
        &self,
        from_api: &ApiVersion,
        to_api: &ApiVersion,
        report: &mut CompatibilityReport,
    ) -> Result<(), CoreError>;
}

/// Built-in compatibility rules
#[derive(Debug, Clone)]
pub enum CompatibilityRule {
    /// Semantic versioning rule
    SemVerRule,
    /// Breaking change detection rule
    BreakingChangeRule,
    /// Feature removal rule
    FeatureRemovalRule,
    /// API signature change rule
    ApiSignatureRule,
    /// Behavior change rule
    BehaviorChangeRule,
}

impl CompatibilityRuleTrait for CompatibilityRule {
    fn apply(
        &self,
        from_api: &ApiVersion,
        to_api: &ApiVersion,
        report: &mut CompatibilityReport,
    ) -> Result<(), CoreError> {
        match self {
            CompatibilityRule::SemVerRule => self.apply_semver_rule(from_api, to_api, report),
            CompatibilityRule::BreakingChangeRule => {
                self.apply_breaking_change_rule(from_api, to_api, report)
            }
            CompatibilityRule::FeatureRemovalRule => {
                self.apply_feature_removal_rule(from_api, to_api, report)
            }
            CompatibilityRule::ApiSignatureRule => {
                self.apply_api_signature_rule(from_api, to_api, report)
            }
            CompatibilityRule::BehaviorChangeRule => {
                self.apply_behavior_change_rule(from_api, to_api, report)
            }
        }
    }
}

impl CompatibilityRule {
    /// Apply semantic versioning rule
    fn apply_semver_rule(
        &self,
        from_api: &ApiVersion,
        to_api: &ApiVersion,
        report: &mut CompatibilityReport,
    ) -> Result<(), CoreError> {
        let from_version = &from_api.version;
        let toversion = &to_api.version;

        if toversion.major() > from_version.major() {
            report.issues.push(CompatibilityIssue {
                severity: IssueSeverity::Warning,
                component: "version".to_string(),
                description: "Major version upgrade detected".to_string(),
                resolution: Some("Review all API usage for breaking changes".to_string()),
                impact: ImpactLevel::High,
            });
        }

        if toversion < from_version {
            report.issues.push(CompatibilityIssue {
                severity: IssueSeverity::Error,
                component: "version".to_string(),
                description: "Downgrade detected".to_string(),
                resolution: Some("Downgrades are not supported".to_string()),
                impact: ImpactLevel::Critical,
            });
        }

        Ok(())
    }

    /// Apply breaking change rule
    fn apply_breaking_change_rule(
        &self,
        _from_api: &ApiVersion,
        to_api: &ApiVersion,
        report: &mut CompatibilityReport,
    ) -> Result<(), CoreError> {
        for breaking_change in &to_api.breakingchanges {
            report.breakingchanges.push(BreakingChange {
                change_type: ChangeType::BehaviorChange, // Default type
                component: "api".to_string(),
                description: breaking_change.clone(),
                migration_path: None,
                introduced_in: to_api.version.clone(),
            });

            report.issues.push(CompatibilityIssue {
                severity: IssueSeverity::Error,
                component: "api".to_string(),
                description: breaking_change.to_string(),
                resolution: Some("Update code to handle the breaking change".to_string()),
                impact: ImpactLevel::High,
            });
        }

        Ok(())
    }

    /// Apply feature removal rule
    fn apply_feature_removal_rule(
        &self,
        from_api: &ApiVersion,
        to_api: &ApiVersion,
        report: &mut CompatibilityReport,
    ) -> Result<(), CoreError> {
        // Check for features that existed in from_api but not in to_api
        for feature in &from_api.features {
            if !to_api.features.contains(feature) {
                report.breakingchanges.push(BreakingChange {
                    change_type: ChangeType::FeatureRemoval,
                    component: feature.clone(),
                    description: format!("Feature '{feature}' has been removed"),
                    migration_path: Some("Remove usage of this feature".to_string()),
                    introduced_in: to_api.version.clone(),
                });

                report.issues.push(CompatibilityIssue {
                    severity: IssueSeverity::Error,
                    component: feature.clone(),
                    description: format!("Feature '{feature}' no longer available"),
                    resolution: Some("Remove or replace feature usage".to_string()),
                    impact: ImpactLevel::High,
                });
            }
        }

        Ok(())
    }

    /// Apply API signature rule
    fn apply_api_signature_rule(
        &self,
        _from_api: &ApiVersion,
        _to_api: &ApiVersion,
        _report: &mut CompatibilityReport,
    ) -> Result<(), CoreError> {
        // This would typically analyze actual API signatures
        // For now, it's a placeholder
        Ok(())
    }

    /// Apply behavior change rule
    fn apply_behavior_change_rule(
        &self,
        _from_api: &ApiVersion,
        _to_api: &ApiVersion,
        _report: &mut CompatibilityReport,
    ) -> Result<(), CoreError> {
        // This would typically analyze behavior changes
        // For now, it's a placeholder
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::versioning::ApiVersionBuilder;

    #[test]
    fn test_compatibility_levels() {
        assert!(CompatibilityLevel::BackwardCompatible < CompatibilityLevel::BreakingChanges);
        assert!(CompatibilityLevel::BreakingChanges.requires_migration());
        assert!(CompatibilityLevel::BackwardCompatible.supports_auto_migration());
    }

    #[test]
    fn test_compatibility_checker() {
        let mut checker = CompatibilityChecker::new();

        let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())
            .feature("feature1")
            .build()
            .unwrap();
        let v2 = ApiVersionBuilder::new(Version::parse("1.1.0").unwrap())
            .feature("feature1")
            .feature("feature2")
            .new_feature("Added feature2")
            .build()
            .unwrap();

        checker.register_version(&v1).unwrap();
        checker.register_version(&v2).unwrap();

        let compatibility = checker
            .check_compatibility(&v1.version, &v2.version)
            .unwrap();
        assert_eq!(compatibility, CompatibilityLevel::BackwardCompatible);
    }

    #[test]
    fn test_breakingchanges() {
        let mut checker = CompatibilityChecker::new();

        let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())
            .feature("feature1")
            .build()
            .unwrap();
        let v2 = ApiVersionBuilder::new(Version::parse("2.0.0").unwrap())
            .breaking_change("Removed feature1")
            .build()
            .unwrap();

        checker.register_version(&v1).unwrap();
        checker.register_version(&v2).unwrap();

        let report = checker
            .get_compatibility_report(&v1.version, &v2.version)
            .unwrap();
        assert!(!report.breakingchanges.is_empty());
        assert!(report.compatibility_level.requires_migration());
    }

    #[test]
    fn test_migration_effort_estimation() {
        let mut checker = CompatibilityChecker::new();

        let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())
            .build()
            .unwrap();
        let v2 = ApiVersionBuilder::new(Version::parse("2.0.0").unwrap())
            .breaking_change("Major API overhaul")
            .build()
            .unwrap();

        checker.register_version(&v1).unwrap();
        checker.register_version(&v2).unwrap();

        let report = checker
            .get_compatibility_report(&v1.version, &v2.version)
            .unwrap();
        assert!(report.estimated_migration_effort.is_some());
        assert!(report.estimated_migration_effort.unwrap() > 0);
    }

    #[test]
    fn test_feature_removal_detection() {
        let mut checker = CompatibilityChecker::new();

        let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())
            .feature("feature1")
            .feature("feature2")
            .build()
            .unwrap();
        let v2 = ApiVersionBuilder::new(Version::parse("1.1.0").unwrap())
            .feature("feature1")
            // feature2 removed
            .build().unwrap();

        checker.register_version(&v1).unwrap();
        checker.register_version(&v2).unwrap();

        let report = checker
            .get_compatibility_report(&v1.version, &v2.version)
            .unwrap();
        assert!(!report.breakingchanges.is_empty());

        let feature_removal = report
            .breakingchanges
            .iter()
            .find(|bc| bc.change_type == ChangeType::FeatureRemoval);
        assert!(feature_removal.is_some());
    }
}
