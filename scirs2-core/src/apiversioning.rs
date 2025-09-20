//! API versioning system for backward compatibility
//!
//! This module provides version management for scirs2-core APIs,
//! ensuring smooth transitions between versions and maintaining
//! backward compatibility.

use std::fmt;
use std::sync::{Mutex, OnceLock};

/// Represents a semantic version number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features, backward compatible)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string like "1.2.3" or "1.2.3-beta.1"
    pub fn parse(versionstr: &str) -> Result<Self, String> {
        // Split on '-' to handle version suffixes like "-beta.1"
        let base_version = versionstr.split('-').next().unwrap_or(versionstr);

        // Split the base version on '.'
        let parts: Vec<&str> = base_version.split('.').collect();

        if parts.len() < 3 {
            return Err(format!("Invalid version format: {versionstr}"));
        }

        let major = parts[0].parse::<u32>().map_err(|_| parts[0].to_string())?;
        let minor = parts[1].parse::<u32>().map_err(|_| parts[1].to_string())?;
        let patch = parts[2].parse::<u32>().map_err(|_| parts[2].to_string())?;

        Ok(Self::new(major, minor, patch))
    }

    /// Current version of scirs2-core (Beta 1)
    pub const CURRENT: Self = Self::new(0, 1, 0);

    /// Current beta release identifier
    pub const CURRENT_BETA: &'static str = "0.1.0-beta.2";

    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Same major version and greater or equal minor/patch
        self.major == other.major
            && (self.minor > other.minor
                || (self.minor == other.minor && self.patch >= other.patch))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Trait for versioned APIs
pub trait Versioned {
    /// Get the version this API was introduced in
    fn since_version() -> Version;

    /// Get the version this API was deprecated in (if any)
    fn deprecated_version() -> Option<Version> {
        None
    }

    /// Check if this API is available in a given version
    fn is_available_in(version: &Version) -> bool {
        version.is_compatible_with(&Self::since_version())
            && Self::deprecated_version()
                .map(|dep| version < &dep)
                .unwrap_or(true)
    }
}

/// Macro to mark APIs with version information
#[macro_export]
macro_rules! since_version {
    ($major:expr, $minor:expr, $patch:expr) => {
        fn since_version() -> $crate::apiversioning::Version {
            $crate::apiversioning::Version::new($major, $minor, $patch)
        }
    };
}

/// Macro to mark deprecated APIs
#[macro_export]
macro_rules! deprecated_in {
    ($major:expr, $minor:expr, $patch:expr) => {
        fn deprecated_version() -> Option<$crate::apiversioning::Version> {
            Some($crate::apiversioning::Version::new($major, $minor, $patch))
        }
    };
}

/// Version registry for tracking API changes
pub struct VersionRegistry {
    entries: Vec<ApiEntry>,
}

#[derive(Debug, Clone)]
pub struct ApiEntry {
    pub name: String,
    pub module: String,
    pub since: Version,
    pub deprecated: Option<Version>,
    pub replacement: Option<String>,
    pub breakingchanges: Vec<BreakingChange>,
    pub example_usage: Option<String>,
    pub migration_example: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BreakingChange {
    pub version: Version,
    pub description: String,
    pub impact: BreakingChangeImpact,
    pub mitigation: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BreakingChangeImpact {
    Low,      // Minor API signature changes
    Medium,   // Functionality changes
    High,     // Major restructuring
    Critical, // Complete removal
}

impl VersionRegistry {
    /// Create a new version registry
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a new API
    pub fn register_api(
        &mut self,
        name: impl Into<String>,
        module: impl Into<String>,
        since: Version,
    ) -> &mut Self {
        let name_str = name.into();
        let module_str = module.into();

        // Check if the API is already registered
        if !self
            .entries
            .iter()
            .any(|e| e.name == name_str && e.module == module_str)
        {
            self.entries.push(ApiEntry {
                name: name_str,
                module: module_str,
                since,
                deprecated: None,
                replacement: None,
                breakingchanges: Vec::new(),
                example_usage: None,
                migration_example: None,
            });
        }
        self
    }

    /// Register an API with usage example
    pub fn register_api_with_example(
        &mut self,
        name: impl Into<String>,
        module: impl Into<String>,
        since: Version,
        example: impl Into<String>,
    ) -> &mut Self {
        let name_str = name.into();
        let module_str = module.into();

        if !self
            .entries
            .iter()
            .any(|e| e.name == name_str && e.module == module_str)
        {
            self.entries.push(ApiEntry {
                name: name_str,
                module: module_str,
                since,
                deprecated: None,
                replacement: None,
                breakingchanges: Vec::new(),
                example_usage: Some(example.into()),
                migration_example: None,
            });
        }
        self
    }

    /// Mark an API as deprecated
    pub fn deprecate_api(
        &mut self,
        name: &str,
        version: Version,
        replacement: Option<String>,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.deprecated = Some(version);
            entry.replacement = replacement;
            Ok(())
        } else {
            Err(format!("API '{name}' not found in registry"))
        }
    }

    /// Mark an API as deprecated with migration example
    pub fn deprecate_api_with_example(
        &mut self,
        name: &str,
        version: Version,
        replacement: Option<String>,
        migration_example: impl Into<String>,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.deprecated = Some(version);
            entry.replacement = replacement;
            entry.migration_example = Some(migration_example.into());
            Ok(())
        } else {
            Err(format!("API '{name}' not found in registry"))
        }
    }

    /// Add a breaking change to an API
    pub fn add_breaking_change(
        &mut self,
        apiname: &str,
        change: BreakingChange,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == apiname) {
            entry.breakingchanges.push(change);
            Ok(())
        } else {
            Err(format!("API '{apiname}' not found in registry"))
        }
    }

    /// Get all APIs available in a specific version
    pub fn apis_in_version(&self, version: &Version) -> Vec<&ApiEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                version.is_compatible_with(&entry.since)
                    && entry.deprecated.map(|dep| version < &dep).unwrap_or(true)
            })
            .collect()
    }

    /// Get all deprecated APIs in a version
    pub fn deprecated_apis_in_version(&self, version: &Version) -> Vec<&ApiEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.deprecated.map(|dep| version >= &dep).unwrap_or(false))
            .collect()
    }

    /// Generate migration guide between versions
    pub fn migration_guide(&self, from: &Version, to: &Version) -> String {
        let mut guide = format!("# Migration Guide: {from} → {to}\n\n");

        guide.push_str(&format!(
            "This guide helps you upgrade from scirs2-core {from} to {to}.\n\n"
        ));

        // Breaking changes analysis
        let breakingchanges: Vec<_> = self
            .entries
            .iter()
            .filter_map(|e| {
                if !e.breakingchanges.is_empty() {
                    let relevant_changes: Vec<_> = e
                        .breakingchanges
                        .iter()
                        .filter(|bc| bc.version > *from && bc.version <= *to)
                        .collect();
                    if !relevant_changes.is_empty() {
                        Some((e, relevant_changes))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        if !breakingchanges.is_empty() {
            guide.push_str("## ⚠️ Breaking Changes\n\n");

            for (api, changes) in breakingchanges {
                guide.push_str(&format!(
                    "### {name} ({module})\n\n",
                    name = api.name,
                    module = api.module
                ));

                for change in changes {
                    let impact_icon = match change.impact {
                        BreakingChangeImpact::Low => "🟡",
                        BreakingChangeImpact::Medium => "🟠",
                        BreakingChangeImpact::High => "🔴",
                        BreakingChangeImpact::Critical => "💥",
                    };

                    guide.push_str(&format!(
                        "{} **{}**: {}\n\n**Mitigation**: {}\n\n",
                        impact_icon, change.version, change.description, change.mitigation
                    ));
                }
            }
        }

        // Find removed APIs
        let removed: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                from.is_compatible_with(&e.since)
                    && e.deprecated.map(|d| to >= &d && from < &d).unwrap_or(false)
            })
            .collect();

        if !removed.is_empty() {
            guide.push_str("## 🗑️ Removed APIs\n\n");

            for api in removed {
                guide.push_str(&format!(
                    "### {name} ({module})\n\n",
                    name = api.name,
                    module = api.module
                ));

                if let Some(ref replacement) = api.replacement {
                    guide.push_str(&format!("**Replacement**: {replacement}\n\n"));
                }

                if let Some(ref migration_example) = api.migration_example {
                    guide.push_str("**Migration Example**:\n\n");
                    guide.push_str("```rust\n");
                    guide.push_str(migration_example);
                    guide.push_str("\n```\n\n");
                }
            }
        }

        // Find new APIs
        let new_apis: Vec<_> = self
            .entries
            .iter()
            .filter(|e| to.is_compatible_with(&e.since) && !from.is_compatible_with(&e.since))
            .collect();

        if !new_apis.is_empty() {
            guide.push_str("## ✨ New APIs\n\n");

            for api in new_apis {
                guide.push_str(&format!(
                    "### {} ({}) - Since {}\n\n",
                    api.name, api.module, api.since
                ));

                if let Some(ref example) = api.example_usage {
                    guide.push_str("**Usage Example**:\n\n");
                    guide.push_str("```rust\n");
                    guide.push_str(example);
                    guide.push_str("\n```\n\n");
                }
            }
        }

        // Migration checklist
        guide.push_str("## 📋 Migration Checklist\n\n");
        guide.push_str("- [ ] Update Cargo.toml dependencies\n");
        guide.push_str("- [ ] Fix compilation errors\n");
        guide.push_str("- [ ] Update deprecated API usage\n");
        guide.push_str("- [ ] Run test suite\n");
        guide.push_str("- [ ] Update documentation\n");
        guide.push_str("- [ ] Performance testing\n\n");

        guide.push_str("## 📚 Additional Resources\n\n");
        guide.push_str(&format!(
            "- [API Documentation](https://docs.rs/scirs2-core/{to})\n"
        ));
        guide.push_str(
            "- [Changelog](https://github.com/cool-japan/scirs/blob/main/CHANGELOG.md)\n",
        );
        guide.push_str("- [Examples](https://github.com/cool-japan/scirs/tree/main/examples)\n");

        guide
    }

    /// Generate a deprecation timeline
    pub fn deprecation_timeline(&self) -> String {
        let mut timeline = String::from("# API Deprecation Timeline\n\n");

        let mut deprecated_apis: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.deprecated.is_some())
            .collect();

        deprecated_apis.sort_by_key(|e| e.deprecated.unwrap());

        let mut current_version: Option<Version> = None;

        for api in deprecated_apis {
            let dep_version = api.deprecated.unwrap();

            if current_version != Some(dep_version) {
                timeline.push_str(&format!("\n## Version {dep_version}\n\n"));
                current_version = Some(dep_version);
            }

            timeline.push_str(&format!(
                "- **{name}** ({module})",
                name = api.name,
                module = api.module
            ));

            if let Some(ref replacement) = api.replacement {
                timeline.push_str(&format!(" → {replacement}"));
            }

            timeline.push('\n');
        }

        timeline
    }
}

impl Default for VersionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// API Freeze status for Beta 1 release
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiFreezeStatus {
    /// APIs are still changing (Alpha phase)
    Unfrozen,
    /// APIs are frozen for this version (Beta phase)
    Frozen,
    /// APIs are stable (Release phase)
    Stable,
}

/// Comprehensive API compatibility checker
#[derive(Debug, Clone)]
pub struct ApiCompatibilityChecker {
    /// Current freeze status
    freeze_status: ApiFreezeStatus,
    /// Frozen API surface for comparison
    frozen_apis: Vec<ApiSignature>,
    /// Compatibility rules
    #[allow(dead_code)]
    compatibility_rules: Vec<CompatibilityRule>,
}

/// API signature for detailed compatibility checking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ApiSignature {
    pub name: String,
    pub module: String,
    pub signature_hash: u64,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
    pub visibility: Visibility,
    pub stability: StabilityLevel,
}

/// Parameter definition for API signatures
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Parameter {
    pub name: String,
    pub type_name: String,
    pub is_optional: bool,
    pub defaultvalue: Option<String>,
}

/// API visibility levels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Visibility {
    Public,
    PublicCrate,
    Private,
}

/// API stability levels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StabilityLevel {
    Stable,
    Unstable,
    Deprecated,
    Experimental,
}

/// Compatibility rules for API changes
#[derive(Debug, Clone)]
pub struct CompatibilityRule {
    pub rule_type: CompatibilityRuleType,
    pub description: String,
    pub severity: CompatibilitySeverity,
    pub auto_fix: Option<String>,
}

/// Types of compatibility rules
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityRuleType {
    /// Function signature cannot change
    SignatureChange,
    /// Public APIs cannot be removed
    PublicApiRemoval,
    /// Parameter types cannot change
    ParameterTypeChange,
    /// Return types cannot change
    ReturnTypeChange,
    /// New required parameters cannot be added
    NewRequiredParameter,
    /// Visibility cannot be reduced
    VisibilityReduction,
}

/// Severity of compatibility violations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompatibilitySeverity {
    /// Will cause compilation failure
    Breaking,
    /// May cause runtime issues
    Warning,
    /// Informational only
    Info,
}

/// Compatibility check result
#[derive(Debug, Clone)]
pub struct CompatibilityCheckResult {
    pub is_compatible: bool,
    pub violations: Vec<CompatibilityViolation>,
    pub warnings: Vec<CompatibilityWarning>,
    pub suggestions: Vec<CompatibilitySuggestion>,
}

/// Compatibility violation details
#[derive(Debug, Clone)]
pub struct CompatibilityViolation {
    pub apiname: String,
    pub violation_type: CompatibilityRuleType,
    pub severity: CompatibilitySeverity,
    pub description: String,
    pub old_signature: Option<ApiSignature>,
    pub new_signature: Option<ApiSignature>,
    pub fix_suggestion: Option<String>,
}

/// Compatibility warning
#[derive(Debug, Clone)]
pub struct CompatibilityWarning {
    pub apiname: String,
    pub message: String,
    pub impact: String,
}

/// Compatibility suggestion
#[derive(Debug, Clone)]
pub struct CompatibilitySuggestion {
    pub apiname: String,
    pub suggestion: String,
    pub rationale: String,
}

impl ApiCompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new() -> Self {
        Self {
            freeze_status: ApiFreezeStatus::Unfrozen,
            frozen_apis: Vec::new(),
            compatibility_rules: Self::default_compatibility_rules(),
        }
    }

    /// Create checker for Beta 1 (frozen APIs)
    pub fn for_beta1() -> Self {
        Self {
            freeze_status: ApiFreezeStatus::Frozen,
            frozen_apis: Vec::new(),
            compatibility_rules: Self::beta1_compatibility_rules(),
        }
    }

    /// Default compatibility rules for general API evolution
    fn default_compatibility_rules() -> Vec<CompatibilityRule> {
        vec![
            CompatibilityRule {
                rule_type: CompatibilityRuleType::PublicApiRemoval,
                description: "Public APIs cannot be removed without deprecation".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: Some("Add deprecation annotation".to_string()),
            },
            CompatibilityRule {
                rule_type: CompatibilityRuleType::SignatureChange,
                description: "Function signatures cannot change in breaking ways".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: None,
            },
            CompatibilityRule {
                rule_type: CompatibilityRuleType::ParameterTypeChange,
                description: "Parameter types cannot change".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: Some("Create new function with different name".to_string()),
            },
        ]
    }

    /// Stricter compatibility rules for Beta 1 API freeze
    fn beta1_compatibility_rules() -> Vec<CompatibilityRule> {
        vec![
            CompatibilityRule {
                rule_type: CompatibilityRuleType::PublicApiRemoval,
                description: "No public APIs can be removed during Beta 1 freeze".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: None,
            },
            CompatibilityRule {
                rule_type: CompatibilityRuleType::SignatureChange,
                description: "No function signatures can change during API freeze".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: None,
            },
            CompatibilityRule {
                rule_type: CompatibilityRuleType::NewRequiredParameter,
                description: "No new required parameters can be added during freeze".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: Some("Make parameter optional with default value".to_string()),
            },
            CompatibilityRule {
                rule_type: CompatibilityRuleType::VisibilityReduction,
                description: "API visibility cannot be reduced during freeze".to_string(),
                severity: CompatibilitySeverity::Breaking,
                auto_fix: None,
            },
        ]
    }

    /// Freeze the current API surface
    pub fn freeze_apis(&mut self, apis: Vec<ApiSignature>) -> Result<(), String> {
        if self.freeze_status == ApiFreezeStatus::Stable {
            return Err("Cannot freeze APIs that are already stable".to_string());
        }

        self.frozen_apis = apis;
        self.freeze_status = ApiFreezeStatus::Frozen;
        Ok(())
    }

    /// Check compatibility between current and frozen APIs
    pub fn check_compatibility(&self, currentapis: &[ApiSignature]) -> CompatibilityCheckResult {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        if self.freeze_status == ApiFreezeStatus::Unfrozen {
            warnings.push(CompatibilityWarning {
                apiname: "*".to_string(),
                message: "APIs are not frozen - no compatibility checking performed".to_string(),
                impact: "API changes may break compatibility".to_string(),
            });

            return CompatibilityCheckResult {
                is_compatible: true,
                violations,
                warnings,
                suggestions,
            };
        }

        // Check for removed APIs
        for frozen_api in &self.frozen_apis {
            if frozen_api.visibility == Visibility::Public
                && frozen_api.stability == StabilityLevel::Stable
                && !currentapis
                    .iter()
                    .any(|api| api.name == frozen_api.name && api.module == frozen_api.module)
            {
                violations.push(CompatibilityViolation {
                    apiname: format!(
                        "{module}::{name}",
                        module = frozen_api.module,
                        name = frozen_api.name
                    ),
                    violation_type: CompatibilityRuleType::PublicApiRemoval,
                    severity: CompatibilitySeverity::Breaking,
                    description: "Public stable API was removed".to_string(),
                    old_signature: Some(frozen_api.clone()),
                    new_signature: None,
                    fix_suggestion: Some("Restore the API or mark as deprecated".to_string()),
                });
            }
        }

        // Check for signature changes
        for current_api in currentapis {
            if let Some(frozen_api) = self
                .frozen_apis
                .iter()
                .find(|api| api.name == current_api.name && api.module == current_api.module)
            {
                // Check parameter changes first (more specific)
                if frozen_api.parameters != current_api.parameters {
                    let has_new_required = current_api.parameters.iter().any(|p| {
                        !p.is_optional && !frozen_api.parameters.iter().any(|fp| fp.name == p.name)
                    });

                    if has_new_required {
                        violations.push(CompatibilityViolation {
                            apiname: format!(
                                "{module}::{name}",
                                module = current_api.module,
                                name = current_api.name
                            ),
                            violation_type: CompatibilityRuleType::NewRequiredParameter,
                            severity: CompatibilitySeverity::Breaking,
                            description: "New required parameter added".to_string(),
                            old_signature: Some(frozen_api.clone()),
                            new_signature: Some(current_api.clone()),
                            fix_suggestion: Some("Make new parameters optional".to_string()),
                        });
                        continue; // Skip signature hash check for this API
                    }
                }

                // Check visibility reduction
                if Self::is_visibility_reduced(&frozen_api.visibility, &current_api.visibility) {
                    violations.push(CompatibilityViolation {
                        apiname: format!(
                            "{module}::{name}",
                            module = current_api.module,
                            name = current_api.name
                        ),
                        violation_type: CompatibilityRuleType::VisibilityReduction,
                        severity: CompatibilitySeverity::Breaking,
                        description: "API visibility was reduced".to_string(),
                        old_signature: Some(frozen_api.clone()),
                        new_signature: Some(current_api.clone()),
                        fix_suggestion: Some("Restore original visibility".to_string()),
                    });
                }

                // Check return type changes
                if frozen_api.return_type != current_api.return_type {
                    violations.push(CompatibilityViolation {
                        apiname: format!(
                            "{module}::{name}",
                            module = current_api.module,
                            name = current_api.name
                        ),
                        violation_type: CompatibilityRuleType::ReturnTypeChange,
                        severity: CompatibilitySeverity::Breaking,
                        description: "Return type changed".to_string(),
                        old_signature: Some(frozen_api.clone()),
                        new_signature: Some(current_api.clone()),
                        fix_suggestion: Some(
                            "Restore original return type or create new API".to_string(),
                        ),
                    });
                }

                // Check for other parameter type changes
                if frozen_api.parameters.len() == current_api.parameters.len() {
                    for (old_param, new_param) in frozen_api
                        .parameters
                        .iter()
                        .zip(current_api.parameters.iter())
                    {
                        if old_param.name == new_param.name
                            && old_param.type_name != new_param.type_name
                        {
                            violations.push(CompatibilityViolation {
                                apiname: {
                                    let module = &current_api.module;
                                    let name = &current_api.name;
                                    format!("{module}::{name}")
                                },
                                violation_type: CompatibilityRuleType::ParameterTypeChange,
                                severity: CompatibilitySeverity::Breaking,
                                description: format!(
                                    "Parameter '{param_name}' type changed from '{old_type}' to '{new_type}'",
                                    param_name = old_param.name, old_type = old_param.type_name, new_type = new_param.type_name
                                ),
                                old_signature: Some(frozen_api.clone()),
                                new_signature: Some(current_api.clone()),
                                fix_suggestion: Some(
                                    "Restore original parameter type or create new API".to_string(),
                                ),
                            });
                        }
                    }
                }

                // Check signature hash changes (catch-all for other changes)
                if frozen_api.signature_hash != current_api.signature_hash
                    && !violations.iter().any(|v| {
                        v.apiname
                            == format!(
                                "{module}::{name}",
                                module = current_api.module,
                                name = current_api.name
                            )
                    })
                {
                    violations.push(CompatibilityViolation {
                        apiname: format!(
                            "{module}::{name}",
                            module = current_api.module,
                            name = current_api.name
                        ),
                        violation_type: CompatibilityRuleType::SignatureChange,
                        severity: CompatibilitySeverity::Breaking,
                        description: "API signature changed in an unspecified way".to_string(),
                        old_signature: Some(frozen_api.clone()),
                        new_signature: Some(current_api.clone()),
                        fix_suggestion: Some(
                            "Revert signature change or create new API".to_string(),
                        ),
                    });
                }
            }
        }

        // Generate suggestions for new APIs
        for current_api in currentapis {
            if !self
                .frozen_apis
                .iter()
                .any(|api| api.name == current_api.name && api.module == current_api.module)
            {
                suggestions.push(CompatibilitySuggestion {
                    apiname: format!(
                        "{module}::{name}",
                        module = current_api.module,
                        name = current_api.name
                    ),
                    suggestion: "New API detected - ensure proper documentation".to_string(),
                    rationale: "New APIs should be well-documented and tested".to_string(),
                });
            }
        }

        let is_compatible = violations
            .iter()
            .all(|v| v.severity != CompatibilitySeverity::Breaking);

        CompatibilityCheckResult {
            is_compatible,
            violations,
            warnings,
            suggestions,
        }
    }

    /// Check if visibility was reduced
    fn is_visibility_reduced(old: &Visibility, new: &Visibility) -> bool {
        use Visibility::*;
        matches!(
            (old, new),
            (Public, PublicCrate) | (Public, Private) | (PublicCrate, Private)
        )
    }

    /// Generate detailed compatibility report
    pub fn generate_compatibility_report(&self, result: &CompatibilityCheckResult) -> String {
        let mut report = String::from("# API Compatibility Report\n\n");

        report.push_str(&format!("**Freeze Status**: {:?}\n", self.freeze_status));
        report.push_str(&format!(
            "**Overall Compatible**: {}\n\n",
            result.is_compatible
        ));

        if !result.violations.is_empty() {
            report.push_str("## ❌ Compatibility Violations\n\n");
            for violation in &result.violations {
                let severity_icon = match violation.severity {
                    CompatibilitySeverity::Breaking => "💥",
                    CompatibilitySeverity::Warning => "⚠️",
                    CompatibilitySeverity::Info => "ℹ️",
                };

                report.push_str(&format!(
                    "{} **{}** ({:?}): {}\n",
                    severity_icon,
                    violation.apiname,
                    violation.violation_type,
                    violation.description
                ));

                if let Some(ref fix) = violation.fix_suggestion {
                    report.push_str(&format!("   - **Fix**: {fix}\n"));
                }
                report.push('\n');
            }
        }

        if !result.warnings.is_empty() {
            report.push_str("## ⚠️ Warnings\n\n");
            for warning in &result.warnings {
                report.push_str(&format!(
                    "- **{}**: {} (Impact: {})\n",
                    warning.apiname, warning.message, warning.impact
                ));
            }
            report.push('\n');
        }

        if !result.suggestions.is_empty() {
            report.push_str("## 💡 Suggestions\n\n");
            for suggestion in &result.suggestions {
                report.push_str(&format!(
                    "- **{}**: {} ({})\n",
                    suggestion.apiname, suggestion.suggestion, suggestion.rationale
                ));
            }
            report.push('\n');
        }

        report.push_str("## 📋 Next Steps\n\n");
        if result.is_compatible {
            report.push_str("✅ All compatibility checks passed. Safe to proceed.\n");
        } else {
            report
                .push_str("❌ Compatibility violations detected. Address issues before release:\n");
            report.push_str("1. Review breaking changes listed above\n");
            report.push_str("2. Apply suggested fixes or revert changes\n");
            report.push_str("3. Re-run compatibility check\n");
            report.push_str("4. Update documentation if needed\n");
        }

        report
    }
}

impl Default for ApiCompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Global version registry
static REGISTRY: OnceLock<Mutex<VersionRegistry>> = OnceLock::new();

/// Global API compatibility checker
static API_CHECKER: OnceLock<Mutex<ApiCompatibilityChecker>> = OnceLock::new();

/// Get the global registry for modification
#[allow(dead_code)]
pub fn global_registry_mut() -> std::sync::MutexGuard<'static, VersionRegistry> {
    REGISTRY
        .get_or_init(|| Mutex::new(VersionRegistry::new()))
        .lock()
        .unwrap()
}

/// Get the global registry for reading
#[allow(dead_code)]
pub fn global_registry() -> &'static Mutex<VersionRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(VersionRegistry::new()))
}

/// Get the global API checker for modification
#[allow(dead_code)]
pub fn global_api_checker_mut() -> std::sync::MutexGuard<'static, ApiCompatibilityChecker> {
    API_CHECKER
        .get_or_init(|| Mutex::new(ApiCompatibilityChecker::for_beta1()))
        .lock()
        .unwrap()
}

/// Get the global API checker for reading
#[allow(dead_code)]
pub fn global_api_checker() -> &'static Mutex<ApiCompatibilityChecker> {
    API_CHECKER.get_or_init(|| Mutex::new(ApiCompatibilityChecker::for_beta1()))
}

/// Helper function to create API signature from function metadata
#[allow(dead_code)]
pub fn create_api_signature(
    name: &str,
    module: &str,
    parameters: Vec<Parameter>,
    return_type: Option<String>,
    visibility: Visibility,
    stability: StabilityLevel,
) -> ApiSignature {
    let name_str = name.to_string();
    let module_str = module.to_string();

    // Create a simple hash based on function signature components
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};

    name_str.hash(&mut hasher);
    module_str.hash(&mut hasher);
    // Hash each parameter
    for param in &parameters {
        param.name.hash(&mut hasher);
        param.type_name.hash(&mut hasher);
        param.is_optional.hash(&mut hasher);
        if let Some(ref default) = param.defaultvalue {
            default.hash(&mut hasher);
        }
    }
    return_type.hash(&mut hasher);
    visibility.hash(&mut hasher);

    ApiSignature {
        name: name_str,
        module: module_str,
        signature_hash: hasher.finish(),
        parameters,
        return_type,
        visibility,
        stability,
    }
}

/// Initialize Beta 1 API freeze with core scirs2 APIs
#[allow(dead_code)]
pub fn initialize_beta1_freeze() -> Result<(), String> {
    let mut checker = global_api_checker_mut();

    // Use the same API definitions for consistency
    let core_apis = get_test_frozen_apis();

    checker.freeze_apis(core_apis)
}

/// Get the frozen API signatures for testing
#[allow(dead_code)]
fn get_test_frozen_apis() -> Vec<ApiSignature> {
    vec![
        create_api_signature(
            "simd_add",
            "simd_ops",
            vec![
                Parameter {
                    name: "a".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "b".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
            ],
            Some("Array1<Self>".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        ),
        create_api_signature(
            "simd_mul",
            "simd_ops",
            vec![
                Parameter {
                    name: "a".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "b".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
            ],
            Some("Array1<Self>".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        ),
        create_api_signature(
            "simd_dot",
            "simd_ops",
            vec![
                Parameter {
                    name: "a".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "b".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
            ],
            Some("Self".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        ),
        // Add new enhanced SIMD operations as experimental
        create_api_signature(
            "simd_add_cache_optimized",
            "simd_ops",
            vec![
                Parameter {
                    name: "a".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "b".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
            ],
            Some("Array1<Self>".to_string()),
            Visibility::Public,
            StabilityLevel::Experimental,
        ),
        create_api_signature(
            "simd_fma_advanced_optimized",
            "simd_ops",
            vec![
                Parameter {
                    name: "a".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "b".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "c".to_string(),
                    type_name: "&ArrayView1<Self>".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
            ],
            Some("Array1<Self>".to_string()),
            Visibility::Public,
            StabilityLevel::Experimental,
        ),
        create_api_signature(
            "Version",
            "apiversioning",
            vec![],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        ),
        create_api_signature(
            "ApiCompatibilityChecker",
            "apiversioning",
            vec![],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        ),
    ]
}

/// Convenience function to check current API compatibility
#[allow(dead_code)]
pub fn check_current_compatibility() -> Result<CompatibilityCheckResult, String> {
    let checker = global_api_checker();
    let checker_guard = checker.lock().map_err(|e| e.to_string())?;

    // For testing, we'll use the same APIs that were frozen
    // In a real implementation, this would be generated from actual code analysis
    let current_apis = get_test_frozen_apis();

    Ok((*checker_guard).check_compatibility(&current_apis))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        assert_eq!(Version::parse("1.2.3").unwrap(), Version::new(1, 2, 3));
        assert_eq!(
            Version::parse("0.1.0-beta.2").unwrap(),
            Version::new(0, 1, 0)
        );
        assert_eq!(
            Version::parse("10.20.30").unwrap(),
            Version::new(10, 20, 30)
        );

        assert!(Version::parse("1.2").is_err());
        assert!(Version::parse("a.b.c").is_err());
        assert!(Version::parse("").is_err());
    }

    #[test]
    fn test_version_compatibility() {
        let v1_0_0 = Version::new(1, 0, 0);
        let v1_1_0 = Version::new(1, 1, 0);
        let v1_0_1 = Version::new(1, 0, 1);
        let v2_0_0 = Version::new(2, 0, 0);

        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(v1_0_1.is_compatible_with(&v1_0_0));
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));
    }

    #[test]
    fn test_version_registry() {
        let mut registry = VersionRegistry::new();

        registry
            .register_api("Array", "core", Version::new(0, 1, 0))
            .register_api("Matrix", "linalg", Version::new(0, 1, 0))
            .register_api("OldArray", "core", Version::new(0, 1, 0));

        registry
            .deprecate_api("OldArray", Version::new(0, 2, 0), Some("Array".to_string()))
            .unwrap();

        let v0_1_0 = Version::new(0, 1, 0);
        let v0_2_0 = Version::new(0, 2, 0);

        // Check APIs in v0.1.0
        let apis_v1 = registry.apis_in_version(&v0_1_0);
        assert_eq!(apis_v1.len(), 3);

        // Check APIs in v0.2.0
        let apis_v2 = registry.apis_in_version(&v0_2_0);
        assert_eq!(apis_v2.len(), 2); // OldArray is deprecated

        // Check deprecated APIs
        let deprecated = registry.deprecated_apis_in_version(&v0_2_0);
        assert_eq!(deprecated.len(), 1);
        assert_eq!(deprecated[0].name, "OldArray");
    }

    #[test]
    fn test_migration_guide() {
        let mut registry = VersionRegistry::new();

        registry
            .register_api("Feature1", "module1", Version::new(0, 1, 0))
            .register_api("Feature2", "module2", Version::new(0, 2, 0))
            .register_api("OldFeature", "module1", Version::new(0, 1, 0));

        registry
            .deprecate_api(
                "OldFeature",
                Version::new(0, 2, 0),
                Some("Feature2".to_string()),
            )
            .unwrap();

        let guide = registry.migration_guide(&Version::new(0, 1, 0), &Version::new(0, 2, 0));

        assert!(guide.contains("Removed APIs"));
        assert!(guide.contains("OldFeature"));
        assert!(guide.contains("Replacement"));
        assert!(guide.contains("Feature2"));
        assert!(guide.contains("New APIs"));
        assert!(guide.contains("Migration Checklist"));
    }

    #[test]
    fn test_api_compatibility_checker() {
        let mut checker = ApiCompatibilityChecker::new();

        // Create initial API set
        let initial_apis = vec![create_api_signature(
            "test_func",
            "test_module",
            vec![Parameter {
                name: "param1".to_string(),
                type_name: "i32".to_string(),
                is_optional: false,
                defaultvalue: None,
            }],
            Some("String".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        )];

        // Freeze the APIs
        checker.freeze_apis(initial_apis.clone()).unwrap();
        assert_eq!(checker.freeze_status, ApiFreezeStatus::Frozen);

        // Test compatibility with same APIs
        let result = checker.check_compatibility(&initial_apis);
        assert!(result.is_compatible);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_api_signature_changes() {
        let mut checker = ApiCompatibilityChecker::for_beta1();

        let original_api = create_api_signature(
            "test_func",
            "test_module",
            vec![Parameter {
                name: "param1".to_string(),
                type_name: "i32".to_string(),
                is_optional: false,
                defaultvalue: None,
            }],
            Some("String".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        );

        checker.freeze_apis(vec![original_api.clone()]).unwrap();

        // Test with modified signature (different parameter type)
        let modified_api = create_api_signature(
            "test_func",
            "test_module",
            vec![Parameter {
                name: "param1".to_string(),
                type_name: "f64".to_string(), // Changed type
                is_optional: false,
                defaultvalue: None,
            }],
            Some("String".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        );

        let result = checker.check_compatibility(&[modified_api]);
        assert!(!result.is_compatible);
        assert!(!result.violations.is_empty());
        assert_eq!(
            result.violations[0].violation_type,
            CompatibilityRuleType::ParameterTypeChange
        );
    }

    #[test]
    fn test_api_removal_detection() {
        let mut checker = ApiCompatibilityChecker::for_beta1();

        let api1 = create_api_signature(
            "func1",
            "module",
            vec![],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        );

        let api2 = create_api_signature(
            "func2",
            "module",
            vec![],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        );

        checker
            .freeze_apis(vec![api1.clone(), api2.clone()])
            .unwrap();

        // Test with one API removed
        let result = checker.check_compatibility(&[api1]); // api2 is missing
        assert!(!result.is_compatible);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(
            result.violations[0].violation_type,
            CompatibilityRuleType::PublicApiRemoval
        );
    }

    #[test]
    fn test_visibility_reduction() {
        let mut checker = ApiCompatibilityChecker::for_beta1();

        let public_api = create_api_signature(
            "func",
            "module",
            vec![],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        );

        checker.freeze_apis(vec![public_api]).unwrap();

        // Test with reduced visibility
        let private_api = create_api_signature(
            "func",
            "module",
            vec![],
            None,
            Visibility::Private, // Reduced from Public
            StabilityLevel::Stable,
        );

        let result = checker.check_compatibility(&[private_api]);
        assert!(!result.is_compatible);
        assert_eq!(
            result.violations[0].violation_type,
            CompatibilityRuleType::VisibilityReduction
        );
    }

    #[test]
    fn test_new_required_parameter() {
        let mut checker = ApiCompatibilityChecker::for_beta1();

        let original_api = create_api_signature(
            "func",
            "module",
            vec![Parameter {
                name: "param1".to_string(),
                type_name: "i32".to_string(),
                is_optional: false,
                defaultvalue: None,
            }],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        );

        checker.freeze_apis(vec![original_api]).unwrap();

        // Test with new required parameter
        let modified_api = create_api_signature(
            "func",
            "module",
            vec![
                Parameter {
                    name: "param1".to_string(),
                    type_name: "i32".to_string(),
                    is_optional: false,
                    defaultvalue: None,
                },
                Parameter {
                    name: "param2".to_string(),
                    type_name: "String".to_string(),
                    is_optional: false, // New required parameter
                    defaultvalue: None,
                },
            ],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        );

        let result = checker.check_compatibility(&[modified_api]);
        assert!(!result.is_compatible);
        assert_eq!(
            result.violations[0].violation_type,
            CompatibilityRuleType::NewRequiredParameter
        );
    }

    #[test]
    fn test_compatibility_report_generation() {
        let mut checker = ApiCompatibilityChecker::for_beta1();

        let original_api = create_api_signature(
            "test_func",
            "test_module",
            vec![],
            None,
            Visibility::Public,
            StabilityLevel::Stable,
        );

        checker.freeze_apis(vec![original_api]).unwrap();

        // Test with removed API
        let result = checker.check_compatibility(&[]); // All APIs removed
        let report = checker.generate_compatibility_report(&result);

        assert!(report.contains("API Compatibility Report"));
        assert!(report.contains("Freeze Status"));
        assert!(report.contains("Compatibility Violations"));
        assert!(report.contains("Next Steps"));
        assert!(report.contains("💥")); // Breaking change icon
    }

    #[test]
    fn test_beta1_initialization() {
        // Test that we can initialize the Beta 1 freeze
        let result = initialize_beta1_freeze();
        assert!(result.is_ok());

        // Test compatibility check
        let compat_result = check_current_compatibility();
        assert!(compat_result.is_ok());

        let result = compat_result.unwrap();
        // Should be compatible since we're checking against the same APIs
        assert!(result.is_compatible);
    }

    #[test]
    fn test_api_signature_creation() {
        let signature = create_api_signature(
            "test_func",
            "test_module",
            vec![Parameter {
                name: "param".to_string(),
                type_name: "i32".to_string(),
                is_optional: false,
                defaultvalue: None,
            }],
            Some("String".to_string()),
            Visibility::Public,
            StabilityLevel::Stable,
        );

        assert_eq!(signature.name, "test_func");
        assert_eq!(signature.module, "test_module");
        assert_eq!(signature.parameters.len(), 1);
        assert_eq!(signature.return_type, Some("String".to_string()));
        assert_eq!(signature.visibility, Visibility::Public);
        assert_eq!(signature.stability, StabilityLevel::Stable);
        assert!(signature.signature_hash != 0); // Should have a hash
    }
}
