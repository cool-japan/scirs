//! # Deprecation Management
//!
//! Comprehensive deprecation management system for API lifecycle
//! management and transition planning in enterprise environments.

use super::Version;
use crate::error::CoreError;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Deprecation policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationPolicy {
    /// Default deprecation period in days
    pub default_deprecation_period: u32,
    /// Grace period after end of life
    pub grace_period: u32,
    /// Deprecation notice period before end of life
    pub notice_period: u32,
    /// Automatic deprecation rules
    pub auto_deprecation_rules: Vec<AutoDeprecationRule>,
    /// Support level during deprecation
    pub deprecation_support_level: SupportLevel,
    /// Migration assistance provided
    pub migration_assistance: bool,
}

impl Default for DeprecationPolicy {
    fn default() -> Self {
        Self {
            default_deprecation_period: 365, // 1 year
            grace_period: 90,                // 3 months
            notice_period: 180,              // 6 months
            auto_deprecation_rules: vec![
                AutoDeprecationRule::MajorVersionSuperseded {
                    versions_to_keep: 2,
                },
                AutoDeprecationRule::AgeBasedDeprecation { maxage_days: 1095 }, // 3 years
            ],
            deprecation_support_level: SupportLevel::SecurityOnly,
            migration_assistance: true,
        }
    }
}

/// Support levels during deprecation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportLevel {
    /// Full support continues
    Full,
    /// Maintenance only (bug fixes)
    MaintenanceOnly,
    /// Security updates only
    SecurityOnly,
    /// No support
    None,
}

/// Automatic deprecation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoDeprecationRule {
    /// Deprecate when superseded by newer major versions
    MajorVersionSuperseded { versions_to_keep: u32 },
    /// Deprecate based on age
    AgeBasedDeprecation { maxage_days: u32 },
    /// Deprecate when usage falls below threshold
    UsageBasedDeprecation { min_usage_percent: f64 },
    /// Deprecate unstable versions after stable release
    StableVersionReleased,
}

/// Deprecation status for a version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationStatus {
    /// Version being deprecated
    pub version: Version,
    /// Current deprecation phase
    pub phase: DeprecationPhase,
    /// Deprecation announcement date
    pub announced_date: chrono::DateTime<chrono::Utc>,
    /// Planned end of life date
    pub end_of_life_date: chrono::DateTime<chrono::Utc>,
    /// Actual end of life date (if reached)
    pub actual_end_of_life: Option<chrono::DateTime<chrono::Utc>>,
    /// Reason for deprecation
    pub reason: DeprecationReason,
    /// Recommended replacement version
    pub replacement_version: Option<Version>,
    /// Migration guide URL or content
    pub migration_guide: Option<String>,
    /// Support level during deprecation
    pub support_level: SupportLevel,
    /// Usage metrics at time of deprecation
    pub usage_metrics: Option<UsageMetrics>,
}

/// Phases of deprecation lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DeprecationPhase {
    /// Actively supported
    Active,
    /// Deprecation announced but still supported
    Announced,
    /// In deprecation period
    Deprecated,
    /// End of life reached, no support
    EndOfLife,
    /// Completely removed
    Removed,
}

impl DeprecationPhase {
    /// Get the string representation
    pub const fn as_str(&self) -> &'static str {
        match self {
            DeprecationPhase::Active => "active",
            DeprecationPhase::Announced => "announced",
            DeprecationPhase::Deprecated => "deprecated",
            DeprecationPhase::EndOfLife => "end_of_life",
            DeprecationPhase::Removed => "removed",
        }
    }

    /// Check if version is still supported
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            DeprecationPhase::Active | DeprecationPhase::Announced | DeprecationPhase::Deprecated
        )
    }
}

/// Reasons for deprecation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeprecationReason {
    /// Superseded by newer version
    SupersededBy(Version),
    /// Security vulnerabilities
    SecurityConcerns,
    /// Performance issues
    PerformanceIssues,
    /// Maintenance burden
    MaintenanceBurden,
    /// Low usage
    LowUsage,
    /// Technology obsolescence
    TechnologyObsolescence,
    /// Business decision
    BusinessDecision(String),
    /// End of vendor support
    VendorEndOfSupport,
}

/// Usage metrics for deprecation decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetrics {
    /// Number of active users/clients
    pub active_users: u64,
    /// Percentage of total API usage
    pub usage_percentage: f64,
    /// Download/installation count
    pub download_count: u64,
    /// Last recorded usage
    pub last_usage: chrono::DateTime<chrono::Utc>,
    /// Usage trend (increasing/decreasing)
    pub trend: UsageTrend,
}

/// Usage trend indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageTrend {
    /// Usage is increasing
    Increasing,
    /// Usage is stable
    Stable,
    /// Usage is decreasing
    Decreasing,
    /// Usage is rapidly declining
    Declining,
}

/// Deprecation announcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationAnnouncement {
    /// Version being deprecated
    pub version: Version,
    /// Announcement date
    pub announcement_date: chrono::DateTime<chrono::Utc>,
    /// Deprecation timeline
    pub timeline: DeprecationTimeline,
    /// Announcement message
    pub message: String,
    /// Migration instructions
    pub migration_instructions: Option<String>,
    /// Contact information for support
    pub support_contact: Option<String>,
    /// Communication channels used
    pub communication_channels: Vec<CommunicationChannel>,
}

/// Deprecation timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationTimeline {
    /// When deprecation was announced
    pub announced: chrono::DateTime<chrono::Utc>,
    /// When version enters deprecated phase
    pub deprecated_date: chrono::DateTime<chrono::Utc>,
    /// When version reaches end of life
    pub end_of_life: chrono::DateTime<chrono::Utc>,
    /// When version will be removed
    pub removal_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Milestone dates
    pub milestones: Vec<DeprecationMilestone>,
}

/// Deprecation milestone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationMilestone {
    /// Milestone date
    pub date: chrono::DateTime<chrono::Utc>,
    /// Milestone description
    pub description: String,
    /// Actions to be taken
    pub actions: Vec<String>,
}

/// Communication channels for deprecation announcements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationChannel {
    /// Email notification
    Email,
    /// Website announcement
    Website,
    /// API response headers
    ApiHeaders,
    /// Documentation update
    Documentation,
    /// Blog post
    BlogPost,
    /// Developer newsletter
    Newsletter,
    /// Social media
    SocialMedia,
    /// Direct notification
    DirectNotification,
}

/// Helper function to create a deprecation timeline
fn deprecation_timeline(
    version: &Version,
    _replacement_version: Option<&Version>,
) -> DeprecationTimeline {
    let now = chrono::Utc::now();
    let deprecated_date = now + chrono::Duration::days(30);
    let end_of_life = now + chrono::Duration::days(180);
    let removal_date = now + chrono::Duration::days(365);

    let milestones = vec![
        DeprecationMilestone {
            date: deprecated_date,
            description: format!("Version {version} will be deprecated"),
            actions: vec!["Update to newer version".to_string()],
        },
        DeprecationMilestone {
            date: end_of_life,
            description: format!("Version {version} reaches end of life"),
            actions: vec!["Support will be discontinued".to_string()],
        },
    ];

    DeprecationTimeline {
        announced: now,
        deprecated_date,
        end_of_life,
        removal_date: Some(removal_date),
        milestones,
    }
}

/// Deprecation manager implementation
pub struct DeprecationManager {
    /// Deprecation policy
    policy: DeprecationPolicy,
    /// Deprecation statuses by version
    deprecations: HashMap<Version, DeprecationStatus>,
    /// Deprecation announcements
    announcements: Vec<DeprecationAnnouncement>,
}

impl DeprecationManager {
    /// Create a new deprecation manager
    pub fn new() -> Self {
        Self {
            policy: DeprecationPolicy::default(),
            deprecations: HashMap::new(),
            announcements: Vec::new(),
        }
    }

    /// Create with custom policy
    pub fn with_policy(policy: DeprecationPolicy) -> Self {
        Self {
            policy,
            deprecations: HashMap::new(),
            announcements: Vec::new(),
        }
    }

    /// Register a version for deprecation management
    pub fn register_version(&mut self, apiversion: &super::ApiVersion) -> Result<(), CoreError> {
        // Create active status for new version
        let status = DeprecationStatus {
            version: apiversion.version.clone(),
            phase: DeprecationPhase::Active,
            announced_date: apiversion.release_date,
            end_of_life_date: apiversion.end_of_life.unwrap_or_else(|| {
                apiversion.release_date
                    + chrono::Duration::days(self.policy.default_deprecation_period as i64)
            }),
            actual_end_of_life: None,
            reason: DeprecationReason::BusinessDecision("Not deprecated".to_string()),
            replacement_version: None,
            migration_guide: None,
            support_level: SupportLevel::Full,
            usage_metrics: None,
        };

        self.deprecations.insert(apiversion.version.clone(), status);
        Ok(())
    }

    /// Announce deprecation of a version
    pub fn announce_deprecation(
        &mut self,
        version: &Version,
        reason: DeprecationReason,
        replacement_version: Option<Version>,
    ) -> Result<DeprecationAnnouncement, CoreError> {
        let status = self.deprecations.get_mut(version).ok_or_else(|| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Version {version} not registered"
            )))
        })?;

        let now = chrono::Utc::now();
        let deprecated_date = now + chrono::Duration::days(self.policy.notice_period as i64);
        let end_of_life =
            deprecated_date + chrono::Duration::days(self.policy.default_deprecation_period as i64);

        // Update status
        status.phase = DeprecationPhase::Announced;
        status.announced_date = now;
        status.end_of_life_date = end_of_life;
        status.reason = reason.clone();
        status.replacement_version = replacement_version.clone();
        status.support_level = self.policy.deprecation_support_level;

        // Create timeline
        let timeline = DeprecationTimeline {
            announced: now,
            deprecated_date,
            end_of_life,
            removal_date: Some(
                end_of_life + chrono::Duration::days(self.policy.grace_period as i64),
            ),
            milestones: vec![
                DeprecationMilestone {
                    date: deprecated_date,
                    description: "Version enters deprecated phase".to_string(),
                    actions: vec!["Migration recommended".to_string()],
                },
                DeprecationMilestone {
                    date: end_of_life - chrono::Duration::days(30),
                    description: "Final warning - 30 days to end of life".to_string(),
                    actions: vec!["Complete migration immediately".to_string()],
                },
            ],
        };

        // Create announcement
        let announcement = DeprecationAnnouncement {
            version: version.clone(),
            announcement_date: now,
            timeline,
            message: self.generate_deprecation_message(
                version,
                &reason,
                replacement_version.as_ref(),
            ),
            migration_instructions: self
                .generate_migration_instructions(version, replacement_version.as_ref()),
            support_contact: Some("support@scirs.dev".to_string()),
            communication_channels: vec![
                CommunicationChannel::Email,
                CommunicationChannel::Website,
                CommunicationChannel::ApiHeaders,
                CommunicationChannel::Documentation,
            ],
        };

        self.announcements.push(announcement.clone());
        Ok(announcement)
    }

    /// Get deprecation status for a version
    pub fn get_deprecation_status(&self, version: &Version) -> Option<DeprecationStatus> {
        self.deprecations.get(version).cloned()
    }

    /// Update deprecation status
    pub fn update_status(
        &mut self,
        version: &Version,
        new_status: DeprecationStatus,
    ) -> Result<(), CoreError> {
        if !self.deprecations.contains_key(version) {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(format!("Version {version} not registered")),
            ));
        }

        self.deprecations.insert(version.clone(), new_status);
        Ok(())
    }

    /// Perform maintenance tasks
    pub fn perform_maintenance(&mut self) -> Result<Vec<MaintenanceAction>, CoreError> {
        let mut actions = Vec::new();
        let now = chrono::Utc::now();

        // Check for automatic deprecations
        for rule in &self.policy.auto_deprecation_rules.clone() {
            actions.extend(self.apply_auto_deprecation_rule(rule, now)?);
        }

        // Update phases based on timeline
        for (version, status) in &mut self.deprecations {
            let old_phase = status.phase;

            if status.phase == DeprecationPhase::Announced
                && now
                    >= status.end_of_life_date
                        - chrono::Duration::days(self.policy.default_deprecation_period as i64)
            {
                status.phase = DeprecationPhase::Deprecated;
                actions.push(MaintenanceAction::PhaseTransition {
                    version: version.clone(),
                    from_phase: old_phase,
                    to_phase: status.phase,
                });
            }

            if status.phase == DeprecationPhase::Deprecated && now >= status.end_of_life_date {
                status.phase = DeprecationPhase::EndOfLife;
                status.actual_end_of_life = Some(now);
                actions.push(MaintenanceAction::PhaseTransition {
                    version: version.clone(),
                    from_phase: old_phase,
                    to_phase: status.phase,
                });
            }
        }

        Ok(actions)
    }

    /// Apply automatic deprecation rule
    fn apply_auto_deprecation_rule(
        &mut self,
        rule: &AutoDeprecationRule,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MaintenanceAction>, CoreError> {
        let mut actions = Vec::new();

        match rule {
            AutoDeprecationRule::MajorVersionSuperseded { versions_to_keep } => {
                actions.extend(self.apply_major_version_rule(*versions_to_keep)?);
            }
            AutoDeprecationRule::AgeBasedDeprecation { maxage_days } => {
                actions.extend(self.apply_agebased_rule(*maxage_days, now)?);
            }
            AutoDeprecationRule::UsageBasedDeprecation { min_usage_percent } => {
                actions.extend(self.apply_usagebased_rule(*min_usage_percent)?);
            }
            AutoDeprecationRule::StableVersionReleased => {
                actions.extend(self.apply_stable_release_rule()?);
            }
        }

        Ok(actions)
    }

    /// Apply major version superseded rule
    fn apply_major_version_rule(
        &self,
        versions_to_keep: u32,
    ) -> Result<Vec<MaintenanceAction>, CoreError> {
        let mut actions = Vec::new();

        // Group versions by major version and collect them to avoid borrowing issues
        let mut majorversions: std::collections::BTreeMap<u64, Vec<Version>> =
            std::collections::BTreeMap::new();
        for version in self.deprecations.keys() {
            majorversions
                .entry(version.major())
                .or_default()
                .push(version.clone());
        }

        // Find majors to deprecate
        let major_keys: Vec<u64> = majorversions.keys().cloned().collect();
        if major_keys.len() > versions_to_keep as usize {
            let to_deprecate = &major_keys[..major_keys.len() - versions_to_keep as usize];

            for &major in to_deprecate {
                if let Some(versions) = majorversions.get(&major) {
                    for version in versions {
                        if let Some(status) = self.deprecations.get(version) {
                            if status.phase == DeprecationPhase::Active {
                                let latest_major = major_keys.last().unwrap();
                                let replacement = Version::new(*latest_major, 0, 0);

                                let _announcement = DeprecationAnnouncement {
                                    version: version.clone(),
                                    announcement_date: chrono::Utc::now(),
                                    timeline: deprecation_timeline(version, Some(&replacement)),
                                    message: format!(
                                        "Version {version} is deprecated in favor of {replacement}"
                                    ),
                                    migration_instructions: Some(format!(
                                        "Please migrate to version {replacement}"
                                    )),
                                    support_contact: None,
                                    communication_channels: vec![],
                                };

                                actions.push(MaintenanceAction::AutoDeprecation {
                                    version: version.clone(),
                                    rule: format!(
                                        "Major version superseded (keep {versions_to_keep})"
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(actions)
    }

    /// Apply age-based deprecation rule
    fn apply_agebased_rule(
        &self,
        maxage_days: u32,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<MaintenanceAction>, CoreError> {
        let mut actions = Vec::new();
        let maxage = chrono::Duration::days(maxage_days as i64);

        for (version, status) in &self.deprecations.clone() {
            if status.phase == DeprecationPhase::Active {
                let age = now.signed_duration_since(status.announced_date);
                if age > maxage {
                    let _announcement = DeprecationAnnouncement {
                        version: version.clone(),
                        timeline: deprecation_timeline(version, None),
                        announcement_date: now,
                        message: format!("Version {version} deprecated due to maintenance burden"),
                        migration_instructions: Some("Please upgrade to newer version".to_string()),
                        support_contact: Some("support@scirs.org".to_string()),
                        communication_channels: vec![
                            CommunicationChannel::Documentation,
                            CommunicationChannel::Email,
                        ],
                    };

                    actions.push(MaintenanceAction::AutoDeprecation {
                        version: version.clone(),
                        rule: format!("Age-based deprecation (max {maxage_days} days)"),
                    });
                }
            }
        }

        Ok(actions)
    }

    /// Apply usage-based deprecation rule
    fn apply_usagebased_rule(
        &self,
        _min_usage_percent: f64,
    ) -> Result<Vec<MaintenanceAction>, CoreError> {
        // This would require actual usage metrics
        // For now, return empty actions
        Ok(Vec::new())
    }

    /// Apply stable version released rule
    fn apply_stable_release_rule(&self) -> Result<Vec<MaintenanceAction>, CoreError> {
        // This would deprecate pre-release versions when stable is available
        // For now, return empty actions
        Ok(Vec::new())
    }

    /// Generate deprecation message
    fn generate_deprecation_message(
        &self,
        version: &Version,
        reason: &DeprecationReason,
        replacement: Option<&Version>,
    ) -> String {
        let reason_str = match reason {
            DeprecationReason::SupersededBy(v) => format!("{v}"),
            DeprecationReason::SecurityConcerns => "security concerns".to_string(),
            DeprecationReason::PerformanceIssues => "performance issues".to_string(),
            DeprecationReason::MaintenanceBurden => "maintenance burden".to_string(),
            DeprecationReason::LowUsage => "low usage".to_string(),
            DeprecationReason::TechnologyObsolescence => "technology obsolescence".to_string(),
            DeprecationReason::BusinessDecision(msg) => msg.clone(),
            DeprecationReason::VendorEndOfSupport => "vendor end of support".to_string(),
        };

        let mut message = format!("Version {version} has been deprecated due to {reason_str}. ");

        if let Some(replacement) = replacement {
            message.push_str(&format!(
                "Please migrate to version {replacement} as soon as possible. "
            ));
        }

        message.push_str(&format!(
            "Support will end on {}. ",
            self.deprecations
                .get(version)
                .map(|s| s.end_of_life_date.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "TBD".to_string())
        ));

        message
    }

    /// Generate migration instructions
    fn generate_migration_instructions(
        &self,
        _current_version: &Version,
        replacement: Option<&Version>,
    ) -> Option<String> {
        replacement.map(|replacement| {
            format!(
                "To migrate to version {replacement}:\n\
                1. Update your dependency to version {replacement}\n\
                2. Review the changelog for breaking changes\n\
                3. Update your code as necessary\n\
                4. Test thoroughly before deploying\n\
                5. Contact support if you need assistance"
            )
        })
    }

    /// Get all deprecated versions
    pub fn get_deprecatedversions(&self) -> Vec<&DeprecationStatus> {
        self.deprecations
            .values()
            .filter(|status| status.phase != DeprecationPhase::Active)
            .collect()
    }

    /// Get versions in specific phase
    pub fn getversions_in_phase(&self, phase: DeprecationPhase) -> Vec<&DeprecationStatus> {
        self.deprecations
            .values()
            .filter(|status| status.phase == phase)
            .collect()
    }
}

impl Default for DeprecationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Maintenance actions performed by the deprecation manager
#[derive(Debug, Clone)]
pub enum MaintenanceAction {
    /// Phase transition
    PhaseTransition {
        version: Version,
        from_phase: DeprecationPhase,
        to_phase: DeprecationPhase,
    },
    /// Automatic deprecation applied
    AutoDeprecation { version: Version, rule: String },
    /// Usage metrics updated
    UsageMetricsUpdated { version: Version },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deprecation_manager_creation() {
        let manager = DeprecationManager::new();
        assert!(manager.deprecations.is_empty());
        assert!(manager.announcements.is_empty());
    }

    #[test]
    fn test_deprecation_phases() {
        assert!(DeprecationPhase::Active < DeprecationPhase::Deprecated);
        assert!(DeprecationPhase::Deprecated < DeprecationPhase::EndOfLife);
        assert!(DeprecationPhase::Active.is_supported());
        assert!(!DeprecationPhase::EndOfLife.is_supported());
    }

    #[test]
    fn test_deprecation_announcement() {
        let mut manager = DeprecationManager::new();
        let version = Version::new(1, 0, 0);

        // Register version first
        let apiversion = super::super::ApiVersionBuilder::new(version.clone())
            .build()
            .unwrap();
        manager.register_version(&apiversion).unwrap();

        // Announce deprecation
        let replacement = Version::new(2, 0, 0);
        let announcement = manager
            .announce_deprecation(
                &version,
                DeprecationReason::SupersededBy(replacement.clone()),
                Some(replacement),
            )
            .unwrap();

        assert_eq!(announcement.version, version);
        assert!(!announcement.message.is_empty());
        assert!(announcement.migration_instructions.is_some());

        // Check status was updated
        let status = manager.get_deprecation_status(&version).unwrap();
        assert_eq!(status.phase, DeprecationPhase::Announced);
    }

    #[test]
    fn test_deprecation_policy() {
        let policy = DeprecationPolicy::default();
        assert_eq!(policy.default_deprecation_period, 365);
        assert_eq!(policy.grace_period, 90);
        assert_eq!(policy.notice_period, 180);
        assert!(policy.migration_assistance);
    }

    #[test]
    fn test_auto_deprecation_rules() {
        let mut manager = DeprecationManager::new();

        // Register multiple major versions
        for major in 1..=5 {
            let version = Version::new(major, 0, 0);
            let apiversion = super::super::ApiVersionBuilder::new(version)
                .build()
                .unwrap();
            manager.register_version(&apiversion).unwrap();
        }

        // Apply major version rule
        let rule = AutoDeprecationRule::MajorVersionSuperseded {
            versions_to_keep: 2,
        };
        let actions = manager
            .apply_auto_deprecation_rule(&rule, chrono::Utc::now())
            .unwrap();

        // Should have deprecated older versions
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_usage_trends() {
        let metrics = UsageMetrics {
            active_users: 100,
            usage_percentage: 5.0,
            download_count: 1000,
            last_usage: chrono::Utc::now(),
            trend: UsageTrend::Decreasing,
        };

        assert_eq!(metrics.trend, UsageTrend::Decreasing);
        assert_eq!(metrics.usage_percentage, 5.0);
    }
}
