//! # Version Negotiation
//!
//! Version negotiation system for client-server compatibility
//! and automatic version selection in distributed systems.

use super::Version;
use crate::error::CoreError;
use std::collections::BTreeSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Client capabilities for version negotiation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClientCapabilities {
    /// Preferred version
    pub preferred_version: Option<Version>,
    /// Supported version range
    pub supported_versions: Vec<Version>,
    /// Required features
    pub required_features: BTreeSet<String>,
    /// Optional features
    pub optional_features: BTreeSet<String>,
    /// Client type identifier
    pub client_type: String,
    /// Client version
    pub client_version: Version,
}

impl ClientCapabilities {
    /// Create new client capabilities
    pub fn new(client_type: String, client_version: Version) -> Self {
        Self {
            preferred_version: None,
            supported_versions: Vec::new(),
            required_features: BTreeSet::new(),
            optional_features: BTreeSet::new(),
            client_type,
            client_version,
        }
    }

    /// Set preferred version
    pub fn preferred_version(mut self, version: Version) -> Self {
        self.preferred_version = Some(version);
        self
    }

    /// Add supported version
    pub fn add_supported_version(mut self, version: Version) -> Self {
        self.supported_versions.push(version);
        self
    }

    /// Add required feature
    pub fn require_feature(mut self, feature: &str) -> Self {
        self.required_features.insert(feature.to_string());
        self
    }

    /// Add optional feature
    pub fn prefer_feature(mut self, feature: &str) -> Self {
        self.optional_features.insert(feature.to_string());
        self
    }
}

/// Result of version negotiation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NegotiationResult {
    /// Negotiated version
    pub negotiated_version: Version,
    /// Available features in negotiated version
    pub available_features: BTreeSet<String>,
    /// Unsupported required features
    pub unsupported_features: BTreeSet<String>,
    /// Negotiation status
    pub status: NegotiationStatus,
    /// Negotiation metadata
    pub metadata: NegotiationMetadata,
}

/// Status of version negotiation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NegotiationStatus {
    /// Negotiation successful
    Success,
    /// Negotiation successful with warnings
    SuccessWithWarnings,
    /// Partial success - some features unavailable
    PartialSuccess,
    /// Failed - no compatible version found
    Failed,
    /// Failed - required features not available
    FeaturesMissing,
}

/// Metadata about the negotiation process
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NegotiationMetadata {
    /// Versions considered during negotiation
    pub considered_versions: Vec<Version>,
    /// Reason for final selection
    pub selection_reason: String,
    /// Warnings generated during negotiation
    pub warnings: Vec<String>,
    /// Negotiation algorithm used
    pub algorithm: String,
}

/// Version negotiation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegotiationStrategy {
    /// Prefer the highest compatible version
    PreferLatest,
    /// Prefer the client's preferred version if compatible
    PreferClientPreference,
    /// Prefer the most stable version
    PreferStable,
    /// Prefer the version with most features
    PreferFeatureRich,
    /// Custom negotiation logic
    Custom,
}

/// Version negotiator implementation
pub struct VersionNegotiator {
    /// Default negotiation strategy
    strategy: NegotiationStrategy,
    /// Feature compatibility matrix
    feature_matrix: FeatureMatrix,
}

impl VersionNegotiator {
    /// Create a new version negotiator
    pub fn new() -> Self {
        Self {
            strategy: NegotiationStrategy::PreferLatest,
            feature_matrix: FeatureMatrix::new(),
        }
    }

    /// Set negotiation strategy
    pub fn with_strategy(mut self, strategy: NegotiationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Negotiate version with client
    pub fn negotiate(
        &self,
        client_capabilities: &ClientCapabilities,
        server_versions: &[&Version],
    ) -> Result<NegotiationResult, CoreError> {
        let mut metadata = NegotiationMetadata {
            considered_versions: server_versions.iter().map(|v| (*v).clone()).collect(),
            selection_reason: String::new(),
            warnings: Vec::new(),
            algorithm: format!("{:?}", self.strategy),
        };

        // Find compatible versions
        let compatible_versions =
            self.find_compatible_versions(client_capabilities, server_versions, &mut metadata)?;

        if compatible_versions.is_empty() {
            return Ok(NegotiationResult {
                negotiated_version: Version::new(0, 0, 0),
                available_features: BTreeSet::new(),
                unsupported_features: client_capabilities.required_features.clone(),
                status: NegotiationStatus::Failed,
                metadata,
            });
        }

        // Select best version based on strategy
        let selected_version =
            self.select_version(&compatible_versions, client_capabilities, &mut metadata)?;

        // Check feature compatibility
        let (available_features, unsupported_features, status) = self.check_feature_compatibility(
            &selected_version,
            client_capabilities,
            &mut metadata,
        )?;

        metadata.selection_reason = format!(
            "Selected {} using {:?} strategy",
            selected_version, self.strategy
        );

        Ok(NegotiationResult {
            negotiated_version: selected_version,
            available_features,
            unsupported_features,
            status,
            metadata,
        })
    }

    /// Find versions compatible with client
    fn find_compatible_versions(
        &self,
        client_capabilities: &ClientCapabilities,
        server_versions: &[&Version],
        metadata: &mut NegotiationMetadata,
    ) -> Result<Vec<Version>, CoreError> {
        let mut compatible = Vec::new();

        for server_version in server_versions {
            // Check if client supports this version
            if client_capabilities.supported_versions.is_empty()
                || client_capabilities
                    .supported_versions
                    .contains(server_version)
            {
                compatible.push((*server_version).clone());
            } else {
                metadata.warnings.push(format!(
                    "Version {} not in client's supported list",
                    server_version
                ));
            }
        }

        Ok(compatible)
    }

    /// Select the best version based on strategy
    fn select_version(
        &self,
        compatible_versions: &[Version],
        client_capabilities: &ClientCapabilities,
        metadata: &mut NegotiationMetadata,
    ) -> Result<Version, CoreError> {
        match self.strategy {
            NegotiationStrategy::PreferLatest => {
                let mut versions = compatible_versions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
            NegotiationStrategy::PreferClientPreference => {
                if let Some(ref preferred) = client_capabilities.preferred_version {
                    if compatible_versions.contains(preferred) {
                        return Ok(preferred.clone());
                    }
                    metadata.warnings.push(
                        "Client preferred version not available, falling back to latest"
                            .to_string(),
                    );
                }
                // Fall back to latest
                let mut versions = compatible_versions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
            NegotiationStrategy::PreferStable => {
                // Prefer non-pre-release versions
                let stable_versions: Vec<_> = compatible_versions
                    .iter()
                    .filter(|v| v.is_stable())
                    .cloned()
                    .collect();

                if !stable_versions.is_empty() {
                    let mut versions = stable_versions;
                    versions.sort();
                    versions.reverse();
                    Ok(versions.into_iter().next().unwrap())
                } else {
                    // No stable versions, pick latest
                    let mut versions = compatible_versions.to_vec();
                    versions.sort();
                    versions.reverse();
                    Ok(versions.into_iter().next().unwrap())
                }
            }
            NegotiationStrategy::PreferFeatureRich => {
                // This would require feature information for each version
                // For now, fall back to latest
                let mut versions = compatible_versions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
            NegotiationStrategy::Custom => {
                // Custom logic would be implemented here
                let mut versions = compatible_versions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
        }
    }

    /// Check feature compatibility
    fn check_feature_compatibility(
        &self,
        selected_version: &Version,
        client_capabilities: &ClientCapabilities,
        _metadata: &mut NegotiationMetadata,
    ) -> Result<(BTreeSet<String>, BTreeSet<String>, NegotiationStatus), CoreError> {
        let available_features = self.feature_matrix.get_features(selected_version);
        let unsupported_features: BTreeSet<String> = client_capabilities
            .required_features
            .difference(&available_features)
            .cloned()
            .collect();

        let status = if unsupported_features.is_empty() {
            NegotiationStatus::Success
        } else {
            NegotiationStatus::FeaturesMissing
        };

        Ok((available_features, unsupported_features, status))
    }
}

impl Default for VersionNegotiator {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature compatibility matrix
struct FeatureMatrix {
    /// Maps versions to their available features
    version_features: std::collections::HashMap<Version, BTreeSet<String>>,
}

impl FeatureMatrix {
    fn new() -> Self {
        Self {
            version_features: std::collections::HashMap::new(),
        }
    }

    fn get_features(&self, version: &Version) -> BTreeSet<String> {
        self.version_features
            .get(version)
            .cloned()
            .unwrap_or_else(BTreeSet::new)
    }

    #[allow(dead_code)]
    fn add_version_features(&mut self, version: Version, features: BTreeSet<String>) {
        self.version_features.insert(version, features);
    }
}

/// Client version requirements builder
pub struct ClientRequirementsBuilder {
    capabilities: ClientCapabilities,
}

impl ClientRequirementsBuilder {
    /// Create a new builder
    pub fn new(client_type: &str, client_version: Version) -> Self {
        Self {
            capabilities: ClientCapabilities::new(client_type.to_string(), client_version),
        }
    }

    /// Set preferred version
    pub fn prefer_version(mut self, version: Version) -> Self {
        self.capabilities.preferred_version = Some(version);
        self
    }

    /// Add supported version range
    pub fn support_versions(mut self, versions: Vec<Version>) -> Self {
        self.capabilities.supported_versions = versions;
        self
    }

    /// Require a feature
    pub fn require_feature(mut self, feature: &str) -> Self {
        self.capabilities
            .required_features
            .insert(feature.to_string());
        self
    }

    /// Prefer a feature (optional)
    pub fn prefer_feature(mut self, feature: &str) -> Self {
        self.capabilities
            .optional_features
            .insert(feature.to_string());
        self
    }

    /// Build the client capabilities
    pub fn build(self) -> ClientCapabilities {
        self.capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_capabilities_builder() {
        let capabilities = ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
            .prefer_version(Version::new(2, 1, 0))
            .support_versions(vec![Version::new(2, 0, 0), Version::new(2, 1, 0)])
            .require_feature("feature1")
            .prefer_feature("feature2")
            .build();

        assert_eq!(capabilities.client_type, "test_client");
        assert_eq!(capabilities.preferred_version, Some(Version::new(2, 1, 0)));
        assert!(capabilities.required_features.contains("feature1"));
        assert!(capabilities.optional_features.contains("feature2"));
    }

    #[test]
    fn test_version_negotiation_prefer_latest() {
        let negotiator = VersionNegotiator::new().with_strategy(NegotiationStrategy::PreferLatest);

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .support_versions(vec![
                    Version::new(1, 0, 0),
                    Version::new(1, 1, 0),
                    Version::new(2, 0, 0),
                ])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);
        let server_versions = vec![&v1, &v2, &v3];

        let result = negotiator
            .negotiate(&client_capabilities, &server_versions)
            .unwrap();
        assert_eq!(result.negotiated_version, Version::new(2, 0, 0));
        assert_eq!(result.status, NegotiationStatus::Success);
    }

    #[test]
    fn test_version_negotiation_prefer_client() {
        let negotiator =
            VersionNegotiator::new().with_strategy(NegotiationStrategy::PreferClientPreference);

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .prefer_version(Version::new(1, 1, 0))
                .support_versions(vec![
                    Version::new(1, 0, 0),
                    Version::new(1, 1, 0),
                    Version::new(2, 0, 0),
                ])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);
        let server_versions = vec![&v1, &v2, &v3];

        let result = negotiator
            .negotiate(&client_capabilities, &server_versions)
            .unwrap();
        assert_eq!(result.negotiated_version, Version::new(1, 1, 0));
    }

    #[test]
    fn test_version_negotiation_prefer_stable() {
        let negotiator = VersionNegotiator::new().with_strategy(NegotiationStrategy::PreferStable);

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .support_versions(vec![
                    Version::new(1, 0, 0),
                    Version::parse("2.0.0-alpha").unwrap(),
                    Version::new(1, 1, 0),
                ])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::parse("2.0.0-alpha").unwrap();
        let v3 = Version::new(1, 1, 0);
        let server_versions = vec![&v1, &v2, &v3];

        let result = negotiator
            .negotiate(&client_capabilities, &server_versions)
            .unwrap();
        // Should prefer stable 1.1.0 over pre-release 2.0.0-alpha
        assert_eq!(result.negotiated_version, Version::new(1, 1, 0));
    }

    #[test]
    fn test_no_compatible_version() {
        let negotiator = VersionNegotiator::new();

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .support_versions(vec![Version::new(3, 0, 0)])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(2, 0, 0);
        let server_versions = vec![&v1, &v2];

        let result = negotiator
            .negotiate(&client_capabilities, &server_versions)
            .unwrap();
        assert_eq!(result.status, NegotiationStatus::Failed);
    }

    #[test]
    fn test_negotiation_metadata() {
        let negotiator = VersionNegotiator::new();

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .support_versions(vec![Version::new(1, 0, 0)])
                .build();

        let v1 = Version::new(1, 0, 0);
        let server_versions = vec![&v1];

        let result = negotiator
            .negotiate(&client_capabilities, &server_versions)
            .unwrap();
        assert!(!result.metadata.selection_reason.is_empty());
        assert_eq!(result.metadata.considered_versions.len(), 1);
    }
}
