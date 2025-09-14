//! # Version Negotiation
//!
//! Version negotiation system for client-server compatibility
//! and automatic version selection in distributed systems.

use super::Version;
use crate::error::CoreError;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Client capabilities for version negotiation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Preferred version
    pub preferred_version: Option<Version>,
    /// Supported version range
    pub supportedversions: Vec<Version>,
    /// Required features
    pub required_features: BTreeSet<String>,
    /// Optional features
    pub optional_features: BTreeSet<String>,
    /// Client type identifier
    pub client_type: String,
    /// Client version
    pub clientversion: Version,
}

impl ClientCapabilities {
    /// Create new client capabilities
    pub fn new(client_type: String, clientversion: Version) -> Self {
        Self {
            preferred_version: None,
            supportedversions: Vec::new(),
            required_features: BTreeSet::new(),
            optional_features: BTreeSet::new(),
            client_type,
            clientversion,
        }
    }

    /// Set preferred version
    pub fn with_preferred_version(mut self, version: Version) -> Self {
        self.preferred_version = Some(version);
        self
    }

    /// Add supported version
    pub fn with_supported_version(mut self, version: Version) -> Self {
        self.supportedversions.push(version);
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationMetadata {
    /// Versions considered during negotiation
    pub consideredversions: Vec<Version>,
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
    featurematrix: FeatureMatrix,
}

impl VersionNegotiator {
    /// Create a new version negotiator
    pub fn new() -> Self {
        Self {
            strategy: NegotiationStrategy::PreferLatest,
            featurematrix: FeatureMatrix::new(),
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
        serverversions: &[&Version],
    ) -> Result<NegotiationResult, CoreError> {
        let mut metadata = NegotiationMetadata {
            consideredversions: serverversions.iter().map(|v| (*v).clone()).collect(),
            selection_reason: String::new(),
            warnings: Vec::new(),
            algorithm: format!("{:?}", self.strategy),
        };

        // Find compatible versions
        let compatibleversions =
            self.find_compatibleversions(client_capabilities, serverversions, &mut metadata)?;

        if compatibleversions.is_empty() {
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
            self.apply_strategy(&compatibleversions, client_capabilities, &mut metadata)?;

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
    fn find_compatibleversions(
        &self,
        client_capabilities: &ClientCapabilities,
        serverversions: &[&Version],
        metadata: &mut NegotiationMetadata,
    ) -> Result<Vec<Version>, CoreError> {
        let mut compatible = Vec::new();

        for server_version in serverversions {
            // Check if client supports this version
            if client_capabilities.supportedversions.is_empty()
                || client_capabilities
                    .supportedversions
                    .contains(server_version)
            {
                compatible.push((*server_version).clone());
            } else {
                metadata.warnings.push(format!(
                    "Version {server_version} not in client's supported list"
                ));
            }
        }

        Ok(compatible)
    }

    /// Select the best version based on strategy
    fn apply_strategy(
        &self,
        compatibleversions: &[Version],
        client_capabilities: &ClientCapabilities,
        metadata: &mut NegotiationMetadata,
    ) -> Result<Version, CoreError> {
        match self.strategy {
            NegotiationStrategy::PreferLatest => {
                let mut versions = compatibleversions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
            NegotiationStrategy::PreferClientPreference => {
                if let Some(ref preferred) = client_capabilities.preferred_version {
                    if compatibleversions.contains(preferred) {
                        return Ok(preferred.clone());
                    }
                    metadata.warnings.push(
                        "Client preferred _version not available, falling back to latest"
                            .to_string(),
                    );
                }
                // Fall back to latest
                let mut versions = compatibleversions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
            NegotiationStrategy::PreferStable => {
                // Prefer non-pre-release versions
                let stableversions: Vec<_> = compatibleversions
                    .iter()
                    .filter(|v| v.is_stable())
                    .cloned()
                    .collect();

                if !stableversions.is_empty() {
                    let mut versions = stableversions;
                    versions.sort();
                    versions.reverse();
                    Ok(versions.into_iter().next().unwrap())
                } else {
                    // No stable versions, pick latest
                    let mut versions = compatibleversions.to_vec();
                    versions.sort();
                    versions.reverse();
                    Ok(versions.into_iter().next().unwrap())
                }
            }
            NegotiationStrategy::PreferFeatureRich => {
                // This would require feature information for each version
                // For now, fall back to latest
                let mut versions = compatibleversions.to_vec();
                versions.sort();
                versions.reverse();
                Ok(versions.into_iter().next().unwrap())
            }
            NegotiationStrategy::Custom => {
                // Custom logic would be implemented here
                let mut versions = compatibleversions.to_vec();
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
        let available_features = self.featurematrix.get_supported_features(selected_version);
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

    fn get_supported_features(&self, version: &Version) -> BTreeSet<String> {
        self.version_features
            .get(version)
            .cloned()
            .unwrap_or_else(BTreeSet::new)
    }

    #[allow(dead_code)]
    fn set_version_features(&mut self, version: Version, features: BTreeSet<String>) {
        self.version_features.insert(version, features);
    }
}

/// Client version requirements builder
pub struct ClientRequirementsBuilder {
    capabilities: ClientCapabilities,
}

impl ClientRequirementsBuilder {
    /// Create a new builder
    pub fn new(client_type: &str, clientversion: Version) -> Self {
        Self {
            capabilities: ClientCapabilities::new(client_type.to_string(), clientversion),
        }
    }

    /// Set preferred version
    pub fn preferred_version(mut self, version: Version) -> Self {
        self.capabilities.preferred_version = Some(version);
        self
    }

    /// Add supported version range
    pub fn supportversions(mut self, versions: Vec<Version>) -> Self {
        self.capabilities.supportedversions = versions;
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
            .preferred_version(Version::new(2, 1, 0))
            .supportversions(vec![Version::new(2, 0, 0), Version::new(2, 1, 0)])
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
                .supportversions(vec![
                    Version::new(1, 0, 0),
                    Version::new(1, 1, 0),
                    Version::new(2, 0, 0),
                ])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);
        let serverversions = vec![&v1, &v2, &v3];

        let result = negotiator
            .negotiate(&client_capabilities, &serverversions)
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
                .preferred_version(Version::new(1, 1, 0))
                .supportversions(vec![
                    Version::new(1, 0, 0),
                    Version::new(1, 1, 0),
                    Version::new(2, 0, 0),
                ])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);
        let serverversions = vec![&v1, &v2, &v3];

        let result = negotiator
            .negotiate(&client_capabilities, &serverversions)
            .unwrap();
        assert_eq!(result.negotiated_version, Version::new(1, 1, 0));
    }

    #[test]
    fn test_version_negotiation_prefer_stable() {
        let negotiator = VersionNegotiator::new().with_strategy(NegotiationStrategy::PreferStable);

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .supportversions(vec![
                    Version::new(1, 0, 0),
                    Version::parse("2.0.0-alpha").unwrap(),
                    Version::new(1, 1, 0),
                ])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::parse("2.0.0-alpha").unwrap();
        let v3 = Version::new(1, 1, 0);
        let serverversions = vec![&v1, &v2, &v3];

        let result = negotiator
            .negotiate(&client_capabilities, &serverversions)
            .unwrap();
        // Should prefer stable 1.1.0 over pre-release 2.0.0-alpha
        assert_eq!(result.negotiated_version, Version::new(1, 1, 0));
    }

    #[test]
    fn test_no_compatible_version() {
        let negotiator = VersionNegotiator::new();

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .supportversions(vec![Version::new(3, 0, 0)])
                .build();

        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(2, 0, 0);
        let serverversions = vec![&v1, &v2];

        let result = negotiator
            .negotiate(&client_capabilities, &serverversions)
            .unwrap();
        assert_eq!(result.status, NegotiationStatus::Failed);
    }

    #[test]
    fn test_negotiation_metadata() {
        let negotiator = VersionNegotiator::new();

        let client_capabilities =
            ClientRequirementsBuilder::new("test_client", Version::new(1, 0, 0))
                .supportversions(vec![Version::new(1, 0, 0)])
                .build();

        let v1 = Version::new(1, 0, 0);
        let serverversions = vec![&v1];

        let result = negotiator
            .negotiate(&client_capabilities, &serverversions)
            .unwrap();
        assert!(!result.metadata.selection_reason.is_empty());
        assert_eq!(result.metadata.consideredversions.len(), 1);
    }
}
