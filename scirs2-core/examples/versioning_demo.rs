//! # API Versioning System Demonstration
//!
//! This example demonstrates the comprehensive API versioning system
//! implemented in SciRS2 Core, showcasing all major features.

use scirs2_core::versioning::{
    deprecation::{DeprecationManager, DeprecationReason},
    ApiVersionBuilder, StabilityLevel, SupportStatus, Version, VersionManager,
};
use scirs2_core::CoreResult;

fn main() -> CoreResult<()> {
    println!("🔧 SciRS2 Core API Versioning System Demo");
    println!("==========================================\n");

    // Initialize version manager
    let mut version_manager = VersionManager::new();
    println!("✅ Created version manager");

    // Create and register API versions
    let v1_0_0 = ApiVersionBuilder::new(Version::parse("1.0.0")?)
        .stability(StabilityLevel::Stable)
        .feature("basic_operations")
        .feature("linear_algebra")
        .new_feature("Initial stable release")
        .new_feature("Basic mathematical operations")
        .new_feature("Linear algebra support")
        .build()?;

    let v1_1_0 = ApiVersionBuilder::new(Version::parse("1.1.0")?)
        .stability(StabilityLevel::Stable)
        .feature("basic_operations")
        .feature("linear_algebra")
        .feature("statistics")
        .new_feature("Statistical functions")
        .new_feature("Regression analysis")
        .bug_fix("Fixed numerical stability issues")
        .build()?;

    let v2_0_0 = ApiVersionBuilder::new(Version::parse("2.0.0")?)
        .stability(StabilityLevel::Beta)
        .support_status(SupportStatus::Active)
        .feature("basic_operations")
        .feature("linear_algebra_v2")
        .feature("statistics")
        .feature("machine_learning")
        .breaking_change("Linear algebra API redesigned for better performance")
        .breaking_change("Removed deprecated basic_math module")
        .new_feature("Machine learning algorithms")
        .new_feature("Neural network primitives")
        .deprecated_feature("legacy_array_ops")
        .min_client_version(Version::parse("1.5.0")?)
        .build()?;

    // Register versions
    version_manager.register_version(v1_0_0)?;
    version_manager.register_version(v1_1_0)?;
    version_manager.register_version(v2_0_0)?;
    println!("✅ Registered 3 API versions");

    // Set current version
    version_manager.set_current_version(Version::parse("2.0.0")?)?;
    println!("✅ Set current version to 2.0.0");

    // Check compatibility between versions
    println!("\n📊 Compatibility Analysis:");
    let from_version = Version::parse("1.0.0")?;
    let to_version = Version::parse("1.1.0")?;

    let compatibility = version_manager.check_compatibility(&from_version, &to_version)?;
    println!("  🔄 1.0.0 → 1.1.0: {:?}", compatibility);

    let report = version_manager.get_compatibility_report(&from_version, &to_version)?;
    println!("    New features: {:?}", report.new_features);
    println!("    Breaking changes: {}", report.breaking_changes.len());

    // Check major version upgrade
    let major_upgrade =
        version_manager.check_compatibility(&from_version, &Version::parse("2.0.0")?)?;
    println!("  🔄 1.0.0 → 2.0.0: {:?}", major_upgrade);

    let major_report =
        version_manager.get_compatibility_report(&from_version, &Version::parse("2.0.0")?)?;
    println!(
        "    Breaking changes: {}",
        major_report.breaking_changes.len()
    );
    if let Some(effort) = major_report.estimated_migration_effort {
        println!("    Estimated migration effort: {} hours", effort);
    }

    // Migration planning
    println!("\n🚀 Migration Planning:");
    let migration_plan =
        version_manager.get_migration_plan(&from_version, &Version::parse("2.0.0")?)?;
    println!("  📋 Migration from 1.0.0 to 2.0.0:");
    println!("    Risk level: {:?}", migration_plan.risk_level);
    println!(
        "    Estimated effort: {} hours",
        migration_plan.estimated_effort
    );
    println!("    Number of steps: {}", migration_plan.steps.len());

    for (i, step) in migration_plan.steps.iter().enumerate() {
        println!(
            "    {}. {} (Priority: {:?})",
            i + 1,
            step.name,
            step.priority
        );
    }

    // Deprecation management
    println!("\n⚠️  Deprecation Management:");
    let mut deprecation_manager = DeprecationManager::new();

    // Register version for deprecation tracking
    let old_version = version_manager
        .get_version(&Version::parse("1.0.0")?)
        .unwrap();
    deprecation_manager.register_version(old_version)?;

    // Announce deprecation
    let announcement = deprecation_manager.announce_deprecation(
        &Version::parse("1.0.0")?,
        DeprecationReason::SupersededBy(Version::parse("2.0.0")?),
        Some(Version::parse("2.0.0")?),
    )?;

    println!("  📢 Deprecation announced for 1.0.0:");
    println!("    Message: {}", announcement.message);
    println!(
        "    Timeline: {} days until deprecated",
        (announcement.timeline.deprecated_date - announcement.timeline.announced).num_days()
    );
    println!(
        "    End of life: {}",
        announcement.timeline.end_of_life.format("%Y-%m-%d")
    );

    // Version statistics
    println!("\n📈 Version Statistics:");
    let stats = version_manager.get_version_statistics();
    println!("  Total versions: {}", stats.total_versions);
    println!("  Stable versions: {}", stats.stable_versions);
    println!("  Beta versions: {}", stats.beta_versions);
    println!("  Active versions: {}", stats.active_versions);

    // Get supported versions
    let supported = version_manager.get_supported_versions();
    println!(
        "  Supported versions: {:?}",
        supported
            .iter()
            .map(|v| v.version.to_string())
            .collect::<Vec<_>>()
    );

    // Latest stable version
    if let Some(latest) = version_manager.get_latest_stable() {
        println!("  Latest stable: {}", latest.version);
    }

    // Version negotiation example
    println!("\n🤝 Version Negotiation:");
    use scirs2_core::versioning::negotiation::ClientRequirementsBuilder;

    let client_capabilities =
        ClientRequirementsBuilder::new("scientific_client", Version::parse("1.0.0")?)
            .prefer_version(Version::parse("1.1.0")?)
            .support_versions(vec![
                Version::parse("1.0.0")?,
                Version::parse("1.1.0")?,
                Version::parse("2.0.0")?,
            ])
            .require_feature("linear_algebra")
            .prefer_feature("statistics")
            .build();

    let _supported_versions: Vec<&Version> = version_manager
        .get_supported_versions()
        .into_iter()
        .map(|v| &v.version)
        .collect();

    let negotiation_result = version_manager.negotiate_version(&client_capabilities)?;
    println!(
        "  📋 Negotiated version: {}",
        negotiation_result.negotiated_version
    );
    println!("  ✅ Negotiation status: {:?}", negotiation_result.status);
    println!(
        "  💡 Selection reason: {}",
        negotiation_result.metadata.selection_reason
    );

    // Maintenance operations
    println!("\n🔧 Maintenance Operations:");
    let maintenance_report = version_manager.perform_maintenance()?;
    println!("  📊 Maintenance completed:");
    println!(
        "    Versions marked EOL: {}",
        maintenance_report.versions_marked_eol.len()
    );
    println!(
        "    Deprecation updates: {}",
        maintenance_report.deprecation_updates
    );
    println!(
        "    Migration plans cleaned: {}",
        maintenance_report.migration_plans_cleaned
    );

    println!("\n✨ Demo completed successfully!");
    println!("\nThe versioning system provides:");
    println!("  🔹 Semantic versioning with SemVer 2.0.0 compliance");
    println!("  🔹 Backward compatibility checking and reporting");
    println!("  🔹 Automated migration planning and guidance");
    println!("  🔹 Deprecation lifecycle management");
    println!("  🔹 Version negotiation for client-server scenarios");
    println!("  🔹 Production-ready maintenance and monitoring");

    Ok(())
}
