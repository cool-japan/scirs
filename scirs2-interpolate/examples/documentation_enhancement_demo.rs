//! Documentation Enhancement Demo
//!
//! This example demonstrates how to use the documentation enhancement tools
//! to analyze and improve documentation quality for the stable release.

use scirs2_interpolate::documentation_enhancement::{DocumentationReport, ValidationStatus};
use scirs2_interpolate::{
    enhance_documentation_for_stable_release, enhance_documentation_with_config,
    quick_documentation_analysis, AudienceLevel, DocumentationConfig, DocumentationReadiness,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Documentation Enhancement Demo ===\n");

    // 1. Quick documentation analysis for development
    println!("1. Running quick documentation analysis...");
    match quick_documentation_analysis() {
        Ok(report) => {
            println!("{}", report);

            match report.readiness {
                DocumentationReadiness::Ready => {
                    println!("✅ Documentation is ready for stable release!");
                }
                DocumentationReadiness::NeedsMinorWork => {
                    println!("⚠️  Documentation needs minor improvements");
                }
                DocumentationReadiness::NeedsSignificantWork => {
                    println!("❌ Documentation needs significant work");
                }
            }
            println!();
        }
        Err(e) => println!("Quick documentation analysis failed: {}\n", e),
    }

    // 2. Comprehensive documentation enhancement
    println!("2. Running comprehensive documentation enhancement...");
    match enhance_documentation_for_stable_release() {
        Ok(report) => {
            println!("{}", report);

            // Analyze the generated content
            analyze_generated_content(&report);
            println!();
        }
        Err(e) => println!("Comprehensive documentation enhancement failed: {}\n", e),
    }

    // 3. Custom documentation enhancement with different configurations
    println!("3. Running documentation enhancement with different configurations...");

    let configs = vec![
        (
            "Basic",
            DocumentationConfig {
                min_coverage_percentage: 70.0,
                min_quality_score: 0.6,
                generate_user_guides: false,
                validate_examples: true,
                create_tutorials: false,
                target_audiences: vec![AudienceLevel::Beginner],
            },
        ),
        (
            "Comprehensive",
            DocumentationConfig {
                min_coverage_percentage: 95.0,
                min_quality_score: 0.9,
                generate_user_guides: true,
                validate_examples: true,
                create_tutorials: true,
                target_audiences: vec![
                    AudienceLevel::Beginner,
                    AudienceLevel::Intermediate,
                    AudienceLevel::Advanced,
                    AudienceLevel::DomainExpert,
                ],
            },
        ),
        (
            "Examples Only",
            DocumentationConfig {
                min_coverage_percentage: 80.0,
                min_quality_score: 0.7,
                generate_user_guides: false,
                validate_examples: true,
                create_tutorials: false,
                target_audiences: vec![AudienceLevel::Intermediate],
            },
        ),
    ];

    for (name, config) in configs {
        println!("  Running {} configuration...", name);
        match enhance_documentation_with_config(config) {
            Ok(report) => {
                println!(
                    "    {}: {:?} ({:.1}% coverage, {} guides, {} tutorials)",
                    name,
                    report.readiness,
                    report.coverage_percentage,
                    report.user_guides.len(),
                    report.tutorials.len()
                );

                if !report.critical_issues.is_empty() {
                    println!("      Critical issues: {}", report.critical_issues.len());
                }
            }
            Err(e) => println!("    {} configuration failed: {}", name, e),
        }
    }
    println!();

    // 4. Analyze documentation by audience level
    println!("4. Analyzing documentation by audience level...");
    match enhance_documentation_for_stable_release() {
        Ok(report) => {
            analyze_by_audience_level(&report);
            println!();
        }
        Err(e) => println!("Audience analysis failed: {}\n", e),
    }

    // 5. Example validation analysis
    println!("5. Example validation analysis...");
    match enhance_documentation_for_stable_release() {
        Ok(report) => {
            analyze_example_validation(&report);
            println!();
        }
        Err(e) => println!("Example validation analysis failed: {}\n", e),
    }

    // 6. Documentation quality assessment
    println!("6. Documentation quality assessment...");
    match enhance_documentation_for_stable_release() {
        Ok(report) => {
            assess_documentation_quality(&report);
            println!();
        }
        Err(e) => println!("Quality assessment failed: {}\n", e),
    }

    // 7. Action plan for documentation improvement
    println!("7. Action plan for documentation improvement:");
    match enhance_documentation_for_stable_release() {
        Ok(report) => {
            provide_documentation_action_plan(&report);
        }
        Err(e) => println!("Failed to generate action plan: {}", e),
    }

    println!("\n=== Documentation Enhancement Complete ===");
    println!("Use these results to improve documentation quality before stable release.");

    Ok(())
}

/// Analyze the generated content from documentation enhancement
#[allow(dead_code)]
fn analyze_generated_content(report: &DocumentationReport) {
    println!("Generated Content Analysis:");

    // User guides analysis
    if !report.user_guides.is_empty() {
        println!("  User Guides Generated ({}):", report.user_guides.len());
        for guide in &report.user_guides {
            println!(
                "    - {} (Target: {:?}, {} sections, ~{} min read)",
                guide.title,
                guide.audience,
                guide.sections.len(),
                guide.reading_time
            );
        }
    }

    // Tutorials analysis
    if !report.tutorials.is_empty() {
        println!("  Tutorials Created ({}):", report.tutorials.len());
        for tutorial in &report.tutorials {
            println!(
                "    - {} (Target: {:?}, {} steps, ~{} min)",
                tutorial.title,
                tutorial.audience,
                tutorial.steps.len(),
                tutorial.completion_time
            );
        }
    }

    // Example validations
    if !report.example_validations.is_empty() {
        println!(
            "  Example Validations ({}):",
            report.example_validations.len()
        );
        let valid_count = report
            .example_validations
            .iter()
            .filter(|v| v.status == ValidationStatus::Valid)
            .count();
        let invalid_count = report
            .example_validations
            .iter()
            .filter(|v| v.status == ValidationStatus::Invalid)
            .count();
        let warning_count = report
            .example_validations
            .iter()
            .filter(|v| v.status == ValidationStatus::ValidWithWarnings)
            .count();

        println!(
            "    - {} valid, {} with warnings, {} invalid",
            valid_count, warning_count, invalid_count
        );

        if invalid_count > 0 {
            println!("    ⚠️  {} examples need fixing", invalid_count);
        }
    }
}

/// Analyze documentation by target audience
#[allow(dead_code)]
fn analyze_by_audience_level(report: &DocumentationReport) {
    use std::collections::HashMap;

    let mut audience_content: HashMap<String, (usize, usize)> = HashMap::new();

    // Count guides by audience
    for guide in &report.user_guides {
        let audience_name = format!("{:?}", guide.audience);
        let entry = audience_content.entry(audience_name).or_insert((0, 0));
        entry.0 += 1;
    }

    // Count tutorials by audience
    for tutorial in &report.tutorials {
        let audience_name = format!("{:?}", tutorial.audience);
        let entry = audience_content.entry(audience_name).or_insert((0, 0));
        entry.1 += 1;
    }

    println!("Content by Audience Level:");
    for (audience, (guides, tutorials)) in &audience_content {
        println!("  {}: {} guides, {} tutorials", audience, guides, tutorials);
    }

    // Identify gaps
    let all_audiences = ["Beginner", "Intermediate", "Advanced", "DomainExpert"];
    let missing_audiences: Vec<_> = all_audiences
        .iter()
        .filter(|&audience| !audience_content.contains_key(*audience))
        .collect();

    if !missing_audiences.is_empty() {
        let missing_str: Vec<String> = missing_audiences.iter().map(|s| s.to_string()).collect();
        println!("  Missing content for: {}", missing_str.join(", "));
    }
}

/// Analyze example validation results
#[allow(dead_code)]
fn analyze_example_validation(report: &DocumentationReport) {
    println!("Example Validation Results:");

    let total_examples = report.example_validations.len();
    if total_examples == 0 {
        println!("  No examples validated");
        return;
    }

    let valid_examples = report
        .example_validations
        .iter()
        .filter(|v| v.status == ValidationStatus::Valid)
        .count();

    let warning_examples = report
        .example_validations
        .iter()
        .filter(|v| v.status == ValidationStatus::ValidWithWarnings)
        .count();

    let invalid_examples = report
        .example_validations
        .iter()
        .filter(|v| v.status == ValidationStatus::Invalid)
        .count();

    println!("  Total Examples: {}", total_examples);
    println!(
        "  Valid: {} ({:.1}%)",
        valid_examples,
        (valid_examples as f32 / total_examples as f32) * 100.0
    );
    println!(
        "  With Warnings: {} ({:.1}%)",
        warning_examples,
        (warning_examples as f32 / total_examples as f32) * 100.0
    );
    println!(
        "  Invalid: {} ({:.1}%)",
        invalid_examples,
        (invalid_examples as f32 / total_examples as f32) * 100.0
    );

    // Show examples that need attention
    let problematic_examples: Vec<_> = report
        .example_validations
        .iter()
        .filter(|v| v.status != ValidationStatus::Valid)
        .collect();

    if !problematic_examples.is_empty() {
        println!("  Examples needing attention:");
        for (i, example) in problematic_examples.iter().enumerate() {
            if i < 5 {
                // Show first 5
                println!("    - {}: {:?}", example.example_id, example.status);
                if !example.issues.is_empty() {
                    println!("      Issues: {}", example.issues.join(", "));
                }
            }
        }
        if problematic_examples.len() > 5 {
            println!("    ... and {} more", problematic_examples.len() - 5);
        }
    }
}

/// Assess overall documentation quality
#[allow(dead_code)]
fn assess_documentation_quality(report: &DocumentationReport) {
    println!("Documentation Quality Assessment:");

    if report.analysis_results.is_empty() {
        println!("  No analysis results available");
        return;
    }

    // Calculate average quality scores
    let total_items = report.analysis_results.len() as f32;
    let avg_overall_score = report
        .analysis_results
        .iter()
        .map(|r| r.quality_assessment.overall_score)
        .sum::<f32>()
        / total_items;

    let avg_clarity = report
        .analysis_results
        .iter()
        .map(|r| r.quality_assessment.clarity_score)
        .sum::<f32>()
        / total_items;

    let avg_completeness = report
        .analysis_results
        .iter()
        .map(|r| r.quality_assessment.completeness_score)
        .sum::<f32>()
        / total_items;

    let avg_usefulness = report
        .analysis_results
        .iter()
        .map(|r| r.quality_assessment.usefulness_score)
        .sum::<f32>()
        / total_items;

    println!("  Average Quality Scores:");
    println!("    Overall: {:.2}/1.0", avg_overall_score);
    println!("    Clarity: {:.2}/1.0", avg_clarity);
    println!("    Completeness: {:.2}/1.0", avg_completeness);
    println!("    Usefulness: {:.2}/1.0", avg_usefulness);

    // Quality distribution
    let high_quality = report
        .analysis_results
        .iter()
        .filter(|r| r.quality_assessment.overall_score >= 0.8)
        .count();

    let medium_quality = report
        .analysis_results
        .iter()
        .filter(|r| {
            r.quality_assessment.overall_score >= 0.6 && r.quality_assessment.overall_score < 0.8
        })
        .count();

    let low_quality = report
        .analysis_results
        .iter()
        .filter(|r| r.quality_assessment.overall_score < 0.6)
        .count();

    println!("  Quality Distribution:");
    println!("    High Quality (≥0.8): {} items", high_quality);
    println!("    Medium Quality (0.6-0.8): {} items", medium_quality);
    println!("    Low Quality (<0.6): {} items", low_quality);

    if low_quality > 0 {
        println!(
            "  ⚠️  {} items need significant documentation improvement",
            low_quality
        );
    }
}

/// Provide actionable documentation improvement plan
#[allow(dead_code)]
fn provide_documentation_action_plan(report: &DocumentationReport) {
    match report.readiness {
        DocumentationReadiness::Ready => {
            println!("✅ DOCUMENTATION READY FOR STABLE RELEASE");
            println!("  The documentation meets quality standards for stable release.");
            println!("  Optional improvements:");
            println!("  - Consider adding more advanced examples");
            println!("  - Create domain-specific tutorials");
            println!("  - Add performance optimization guides");
        }
        DocumentationReadiness::NeedsMinorWork => {
            println!("⚠️  MINOR DOCUMENTATION IMPROVEMENTS NEEDED");
            println!("  Priority Actions:");

            let mut action_count = 1;

            // Critical issues first
            if !report.critical_issues.is_empty() {
                println!(
                    "    {}. Fix {} critical documentation issues",
                    action_count,
                    report.critical_issues.len()
                );
                action_count += 1;

                for (i, issue) in report.critical_issues.iter().enumerate() {
                    if i < 3 {
                        // Show top 3
                        println!("       - {}: {}", issue.location, issue.description);
                    }
                }
                if report.critical_issues.len() > 3 {
                    println!("       ... and {} more", report.critical_issues.len() - 3);
                }
            }

            // Coverage improvements
            if report.coverage_percentage < 95.0 {
                println!(
                    "    {}. Improve documentation coverage from {:.1}% to 95%",
                    action_count, report.coverage_percentage
                );
                action_count += 1;
            }

            // Example issues
            let broken_examples = report
                .example_validations
                .iter()
                .filter(|v| v.status == ValidationStatus::Invalid)
                .count();

            if broken_examples > 0 {
                println!(
                    "    {}. Fix {} broken examples",
                    action_count, broken_examples
                );
            }
        }
        DocumentationReadiness::NeedsSignificantWork => {
            println!("❌ SIGNIFICANT DOCUMENTATION WORK REQUIRED");
            println!("  IMMEDIATE ACTIONS REQUIRED:");

            println!(
                "    1. Increase documentation coverage from {:.1}% to 95%",
                report.coverage_percentage
            );

            if !report.critical_issues.is_empty() {
                println!(
                    "    2. Resolve {} critical documentation issues",
                    report.critical_issues.len()
                );
            }

            let missing_examples = report
                .analysis_results
                .iter()
                .filter(|r| !r.examples_status.has_examples)
                .count();

            if missing_examples > 0 {
                println!("    3. Add examples to {} API items", missing_examples);
            }

            let broken_examples = report
                .example_validations
                .iter()
                .filter(|v| v.status == ValidationStatus::Invalid)
                .count();

            if broken_examples > 0 {
                println!("    4. Fix {} broken examples", broken_examples);
            }

            println!("    5. Create comprehensive user guides");
            println!("    6. Develop step-by-step tutorials");

            println!("  WORKFLOW:");
            println!("    - Focus on high-impact API items first");
            println!("    - Prioritize user-facing functions and types");
            println!("    - Validate all examples before submission");
            println!("    - Get feedback from target users");
        }
    }

    // General recommendations
    println!("  Additional Recommendations:");
    for (i, recommendation) in report.recommendations.iter().enumerate() {
        if i < 5 {
            // Show top 5 recommendations
            println!("    - {}", recommendation);
        }
    }
    if report.recommendations.len() > 5 {
        println!(
            "    ... and {} more recommendations",
            report.recommendations.len() - 5
        );
    }

    // Timeline estimation
    let estimated_hours = match report.readiness {
        DocumentationReadiness::Ready => 2,
        DocumentationReadiness::NeedsMinorWork => 8,
        DocumentationReadiness::NeedsSignificantWork => 40,
    };

    println!("  Estimated Work: ~{} hours", estimated_hours);
}
