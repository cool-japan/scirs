//! Demonstration of Python/SciPy migration assistance
//!
//! This example shows how to use the migration tools to help port
//! SciPy special functions code to SciRS2.

use scirs2_core::ndarray::Array1;
use scirs2_special::python_interop::{codegen, compat, examples, MigrationGuide};

#[allow(dead_code)]
fn main() {
    println!("=== Python/SciPy Migration Assistant Demo ===\n");

    // 1. Create migration guide
    let guide = MigrationGuide::new();

    println!("1. Function Mapping Examples");
    println!("----------------------------");

    // Look up some common functions
    let functions_to_check = vec![
        "scipy.special.gamma",
        "scipy.special.erf",
        "scipy.special.j0",
        "scipy.special.expit",
        "scipy.special.airy",
        "scipy.special.unknown_function",
    ];

    for func in &functions_to_check {
        match guide.get_mapping(func) {
            Some(mapping) => {
                println!("✓ {}", func);
                println!("  → SciRS2: {}", mapping.module_path);
                println!("  → Function: {}", mapping.scirs2_name);
                if !mapping.notes.is_empty() {
                    println!("  → Notes: {}", mapping.notes[0]);
                }
            }
            None => {
                println!("✗ {} - No direct mapping found", func);
            }
        }
        println!();
    }

    // 2. Generate migration report
    println!("\n2. Migration Report");
    println!("-------------------");
    let report = guide.generate_migration_report(&functions_to_check);
    println!("{}", &report[..500]); // Show first 500 chars
    println!("... (truncated)\n");

    // 3. Code translation examples
    println!("3. Code Translation");
    println!("-------------------");

    let scipy_code = r#"
from scipy.special import gamma, erf, j0
result1 = gamma(5.5)
result2 = erf(1.0)
result3 = j0(2.5)
"#;

    println!("SciPy code:");
    println!("{}", scipy_code);

    match codegen::generate_rust_equivalent(scipy_code) {
        Ok(rust_code) => {
            println!("\nGenerated Rust hints:");
            println!("{}", rust_code);
        }
        Err(e) => {
            println!("Translation error: {}", e);
        }
    }

    // 4. Compatibility layer usage
    println!("\n4. Compatibility Layer");
    println!("----------------------");

    // Create test data
    let x = Array1::linspace(0.1, 5.0, 10);
    println!("Input array: {:?}", &x.as_slice().unwrap()[..5]); // Show first 5 elements

    // Use compatibility functions
    let gamma_results = compat::gamma_array(&x.view());
    let erf_results = compat::erf_array(&x.view());
    let j0_results = compat::j0_array(&x.view());

    println!(
        "Gamma results: {:?}",
        &gamma_results.as_slice().unwrap()[..5]
    );
    println!("Erf results: {:?}", &erf_results.as_slice().unwrap()[..5]);
    println!("J0 results: {:?}", &j0_results.as_slice().unwrap()[..5]);

    // 5. Migration examples
    println!("\n5. Migration Examples");
    println!("---------------------");

    println!("Gamma function migration:");
    println!("{}", examples::gamma_migration_example());

    println!("\nBessel function migration:");
    println!("{}", examples::bessel_migration_example());

    println!("\nStatistical function migration:");
    println!("{}", examples::statistical_migration_example());

    // 6. Performance comparison (simulated)
    println!("\n6. Performance Comparison");
    println!("-------------------------");

    use scirs2_special::python_interop::performance::PerformanceComparison;

    let comparisons = vec![
        PerformanceComparison {
            function_name: "gamma".to_string(),
            scipy_time_ms: 10.5,
            scirs2_time_ms: 2.1,
            speedup: 5.0,
            accuracy_difference: 1e-15,
        },
        PerformanceComparison {
            function_name: "j0".to_string(),
            scipy_time_ms: 8.3,
            scirs2_time_ms: 1.2,
            speedup: 6.9,
            accuracy_difference: 2e-16,
        },
        PerformanceComparison {
            function_name: "erf".to_string(),
            scipy_time_ms: 5.7,
            scirs2_time_ms: 0.9,
            speedup: 6.3,
            accuracy_difference: 5e-16,
        },
    ];

    for comp in comparisons {
        println!("{}", comp.report());
    }

    // 7. Import generation
    println!("\n7. Import Generation");
    println!("--------------------");

    let scipy_imports = vec!["gamma", "erf", "j0", "beta"];
    let rust_imports = codegen::generate_imports(&scipy_imports);
    println!("For SciPy imports: {:?}", scipy_imports);
    println!("Generate Rust imports:");
    println!("{}", rust_imports);

    // 8. List all available mappings
    println!("\n8. Available Function Mappings");
    println!("------------------------------");

    let all_mappings = guide.list_all_mappings();
    println!("Total mappings available: {}", all_mappings.len());
    println!("\nFirst 10 mappings:");
    for (scipy_name, mapping) in all_mappings.iter().take(10) {
        println!("  {} → {}", scipy_name, mapping.scirs2_name);
    }

    println!("\n=== Demo Complete ===");
    println!("\nTip: Use the MigrationGuide to help port your SciPy code to SciRS2!");
    println!("The compatibility layer (compat module) provides SciPy-like interfaces.");
}
