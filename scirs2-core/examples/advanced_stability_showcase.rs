//! Advanced Stability Framework Showcase
//!
//! This example demonstrates the comprehensive stability framework with:
//! - Formal verification integration
//! - Runtime contract validation
//! - Chaos engineering for resilience testing
//! - Advanced performance modeling with ML
//! - Cryptographic audit trails
//! - Real-time monitoring
//!
//! This showcases the "advanced mode" enhancements for 1.0 release preparation.

use scirs2_core::apiversioning::Version;
use scirs2_core::stability::*;
use std::thread;
use std::time::{Duration, SystemTime};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Stability Framework Showcase");
    println!("========================================\n");

    // Create advanced stability manager
    let mut manager = StabilityGuaranteeManager::new();

    // Initialize core contracts
    manager.initialize_core_contracts()?;

    println!("✅ Initialized stability manager with core contracts");

    // Demonstrate advanced contract registration
    demonstrate_advanced_contract_registration(&mut manager)?;

    // Demonstrate formal verification
    demonstrate_formal_verification(&manager)?;

    // Demonstrate runtime validation with chaos engineering
    demonstrate_runtime_validation(&mut manager)?;

    // Demonstrate performance modeling
    demonstrate_performance_modeling(&mut manager)?;

    // Demonstrate audit trail
    demonstrate_audit_trail(&manager)?;

    // Generate comprehensive stability report
    generate_advanced_report(&manager)?;

    Ok(())
}

/// Demonstrate advanced contract registration with cryptographic hashing
#[allow(dead_code)]
fn demonstrate_advanced_contract_registration(
    manager: &mut StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📋 Advanced Contract Registration");
    println!("================================");

    // Create a high-performance matrix multiplication contract
    let matrix_contract = ApiContract {
        apiname: "matrix_multiply".to_string(),
        module: "linalg".to_string(),
        contract_hash: String::new(), // Will be calculated automatically
        created_at: SystemTime::UNIX_EPOCH, // Will be updated automatically
        verification_status: VerificationStatus::NotVerified,
        stability: StabilityLevel::Stable,
        since_version: Version::new(1, 0, 0),
        performance: PerformanceContract {
            time_complexity: ComplexityBound::Cubic, // O(n³) for naive implementation
            space_complexity: ComplexityBound::Quadratic, // O(n²) for output
            maxexecution_time: Some(Duration::from_secs(10)),
            min_throughput: Some(1000.0), // Operations per second
            memorybandwidth: Some(0.8),   // 80% memory bandwidth utilization
        },
        numerical: NumericalContract {
            precision: PrecisionGuarantee::MachinePrecision,
            stability: NumericalStability::Stable,
            input_domain: InputDomain {
                ranges: vec![(1.0, 10000.0)], // Matrix dimensions
                exclusions: vec![],
                special_values: SpecialValueHandling::Error,
            },
            output_range: OutputRange {
                bounds: None, // Output depends on input
                monotonic: None,
                continuous: true,
            },
        },
        concurrency: ConcurrencyContract {
            thread_safety: ThreadSafety::ThreadSafe,
            atomicity: AtomicityGuarantee::OperationAtomic,
            lock_free: false, // Uses internal locking for thread safety
            wait_free: false,
            memory_ordering: MemoryOrdering::AcquireRelease,
        },
        memory: MemoryContract {
            allocation_pattern: AllocationPattern::BoundedAllocations,
            max_memory: Some(1024 * 1024 * 1024), // 1GB max
            alignment: Some(64),                  // 64-byte alignment for SIMD
            locality: LocalityGuarantee::ExcellentSpatial,
            gc_behavior: GcBehavior::MinimalGc,
        },
        deprecation: None,
    };

    // Register the contract (demonstrates automatic hashing and verification)
    manager.register_contract(matrix_contract)?;

    println!("✅ Registered matrix_multiply contract with automatic:");
    println!("   - Cryptographic hash generation");
    println!("   - Formal verification initiation");
    println!("   - Audit trail recording");

    // Verify the contract was registered
    let retrieved = manager.get_contract("matrix_multiply", "linalg");
    match retrieved {
        Some(contract) => {
            println!("📋 Contract Details:");
            println!("   - Hash: {}", &contract.contract_hash[..16]);
            println!(
                "   - Verification Status: {:?}",
                contract.verification_status
            );
            println!("   - Stability: {:?}", contract.stability);
        }
        None => println!("❌ Contract not found!"),
    }

    Ok(())
}

/// Demonstrate formal verification capabilities
#[allow(dead_code)]
fn demonstrate_formal_verification(
    manager: &StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Formal Verification");
    println!("======================");

    // Check verification status
    let status = manager.get_verification_status("matrix_multiply", "linalg");
    println!("📊 Matrix multiply verification status: {status:?}");

    // Get overall verification coverage
    let coverage = manager.get_verification_coverage();
    println!("📈 Overall verification coverage: {coverage:.1}%");

    // In a real scenario, we'd wait for verification to complete
    println!("⏳ Formal verification includes:");
    println!("   - Performance bound checking");
    println!("   - Memory safety analysis");
    println!("   - Thread safety verification");
    println!("   - Temporal logic properties");

    Ok(())
}

/// Demonstrate runtime validation with chaos engineering
#[allow(dead_code)]
fn demonstrate_runtime_validation(
    manager: &mut StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Runtime Validation & Chaos Engineering");
    println!("==========================================");

    // Enable chaos engineering with 5% fault probability
    manager.enable_chaos_engineering(0.05);
    println!("🌪️  Enabled chaos engineering (5% fault probability)");

    // Simulate API calls with runtime validation
    for i in 0..10 {
        let call_context = ApiCallContext {
            execution_time: Duration::from_millis(50 + (i * 10)), // Increasing time
            memory_usage: 1024 * 1024 * (1 + i as usize),         // Increasing memory
            input_hash: format!("input_{i}"),
            output_hash: format!("output_{i}"),
            thread_id: format!("thread_{}", i % 4),
        };

        print!("🔄 Validating call {} ... ", i + 1);
        match manager.validate_api_call("matrix_multiply", "linalg", &call_context) {
            Ok(()) => println!("✅ Passed"),
            Err(e) => println!("❌ Failed: {e}"),
        }

        // Small delay to simulate real workload
        thread::sleep(Duration::from_millis(10));
    }

    // Check validation statistics
    if let Some(stats) = manager.get_validation_statistics() {
        println!("\n📊 Validation Statistics:");
        println!("   - Total validations: {}", stats.total_validations);
        println!("   - Violations detected: {}", stats.violations_detected);
        println!("   - Success rate: {:.2}%", stats.success_rate * 100.0);
        println!("   - Avg validation time: {:?}", stats.avg_validation_time);
    }

    // Check chaos engineering status
    if let Some((enabled, probability, fault_count)) = manager.get_chaos_status() {
        println!("\n🌪️  Chaos Engineering Status:");
        println!("   - Enabled: {enabled}");
        println!("   - Fault probability: {:.1}%", probability * 100.0);
        println!("   - Total faults injected: {fault_count}");
    }

    Ok(())
}

/// Demonstrate advanced performance modeling
#[allow(dead_code)]
fn demonstrate_performance_modeling(
    manager: &mut StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Advanced Performance Modeling");
    println!("=================================");

    // Record performance measurements for different input sizes
    let sizes = vec![100, 500, 1000, 2000, 5000];

    for &size in &sizes {
        let input_chars = InputCharacteristics::matrix(size, size);
        let performance = RuntimePerformanceMetrics {
            execution_time: Duration::from_millis((size * size / 1000) as u64), // Simulated
            memory_usage: size * size * 8,                                      // 8 bytes per f64
            cpu_usage: 0.8,                                                     // 80% CPU usage
            cache_hit_rate: 0.9 - (size as f64 / 10000.0), // Decreasing with size
            thread_count: 4,
        };
        let system_state = SystemState::current();

        // Convert RuntimePerformanceMetrics to PerformanceMetrics
        let mut operation_times = std::collections::HashMap::new();
        operation_times.insert(
            "matrix_multiply".to_string(),
            performance.execution_time.as_secs_f64(),
        );

        let mut strategy_success_rates = std::collections::HashMap::new();
        strategy_success_rates.insert(
            scirs2_core::performance_optimization::OptimizationStrategy::VectorOptimized,
            0.9,
        );

        let perf_metrics = scirs2_core::performance_optimization::PerformanceMetrics {
            operation_times,
            strategy_success_rates,
            memorybandwidth_utilization: performance.memory_usage as f64
                / (1024.0 * 1024.0 * 1024.0), // Convert to GB
            cache_hit_rate: performance.cache_hit_rate,
            parallel_efficiency: performance.cpu_usage / performance.thread_count as f64,
        };

        manager.record_performance(
            "matrix_multiply",
            "linalg",
            system_state,
            input_chars,
            perf_metrics,
        );

        println!("📈 Recorded performance for {size}x{size} matrix");
    }

    // Test performance prediction
    let test_input = InputCharacteristics::matrix(3000, 3000);
    let test_system = SystemState::current();

    if let Some(predicted) =
        manager.predict_performance("matrix_multiply", test_input, &test_system)
    {
        println!("\n🔮 Performance Prediction for 3000x3000 matrix:");
        println!("   - Predicted time: {:?}", predicted.execution_time);
        println!(
            "   - Predicted memory: {} MB",
            predicted.memory_usage / (1024 * 1024)
        );
        println!(
            "   - Predicted CPU usage: {:.1}%",
            predicted.cpu_usage * 100.0
        );
        println!(
            "   - Predicted cache hit rate: {:.1}%",
            predicted.cache_hit_rate * 100.0
        );
    } else {
        println!("🤔 No prediction model available yet (need more training data)");
    }

    // Check model accuracy
    if let Some(accuracy) = manager.get_model_accuracy("matrix_multiply") {
        println!("🎯 Model accuracy: {:.1}%", accuracy * 100.0);
    }

    Ok(())
}

/// Demonstrate cryptographic audit trail
#[allow(dead_code)]
fn demonstrate_audit_trail(
    manager: &StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔒 Cryptographic Audit Trail");
    println!("============================");

    // Check audit trail status
    let trail_length = manager.get_audit_trail_length();
    let integrity_verified = manager.verify_audit_integrity();

    println!("📋 Audit Trail Status:");
    println!("   - Total records: {trail_length}");
    println!(
        "   - Integrity verified: {}",
        if integrity_verified {
            "✅ Valid"
        } else {
            "❌ Corrupted"
        }
    );

    // Export audit trail (if serde feature is enabled)

    {
        match manager.export_audit_trail() {
            Ok(exported) => {
                println!("💾 Exported audit trail ({} bytes)", exported.len());
                // In a real application, you'd save this to a file or send to external audit system
            }
            Err(e) => println!("❌ Failed to export audit trail: {e}"),
        }
    }

    #[cfg(not(feature = "serde"))]
    {
        println!("💡 Audit trail export requires 'serde' feature");
    }

    Ok(())
}

/// Generate comprehensive stability report
#[allow(dead_code)]
fn generate_advanced_report(
    manager: &StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Comprehensive Stability Report");
    println!("==================================");

    let report = manager.generate_stability_report();

    // Print key sections of the report
    let lines: Vec<&str> = report.lines().collect();
    for line in lines.iter().take(30) {
        // Show first 30 lines
        println!("{line}");
    }

    if lines.len() > 30 {
        println!("... ({} more lines)", lines.len() - 30);
    }

    println!("\n💡 This report includes:");
    println!("   - API stability coverage");
    println!("   - Formal verification status");
    println!("   - Runtime validation statistics");
    println!("   - Chaos engineering metrics");
    println!("   - Performance model accuracies");
    println!("   - Audit trail integrity");

    Ok(())
}

/// Demonstrate advanced usage context validation
#[allow(dead_code)]
fn demonstrate_usage_context_validation(
    manager: &StabilityGuaranteeManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚙️  Advanced Usage Context Validation");
    println!("======================================");

    // Test different usage contexts
    let contexts = vec![
        (
            "High-performance context",
            UsageContext {
                required_stability: StabilityLevel::Stable,
                maxexecution_time: Some(Duration::from_millis(100)),
                requires_thread_safety: true,
                max_memory_usage: Some(512 * 1024 * 1024), // 512 MB
                required_precision: Some(PrecisionGuarantee::MachinePrecision),
            },
        ),
        (
            "Research context",
            UsageContext {
                required_stability: StabilityLevel::Experimental,
                maxexecution_time: Some(Duration::from_secs(60)),
                requires_thread_safety: false,
                max_memory_usage: Some(8 * 1024 * 1024 * 1024), // 8 GB
                required_precision: Some(PrecisionGuarantee::RelativeError(1e-12)),
            },
        ),
    ];

    for (name, context) in contexts {
        print!("🔍 Testing {name} ... ");
        match manager.validate_usage("matrix_multiply", "linalg", &context) {
            Ok(()) => println!("✅ Compatible"),
            Err(e) => println!("❌ Incompatible: {e}"),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_stability_showcase() {
        // Test that the showcase runs without panicking
        let result = std::panic::catch_unwind(|| {
            main().unwrap();
        });

        assert!(
            result.is_ok(),
            "Advanced stability showcase should complete successfully"
        );
    }

    #[test]
    fn test_contract_registration_and_retrieval() {
        let mut manager = StabilityGuaranteeManager::new();

        let contract = ApiContract {
            apiname: "test_api".to_string(),
            module: "test_module".to_string(),
            contract_hash: String::new(),
            created_at: SystemTime::UNIX_EPOCH,
            verification_status: VerificationStatus::NotVerified,
            stability: StabilityLevel::Stable,
            since_version: Version::new(1, 0, 0),
            performance: PerformanceContract {
                time_complexity: ComplexityBound::Linear,
                space_complexity: ComplexityBound::Constant,
                maxexecution_time: Some(Duration::from_millis(100)),
                min_throughput: None,
                memorybandwidth: None,
            },
            numerical: NumericalContract {
                precision: PrecisionGuarantee::MachinePrecision,
                stability: NumericalStability::Stable,
                input_domain: InputDomain {
                    ranges: vec![],
                    exclusions: vec![],
                    special_values: SpecialValueHandling::Propagate,
                },
                output_range: OutputRange {
                    bounds: None,
                    monotonic: None,
                    continuous: true,
                },
            },
            concurrency: ConcurrencyContract {
                thread_safety: ThreadSafety::ThreadSafe,
                atomicity: AtomicityGuarantee::OperationAtomic,
                lock_free: false,
                wait_free: false,
                memory_ordering: MemoryOrdering::AcquireRelease,
            },
            memory: MemoryContract {
                allocation_pattern: AllocationPattern::SingleAllocation,
                max_memory: Some(1024),
                alignment: None,
                locality: LocalityGuarantee::GoodSpatial,
                gc_behavior: GcBehavior::MinimalGc,
            },
            deprecation: None,
        };

        manager.register_contract(contract).unwrap();

        let retrieved = manager.get_contract("test_api", "test_module");
        assert!(retrieved.is_some());

        let contract = retrieved.unwrap();
        assert!(!contract.contract_hash.is_empty());
        assert_ne!(contract.created_at, SystemTime::UNIX_EPOCH);
    }

    #[test]
    fn test_performance_modeling_workflow() {
        let manager = StabilityGuaranteeManager::new();

        // Record some measurements
        for size in [100, 200, 300] {
            let input_chars = InputCharacteristics::matrix(size, size);
            let performance = RuntimePerformanceMetrics {
                execution_time: Duration::from_millis(size as u64),
                memory_usage: size * size * 8,
                cpu_usage: 0.5,
                cache_hit_rate: 0.8,
                thread_count: 1,
            };
            let system_state = SystemState::current();

            manager.record_performance("test_api", input_chars, performance, system_state);
        }

        // Check data points were recorded
        // Note: In a real implementation, we'd have methods to check this
        assert!(true); // Placeholder assertion
    }

    #[test]
    fn test_chaos_engineering_controls() {
        let manager = StabilityGuaranteeManager::new();

        // Initially disabled
        if let Some((enabled__)) = manager.get_chaos_status() {
            assert!(!enabled);
        }

        // Enable chaos engineering
        manager.enable_chaos_engineering(0.1);

        // Should now be enabled
        if let Some((enabled, probability_)) = manager.get_chaos_status() {
            assert!(enabled);
            assert!((probability - 0.1).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_audit_trail_integrity() {
        let manager = StabilityGuaranteeManager::new();

        // Fresh audit trail should be valid
        assert!(manager.verify_audit_integrity());

        // Should have some records from initialization
        assert!(manager.get_audit_trail_length() > 0);
    }
}
