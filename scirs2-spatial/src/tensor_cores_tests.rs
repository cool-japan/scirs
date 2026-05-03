use super::*;
use scirs2_core::ndarray::array;
use std::time::Duration;

#[test]
fn test_precision_mode() {
    assert_eq!(PrecisionMode::Mixed16, PrecisionMode::Mixed16);
    assert_ne!(PrecisionMode::Mixed16, PrecisionMode::Full32);
}

#[test]
fn test_tensor_core_capabilities() {
    let capabilities = detect_tensor_core_capabilities();
    assert!(capabilities.is_ok());

    let caps = capabilities.expect("Operation failed");
    assert!(!caps.tensor_core_types.is_empty());
    assert!(!caps.supported_precisions.is_empty());
}

#[test]
fn test_tensor_core_distance_matrix_creation() {
    let result = TensorCoreDistanceMatrix::new();
    assert!(result.is_ok());

    let matrix_computer = result.expect("Operation failed");
    assert_eq!(matrix_computer.precision_mode, PrecisionMode::Mixed16);
}

#[test]
fn test_tensor_core_clustering_creation() {
    let result = TensorCoreClustering::new(3);
    assert!(result.is_ok());

    let clustering = result.expect("Operation failed");
    assert_eq!(clustering._numclusters, 3);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_tensor_core_distance_computation() {
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let mut matrix_computer = TensorCoreDistanceMatrix::new().expect("Operation failed");

    let result = matrix_computer.compute_parallel(&points.view()).await;
    assert!(result.is_ok());

    let distances = result.expect("Operation failed");
    assert_eq!(distances.shape(), &[3, 3]);

    // Check diagonal is zero
    for i in 0..3 {
        assert!((distances[[i, i]]).abs() < 1e-10);
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_tensor_core_clustering() {
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let mut clustering = TensorCoreClustering::new(2).expect("Operation failed");

    let result = clustering.fit(&points.view()).await;
    assert!(result.is_ok());

    let (centroids, assignments) = result.expect("Operation failed");
    assert_eq!(centroids.shape(), &[2, 2]);
    assert_eq!(assignments.len(), 4);
}

#[test]
fn test_stability_metrics_creation() {
    let metrics = StabilityMetrics::new();
    assert_eq!(metrics.condition_number, 1.0);
    assert_eq!(metrics.relative_error, 0.0);
    assert_eq!(metrics.stability_level, StabilityLevel::Excellent);
    assert!(metrics.error_types.is_empty());
}

#[test]
fn test_stability_level_update() {
    let mut metrics = StabilityMetrics::new();

    // Test critical stability
    metrics.condition_number = 1e15;
    metrics.update_stability_level();
    assert_eq!(metrics.stability_level, StabilityLevel::Critical);

    // Test poor stability
    metrics.condition_number = 1e9;
    metrics.relative_error = 1e-7;
    metrics.update_stability_level();
    assert_eq!(metrics.stability_level, StabilityLevel::Poor);

    // Test good stability
    metrics.condition_number = 1e3;
    metrics.relative_error = 1e-10;
    metrics.update_stability_level();
    assert_eq!(metrics.stability_level, StabilityLevel::Good);
}

#[test]
fn test_error_detection() {
    let mut metrics = StabilityMetrics::new();

    // Test NaN detection
    let data_with_nan = array![[1.0, 2.0], [f64::NAN, 4.0]];
    metrics.detect_errors(&data_with_nan);
    assert!(metrics
        .error_types
        .contains(&NumericalErrorType::InvalidValues));

    // Test overflow detection
    let data_with_overflow = array![[1e150, 2.0], [3.0, 4.0]];
    metrics.detect_errors(&data_with_overflow);
    assert!(metrics.error_types.contains(&NumericalErrorType::Overflow));

    // Test underflow detection - all values must be small for underflow detection
    let data_with_underflow = array![[1e-150, 1e-120], [1e-130, 1e-140]];
    metrics.detect_errors(&data_with_underflow);
    assert!(metrics.error_types.contains(&NumericalErrorType::Underflow));
}

#[test]
fn test_dynamic_precision_config() {
    let config = DynamicPrecisionConfig::default();
    assert_eq!(config.strategy, ScalingStrategy::Balanced);
    assert_eq!(config.min_precision, PrecisionMode::Int8Dynamic);
    assert_eq!(config.max_precision, PrecisionMode::Full32);
    assert_eq!(config.performance_weight, 0.6);
    assert_eq!(config.accuracy_weight, 0.4);
}

#[test]
fn test_numerical_stability_monitor_creation() {
    let config = DynamicPrecisionConfig::default();
    let monitor = NumericalStabilityMonitor::new(config);

    assert_eq!(monitor.current_precision, PrecisionMode::Mixed16);
    assert!(monitor.stability_history.is_empty());
    assert_eq!(monitor.recovery_attempts, 0);
}

#[test]
fn test_precision_increase_decrease() {
    let config = DynamicPrecisionConfig::default();
    let monitor = NumericalStabilityMonitor::new(config);

    // Test precision increase
    let increased = NumericalStabilityMonitor::increase_precision(PrecisionMode::Int8Dynamic);
    assert_eq!(increased, PrecisionMode::Mixed16);

    let max_increased = NumericalStabilityMonitor::increase_precision(PrecisionMode::Full32);
    assert_eq!(max_increased, PrecisionMode::Full32); // Should stay at max

    // Test precision decrease
    let decreased = NumericalStabilityMonitor::decrease_precision(PrecisionMode::Mixed16);
    assert_eq!(decreased, PrecisionMode::Int8Dynamic);

    let min_decreased = NumericalStabilityMonitor::decrease_precision(PrecisionMode::Int4Advanced);
    assert_eq!(min_decreased, PrecisionMode::Int4Advanced); // Should stay at min
}

#[test]
fn test_condition_number_estimation() {
    let config = DynamicPrecisionConfig::default();
    let monitor = NumericalStabilityMonitor::new(config);

    // Well-conditioned data
    let well_conditioned = array![[1.0, 2.0], [3.0, 4.0]];
    let condition_1 = NumericalStabilityMonitor::estimate_condition_number(&well_conditioned);
    assert!(condition_1 > 1.0 && condition_1 < 100.0);

    // Ill-conditioned data (large range)
    let ill_conditioned = array![[1e-10, 2.0], [3.0, 1e10]];
    let condition_2 = NumericalStabilityMonitor::estimate_condition_number(&ill_conditioned);
    assert!(condition_2 > 1e15);
}

#[test]
fn test_error_recovery_system_creation() {
    let recovery_system = ErrorRecoverySystem::new();

    // Check that recovery strategies are defined
    assert!(!recovery_system.recovery_strategies.is_empty());
    assert!(recovery_system
        .recovery_strategies
        .contains_key(&NumericalErrorType::Overflow));
    assert!(recovery_system
        .recovery_strategies
        .contains_key(&NumericalErrorType::IllConditioned));
    assert_eq!(recovery_system.max_recovery_attempts, 3);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_recovery_action_selection() {
    let mut recovery_system = ErrorRecoverySystem::new();

    let action = recovery_system
        .attempt_recovery(NumericalErrorType::Overflow)
        .await;
    assert!(action.is_ok());

    let recovery_action = action.expect("Operation failed");
    assert!(matches!(
        recovery_action,
        RecoveryAction::IncreasePrecision
            | RecoveryAction::ReduceTileSize
            | RecoveryAction::NumericalStabilization
    ));
}

#[test]
fn test_success_rate_update() {
    let mut recovery_system = ErrorRecoverySystem::new();

    // Test successful recovery
    recovery_system.update_success_rate(RecoveryAction::IncreasePrecision, true);
    let rate = recovery_system
        .success_rates
        .get(&RecoveryAction::IncreasePrecision);
    assert!(rate.is_some());
    assert!(*rate.expect("Operation failed") > 0.5);

    // Test failed recovery
    recovery_system.update_success_rate(RecoveryAction::ReduceTileSize, false);
    let rate = recovery_system
        .success_rates
        .get(&RecoveryAction::ReduceTileSize);
    assert!(rate.is_some());
    assert!(*rate.expect("Operation failed") < 0.5);
}

#[test]
fn test_performance_accuracy_analyzer() {
    let params = TradeOffParams {
        performance_weight: 0.7,
        accuracy_weight: 0.3,
        energy_weight: 0.0,
        min_accuracy: 0.9,
        max_time: Duration::from_secs(10),
        objective: OptimizationObjective::Balanced,
    };

    let mut analyzer = PerformanceAccuracyAnalyzer::new(params);

    // Record some performance data
    analyzer.record_performance(PrecisionMode::Mixed16, Duration::from_millis(100));
    analyzer.record_performance(PrecisionMode::Full32, Duration::from_millis(200));

    // Record some accuracy data
    analyzer.record_accuracy(PrecisionMode::Mixed16, 0.95);
    analyzer.record_accuracy(PrecisionMode::Full32, 0.99);

    // Test optimization
    let optimal_precision = analyzer.optimize_precision();
    assert!(matches!(
        optimal_precision,
        PrecisionMode::Mixed16 | PrecisionMode::Full32
    ));
}

#[test]
fn test_pareto_frontier_update() {
    let params = TradeOffParams {
        performance_weight: 0.5,
        accuracy_weight: 0.5,
        energy_weight: 0.0,
        min_accuracy: 0.8,
        max_time: Duration::from_secs(5),
        objective: OptimizationObjective::Balanced,
    };

    let mut analyzer = PerformanceAccuracyAnalyzer::new(params);

    // Add data for multiple precision modes
    analyzer.record_performance(PrecisionMode::Int8Dynamic, Duration::from_millis(50));
    analyzer.record_accuracy(PrecisionMode::Int8Dynamic, 0.85);

    analyzer.record_performance(PrecisionMode::Mixed16, Duration::from_millis(100));
    analyzer.record_accuracy(PrecisionMode::Mixed16, 0.95);

    analyzer.record_performance(PrecisionMode::Full32, Duration::from_millis(200));
    analyzer.record_accuracy(PrecisionMode::Full32, 0.99);

    analyzer.update_pareto_frontier();
    assert!(!analyzer.pareto_frontier.is_empty());
    assert_eq!(analyzer.pareto_frontier.len(), 3);
}

#[test]
fn test_weighted_score_computation() {
    let params = TradeOffParams {
        performance_weight: 0.6,
        accuracy_weight: 0.4,
        energy_weight: 0.0,
        min_accuracy: 0.8,
        max_time: Duration::from_secs(5),
        objective: OptimizationObjective::Custom,
    };

    let mut analyzer = PerformanceAccuracyAnalyzer::new(params);

    // Test different performance-accuracy combinations
    let score1 = analyzer.compute_weighted_score(0.1, 0.9); // Fast, accurate
    let score2 = analyzer.compute_weighted_score(0.2, 0.95); // Slower, more accurate

    assert!(score1 > 0.0);
    assert!(score2 > 0.0);
}

#[test]
fn test_advanced_tensor_core_distance_matrix_creation() {
    let result = AdvancedTensorCoreDistanceMatrix::new();
    assert!(result.is_ok());

    let advanced_computer = result.expect("Operation failed");
    assert!(advanced_computer.dynamic_precision_enabled);
    assert!(advanced_computer.auto_recovery_enabled);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_stability_monitoring_computation() {
    let mut advanced_computer = AdvancedTensorCoreDistanceMatrix::new().expect("Operation failed");
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

    let result = advanced_computer
        .compute_with_stability_monitoring(&points.view())
        .await;
    assert!(result.is_ok());

    let distances = result.expect("Operation failed");
    assert_eq!(distances.shape(), &[3, 3]);

    // Check that stability monitoring was performed
    let monitor = advanced_computer
        .stability_monitor
        .lock()
        .expect("Operation failed");
    assert!(!monitor.stability_history.is_empty());
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_recovery_action_application() {
    let mut advanced_computer = AdvancedTensorCoreDistanceMatrix::new().expect("Operation failed");
    let original_precision = advanced_computer.base_computer.precision_mode;

    // Test precision increase recovery
    let result = advanced_computer
        .apply_recovery_action(RecoveryAction::IncreasePrecision)
        .await;
    assert!(result.is_ok());

    // Precision should have increased (unless already at max)
    if original_precision != PrecisionMode::Full32 {
        assert_ne!(
            advanced_computer.base_computer.precision_mode,
            original_precision
        );
    }

    // Test tile size reduction recovery
    let original_tile_size = advanced_computer.base_computer.tile_size;
    let result = advanced_computer
        .apply_recovery_action(RecoveryAction::ReduceTileSize)
        .await;
    assert!(result.is_ok());

    let new_tile_size = advanced_computer.base_computer.tile_size;
    assert!(new_tile_size.0 <= original_tile_size.0);
    assert!(new_tile_size.1 <= original_tile_size.1);
}

#[test]
fn test_result_accuracy_estimation() {
    let advanced_computer = AdvancedTensorCoreDistanceMatrix::new().expect("Operation failed");

    // Test with valid data
    let valid_result = array![[0.0, 1.0], [1.0, 0.0]];
    let accuracy = advanced_computer.estimate_result_accuracy(&valid_result);
    assert!(accuracy > 0.8 && accuracy <= 1.0);

    // Test with invalid data (NaN)
    let invalid_result = array![[0.0, f64::NAN], [1.0, 0.0]];
    let accuracy = advanced_computer.estimate_result_accuracy(&invalid_result);
    assert_eq!(accuracy, 0.0);

    // Test with high dynamic range data
    let high_range_result = array![[1e-10, 1e10], [1e5, 1e-5]];
    let accuracy = advanced_computer.estimate_result_accuracy(&high_range_result);
    assert!(accuracy > 0.0 && accuracy < 1.0);
}

#[test]
fn test_precision_mode_ordering() {
    // Test AdvancedAdaptive mode
    assert!(matches!(
        PrecisionMode::AdvancedAdaptive,
        PrecisionMode::AdvancedAdaptive
    ));
    assert_ne!(PrecisionMode::AdvancedAdaptive, PrecisionMode::Adaptive);
}

#[test]
fn test_stability_levels() {
    assert!(matches!(StabilityLevel::Critical, StabilityLevel::Critical));
    assert_ne!(StabilityLevel::Critical, StabilityLevel::Excellent);
}

#[test]
fn test_error_types() {
    let error_types = [
        NumericalErrorType::Overflow,
        NumericalErrorType::Underflow,
        NumericalErrorType::PrecisionLoss,
        NumericalErrorType::ConvergenceFailure,
        NumericalErrorType::IllConditioned,
        NumericalErrorType::InvalidValues,
    ];

    assert_eq!(error_types.len(), 6);
    assert!(error_types.contains(&NumericalErrorType::Overflow));
}

#[test]
fn test_scaling_strategies() {
    let strategies = [
        ScalingStrategy::Conservative,
        ScalingStrategy::Balanced,
        ScalingStrategy::Aggressive,
        ScalingStrategy::Custom,
    ];

    assert_eq!(strategies.len(), 4);
    assert!(strategies.contains(&ScalingStrategy::Balanced));
}

#[test]
fn test_recovery_actions() {
    let actions = [
        RecoveryAction::IncreasePrecision,
        RecoveryAction::ReduceTileSize,
        RecoveryAction::FallbackAlgorithm,
        RecoveryAction::NumericalStabilization,
        RecoveryAction::RetryWithNewParams,
        RecoveryAction::SwitchToCPU,
    ];

    assert_eq!(actions.len(), 6);
    assert!(actions.contains(&RecoveryAction::IncreasePrecision));
}

#[test]
fn test_optimization_objectives() {
    let objectives = [
        OptimizationObjective::MaxPerformance,
        OptimizationObjective::MaxAccuracy,
        OptimizationObjective::Balanced,
        OptimizationObjective::MinEnergy,
        OptimizationObjective::Custom,
    ];

    assert_eq!(objectives.len(), 5);
    assert!(objectives.contains(&OptimizationObjective::Balanced));
}
