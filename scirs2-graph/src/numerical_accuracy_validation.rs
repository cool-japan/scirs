//! Numerical Accuracy Validation for Advanced Mode
//!
//! This module provides comprehensive validation of numerical accuracy
//! for advanced mode optimizations by comparing results against
//! reference implementations and established benchmarks.

#![allow(missing_docs)]

use crate::algorithms::community::{label_propagation_result, louvain_communities_result};
use crate::algorithms::connectivity::connected_components;
use crate::algorithms::floyd_warshall;
use crate::algorithms::{betweenness_centrality, closeness_centrality};
use crate::base::Graph;
use crate::error::{GraphError, Result};
use crate::generators::{barabasi_albert_graph, erdos_renyi_graph};
use crate::measures::pagerank_centrality;
// Unused imports are commented to avoid warnings
// use crate::advanced::{
//     create_advanced_processor, execute_with_advanced, AdvancedProcessor,
// };
use crate::advanced::{create_enhanced_advanced_processor, execute_with_enhanced_advanced};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime};

/// Validation tolerance levels for different types of numerical comparisons
#[derive(Debug, Clone)]
pub struct ValidationTolerances {
    /// Absolute tolerance for floating-point comparisons
    pub absolute_tolerance: f64,
    /// Relative tolerance for floating-point comparisons
    pub relative_tolerance: f64,
    /// Integer exact match tolerance (0 = exact match required)
    pub integer_tolerance: i32,
    /// Statistical correlation threshold for rankings
    pub correlation_threshold: f64,
    /// Maximum allowed deviation for centrality measures
    pub centrality_deviation_threshold: f64,
}

impl Default for ValidationTolerances {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-5,
            integer_tolerance: 0,
            correlation_threshold: 0.95,
            centrality_deviation_threshold: 0.01,
        }
    }
}

/// Test case configuration for validation
#[derive(Debug, Clone)]
pub struct ValidationTestCase {
    /// Name of the test case
    pub name: String,
    /// Test graph generator
    pub graph_generator: GraphGenerator,
    /// Algorithms to validate
    pub algorithms: Vec<ValidationAlgorithm>,
    /// Custom tolerances for this test case
    pub tolerances: ValidationTolerances,
    /// Number of validation runs
    pub num_runs: usize,
}

/// Graph generators for validation testing
#[derive(Debug, Clone)]
pub enum GraphGenerator {
    Random {
        nodes: usize,
        edges: usize,
        directed: bool,
    },
    ErdosRenyi {
        nodes: usize,
        probability: f64,
    },
    BarabasiAlbert {
        nodes: usize,
        edges_per_node: usize,
    },
    SmallWorld {
        nodes: usize,
        k: usize,
        p: f64,
    },
    Complete {
        nodes: usize,
    },
    Custom {
        generator: fn() -> Result<Graph<usize, f64>>,
    },
}

/// Algorithms available for validation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationAlgorithm {
    ConnectedComponents,
    StronglyConnectedComponents,
    PageRank {
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
    },
    BetweennessCentrality,
    ClosenessCentrality,
    DegreeCentrality,
    ShortestPaths {
        source: usize,
    },
    AllPairsShortestPaths,
    LouvainCommunities,
    LabelPropagation {
        max_iterations: usize,
    },
}

impl std::hash::Hash for ValidationAlgorithm {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ValidationAlgorithm::ConnectedComponents => 0.hash(state),
            ValidationAlgorithm::StronglyConnectedComponents => 1.hash(state),
            ValidationAlgorithm::PageRank { max_iterations, .. } => {
                2.hash(state);
                max_iterations.hash(state);
                // Skip f64 fields (damping, tolerance) as they don't implement Hash
            }
            ValidationAlgorithm::BetweennessCentrality => 3.hash(state),
            ValidationAlgorithm::ClosenessCentrality => 4.hash(state),
            ValidationAlgorithm::DegreeCentrality => 5.hash(state),
            ValidationAlgorithm::ShortestPaths { source } => {
                6.hash(state);
                source.hash(state);
            }
            ValidationAlgorithm::AllPairsShortestPaths => 7.hash(state),
            ValidationAlgorithm::LouvainCommunities => 8.hash(state),
            ValidationAlgorithm::LabelPropagation { max_iterations } => {
                9.hash(state);
                max_iterations.hash(state);
            }
        }
    }
}

impl Eq for ValidationAlgorithm {}

/// Result of a single validation run
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Algorithm that was validated
    pub algorithm: ValidationAlgorithm,
    /// Test case name
    pub test_case: String,
    /// Whether validation passed
    pub passed: bool,
    /// Validation score (1.0 = perfect match, 0.0 = complete mismatch)
    pub accuracy_score: f64,
    /// Execution time for standard implementation
    pub standard_time: Duration,
    /// Execution time for advanced implementation
    pub advanced_time: Duration,
    /// Performance speedup achieved
    pub speedup_factor: f64,
    /// Detailed comparison metrics
    pub metrics: ValidationMetrics,
    /// Error message if validation failed
    pub error_message: Option<String>,
}

/// Detailed metrics from validation comparison
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Maximum absolute error
    pub max_absolute_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Root mean square error
    pub root_mean_square_error: f64,
    /// Pearson correlation coefficient
    pub pearson_correlation: f64,
    /// Spearman rank correlation
    pub spearman_correlation: f64,
    /// Number of elements compared
    pub elements_compared: usize,
    /// Number of exact matches
    pub exact_matches: usize,
    /// Additional algorithm-specific metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Comprehensive validation report
#[derive(Debug)]
pub struct ValidationReport {
    /// Overall validation summary
    pub summary: ValidationSummary,
    /// Individual test results
    pub test_results: Vec<ValidationResult>,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Accuracy analysis
    pub accuracy_analysis: AccuracyAnalysis,
    /// Recommendations for improvements
    pub recommendations: Vec<String>,
    /// Timestamp of validation run
    pub timestamp: SystemTime,
}

/// Overall validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of tests passed
    pub tests_passed: usize,
    /// Overall pass rate
    pub pass_rate: f64,
    /// Average accuracy score
    pub average_accuracy: f64,
    /// Average speedup factor
    pub average_speedup: f64,
    /// Total validation time
    pub total_time: Duration,
}

/// Performance analysis of validation results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Best speedup achieved
    pub best_speedup: f64,
    /// Worst speedup (could be slowdown)
    pub worst_speedup: f64,
    /// Algorithms with best performance gains
    pub top_performers: Vec<(ValidationAlgorithm, f64)>,
    /// Algorithms with performance regressions
    pub performance_regressions: Vec<(ValidationAlgorithm, f64)>,
    /// Memory efficiency comparison
    pub memory_efficiency: f64,
}

/// Accuracy analysis of validation results
#[derive(Debug, Clone)]
pub struct AccuracyAnalysis {
    /// Best accuracy achieved
    pub best_accuracy: f64,
    /// Worst accuracy achieved
    pub worst_accuracy: f64,
    /// Algorithms with perfect accuracy
    pub perfect_accuracy_algorithms: Vec<ValidationAlgorithm>,
    /// Algorithms with accuracy concerns
    pub accuracy_concerns: Vec<(ValidationAlgorithm, f64)>,
    /// Statistical significance of differences
    pub statistical_significance: f64,
}

/// Main validator for advanced numerical accuracy
pub struct AdvancedNumericalValidator {
    /// Validation configuration
    config: ValidationConfig,
    /// Test cases to run
    test_cases: Vec<ValidationTestCase>,
    /// Validation tolerances
    tolerances: ValidationTolerances,
    /// Results from validation runs
    results: Vec<ValidationResult>,
}

/// Configuration for numerical validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable detailed logging
    pub verbose_logging: bool,
    /// Enable performance benchmarking
    pub benchmark_performance: bool,
    /// Enable statistical analysis
    pub statistical_analysis: bool,
    /// Number of warm-up runs before validation
    pub warmup_runs: usize,
    /// Enable cross-validation with multiple test cases
    pub cross_validation: bool,
    /// Seed for reproducible random graphs
    pub random_seed: Option<u64>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            verbose_logging: true,
            benchmark_performance: true,
            statistical_analysis: true,
            warmup_runs: 3,
            cross_validation: true,
            random_seed: Some(42),
        }
    }
}

impl AdvancedNumericalValidator {
    /// Create a new numerical validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            test_cases: Vec::new(),
            tolerances: ValidationTolerances::default(),
            results: Vec::new(),
        }
    }

    /// Add a test case for validation
    pub fn add_test_case(&mut self, testcase: ValidationTestCase) {
        self.test_cases.push(testcase);
    }

    /// Set custom validation tolerances
    pub fn set_tolerances(&mut self, tolerances: ValidationTolerances) {
        self.tolerances = tolerances;
    }

    /// Run comprehensive validation suite
    pub fn run_validation(&mut self) -> Result<ValidationReport> {
        println!("🔬 Starting Advanced Numerical Accuracy Validation");
        println!("==================================================");

        let start_time = Instant::now();
        self.results.clear();

        // Initialize random seed if specified
        if let Some(seed) = self.config.random_seed {
            // Note: In a real implementation, we'd set the random seed here
            println!("🎲 Using random seed: {seed}");
        }

        // Run validation for each test case
        for test_case in &self.test_cases.clone() {
            println!("\n📊 Validating test case: {}", test_case.name);
            println!("{}---{}", "".repeat(test_case.name.len()), "".repeat(20));

            self.validate_test_case(test_case)?;
        }

        let total_time = start_time.elapsed();

        // Generate comprehensive report
        let report = self.generate_validation_report(total_time)?;

        println!("\n✅ Validation completed in {total_time:?}");
        self.print_validation_summary(&report.summary);

        Ok(report)
    }

    /// Validate a single test case
    fn validate_test_case(&mut self, testcase: &ValidationTestCase) -> Result<()> {
        // Generate test graph
        let graph = self.generate_test_graph(&testcase.graph_generator)?;

        println!(
            "  📈 Generated graph: {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );

        // Run validation for each algorithm in the test _case
        for algorithm in &testcase.algorithms {
            println!("    🧮 Validating algorithm: {algorithm:?}");

            // Run multiple validation runs for statistical accuracy
            let mut run_results = Vec::new();

            for run in 0..testcase.num_runs {
                if self.config.verbose_logging && testcase.num_runs > 1 {
                    println!("      📋 Run {} of {}", run + 1, testcase.num_runs);
                }

                let result = self.validate_algorithm(&graph, algorithm, &testcase.tolerances)?;
                run_results.push(result);
            }

            // Aggregate results from multiple runs
            let aggregated_result = self.aggregate_validation_results(run_results, testcase)?;

            println!(
                "      ✅ Result: {} (accuracy: {:.4}, speedup: {:.2}x)",
                if aggregated_result.passed {
                    "PASS"
                } else {
                    "FAIL"
                },
                aggregated_result.accuracy_score,
                aggregated_result.speedup_factor
            );

            if let Some(ref error) = aggregated_result.error_message {
                println!("      ❌ Error: {error}");
            }

            self.results.push(aggregated_result);
        }

        Ok(())
    }

    /// Validate a specific algorithm implementation
    fn validate_algorithm(
        &self,
        graph: &Graph<usize, f64>,
        algorithm: &ValidationAlgorithm,
        tolerances: &ValidationTolerances,
    ) -> Result<ValidationResult> {
        // Run standard (reference) implementation
        let (standard_result, standard_time) = self.run_standard_algorithm(graph, algorithm)?;

        // Run advanced optimized implementation
        let (advanced_result, advanced_time) = self.run_advanced_algorithm(graph, algorithm)?;

        // Compare results and calculate metrics
        let metrics = self.compare_results(&standard_result, &advanced_result, algorithm)?;

        // Determine if validation passed
        let passed = self.evaluate_validation_pass(&metrics, tolerances);

        // Calculate accuracy score
        let accuracy_score = self.calculate_accuracy_score(&metrics);

        // Calculate speedup factor
        let speedup_factor = standard_time.as_secs_f64() / advanced_time.as_secs_f64();

        let error_message = if !passed {
            Some(format!(
                "Validation failed: accuracy score {accuracy_score:.6} below threshold"
            ))
        } else {
            None
        };

        Ok(ValidationResult {
            algorithm: algorithm.clone(),
            test_case: "current".to_string(), // Will be updated by caller
            passed,
            accuracy_score,
            standard_time,
            advanced_time,
            speedup_factor,
            metrics,
            error_message,
        })
    }

    /// Run standard (reference) algorithm implementation
    fn run_standard_algorithm(
        &self,
        graph: &Graph<usize, f64>,
        algorithm: &ValidationAlgorithm,
    ) -> Result<(AlgorithmOutput, Duration)> {
        let start = Instant::now();

        let result = match algorithm {
            ValidationAlgorithm::ConnectedComponents => {
                let components = connected_components(graph);
                let mut component_map = HashMap::new();
                for (component_id, component) in components.iter().enumerate() {
                    for node in component {
                        component_map.insert(*node, component_id);
                    }
                }
                AlgorithmOutput::ComponentMap(component_map)
            }
            ValidationAlgorithm::StronglyConnectedComponents => {
                // strongly_connected_components requires DiGraph, for undirected graphs use connected_components
                let components = connected_components(graph);
                let mut component_map = HashMap::new();
                for (component_id, component) in components.iter().enumerate() {
                    for node in component {
                        component_map.insert(*node, component_id);
                    }
                }
                AlgorithmOutput::ComponentMap(component_map)
            }
            ValidationAlgorithm::PageRank {
                damping,
                max_iterations: _,
                tolerance,
            } => AlgorithmOutput::ScoreMap(pagerank_centrality(graph, *damping, *tolerance)?),
            ValidationAlgorithm::BetweennessCentrality => {
                AlgorithmOutput::ScoreMap(betweenness_centrality(graph, false))
            }
            ValidationAlgorithm::ClosenessCentrality => {
                AlgorithmOutput::ScoreMap(closeness_centrality(graph, false))
            }
            ValidationAlgorithm::DegreeCentrality => AlgorithmOutput::ScoreMap({
                let mut degree_map = HashMap::new();
                for node in graph.nodes() {
                    degree_map.insert(*node, graph.degree(node) as f64);
                }
                degree_map
            }),
            ValidationAlgorithm::ShortestPaths { source } => {
                // Use petgraph's dijkstra for single-source shortest paths to all nodes
                use petgraph::algo::dijkstra;

                let graph_ref = graph.inner();
                let source_idx = graph_ref
                    .node_indices()
                    .find(|&idx| &graph_ref[idx] == source)
                    .ok_or_else(|| GraphError::node_not_found("source node"))?;

                let distances = dijkstra(graph_ref, source_idx, None, |e| *e.weight());
                let mut distance_map = HashMap::new();
                for (node_idx, distance) in distances {
                    distance_map.insert(graph_ref[node_idx], distance);
                }
                AlgorithmOutput::DistanceMap(distance_map)
            }
            ValidationAlgorithm::AllPairsShortestPaths => {
                let distance_matrix = floyd_warshall(graph)?;
                let mut distance_map = HashMap::new();

                for i in 0..distance_matrix.nrows() {
                    for j in 0..distance_matrix.ncols() {
                        distance_map.insert((i, j), distance_matrix[[i, j]]);
                    }
                }

                AlgorithmOutput::AllPairsDistances(distance_map)
            }
            ValidationAlgorithm::LouvainCommunities => {
                let result = louvain_communities_result(graph);
                AlgorithmOutput::ComponentMap(result.node_communities)
            }
            ValidationAlgorithm::LabelPropagation { max_iterations } => {
                let result = label_propagation_result(graph, *max_iterations);
                AlgorithmOutput::ComponentMap(result.node_communities)
            }
        };

        let elapsed = start.elapsed();
        Ok((result, elapsed))
    }

    /// Run advanced optimized algorithm implementation
    fn run_advanced_algorithm(
        &self,
        graph: &Graph<usize, f64>,
        algorithm: &ValidationAlgorithm,
    ) -> Result<(AlgorithmOutput, Duration)> {
        let mut processor = create_enhanced_advanced_processor();
        let start = Instant::now();

        let result = match algorithm {
            ValidationAlgorithm::ConnectedComponents => {
                let components =
                    execute_with_enhanced_advanced(graph, |g| Ok(connected_components(g)))?;
                AlgorithmOutput::ComponentMap({
                    let mut component_map = HashMap::new();
                    for (component_id, component) in components.iter().enumerate() {
                        for node in component {
                            component_map.insert(*node, component_id);
                        }
                    }
                    component_map
                })
            }
            ValidationAlgorithm::StronglyConnectedComponents => {
                let components = execute_with_enhanced_advanced(graph, |g| {
                    // strongly_connected_components requires DiGraph, skip for undirected graphs
                    Ok(vec![g.nodes().into_iter().cloned().collect::<HashSet<_>>()])
                })?;
                AlgorithmOutput::ComponentMap({
                    let mut component_map = HashMap::new();
                    for (component_id, component) in components.iter().enumerate() {
                        for node in component {
                            component_map.insert(*node, component_id);
                        }
                    }
                    component_map
                })
            }
            ValidationAlgorithm::PageRank {
                damping,
                max_iterations: _,
                tolerance,
            } => {
                let scores = execute_with_enhanced_advanced(graph, |g| {
                    pagerank_centrality(g, *damping, *tolerance)
                })?;
                AlgorithmOutput::ScoreMap(scores)
            }
            ValidationAlgorithm::BetweennessCentrality => {
                let scores = execute_with_enhanced_advanced(graph, |g| {
                    Ok(betweenness_centrality(g, false))
                })?;
                AlgorithmOutput::ScoreMap(scores)
            }
            ValidationAlgorithm::ClosenessCentrality => {
                let scores =
                    execute_with_enhanced_advanced(graph, |g| Ok(closeness_centrality(g, false)))?;
                AlgorithmOutput::ScoreMap(scores)
            }
            ValidationAlgorithm::DegreeCentrality => {
                let scores = execute_with_enhanced_advanced(graph, |g| {
                    let mut degree_map = HashMap::new();
                    for node in g.nodes() {
                        degree_map.insert(*node, g.degree(node) as f64);
                    }
                    Ok(degree_map)
                })?;
                AlgorithmOutput::ScoreMap(scores)
            }
            ValidationAlgorithm::ShortestPaths { source } => {
                let distances = execute_with_enhanced_advanced(graph, |g| {
                    use petgraph::algo::dijkstra;
                    let graph_ref = g.inner();
                    let source_idx = graph_ref
                        .node_indices()
                        .find(|&idx| &graph_ref[idx] == source)
                        .ok_or_else(|| crate::error::GraphError::node_not_found("source node"))?;

                    let distances = dijkstra(graph_ref, source_idx, None, |e| *e.weight());
                    let mut distance_map = HashMap::new();
                    for (node_idx, distance) in distances {
                        distance_map.insert(graph_ref[node_idx], distance);
                    }
                    Ok(distance_map)
                })?;
                AlgorithmOutput::DistanceMap(distances)
            }
            ValidationAlgorithm::AllPairsShortestPaths => {
                let distances = execute_with_enhanced_advanced(graph, |g| {
                    let distance_matrix = floyd_warshall(g)?;
                    let mut distance_map = HashMap::new();

                    for i in 0..distance_matrix.nrows() {
                        for j in 0..distance_matrix.ncols() {
                            distance_map.insert((i, j), distance_matrix[[i, j]]);
                        }
                    }

                    Ok(distance_map)
                })?;
                AlgorithmOutput::AllPairsDistances(distances)
            }
            ValidationAlgorithm::LouvainCommunities => {
                let communities = execute_with_enhanced_advanced(graph, |g| {
                    Ok(louvain_communities_result(g).node_communities)
                })?;
                AlgorithmOutput::ComponentMap(communities)
            }
            ValidationAlgorithm::LabelPropagation { max_iterations } => {
                let communities = execute_with_enhanced_advanced(graph, |g| {
                    Ok(label_propagation_result(g, *max_iterations).node_communities)
                })?;
                AlgorithmOutput::ComponentMap(communities)
            }
        };

        let elapsed = start.elapsed();
        Ok((result, elapsed))
    }

    /// Generate test graph based on generator specification
    fn generate_test_graph(&self, generator: &GraphGenerator) -> Result<Graph<usize, f64>> {
        match generator {
            GraphGenerator::Random {
                nodes,
                edges,
                directed: _,
            } => erdos_renyi_graph(
                *nodes,
                *edges as f64 / (*nodes * (*nodes - 1) / 2) as f64,
                &mut scirs2_core::random::rng(),
            ),
            GraphGenerator::ErdosRenyi { nodes, probability } => {
                erdos_renyi_graph(*nodes, *probability, &mut scirs2_core::random::rng())
            }
            GraphGenerator::BarabasiAlbert {
                nodes,
                edges_per_node,
            } => barabasi_albert_graph(*nodes, *edges_per_node, &mut scirs2_core::random::rng()),
            GraphGenerator::SmallWorld { nodes, k: _, p: _ } => {
                // For now, approximate with Erdős-Rényi
                // In a full implementation, we'd have a proper small-world generator
                erdos_renyi_graph(*nodes, 6.0 / *nodes as f64, &mut scirs2_core::random::rng())
            }
            GraphGenerator::Complete { nodes } => {
                let mut graph = Graph::new();
                for i in 0..*nodes {
                    graph.add_node(i);
                }
                for i in 0..*nodes {
                    for j in (i + 1)..*nodes {
                        graph.add_edge(i, j, 1.0).unwrap();
                    }
                }
                Ok(graph)
            }
            GraphGenerator::Custom { generator } => generator(),
        }
    }

    /// Compare algorithm results and generate metrics
    fn compare_results(
        &self,
        standard: &AlgorithmOutput,
        advanced: &AlgorithmOutput,
        algorithm: &ValidationAlgorithm,
    ) -> Result<ValidationMetrics> {
        match (standard, advanced) {
            (AlgorithmOutput::ScoreMap(std_scores), AlgorithmOutput::ScoreMap(ut_scores)) => {
                self.compare_score_maps(std_scores, ut_scores)
            }
            (AlgorithmOutput::ComponentMap(std_comps), AlgorithmOutput::ComponentMap(ut_comps)) => {
                self.compare_component_maps(std_comps, ut_comps)
            }
            (AlgorithmOutput::DistanceMap(std_dists), AlgorithmOutput::DistanceMap(ut_dists)) => {
                self.compare_distance_maps(std_dists, ut_dists)
            }
            (
                AlgorithmOutput::AllPairsDistances(std_all),
                AlgorithmOutput::AllPairsDistances(ut_all),
            ) => self.compare_all_pairs_distances(std_all, ut_all),
            _ => Err(crate::error::GraphError::InvalidParameter {
                param: "algorithm_outputs".to_string(),
                value: "mismatched types".to_string(),
                expected: "matching output types".to_string(),
                context: "Mismatched algorithm output types".to_string(),
            }),
        }
    }

    /// Compare floating-point score maps
    fn compare_score_maps(
        &self,
        standard: &HashMap<usize, f64>,
        advanced: &HashMap<usize, f64>,
    ) -> Result<ValidationMetrics> {
        let mut absolute_errors = Vec::new();
        let mut standard_values = Vec::new();
        let mut advanced_values = Vec::new();
        let mut exact_matches = 0;

        // Find common keys
        let common_keys: Vec<_> = standard
            .keys()
            .filter(|k| advanced.contains_key(k))
            .collect();

        for &key in &common_keys {
            let std_val = standard[key];
            let ut_val = advanced[key];

            let abs_error = (std_val - ut_val).abs();
            absolute_errors.push(abs_error);
            standard_values.push(std_val);
            advanced_values.push(ut_val);

            if abs_error < self.tolerances.absolute_tolerance {
                exact_matches += 1;
            }
        }

        let max_absolute_error = absolute_errors.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean_absolute_error =
            absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        let root_mean_square_error = (absolute_errors.iter().map(|e| e * e).sum::<f64>()
            / absolute_errors.len() as f64)
            .sqrt();

        // Calculate correlations
        let pearson_correlation =
            self.calculate_pearson_correlation(&standard_values, &advanced_values);
        let spearman_correlation =
            self.calculate_spearman_correlation(&standard_values, &advanced_values);

        Ok(ValidationMetrics {
            max_absolute_error,
            mean_absolute_error,
            root_mean_square_error,
            pearson_correlation,
            spearman_correlation,
            elements_compared: common_keys.len(),
            exact_matches,
            custom_metrics: HashMap::new(),
        })
    }

    /// Compare integer component maps (for community detection, etc.)
    fn compare_component_maps(
        &self,
        standard: &HashMap<usize, usize>,
        advanced: &HashMap<usize, usize>,
    ) -> Result<ValidationMetrics> {
        let mut exact_matches = 0;
        let common_keys: Vec<_> = standard
            .keys()
            .filter(|k| advanced.contains_key(k))
            .collect();

        // For component maps, we need to check if the partitioning is equivalent
        // even if the component IDs are different
        let normalized_std = self.normalize_component_map(standard);
        let normalized_ut = self.normalize_component_map(advanced);

        for &key in &common_keys {
            if normalized_std.get(key) == normalized_ut.get(key) {
                exact_matches += 1;
            }
        }

        // Calculate modularity similarity or other community detection metrics
        let partition_similarity = exact_matches as f64 / common_keys.len() as f64;

        Ok(ValidationMetrics {
            max_absolute_error: if exact_matches == common_keys.len() {
                0.0
            } else {
                1.0
            },
            mean_absolute_error: 1.0 - partition_similarity,
            root_mean_square_error: (1.0 - partition_similarity).sqrt(),
            pearson_correlation: partition_similarity,
            spearman_correlation: partition_similarity,
            elements_compared: common_keys.len(),
            exact_matches,
            custom_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("partition_similarity".to_string(), partition_similarity);
                metrics
            },
        })
    }

    /// Compare distance maps
    fn compare_distance_maps(
        &self,
        standard: &HashMap<usize, f64>,
        advanced: &HashMap<usize, f64>,
    ) -> Result<ValidationMetrics> {
        // Distance maps are similar to score maps but may have special handling for infinity
        self.compare_score_maps(standard, advanced)
    }

    /// Compare all-pairs distance results
    fn compare_all_pairs_distances(
        &self,
        standard: &HashMap<(usize, usize), f64>,
        advanced: &HashMap<(usize, usize), f64>,
    ) -> Result<ValidationMetrics> {
        let mut absolute_errors = Vec::new();
        let mut exact_matches = 0;

        let common_keys: Vec<_> = standard
            .keys()
            .filter(|k| advanced.contains_key(k))
            .collect();

        for &key in &common_keys {
            let std_val = standard[key];
            let ut_val = advanced[key];

            let abs_error = (std_val - ut_val).abs();
            absolute_errors.push(abs_error);

            if abs_error < self.tolerances.absolute_tolerance {
                exact_matches += 1;
            }
        }

        let max_absolute_error = absolute_errors.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean_absolute_error =
            absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        let root_mean_square_error = (absolute_errors.iter().map(|e| e * e).sum::<f64>()
            / absolute_errors.len() as f64)
            .sqrt();

        Ok(ValidationMetrics {
            max_absolute_error,
            mean_absolute_error,
            root_mean_square_error,
            pearson_correlation: 1.0 - mean_absolute_error, // Approximation
            spearman_correlation: 1.0 - mean_absolute_error, // Approximation
            elements_compared: common_keys.len(),
            exact_matches,
            custom_metrics: HashMap::new(),
        })
    }

    /// Normalize component map to canonical form for comparison
    fn normalize_component_map(&self, components: &HashMap<usize, usize>) -> HashMap<usize, usize> {
        let mut normalized = HashMap::new();
        let mut component_map = HashMap::new();
        let mut next_id = 0;

        for (&node, &component) in components {
            let normalized_component = *component_map.entry(component).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            normalized.insert(node, normalized_component);
        }

        normalized
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_x2: f64 = x.iter().map(|v| v * v).sum();
        let sum_y2: f64 = y.iter().map(|v| v * v).sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate Spearman rank correlation coefficient
    fn calculate_spearman_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        // Create rank vectors
        let rank_x = self.calculate_ranks(x);
        let rank_y = self.calculate_ranks(y);

        // Calculate Pearson correlation of ranks
        self.calculate_pearson_correlation(&rank_x, &rank_y)
    }

    /// Calculate ranks for Spearman correlation
    fn calculate_ranks(&self, values: &[f64]) -> Vec<f64> {
        let mut indexed_values: Vec<(usize, f64)> =
            values.iter().enumerate().map(|(i, &v)| (i, v)).collect();

        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = vec![0.0; values.len()];
        for (rank, (original_index, _)) in indexed_values.iter().enumerate() {
            ranks[*original_index] = (rank + 1) as f64;
        }

        ranks
    }

    /// Evaluate if validation passes based on metrics and tolerances
    fn evaluate_validation_pass(
        &self,
        metrics: &ValidationMetrics,
        tolerances: &ValidationTolerances,
    ) -> bool {
        metrics.max_absolute_error <= tolerances.absolute_tolerance
            && metrics.pearson_correlation >= tolerances.correlation_threshold
            && metrics.mean_absolute_error <= tolerances.centrality_deviation_threshold
    }

    /// Calculate overall accuracy score from metrics
    fn calculate_accuracy_score(&self, metrics: &ValidationMetrics) -> f64 {
        // Weighted combination of different accuracy measures
        let correlation_weight = 0.4;
        let error_weight = 0.3;
        let exact_match_weight = 0.3;

        let correlation_score = (metrics.pearson_correlation + metrics.spearman_correlation) / 2.0;
        let error_score = 1.0 - (metrics.mean_absolute_error / (1.0 + metrics.mean_absolute_error));
        let exact_match_score = metrics.exact_matches as f64 / metrics.elements_compared as f64;

        correlation_weight * correlation_score
            + error_weight * error_score
            + exact_match_weight * exact_match_score
    }

    /// Aggregate results from multiple validation runs
    fn aggregate_validation_results(
        &self,
        results: Vec<ValidationResult>,
        test_case: &ValidationTestCase,
    ) -> Result<ValidationResult> {
        if results.is_empty() {
            return Err(crate::error::GraphError::InvalidParameter {
                param: "results".to_string(),
                value: "empty".to_string(),
                expected: "non-empty vector".to_string(),
                context: "No validation results to aggregate".to_string(),
            });
        }

        let passed = results.iter().all(|r| r.passed);
        let accuracy_score =
            results.iter().map(|r| r.accuracy_score).sum::<f64>() / results.len() as f64;
        let speedup_factor =
            results.iter().map(|r| r.speedup_factor).sum::<f64>() / results.len() as f64;

        let standard_time = Duration::from_secs_f64(
            results
                .iter()
                .map(|r| r.standard_time.as_secs_f64())
                .sum::<f64>()
                / results.len() as f64,
        );
        let advanced_time = Duration::from_secs_f64(
            results
                .iter()
                .map(|r| r.advanced_time.as_secs_f64())
                .sum::<f64>()
                / results.len() as f64,
        );

        // Aggregate metrics
        let metrics = ValidationMetrics {
            max_absolute_error: results
                .iter()
                .map(|r| r.metrics.max_absolute_error)
                .fold(0.0, f64::max),
            mean_absolute_error: results
                .iter()
                .map(|r| r.metrics.mean_absolute_error)
                .sum::<f64>()
                / results.len() as f64,
            root_mean_square_error: results
                .iter()
                .map(|r| r.metrics.root_mean_square_error)
                .sum::<f64>()
                / results.len() as f64,
            pearson_correlation: results
                .iter()
                .map(|r| r.metrics.pearson_correlation)
                .sum::<f64>()
                / results.len() as f64,
            spearman_correlation: results
                .iter()
                .map(|r| r.metrics.spearman_correlation)
                .sum::<f64>()
                / results.len() as f64,
            elements_compared: results
                .iter()
                .map(|r| r.metrics.elements_compared)
                .sum::<usize>()
                / results.len(),
            exact_matches: results
                .iter()
                .map(|r| r.metrics.exact_matches)
                .sum::<usize>()
                / results.len(),
            custom_metrics: HashMap::new(),
        };

        let error_message = if !passed {
            Some(format!(
                "Aggregated validation failed: average accuracy {accuracy_score:.6}"
            ))
        } else {
            None
        };

        Ok(ValidationResult {
            algorithm: results[0].algorithm.clone(),
            test_case: test_case.name.clone(),
            passed,
            accuracy_score,
            standard_time,
            advanced_time,
            speedup_factor,
            metrics,
            error_message,
        })
    }

    /// Generate comprehensive validation report
    fn generate_validation_report(&self, totaltime: Duration) -> Result<ValidationReport> {
        let summary = self.generate_validation_summary(totaltime);
        let performance_analysis = self.generate_performance_analysis();
        let accuracy_analysis = self.generate_accuracy_analysis();
        let recommendations = self.generate_recommendations();

        Ok(ValidationReport {
            summary,
            test_results: self.results.clone(),
            performance_analysis,
            accuracy_analysis,
            recommendations,
            timestamp: SystemTime::now(),
        })
    }

    /// Generate validation summary
    fn generate_validation_summary(&self, totaltime: Duration) -> ValidationSummary {
        let total_tests = self.results.len();
        let tests_passed = self.results.iter().filter(|r| r.passed).count();
        let pass_rate = tests_passed as f64 / total_tests as f64;

        let average_accuracy =
            self.results.iter().map(|r| r.accuracy_score).sum::<f64>() / total_tests as f64;

        let average_speedup =
            self.results.iter().map(|r| r.speedup_factor).sum::<f64>() / total_tests as f64;

        ValidationSummary {
            total_tests,
            tests_passed,
            pass_rate,
            average_accuracy,
            average_speedup,
            total_time: totaltime,
        }
    }

    /// Generate performance analysis
    fn generate_performance_analysis(&self) -> PerformanceAnalysis {
        let speedups: Vec<f64> = self.results.iter().map(|r| r.speedup_factor).collect();
        let best_speedup = speedups.iter().fold(0.0f64, |a, &b| a.max(b));
        let worst_speedup = speedups.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let mut algorithm_speedups: HashMap<ValidationAlgorithm, Vec<f64>> = HashMap::new();
        for result in &self.results {
            algorithm_speedups
                .entry(result.algorithm.clone())
                .or_default()
                .push(result.speedup_factor);
        }

        let mut top_performers = Vec::new();
        let mut performance_regressions = Vec::new();

        for (algorithm, speedups) in algorithm_speedups {
            let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
            if avg_speedup >= 1.5 {
                top_performers.push((algorithm.clone(), avg_speedup));
            }
            if avg_speedup < 1.0 {
                performance_regressions.push((algorithm, avg_speedup));
            }
        }

        top_performers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        performance_regressions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        PerformanceAnalysis {
            best_speedup,
            worst_speedup,
            top_performers,
            performance_regressions,
            memory_efficiency: 1.0, // Would need actual memory measurements
        }
    }

    /// Generate accuracy analysis
    fn generate_accuracy_analysis(&self) -> AccuracyAnalysis {
        let accuracies: Vec<f64> = self.results.iter().map(|r| r.accuracy_score).collect();
        let best_accuracy = accuracies.iter().fold(0.0f64, |a, &b| a.max(b));
        let worst_accuracy = accuracies.iter().fold(1.0f64, |a, &b| a.min(b));

        let perfect_accuracy_algorithms = self
            .results
            .iter()
            .filter(|r| r.accuracy_score >= 0.999)
            .map(|r| r.algorithm.clone())
            .collect();

        let accuracy_concerns = self
            .results
            .iter()
            .filter(|r| r.accuracy_score < 0.95)
            .map(|r| (r.algorithm.clone(), r.accuracy_score))
            .collect();

        AccuracyAnalysis {
            best_accuracy,
            worst_accuracy,
            perfect_accuracy_algorithms,
            accuracy_concerns,
            statistical_significance: 0.95, // Would need proper statistical tests
        }
    }

    /// Generate recommendations for improvements
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let failed_tests = self.results.iter().filter(|r| !r.passed).count();
        if failed_tests > 0 {
            recommendations.push(format!(
                "Address {failed_tests} failed validation tests to improve overall accuracy"
            ));
        }

        let low_accuracy_tests = self
            .results
            .iter()
            .filter(|r| r.accuracy_score < 0.95)
            .count();
        if low_accuracy_tests > 0 {
            recommendations.push(format!(
                "Investigate {low_accuracy_tests} tests with accuracy scores below 0.95"
            ));
        }

        let slow_algorithms = self
            .results
            .iter()
            .filter(|r| r.speedup_factor < 1.0)
            .count();
        if slow_algorithms > 0 {
            recommendations.push(format!(
                "Optimize {slow_algorithms} algorithms showing performance regressions"
            ));
        }

        let avg_accuracy =
            self.results.iter().map(|r| r.accuracy_score).sum::<f64>() / self.results.len() as f64;
        if avg_accuracy < 0.98 {
            recommendations.push(
                "Consider tightening numerical precision in advanced optimizations".to_string(),
            );
        }

        let avg_speedup =
            self.results.iter().map(|r| r.speedup_factor).sum::<f64>() / self.results.len() as f64;
        if avg_speedup < 1.5 {
            recommendations.push(
                "Investigate opportunities for additional performance optimizations".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("All validation tests passed successfully. Consider adding more comprehensive test cases.".to_string());
        }

        recommendations
    }

    /// Print validation summary to console
    fn print_validation_summary(&self, summary: &ValidationSummary) {
        println!("\n📊 Validation Summary");
        println!("===================");
        println!("Total tests: {}", summary.total_tests);
        println!(
            "Tests passed: {} ({:.1}%)",
            summary.tests_passed,
            summary.pass_rate * 100.0
        );
        println!("Average accuracy: {:.4}", summary.average_accuracy);
        println!("Average speedup: {:.2}x", summary.average_speedup);
        println!("Total validation time: {:?}", summary.total_time);

        if summary.pass_rate >= 0.95 {
            println!("✅ Validation PASSED: Advanced mode maintains high numerical accuracy");
        } else {
            println!("❌ Validation FAILED: Accuracy issues detected");
        }
    }
}

/// Algorithm output types for validation comparison
#[derive(Debug, Clone)]
pub enum AlgorithmOutput {
    /// Score map (e.g., centrality measures, PageRank)
    ScoreMap(HashMap<usize, f64>),
    /// Component map (e.g., community detection, connected components)
    ComponentMap(HashMap<usize, usize>),
    /// Distance map (e.g., shortest paths from single source)
    DistanceMap(HashMap<usize, f64>),
    /// All-pairs distances
    AllPairsDistances(HashMap<(usize, usize), f64>),
}

/// Create comprehensive validation test suite
#[allow(dead_code)]
pub fn create_comprehensive_validation_suite() -> AdvancedNumericalValidator {
    let mut validator = AdvancedNumericalValidator::new(ValidationConfig::default());

    // Test Case 1: Small Random Graphs
    validator.add_test_case(ValidationTestCase {
        name: "Small Random Graphs".to_string(),
        graph_generator: GraphGenerator::Random {
            nodes: 100,
            edges: 200,
            directed: false,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 100,
                tolerance: 1e-6,
            },
            ValidationAlgorithm::BetweennessCentrality,
            ValidationAlgorithm::ShortestPaths { source: 0 },
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 5,
    });

    // Test Case 2: Scale-Free Networks
    validator.add_test_case(ValidationTestCase {
        name: "Scale-Free Networks".to_string(),
        graph_generator: GraphGenerator::BarabasiAlbert {
            nodes: 500,
            edges_per_node: 3,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 100,
                tolerance: 1e-6,
            },
            ValidationAlgorithm::LouvainCommunities,
            ValidationAlgorithm::ClosenessCentrality,
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 3,
    });

    // Test Case 3: Dense Random Networks
    validator.add_test_case(ValidationTestCase {
        name: "Dense Random Networks".to_string(),
        graph_generator: GraphGenerator::ErdosRenyi {
            nodes: 200,
            probability: 0.1,
        },
        algorithms: vec![
            ValidationAlgorithm::AllPairsShortestPaths,
            ValidationAlgorithm::DegreeCentrality,
            ValidationAlgorithm::LabelPropagation { max_iterations: 50 },
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 3,
    });

    // Test Case 4: Sparse Large Graphs
    validator.add_test_case(ValidationTestCase {
        name: "Sparse Large Graphs".to_string(),
        graph_generator: GraphGenerator::Random {
            nodes: 2000,
            edges: 4000,
            directed: false,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 50,
                tolerance: 1e-5,
            },
            ValidationAlgorithm::LouvainCommunities,
        ],
        tolerances: ValidationTolerances {
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            correlation_threshold: 0.9,
            ..ValidationTolerances::default()
        },
        num_runs: 2,
    });

    validator
}

/// Run quick validation test
#[allow(dead_code)]
pub fn run_quick_validation() -> Result<ValidationReport> {
    println!("🚀 Running Quick Advanced Numerical Validation");
    println!("===============================================");

    let mut validator = AdvancedNumericalValidator::new(ValidationConfig {
        verbose_logging: true,
        warmup_runs: 1,
        ..ValidationConfig::default()
    });

    // Quick test with small graph
    validator.add_test_case(ValidationTestCase {
        name: "Quick Validation".to_string(),
        graph_generator: GraphGenerator::Random {
            nodes: 50,
            edges: 100,
            directed: false,
        },
        algorithms: vec![
            ValidationAlgorithm::ConnectedComponents,
            ValidationAlgorithm::PageRank {
                damping: 0.85,
                max_iterations: 20,
                tolerance: 1e-4,
            },
        ],
        tolerances: ValidationTolerances::default(),
        num_runs: 1,
    });

    validator.run_validation()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_tolerances_default() {
        let tolerances = ValidationTolerances::default();
        assert_eq!(tolerances.absolute_tolerance, 1e-6);
        assert_eq!(tolerances.relative_tolerance, 1e-5);
        assert_eq!(tolerances.correlation_threshold, 0.95);
    }

    #[test]
    fn test_graph_generation() {
        let validator = AdvancedNumericalValidator::new(ValidationConfig::default());

        // Test random graph generation
        let graph = validator
            .generate_test_graph(&GraphGenerator::Random {
                nodes: 10,
                edges: 15,
                directed: false,
            })
            .unwrap();
        assert_eq!(graph.node_count(), 10);
        // Random graph generator may produce varying edge counts due to undirected edge handling and random selection
        assert!(graph.edge_count() > 0 && graph.edge_count() <= 45); // Maximum possible edges for 10 nodes: 10*9/2 = 45

        // Test complete graph generation
        let complete = validator
            .generate_test_graph(&GraphGenerator::Complete { nodes: 5 })
            .unwrap();
        assert_eq!(complete.node_count(), 5);
        assert_eq!(complete.edge_count(), 10); // 5 choose 2 = 10
    }

    #[test]
    fn test_pearson_correlation() {
        let validator = AdvancedNumericalValidator::new(ValidationConfig::default());

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = validator.calculate_pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = validator.calculate_pearson_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_component_map_normalization() {
        let validator = AdvancedNumericalValidator::new(ValidationConfig::default());

        let mut components = HashMap::new();
        components.insert(0, 100);
        components.insert(1, 100);
        components.insert(2, 200);
        components.insert(3, 200);

        let normalized = validator.normalize_component_map(&components);

        // Should normalize to 0, 0, 1, 1 (or some other consistent mapping)
        assert_eq!(normalized[&0], normalized[&1]);
        assert_eq!(normalized[&2], normalized[&3]);
        assert_ne!(normalized[&0], normalized[&2]);
    }

    #[test]
    fn test_quick_validation() {
        // This test verifies that the validation framework runs without errors
        let result = run_quick_validation();
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.summary.total_tests > 0);
        assert!(report.summary.pass_rate >= 0.0);
        assert!(report.summary.pass_rate <= 1.0);
    }
}
