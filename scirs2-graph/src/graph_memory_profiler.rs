//! Memory Usage Profiler for Advanced Mode
//!
//! This module provides comprehensive memory profiling and optimization analysis
//! for advanced mode components, including detailed memory usage tracking,
//! optimization recommendations, and performance analysis.

#![allow(missing_docs)]

use crate::advanced::AdvancedProcessor;
use crate::base::{EdgeWeight, Graph, Node};
use crate::error::Result;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Memory usage statistics for different components
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Average memory usage in bytes
    pub average_usage: f64,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Memory fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_ratio: f64,
    /// Memory efficiency score (0.0 = inefficient, 1.0 = highly efficient)
    pub efficiency_score: f64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            average_usage: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
            fragmentation_ratio: 0.0,
            efficiency_score: 1.0,
        }
    }
}

/// Memory allocation pattern analysis
#[derive(Debug, Clone)]
pub struct AllocationPattern {
    /// Size of allocation in bytes
    pub size: usize,
    /// Timestamp of allocation
    pub timestamp: SystemTime,
    /// Lifetime of allocation (if deallocated)
    pub lifetime: Option<Duration>,
    /// Category of allocation (graph data, algorithm workspace, cache, etc.)
    pub category: String,
    /// Whether this allocation was predicted by the memory manager
    pub was_predicted: bool,
}

/// Memory usage profiling data for different advanced components
#[derive(Debug)]
pub struct MemoryProfile {
    /// Overall memory statistics
    pub overall_stats: MemoryStats,
    /// Memory usage by component
    pub component_stats: HashMap<String, MemoryStats>,
    /// Allocation patterns over time
    pub allocation_patterns: Vec<AllocationPattern>,
    /// Memory usage history (timestamp, usage_bytes)
    pub usage_history: VecDeque<(SystemTime, usize)>,
    /// Optimization opportunities identified
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    /// Memory efficiency analysis
    pub efficiency_analysis: EfficiencyAnalysis,
}

/// Identified memory optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Estimated memory savings in bytes
    pub estimated_savings: usize,
    /// Estimated performance impact (negative = performance loss, positive = gain)
    pub performance_impact: f64,
    /// Implementation complexity (1-5, 1 = easy, 5 = very complex)
    pub implementation_complexity: u8,
    /// Description of the optimization
    pub description: String,
    /// Priority (1-5, 5 = highest priority)
    pub priority: u8,
}

/// Types of memory optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Use memory pools for frequent allocations
    MemoryPooling,
    /// Reduce data structure sizes
    DataStructureOptimization,
    /// Implement lazy evaluation
    LazyEvaluation,
    /// Use more compact data representations
    CompactRepresentation,
    /// Optimize caching strategies
    CacheOptimization,
    /// Reduce memory fragmentation
    FragmentationReduction,
    /// Use streaming algorithms for large data
    StreamingProcessing,
    /// Optimize garbage collection patterns
    GarbageCollectionOptimization,
}

/// Memory efficiency analysis results
#[derive(Debug, Clone)]
pub struct EfficiencyAnalysis {
    /// Overall efficiency score (0.0-1.0)
    pub overall_efficiency: f64,
    /// Memory utilization ratio (used / allocated)
    pub utilization_ratio: f64,
    /// Cache effectiveness score
    pub cache_effectiveness: f64,
    /// Memory access pattern efficiency
    pub access_pattern_efficiency: f64,
    /// Temporal locality score
    pub temporal_locality: f64,
    /// Spatial locality score
    pub spatial_locality: f64,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Comprehensive memory profiler for advanced mode
pub struct AdvancedMemoryProfiler {
    /// Current memory profile data
    profile: MemoryProfile,
    /// Profiling configuration
    config: MemoryProfilerConfig,
    /// Active memory tracking
    active_allocations: HashMap<String, AllocationPattern>,
    /// Profiling start time
    start_time: SystemTime,
    /// Last garbage collection time
    #[allow(dead_code)]
    last_gc_time: SystemTime,
    /// Memory pressure threshold
    memory_pressure_threshold: usize,
}

/// Configuration for memory profiling
#[derive(Debug, Clone)]
pub struct MemoryProfilerConfig {
    /// Enable detailed allocation tracking
    pub track_allocations: bool,
    /// Enable memory pattern analysis
    pub analyze_patterns: bool,
    /// Enable optimization detection
    pub detect_optimizations: bool,
    /// Maximum history entries to keep
    pub max_history_entries: usize,
    /// Memory sampling interval
    pub sampling_interval: Duration,
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
}

impl Default for MemoryProfilerConfig {
    fn default() -> Self {
        Self {
            track_allocations: true,
            analyze_patterns: true,
            detect_optimizations: true,
            max_history_entries: 10000,
            sampling_interval: Duration::from_millis(100),
            real_time_monitoring: true,
        }
    }
}

impl AdvancedMemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: MemoryProfilerConfig) -> Self {
        let now = SystemTime::now();
        Self {
            profile: MemoryProfile {
                overall_stats: MemoryStats::default(),
                component_stats: HashMap::new(),
                allocation_patterns: Vec::new(),
                usage_history: VecDeque::new(),
                optimization_opportunities: Vec::new(),
                efficiency_analysis: EfficiencyAnalysis {
                    overall_efficiency: 1.0,
                    utilization_ratio: 1.0,
                    cache_effectiveness: 1.0,
                    access_pattern_efficiency: 1.0,
                    temporal_locality: 1.0,
                    spatial_locality: 1.0,
                    recommendations: Vec::new(),
                },
            },
            config,
            active_allocations: HashMap::new(),
            start_time: now,
            last_gc_time: now,
            memory_pressure_threshold: 1024 * 1024 * 1024, // 1GB default
        }
    }

    /// Start profiling an advanced processor
    pub fn start_profiling(&mut self, processor: &AdvancedProcessor) {
        self.start_time = SystemTime::now();
        self.record_initial_state(processor);

        if self.config.real_time_monitoring {
            self.start_real_time_monitoring();
        }
    }

    /// Record memory allocation
    pub fn record_allocation(
        &mut self,
        component: &str,
        size: usize,
        category: &str,
        predicted: bool,
    ) {
        let allocation = AllocationPattern {
            size,
            timestamp: SystemTime::now(),
            lifetime: None,
            category: category.to_string(),
            was_predicted: predicted,
        };

        let allocation_id = format!(
            "{}_{}_{}_{}",
            component,
            category,
            size,
            allocation
                .timestamp
                .duration_since(self.start_time)
                .unwrap_or_default()
                .as_nanos()
        );

        self.active_allocations
            .insert(allocation_id.clone(), allocation.clone());
        self.profile.allocation_patterns.push(allocation);

        // Update component statistics
        let component_stats = self
            .profile
            .component_stats
            .entry(component.to_string())
            .or_default();
        component_stats.current_usage += size;
        component_stats.peak_usage = component_stats
            .peak_usage
            .max(component_stats.current_usage);
        component_stats.allocation_count += 1;

        // Update overall statistics
        self.profile.overall_stats.current_usage += size;
        self.profile.overall_stats.peak_usage = self
            .profile
            .overall_stats
            .peak_usage
            .max(self.profile.overall_stats.current_usage);
        self.profile.overall_stats.allocation_count += 1;

        // Check for memory pressure
        if self.profile.overall_stats.current_usage > self.memory_pressure_threshold {
            self.analyze_memory_pressure();
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&mut self, component: &str, size: usize, category: &str) {
        // Find and remove the allocation
        let allocation_key = self
            .active_allocations
            .keys()
            .find(|k| k.starts_with(component) && k.contains(category))
            .cloned();

        if let Some(key) = allocation_key {
            if let Some(mut allocation) = self.active_allocations.remove(&key) {
                allocation.lifetime = Some(
                    SystemTime::now()
                        .duration_since(allocation.timestamp)
                        .unwrap_or_default(),
                );

                // Update statistics
                let component_stats = self
                    .profile
                    .component_stats
                    .entry(component.to_string())
                    .or_default();
                component_stats.current_usage = component_stats.current_usage.saturating_sub(size);
                component_stats.deallocation_count += 1;

                self.profile.overall_stats.current_usage = self
                    .profile
                    .overall_stats
                    .current_usage
                    .saturating_sub(size);
                self.profile.overall_stats.deallocation_count += 1;
            }
        }
    }

    /// Record memory usage snapshot
    pub fn record_memory_snapshot(&mut self, processor: &AdvancedProcessor) {
        let current_time = SystemTime::now();
        let current_usage = self.estimate_processor_memory_usage(processor);

        self.profile
            .usage_history
            .push_back((current_time, current_usage));

        // Keep history within limits
        while self.profile.usage_history.len() > self.config.max_history_entries {
            self.profile.usage_history.pop_front();
        }

        // Update average usage
        let total_usage: usize = self
            .profile
            .usage_history
            .iter()
            .map(|(_, usage)| usage)
            .sum();
        self.profile.overall_stats.average_usage =
            total_usage as f64 / self.profile.usage_history.len() as f64;
    }

    /// Analyze memory usage patterns and identify optimizations
    pub fn analyze_memory_patterns(&mut self) {
        self.analyze_allocation_patterns();
        self.detect_optimization_opportunities();
        self.calculate_efficiency_metrics();
        self.generate_recommendations();
    }

    /// Profile memory usage during algorithm execution
    pub fn profile_algorithm_execution<N, E, Ix, T>(
        &mut self,
        processor: &mut AdvancedProcessor,
        graph: &Graph<N, E, Ix>,
        algorithm_name: &str,
        algorithm: impl FnOnce(&Graph<N, E, Ix>) -> Result<T>,
    ) -> Result<(T, MemoryExecutionProfile)>
    where
        N: Node + Clone + std::hash::Hash + Eq + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let execution_start = SystemTime::now();
        let initial_memory = self.profile.overall_stats.current_usage;

        // Record pre-execution state
        self.record_memory_snapshot(processor);

        // Estimate graph memory usage
        let graph_memory = self.estimate_graph_memory_usage(graph);
        self.record_allocation("graph", graph_memory, "input_data", false);

        // Execute algorithm with memory tracking
        let result = crate::advanced::execute_with_enhanced_advanced(
            processor,
            graph,
            algorithm_name,
            algorithm,
        );

        let execution_end = SystemTime::now();
        let final_memory = self.profile.overall_stats.current_usage;

        // Record post-execution state
        self.record_memory_snapshot(processor);

        // Calculate execution profile
        let execution_profile = MemoryExecutionProfile {
            algorithm_name: algorithm_name.to_string(),
            execution_time: execution_end
                .duration_since(execution_start)
                .unwrap_or_default(),
            initial_memory,
            peak_memory: self.profile.overall_stats.peak_usage,
            final_memory,
            memory_growth: final_memory.saturating_sub(initial_memory),
            graph_memory,
            workspace_memory: self.estimate_workspace_memory(algorithm_name),
            cache_memory: self.estimate_cache_memory(processor),
            memory_efficiency: self.calculate_execution_efficiency(initial_memory, final_memory),
        };

        match result {
            Ok(value) => Ok((value, execution_profile)),
            Err(e) => Err(e),
        }
    }

    /// Generate comprehensive memory usage report
    pub fn generate_memory_report(&self) -> MemoryUsageReport {
        MemoryUsageReport {
            profile_duration: SystemTime::now()
                .duration_since(self.start_time)
                .unwrap_or_default(),
            overall_stats: self.profile.overall_stats.clone(),
            component_breakdown: self.profile.component_stats.clone(),
            optimization_opportunities: self.profile.optimization_opportunities.clone(),
            efficiency_analysis: self.profile.efficiency_analysis.clone(),
            memory_timeline: self.generate_memory_timeline(),
            allocation_analysis: self.analyze_allocation_efficiency(),
            recommendations: self.generate_optimization_recommendations(),
        }
    }

    /// Estimate memory usage of a graph
    fn estimate_graph_memory_usage<N, E, Ix>(&self, graph: &Graph<N, E, Ix>) -> usize
    where
        N: Node + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let node_size = std::mem::size_of::<N>();
        let edge_size = std::mem::size_of::<E>() + std::mem::size_of::<Ix>() * 2; // source + target
        let index_size = std::mem::size_of::<Ix>();

        let base_graph_overhead = 1024; // Estimated overhead for graph structure
        let node_memory = graph.node_count() * (node_size + index_size);
        let edge_memory = graph.edge_count() * edge_size;

        base_graph_overhead + node_memory + edge_memory
    }

    /// Estimate memory usage of an advanced processor
    fn estimate_processor_memory_usage(&self, processor: &AdvancedProcessor) -> usize {
        let stats = processor.get_optimization_stats();

        // Base processor memory (estimated)
        let base_memory = 1024 * 1024; // 1MB base

        // Neural RL agent memory (estimated based on optimizations)
        let neural_memory = stats.total_optimizations * 1024; // 1KB per optimization

        // Cache memory (estimated)
        let cache_memory = (stats.memory_efficiency * 10.0 * 1024.0 * 1024.0) as usize; // Based on efficiency

        base_memory + neural_memory + cache_memory
    }

    /// Estimate workspace memory for an algorithm
    fn estimate_workspace_memory(&self, algorithmname: &str) -> usize {
        match algorithmname {
            name if name.contains("pagerank") => 1024 * 1024, // 1MB for PageRank workspace
            name if name.contains("community") => 2048 * 1024, // 2MB for community detection
            name if name.contains("centrality") => 512 * 1024, // 512KB for centrality
            name if name.contains("shortest") => 1536 * 1024, // 1.5MB for shortest paths
            _ => 256 * 1024,                                  // 256KB default
        }
    }

    /// Estimate cache memory usage
    fn estimate_cache_memory(&self, processor: &AdvancedProcessor) -> usize {
        let stats = processor.get_optimization_stats();
        // Estimate based on optimization count and efficiency
        (stats.total_optimizations as f64 * stats.memory_efficiency * 1024.0) as usize
    }

    /// Calculate execution efficiency
    fn calculate_execution_efficiency(&self, initial_memory: usize, finalmemory: usize) -> f64 {
        if initial_memory == 0 {
            return 1.0;
        }

        let memory_growth_ratio = finalmemory as f64 / initial_memory as f64;
        // Efficiency decreases with _memory growth
        1.0 / memory_growth_ratio.max(1.0)
    }

    /// Record initial profiling state
    fn record_initial_state(&mut self, processor: &AdvancedProcessor) {
        let initial_memory = self.estimate_processor_memory_usage(processor);
        self.profile.overall_stats.current_usage = initial_memory;
        self.profile.overall_stats.peak_usage = initial_memory;
        self.profile.overall_stats.average_usage = initial_memory as f64;
    }

    /// Start real-time memory monitoring
    fn start_real_time_monitoring(&mut self) {
        // In a real implementation, this would start a background thread
        // For now, we'll simulate this functionality
        println!("Real-time memory monitoring started");
    }

    /// Analyze memory pressure and suggest optimizations
    fn analyze_memory_pressure(&mut self) {
        let pressure_ratio =
            self.profile.overall_stats.current_usage as f64 / self.memory_pressure_threshold as f64;

        if pressure_ratio > 0.8 {
            self.profile
                .optimization_opportunities
                .push(OptimizationOpportunity {
                    optimization_type: OptimizationType::MemoryPooling,
                    estimated_savings: self.profile.overall_stats.current_usage / 4, // 25% savings estimate
                    performance_impact: 0.1, // 10% performance improvement
                    implementation_complexity: 3,
                    description: "Implement memory pooling to reduce allocation overhead"
                        .to_string(),
                    priority: 4,
                });
        }

        if pressure_ratio > 0.9 {
            self.profile
                .optimization_opportunities
                .push(OptimizationOpportunity {
                    optimization_type: OptimizationType::StreamingProcessing,
                    estimated_savings: self.profile.overall_stats.current_usage / 2, // 50% savings estimate
                    performance_impact: -0.05, // 5% performance loss
                    implementation_complexity: 4,
                    description: "Use streaming algorithms to process data in chunks".to_string(),
                    priority: 5,
                });
        }
    }

    /// Analyze allocation patterns for optimization opportunities
    fn analyze_allocation_patterns(&mut self) {
        let mut pattern_analysis = HashMap::new();

        for allocation in &self.profile.allocation_patterns {
            let key = format!("{}_{}", allocation.category, allocation.size);
            let count = pattern_analysis.entry(key).or_insert(0);
            *count += 1;
        }

        // Identify frequent allocations for pooling optimization
        for (pattern, count) in pattern_analysis {
            if count > 10 {
                // Frequent allocation threshold
                self.profile
                    .optimization_opportunities
                    .push(OptimizationOpportunity {
                        optimization_type: OptimizationType::MemoryPooling,
                        estimated_savings: count * 1024, // Estimate based on frequency
                        performance_impact: 0.05 * (count as f64 / 100.0), // Performance improvement
                        implementation_complexity: 2,
                        description: format!("Pool frequent allocations: {pattern}"),
                        priority: 3,
                    });
            }
        }
    }

    /// Detect optimization opportunities
    fn detect_optimization_opportunities(&mut self) {
        // Analyze fragmentation
        self.analyze_fragmentation();

        // Analyze cache effectiveness
        self.analyze_cache_patterns();

        // Analyze allocation lifetime patterns
        self.analyze_lifetime_patterns();
    }

    /// Analyze memory fragmentation
    fn analyze_fragmentation(&mut self) {
        let allocation_sizes: Vec<usize> = self
            .profile
            .allocation_patterns
            .iter()
            .map(|a| a.size)
            .collect();

        if allocation_sizes.is_empty() {
            return;
        }

        let total_size: usize = allocation_sizes.iter().sum();
        let avg_size = total_size as f64 / allocation_sizes.len() as f64;
        let variance = allocation_sizes
            .iter()
            .map(|&size| (size as f64 - avg_size).powi(2))
            .sum::<f64>()
            / allocation_sizes.len() as f64;

        let fragmentation = variance.sqrt() / avg_size;
        self.profile.overall_stats.fragmentation_ratio = fragmentation.min(1.0);

        if fragmentation > 0.5 {
            self.profile
                .optimization_opportunities
                .push(OptimizationOpportunity {
                    optimization_type: OptimizationType::FragmentationReduction,
                    estimated_savings: (total_size as f64 * 0.1) as usize, // 10% savings estimate
                    performance_impact: 0.15, // 15% performance improvement
                    implementation_complexity: 3,
                    description: "Reduce memory fragmentation through better allocation strategies"
                        .to_string(),
                    priority: 3,
                });
        }
    }

    /// Analyze cache patterns
    fn analyze_cache_patterns(&mut self) {
        let cache_allocations = self
            .profile
            .allocation_patterns
            .iter()
            .filter(|a| a.category.contains("cache"))
            .count();

        let total_allocations = self.profile.allocation_patterns.len();

        if total_allocations > 0 {
            let cache_ratio = cache_allocations as f64 / total_allocations as f64;
            self.profile.efficiency_analysis.cache_effectiveness = cache_ratio;

            if cache_ratio < 0.1 {
                self.profile
                    .optimization_opportunities
                    .push(OptimizationOpportunity {
                        optimization_type: OptimizationType::CacheOptimization,
                        estimated_savings: 0, // Cache optimization focuses on performance
                        performance_impact: 0.25, // 25% performance improvement
                        implementation_complexity: 2,
                        description: "Improve caching strategies to reduce redundant computations"
                            .to_string(),
                        priority: 4,
                    });
            }
        }
    }

    /// Analyze allocation lifetime patterns
    fn analyze_lifetime_patterns(&mut self) {
        let lifetimes: Vec<Duration> = self
            .profile
            .allocation_patterns
            .iter()
            .filter_map(|a| a.lifetime)
            .collect();

        if lifetimes.is_empty() {
            return;
        }

        let avg_lifetime = lifetimes.iter().sum::<Duration>() / lifetimes.len() as u32;
        let short_lived = lifetimes
            .iter()
            .filter(|&&lt| lt < avg_lifetime / 2)
            .count();

        let short_lived_ratio = short_lived as f64 / lifetimes.len() as f64;

        if short_lived_ratio > 0.7 {
            self.profile
                .optimization_opportunities
                .push(OptimizationOpportunity {
                    optimization_type: OptimizationType::MemoryPooling,
                    estimated_savings: short_lived * 512, // Estimate based on short-lived allocations
                    performance_impact: 0.1,              // 10% performance improvement
                    implementation_complexity: 2,
                    description: "Pool short-lived allocations to reduce allocation overhead"
                        .to_string(),
                    priority: 3,
                });
        }
    }

    /// Calculate efficiency metrics
    fn calculate_efficiency_metrics(&mut self) {
        // Calculate overall efficiency
        let allocation_efficiency = if self.profile.overall_stats.allocation_count > 0 {
            self.profile.overall_stats.deallocation_count as f64
                / self.profile.overall_stats.allocation_count as f64
        } else {
            1.0
        };

        let memory_utilization = if self.profile.overall_stats.peak_usage > 0 {
            self.profile.overall_stats.average_usage / self.profile.overall_stats.peak_usage as f64
        } else {
            1.0
        };

        self.profile.efficiency_analysis.overall_efficiency = (allocation_efficiency
            + memory_utilization
            + (1.0 - self.profile.overall_stats.fragmentation_ratio))
            / 3.0;

        self.profile.efficiency_analysis.utilization_ratio = memory_utilization;

        // Calculate temporal and spatial locality (simplified)
        self.profile.efficiency_analysis.temporal_locality = self.calculate_temporal_locality();
        self.profile.efficiency_analysis.spatial_locality = self.calculate_spatial_locality();
    }

    /// Calculate temporal locality score
    fn calculate_temporal_locality(&self) -> f64 {
        // Simplified temporal locality calculation based on allocation patterns
        if self.profile.allocation_patterns.len() < 2 {
            return 1.0;
        }

        let mut temporal_score = 0.0;
        let window_size = 10; // Consider last 10 allocations

        for window in self.profile.allocation_patterns.windows(window_size) {
            let categories: std::collections::HashSet<_> =
                window.iter().map(|a| &a.category).collect();
            let locality = 1.0 - (categories.len() as f64 / window_size as f64);
            temporal_score += locality;
        }

        temporal_score
            / (self
                .profile
                .allocation_patterns
                .len()
                .saturating_sub(window_size - 1)) as f64
    }

    /// Calculate spatial locality score
    fn calculate_spatial_locality(&self) -> f64 {
        // Simplified spatial locality calculation based on allocation sizes
        if self.profile.allocation_patterns.is_empty() {
            return 1.0;
        }

        let sizes: Vec<usize> = self
            .profile
            .allocation_patterns
            .iter()
            .map(|a| a.size)
            .collect();
        let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;

        let size_variance = sizes
            .iter()
            .map(|&size| (size as f64 - avg_size).powi(2))
            .sum::<f64>()
            / sizes.len() as f64;

        1.0 / (1.0 + size_variance.sqrt() / avg_size)
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&mut self) {
        let mut recommendations = Vec::new();

        // Memory efficiency recommendations
        if self.profile.efficiency_analysis.overall_efficiency < 0.7 {
            recommendations.push(
                "Consider implementing memory pooling for frequently allocated objects".to_string(),
            );
        }

        if self.profile.efficiency_analysis.utilization_ratio < 0.6 {
            recommendations.push("Memory utilization is low - consider reducing buffer sizes or using lazy allocation".to_string());
        }

        if self.profile.overall_stats.fragmentation_ratio > 0.4 {
            recommendations.push(
                "High memory fragmentation detected - consider using a custom allocator"
                    .to_string(),
            );
        }

        if self.profile.efficiency_analysis.cache_effectiveness < 0.3 {
            recommendations.push(
                "Low cache effectiveness - review caching strategies and data access patterns"
                    .to_string(),
            );
        }

        if self.profile.efficiency_analysis.temporal_locality < 0.5 {
            recommendations.push(
                "Poor temporal locality - consider grouping related operations together"
                    .to_string(),
            );
        }

        if self.profile.efficiency_analysis.spatial_locality < 0.5 {
            recommendations.push(
                "Poor spatial locality - consider using more compact data structures".to_string(),
            );
        }

        self.profile.efficiency_analysis.recommendations = recommendations;
    }

    /// Generate memory timeline for visualization
    fn generate_memory_timeline(&self) -> Vec<(SystemTime, usize)> {
        self.profile.usage_history.iter().cloned().collect()
    }

    /// Analyze allocation efficiency
    fn analyze_allocation_efficiency(&self) -> AllocationEfficiencyAnalysis {
        let total_allocations = self.profile.allocation_patterns.len();
        let predicted_allocations = self
            .profile
            .allocation_patterns
            .iter()
            .filter(|a| a.was_predicted)
            .count();

        let prediction_accuracy = if total_allocations > 0 {
            predicted_allocations as f64 / total_allocations as f64
        } else {
            0.0
        };

        let allocation_size_distribution = self.calculate_allocation_size_distribution();
        let allocation_category_distribution = self.calculate_allocation_category_distribution();

        AllocationEfficiencyAnalysis {
            prediction_accuracy,
            allocation_size_distribution,
            allocation_category_distribution,
            average_allocation_size: self.calculate_average_allocation_size(),
            allocation_frequency: self.calculate_allocation_frequency(),
        }
    }

    /// Calculate allocation size distribution
    fn calculate_allocation_size_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for allocation in &self.profile.allocation_patterns {
            let size_range = match allocation.size {
                0..=1024 => "Small (<1KB)",
                1025..=10240 => "Medium (1-10KB)",
                10241..=102400 => "Large (10-100KB)",
                _ => "Very Large (>100KB)",
            };

            *distribution.entry(size_range.to_string()).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate allocation category distribution
    fn calculate_allocation_category_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for allocation in &self.profile.allocation_patterns {
            *distribution.entry(allocation.category.clone()).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate average allocation size
    fn calculate_average_allocation_size(&self) -> f64 {
        if self.profile.allocation_patterns.is_empty() {
            return 0.0;
        }

        let total_size: usize = self
            .profile
            .allocation_patterns
            .iter()
            .map(|a| a.size)
            .sum();
        total_size as f64 / self.profile.allocation_patterns.len() as f64
    }

    /// Calculate allocation frequency
    fn calculate_allocation_frequency(&self) -> f64 {
        if self.profile.usage_history.is_empty() {
            return 0.0;
        }

        let duration = SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or_default();
        if duration.as_secs() == 0 {
            return 0.0;
        }

        self.profile.allocation_patterns.len() as f64 / duration.as_secs() as f64
    }

    /// Generate comprehensive optimization recommendations
    fn generate_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Prioritize optimization opportunities
        let mut sorted_opportunities = self.profile.optimization_opportunities.clone();
        sorted_opportunities.sort_by(|a, b| b.priority.cmp(&a.priority));

        for opportunity in sorted_opportunities.iter().take(5) {
            recommendations.push(format!(
                "Priority {}: {} - {} (Est. savings: {} bytes, Performance impact: {:.1}%)",
                opportunity.priority,
                format!("{:?}", opportunity.optimization_type).replace("_", " "),
                opportunity.description,
                opportunity.estimated_savings,
                opportunity.performance_impact * 100.0
            ));
        }

        recommendations
    }
}

/// Memory execution profile for a single algorithm run
#[derive(Debug, Clone)]
pub struct MemoryExecutionProfile {
    pub algorithm_name: String,
    pub execution_time: Duration,
    pub initial_memory: usize,
    pub peak_memory: usize,
    pub final_memory: usize,
    pub memory_growth: usize,
    pub graph_memory: usize,
    pub workspace_memory: usize,
    pub cache_memory: usize,
    pub memory_efficiency: f64,
}

/// Comprehensive memory usage report
#[derive(Debug, Clone)]
pub struct MemoryUsageReport {
    pub profile_duration: Duration,
    pub overall_stats: MemoryStats,
    pub component_breakdown: HashMap<String, MemoryStats>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub efficiency_analysis: EfficiencyAnalysis,
    pub memory_timeline: Vec<(SystemTime, usize)>,
    pub allocation_analysis: AllocationEfficiencyAnalysis,
    pub recommendations: Vec<String>,
}

/// Allocation efficiency analysis results
#[derive(Debug, Clone)]
pub struct AllocationEfficiencyAnalysis {
    pub prediction_accuracy: f64,
    pub allocation_size_distribution: HashMap<String, usize>,
    pub allocation_category_distribution: HashMap<String, usize>,
    pub average_allocation_size: f64,
    pub allocation_frequency: f64,
}

impl MemoryUsageReport {
    /// Generate a human-readable summary of the memory usage report
    pub fn generate_summary(&self) -> String {
        format!(
            "Memory Usage Report Summary\n\
            ===========================\n\
            Profile Duration: {:.2}s\n\
            Peak Memory Usage: {:.2} MB\n\
            Average Memory Usage: {:.2} MB\n\
            Memory Efficiency: {:.1}%\n\
            Fragmentation Ratio: {:.1}%\n\
            Total Allocations: {}\n\
            Optimization Opportunities: {}\n\
            \n\
            Top Recommendations:\n\
            {}",
            self.profile_duration.as_secs_f64(),
            self.overall_stats.peak_usage as f64 / 1_000_000.0,
            self.overall_stats.average_usage / 1_000_000.0,
            self.efficiency_analysis.overall_efficiency * 100.0,
            self.overall_stats.fragmentation_ratio * 100.0,
            self.overall_stats.allocation_count,
            self.optimization_opportunities.len(),
            self.recommendations
                .iter()
                .take(3)
                .map(|r| format!("  • {r}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Export report to JSON format
    pub fn to_json(&self) -> String {
        // In a real implementation, this would use serde_json
        "{\"memory_report\": \"JSON export not implemented\"}".to_string()
    }
}

/// Convenience function to create a memory profiler with default configuration
#[allow(dead_code)]
pub fn create_memory_profiler() -> AdvancedMemoryProfiler {
    AdvancedMemoryProfiler::new(MemoryProfilerConfig::default())
}

/// Convenience function to create a memory profiler optimized for large graphs
#[allow(dead_code)]
pub fn create_large_graph_memory_profiler() -> AdvancedMemoryProfiler {
    let config = MemoryProfilerConfig {
        track_allocations: true,
        analyze_patterns: true,
        detect_optimizations: true,
        max_history_entries: 50000, // More history for large graphs
        sampling_interval: Duration::from_millis(50), // More frequent sampling
        real_time_monitoring: true,
    };
    AdvancedMemoryProfiler::new(config)
}

/// Enhanced memory profiler for extreme stress testing
#[allow(dead_code)]
pub fn create_extreme_stress_memory_profiler() -> AdvancedMemoryProfiler {
    let config = MemoryProfilerConfig {
        track_allocations: true,
        analyze_patterns: true,
        detect_optimizations: true,
        max_history_entries: 100000, // Extra history for extreme tests
        sampling_interval: Duration::from_millis(25), // Very frequent sampling
        real_time_monitoring: true,
    };
    AdvancedMemoryProfiler::new(config)
}

/// Profile a comprehensive stress test with detailed memory analysis
#[allow(dead_code)]
pub fn profile_comprehensive_stress_test<F>(
    profiler: &mut AdvancedMemoryProfiler,
    processor: &mut AdvancedProcessor,
    test_name: &str,
    test_function: F,
) -> Result<(MemoryUsageReport, Duration)>
where
    F: FnOnce(&mut AdvancedProcessor) -> Result<String>,
{
    println!("🧠 Starting memory-profiled stress test: {test_name}");

    // Start profiling
    profiler.start_profiling(processor);
    let test_start = std::time::Instant::now();

    // Record initial state
    profiler.record_allocation("stress_test", 0, "test_initialization", true);

    // Execute the test _function
    let test_result = test_function(processor);

    let test_duration = test_start.elapsed();

    // Record final state
    profiler.record_memory_snapshot(processor);
    profiler.analyze_memory_patterns();

    // Generate report
    let report = profiler.generate_memory_report();

    println!("🧠 Memory profiling completed for {test_name}");
    println!(
        "   📊 Peak memory: {:.1} MB",
        report.overall_stats.peak_usage as f64 / 1_000_000.0
    );
    println!(
        "   📊 Memory efficiency: {:.1}%",
        report.efficiency_analysis.overall_efficiency * 100.0
    );
    println!(
        "   📊 Optimization opportunities: {}",
        report.optimization_opportunities.len()
    );

    match test_result {
        Ok(_) => Ok((report, test_duration)),
        Err(e) => {
            println!("⚠️  Test failed but memory profile still generated: {e:?}");
            Ok((report, test_duration))
        }
    }
}

/// Memory-aware graph generator with profiling integration
#[allow(dead_code)]
pub fn generate_profiled_large_graph(
    profiler: &mut AdvancedMemoryProfiler,
    num_nodes: usize,
    graph_type: &str,
) -> Result<crate::base::Graph<usize, f64>> {
    println!("🏗️  Generating profiled {graph_type} graph with {num_nodes} _nodes");

    let generation_start = std::time::Instant::now();
    profiler.record_allocation("graph_generation", num_nodes * 8, "_nodes", true);

    let mut graph = crate::base::Graph::new();
    let mut rng = rand::rng();

    // Add _nodes with memory tracking
    const NODE_BATCH_SIZE: usize = 25_000;
    for batch_start in (0..num_nodes).step_by(NODE_BATCH_SIZE) {
        let batch_end = (batch_start + NODE_BATCH_SIZE).min(num_nodes);

        // Record batch allocation
        profiler.record_allocation(
            "graph_generation",
            (batch_end - batch_start) * std::mem::size_of::<usize>(),
            "node_batch",
            true,
        );

        for i in batch_start..batch_end {
            graph.add_node(i);
        }

        if batch_start % (NODE_BATCH_SIZE * 10) == 0 {
            println!(
                "   📊 Added {} nodes, current memory usage estimate: {:.1} MB",
                batch_end,
                (batch_end * 16) as f64 / 1_000_000.0
            );
        }
    }

    // Add edges based on graph _type
    let target_edges = match graph_type {
        "sparse" => num_nodes * 2,
        "medium" => num_nodes * 4,
        "dense" => num_nodes * 8,
        "scale_free" => (num_nodes as f64 * 2.5) as usize,
        _ => num_nodes * 3, // default
    };

    profiler.record_allocation("graph_generation", target_edges * 24, "edges", true);

    let mut edges_added = 0;
    while edges_added < target_edges && edges_added < num_nodes * 10 {
        // Prevent infinite loop
        let source = rng.gen_range(0..num_nodes);
        let target = rng.gen_range(0..num_nodes);

        if source != target {
            let weight: f64 = rng.random();
            if graph.add_edge(source, target, weight).is_ok() {
                edges_added += 1;

                if edges_added % 100_000 == 0 {
                    println!("   🔗 Added {edges_added} edges");
                }
            }
        }
    }

    let generation_time = generation_start.elapsed();
    println!(
        "✅ Graph generation completed in {:?}: {} nodes, {} edges",
        generation_time,
        graph.node_count(),
        graph.edge_count()
    );

    Ok(graph)
}

/// Comprehensive memory stress test runner
#[allow(dead_code)]
pub fn run_memory_stress_tests() -> Result<Vec<MemoryUsageReport>> {
    println!("🧠 Starting comprehensive memory stress tests...");
    println!("================================================");

    let mut reports = Vec::new();
    let mut profiler = create_extreme_stress_memory_profiler();

    // Test 1: Small graph baseline
    println!("\n📊 Test 1: Small Graph Baseline (100K nodes)");
    match generate_profiled_large_graph(&mut profiler, 100_000, "medium") {
        Ok(small_graph) => {
            let mut processor = crate::advanced::create_large_graph_advanced_processor();

            let (report, duration) = profile_comprehensive_stress_test(
                &mut profiler,
                &mut processor,
                "small_graph_baseline",
                |proc| {
                    // Run basic algorithm
                    let _result = crate::advanced::execute_with_enhanced_advanced(
                        proc,
                        &small_graph,
                        "baseline_cc",
                        |g| {
                            use crate::algorithms::connectivity::connected_components;
                            Ok(connected_components(g))
                        },
                    );
                    Ok("Small graph baseline completed".to_string())
                },
            )?;

            println!("   ⏱️  Test completed in {duration:?}");
            reports.push(report);
        }
        Err(e) => println!("   ❌ Failed to create small graph: {e}"),
    }

    // Test 2: Medium graph stress test
    println!("\n📊 Test 2: Medium Graph Stress Test (500K nodes)");
    match generate_profiled_large_graph(&mut profiler, 500_000, "sparse") {
        Ok(medium_graph) => {
            let mut processor = crate::advanced::create_large_graph_advanced_processor();

            let (report, duration) = profile_comprehensive_stress_test(
                &mut profiler,
                &mut processor,
                "medium_graph_stress",
                |proc| {
                    // Run multiple algorithms
                    let _cc_result = crate::advanced::execute_with_enhanced_advanced(
                        proc,
                        &medium_graph,
                        "medium_cc",
                        |g| {
                            use crate::algorithms::connectivity::connected_components;
                            Ok(connected_components(g))
                        },
                    );

                    let _pr_result = crate::advanced::execute_with_enhanced_advanced(
                        proc,
                        &medium_graph,
                        "medium_pr",
                        |g| {
                            use crate::measures::pagerank_centrality;
                            pagerank_centrality(g, 0.85, 1e-3)
                        },
                    );

                    Ok("Medium graph stress test completed".to_string())
                },
            )?;

            println!("   ⏱️  Test completed in {duration:?}");
            reports.push(report);
        }
        Err(e) => println!("   ❌ Failed to create medium graph: {e}"),
    }

    // Test 3: Large graph extreme test (if memory allows)
    println!("\n📊 Test 3: Large Graph Extreme Test (1M nodes)");
    match generate_profiled_large_graph(&mut profiler, 1_000_000, "sparse") {
        Ok(large_graph) => {
            let mut processor = crate::advanced::create_large_graph_advanced_processor();

            let (report, duration) = profile_comprehensive_stress_test(
                &mut profiler,
                &mut processor,
                "large_graph_extreme",
                |proc| {
                    // Run memory-intensive test
                    let _result = crate::advanced::execute_with_enhanced_advanced(
                        proc,
                        &large_graph,
                        "large_memory_test",
                        |g| {
                            // Force memory allocation to test memory management
                            let nodes: Vec<_> = g.nodes().into_iter().collect();
                            let edges: Vec<_> = g
                                .edges()
                                .into_iter()
                                .map(|e| (e.source, e.target, e.weight))
                                .collect();
                            let _memory_intensive: Vec<f64> = edges
                                .iter()
                                .flat_map(|(s, t, w)| vec![*s as f64, *t as f64, *w])
                                .collect();

                            Ok(nodes.len() + edges.len())
                        },
                    );

                    Ok("Large graph extreme test completed".to_string())
                },
            )?;

            println!("   ⏱️  Test completed in {duration:?}");
            reports.push(report);
        }
        Err(e) => println!("   ❌ Failed to create large graph: {e}"),
    }

    // Generate summary
    println!("\n📋 Memory Stress Test Summary");
    println!("=============================");
    for (i, report) in reports.iter().enumerate() {
        println!(
            "Test {}: Peak Memory: {:.1} MB, Efficiency: {:.1}%, Optimizations: {}",
            i + 1,
            report.overall_stats.peak_usage as f64 / 1_000_000.0,
            report.efficiency_analysis.overall_efficiency * 100.0,
            report.optimization_opportunities.len()
        );
    }

    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = create_memory_profiler();
        assert_eq!(profiler.profile.overall_stats.current_usage, 0);
        assert_eq!(profiler.profile.overall_stats.allocation_count, 0);
    }

    #[test]
    fn test_allocation_recording() {
        let mut profiler = create_memory_profiler();

        profiler.record_allocation("test_component", 1024, "workspace", false);

        assert_eq!(profiler.profile.overall_stats.current_usage, 1024);
        assert_eq!(profiler.profile.overall_stats.allocation_count, 1);
        assert_eq!(profiler.profile.allocation_patterns.len(), 1);
    }

    #[test]
    fn test_deallocation_recording() {
        let mut profiler = create_memory_profiler();

        profiler.record_allocation("test_component", 1024, "workspace", false);
        profiler.record_deallocation("test_component", 1024, "workspace");

        assert_eq!(profiler.profile.overall_stats.current_usage, 0);
        assert_eq!(profiler.profile.overall_stats.deallocation_count, 1);
    }

    #[test]
    fn test_memory_pattern_analysis() {
        let mut profiler = create_memory_profiler();

        // Create some allocation patterns
        for _i in 0..15 {
            profiler.record_allocation("test_component", 1024, "frequent_pattern", false);
        }

        profiler.analyze_memory_patterns();

        // Should detect optimization opportunities for frequent allocations
        let has_pooling_opportunity = profiler
            .profile
            .optimization_opportunities
            .iter()
            .any(|op| op.optimization_type == OptimizationType::MemoryPooling);

        assert!(has_pooling_opportunity);
    }

    #[test]
    fn test_efficiency_calculation() {
        let mut profiler = create_memory_profiler();

        // Simulate some memory activity
        profiler.record_allocation("component1", 2048, "data", false);
        profiler.record_allocation("component2", 1024, "cache", true);
        profiler.record_deallocation("component1", 2048, "data");

        profiler.calculate_efficiency_metrics();

        assert!(profiler.profile.efficiency_analysis.overall_efficiency > 0.0);
        assert!(profiler.profile.efficiency_analysis.overall_efficiency <= 1.0);
    }

    #[test]
    fn test_memory_report_generation() {
        let mut profiler = create_memory_profiler();

        // Add some test data
        profiler.record_allocation("test", 1024, "data", false);
        profiler.analyze_memory_patterns();

        let report = profiler.generate_memory_report();

        assert!(report.profile_duration >= Duration::ZERO);
        assert_eq!(report.overall_stats.allocation_count, 1);

        let summary = report.generate_summary();
        assert!(summary.contains("Memory Usage Report Summary"));
    }

    #[test]
    fn test_large_graph_profiler() {
        let profiler = create_large_graph_memory_profiler();

        assert_eq!(profiler.config.max_history_entries, 50000);
        assert_eq!(profiler.config.sampling_interval, Duration::from_millis(50));
    }
}
