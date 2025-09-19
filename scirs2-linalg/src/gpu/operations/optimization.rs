//! GPU operation optimization and batch size optimization

use std::collections::HashMap;

/// Batch size optimizer for GPU operations
#[derive(Debug)]
pub struct BatchSizeOptimizer {
    /// Optimal batch sizes for different operations
    pub optimalsizes: std::collections::HashMap<String, usize>,
    /// Memory constraints
    pub memory_limit: usize,
    /// Performance history
    performance_history: Vec<BatchPerformanceRecord>,
}

#[derive(Debug, Clone)]
pub struct BatchPerformanceRecord {
    pub operation: String,
    pub batchsize: usize,
    pub execution_time: f64,
    pub memory_usage: usize,
    pub throughput: f64,
}

impl BatchSizeOptimizer {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            optimalsizes: std::collections::HashMap::new(),
            memory_limit,
            performance_history: Vec::new(),
        }
    }

    /// Find optimal batch size for an operation
    pub fn optimize_batchsize(&mut self, operation: &str, datasize: usize) -> usize {
        // Check if we have historical data
        if let Some(&optimal) = self.optimalsizes.get(operation) {
            return optimal.min(datasize);
        }

        // Default heuristics based on operation type
        let default_batch = match operation {
            "matrix_multiply" => (self.memory_limit / 8).min(1024), // Conservative for GEMM
            "matrix_vector" => (self.memory_limit / 4).min(2048),   // Less memory intensive
            "element_wise" => (self.memory_limit / 2).min(4096),    // Most memory efficient
            "decomposition" => (self.memory_limit / 16).min(512),   // Most compute intensive
            _ => (self.memory_limit / 8).min(1024),
        };

        default_batch.min(datasize)
    }

    /// Record performance for batch size optimization
    pub fn record_performance(&mut self, record: BatchPerformanceRecord) {
        self.performance_history.push(record.clone());

        // Update optimal size if this is better
        let _current_optimal = self
            .optimalsizes
            .get(&record.operation)
            .copied()
            .unwrap_or(0);
        if record.throughput > 0.0 {
            // Find best throughput for this operation
            let best_record = self
                .performance_history
                .iter()
                .filter(|r| r.operation == record.operation)
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap());

            if let Some(best) = best_record {
                self.optimalsizes
                    .insert(record.operation.clone(), best.batchsize);
            }
        }
    }

    /// Get performance history for a specific operation
    pub fn get_performance_history(&self, operation: &str) -> Vec<&BatchPerformanceRecord> {
        self.performance_history
            .iter()
            .filter(|record| record.operation == operation)
            .collect()
    }

    /// Clear performance history
    pub fn clear_history(&mut self) {
        self.performance_history.clear();
    }

    /// Get current optimal batch sizes
    pub fn get_optimal_sizes(&self) -> &HashMap<String, usize> {
        &self.optimalsizes
    }
}
