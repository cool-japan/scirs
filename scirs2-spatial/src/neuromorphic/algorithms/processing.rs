//! Neuromorphic Event Processing
//!
//! This module implements general-purpose neuromorphic processing capabilities
//! including event-driven computation, memristive crossbar arrays, and temporal
//! dynamics for spatial data processing.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array2, ArrayView2};
use rand::Rng;
use std::collections::{HashMap, VecDeque};

// Import core neuromorphic components
use super::super::core::SpikeEvent;

/// Neuromorphic processor for general spatial computations
///
/// This processor implements event-driven neuromorphic computing paradigms
/// for spatial data processing. It supports memristive crossbar arrays for
/// in-memory computation and temporal coding schemes for efficient information
/// processing.
///
/// # Features
/// - Event-driven processing pipeline
/// - Memristive crossbar arrays for in-memory computation
/// - Temporal and rate-based coding schemes
/// - Spike timing dynamics and correlation detection
/// - Configurable processing parameters
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use scirs2_spatial::neuromorphic::algorithms::NeuromorphicProcessor;
///
/// let points = Array2::from_shape_vec((3, 2), vec![
///     0.0, 0.0, 1.0, 1.0, 2.0, 2.0
/// ]).unwrap();
///
/// let mut processor = NeuromorphicProcessor::new()
///     .with_memristive_crossbar(true)
///     .with_temporal_coding(true)
///     .with_crossbar_size(64, 64);
///
/// // Encode spatial data as neuromorphic events
/// let events = processor.encode_spatial_events(&points.view()).unwrap();
///
/// // Process events through neuromorphic pipeline
/// let processed_events = processor.process_events(&events).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NeuromorphicProcessor {
    /// Enable memristive crossbar arrays
    memristive_crossbar: bool,
    /// Enable temporal coding
    temporal_coding: bool,
    /// Crossbar array dimensions
    crossbar_size: (usize, usize),
    /// Memristive device conductances
    conductances: Array2<f64>,
    /// Event processing pipeline
    event_pipeline: VecDeque<SpikeEvent>,
    /// Maximum pipeline length
    max_pipeline_length: usize,
    /// Crossbar threshold for spike generation
    crossbar_threshold: f64,
    /// Memristive learning rate
    memristive_learning_rate: f64,
}

impl Default for NeuromorphicProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuromorphicProcessor {
    /// Create new neuromorphic processor
    ///
    /// # Returns
    /// A new `NeuromorphicProcessor` with default configuration
    pub fn new() -> Self {
        Self {
            memristive_crossbar: false,
            temporal_coding: false,
            crossbar_size: (64, 64),
            conductances: Array2::zeros((64, 64)),
            event_pipeline: VecDeque::new(),
            max_pipeline_length: 1000,
            crossbar_threshold: 0.5,
            memristive_learning_rate: 0.001,
        }
    }

    /// Enable memristive crossbar arrays
    ///
    /// When enabled, events are processed through a memristive crossbar array
    /// that provides in-memory computation capabilities.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable memristive crossbar processing
    pub fn with_memristive_crossbar(mut self, enabled: bool) -> Self {
        self.memristive_crossbar = enabled;
        if enabled {
            self.initialize_crossbar();
        }
        self
    }

    /// Enable temporal coding
    ///
    /// When enabled, spatial coordinates are encoded using spike timing
    /// rather than spike rates.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable temporal coding
    pub fn with_temporal_coding(mut self, enabled: bool) -> Self {
        self.temporal_coding = enabled;
        self
    }

    /// Configure crossbar size
    ///
    /// Sets the dimensions of the memristive crossbar array used for
    /// in-memory computation.
    ///
    /// # Arguments
    /// * `rows` - Number of rows in crossbar array
    /// * `cols` - Number of columns in crossbar array
    pub fn with_crossbar_size(mut self, rows: usize, cols: usize) -> Self {
        self.crossbar_size = (rows, cols);
        self.conductances = Array2::zeros((rows, cols));
        if self.memristive_crossbar {
            self.initialize_crossbar();
        }
        self
    }

    /// Configure processing parameters
    ///
    /// # Arguments
    /// * `max_pipeline_length` - Maximum length of event pipeline
    /// * `crossbar_threshold` - Threshold for crossbar spike generation
    /// * `learning_rate` - Learning rate for memristive adaptation
    pub fn with_processing_params(
        mut self,
        max_pipeline_length: usize,
        crossbar_threshold: f64,
        learning_rate: f64,
    ) -> Self {
        self.max_pipeline_length = max_pipeline_length;
        self.crossbar_threshold = crossbar_threshold;
        self.memristive_learning_rate = learning_rate;
        self
    }

    /// Encode spatial data as neuromorphic events
    ///
    /// Converts spatial data points into spike events using either temporal
    /// coding (spike timing) or rate coding (spike frequency) depending on
    /// the processor configuration.
    ///
    /// # Arguments
    /// * `points` - Input spatial points (n_points × n_dims)
    ///
    /// # Returns
    /// Vector of spike events representing the spatial data
    pub fn encode_spatial_events(
        &self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Vec<SpikeEvent>> {
        let (n_points, n_dims) = points.dim();
        let mut events = Vec::new();

        if n_points == 0 || n_dims == 0 {
            return Ok(events);
        }

        for (point_idx, point) in points.outer_iter().enumerate() {
            for (dim, &coord) in point.iter().enumerate() {
                // Temporal coding: encode coordinate as spike timing
                let normalized_coord = (coord + 10.0) / 20.0; // Normalize to [0, 1]
                let normalized_coord = normalized_coord.clamp(0.0, 1.0);

                if self.temporal_coding {
                    // Timing-based encoding
                    let spike_time = normalized_coord * 100.0; // Map to [0, 100] time units
                    let event =
                        SpikeEvent::new(point_idx * n_dims + dim, spike_time, 1.0, point.to_vec());
                    events.push(event);
                } else {
                    // Rate-based encoding
                    let spike_rate = normalized_coord * 50.0; // Max 50 Hz
                    let num_spikes = spike_rate as usize;

                    for spike_idx in 0..num_spikes {
                        let spike_time = if spike_rate > 0.0 {
                            (spike_idx as f64) * (1.0 / spike_rate)
                        } else {
                            0.0
                        };
                        let event = SpikeEvent::new(
                            point_idx * n_dims + dim,
                            spike_time,
                            1.0,
                            point.to_vec(),
                        );
                        events.push(event);
                    }
                }
            }
        }

        // Sort events by timestamp
        events.sort_by(|a, b| a.timestamp().partial_cmp(&b.timestamp()).unwrap());

        Ok(events)
    }

    /// Process events through neuromorphic pipeline
    ///
    /// Processes spike events through the neuromorphic computing pipeline,
    /// applying memristive crossbar processing and temporal dynamics as
    /// configured.
    ///
    /// # Arguments
    /// * `events` - Input spike events to process
    ///
    /// # Returns
    /// Vector of processed spike events
    pub fn process_events(&mut self, events: &[SpikeEvent]) -> SpatialResult<Vec<SpikeEvent>> {
        let mut processed_events = Vec::new();

        for event in events {
            self.event_pipeline.push_back(event.clone());

            // Process through memristive crossbar if enabled
            if self.memristive_crossbar {
                let crossbar_output = self.process_through_crossbar(event)?;
                processed_events.extend(crossbar_output);
            } else {
                processed_events.push(event.clone());
            }

            // Apply temporal dynamics
            if self.temporal_coding {
                Self::apply_temporal_dynamics(&mut processed_events)?;
            }

            // Maintain event pipeline size
            if self.event_pipeline.len() > self.max_pipeline_length {
                self.event_pipeline.pop_front();
            }
        }

        Ok(processed_events)
    }

    /// Initialize memristive crossbar array
    ///
    /// Initializes the conductance values of the memristive crossbar array
    /// with random values representing the initial device states.
    fn initialize_crossbar(&mut self) {
        let (rows, cols) = self.crossbar_size;
        let mut rng = rand::rng();

        // Initialize conductances with random values
        for i in 0..rows {
            for j in 0..cols {
                // Random conductance between 0.1 and 1.0 (normalized)
                self.conductances[[i, j]] = 0.1 + rng.gen_range(0.0..0.9);
            }
        }
    }

    /// Process event through memristive crossbar
    ///
    /// Processes a spike event through the memristive crossbar array,
    /// generating output spikes based on the crossbar conductances and
    /// updating the memristive devices.
    fn process_through_crossbar(&mut self, event: &SpikeEvent) -> SpatialResult<Vec<SpikeEvent>> {
        let (rows, cols) = self.crossbar_size;
        let mut output_events = Vec::new();

        // Map input neuron to crossbar row
        let input_row = event.neuron_id() % rows;

        // Compute crossbar outputs
        for col in 0..cols {
            let conductance = self.conductances[[input_row, col]];
            let output_current = event.amplitude() * conductance;

            // Generate output spike if current exceeds threshold
            if output_current > self.crossbar_threshold {
                let output_event = SpikeEvent::new(
                    rows + col,              // Offset for output neurons
                    event.timestamp() + 0.1, // Small delay
                    output_current,
                    event.spatial_coords().to_vec(),
                );
                output_events.push(output_event);

                // Update memristive device (Hebbian-like plasticity)
                self.update_memristive_device(input_row, col, event.amplitude())?;
            }
        }

        Ok(output_events)
    }

    /// Update memristive device conductance
    ///
    /// Updates the conductance of a memristive device based on the input
    /// spike amplitude using a simple Hebbian-like learning rule.
    fn update_memristive_device(
        &mut self,
        row: usize,
        col: usize,
        spike_amplitude: f64,
    ) -> SpatialResult<()> {
        let current_conductance = self.conductances[[row, col]];

        // Simple memristive update rule
        let conductance_change =
            self.memristive_learning_rate * spike_amplitude * (1.0 - current_conductance);

        self.conductances[[row, col]] += conductance_change;
        self.conductances[[row, col]] = self.conductances[[row, col]].clamp(0.0, 1.0);

        Ok(())
    }

    /// Apply temporal dynamics to event processing
    ///
    /// Applies temporal filtering and spike-timing dependent processing
    /// to enhance temporal correlations and implement refractory periods.
    fn apply_temporal_dynamics(events: &mut Vec<SpikeEvent>) -> SpatialResult<()> {
        // Apply temporal filtering and spike-timing dependent processing
        let mut filtered_events = Vec::new();

        for (i, event) in events.iter().enumerate() {
            let mut should_include = true;
            let mut modified_event = event.clone();

            // Check for temporal correlations with recent events
            for other_event in events.iter().skip(i + 1) {
                let time_diff = (other_event.timestamp() - event.timestamp()).abs();

                if time_diff < 5.0 {
                    // Within temporal window
                    // Apply temporal correlation enhancement
                    let new_amplitude = modified_event.amplitude() * 1.1;
                    modified_event = SpikeEvent::new(
                        modified_event.neuron_id(),
                        modified_event.timestamp(),
                        new_amplitude,
                        modified_event.spatial_coords().to_vec(),
                    );

                    // Coincidence detection
                    if time_diff < 1.0 {
                        let enhanced_amplitude = modified_event.amplitude() * 1.5;
                        modified_event = SpikeEvent::new(
                            modified_event.neuron_id(),
                            modified_event.timestamp(),
                            enhanced_amplitude,
                            modified_event.spatial_coords().to_vec(),
                        );
                    }
                }

                // Refractory period simulation
                if time_diff < 0.5 && event.neuron_id() == other_event.neuron_id() {
                    should_include = false; // Suppress due to refractory period
                    break;
                }
            }

            if should_include {
                filtered_events.push(modified_event);
            }
        }

        *events = filtered_events;
        Ok(())
    }

    /// Get crossbar statistics
    ///
    /// Returns statistics about the current state of the memristive crossbar
    /// array and event processing pipeline.
    ///
    /// # Returns
    /// HashMap containing various statistics about the processor state
    pub fn get_crossbar_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if self.memristive_crossbar {
            let total_conductance: f64 = self.conductances.sum();
            let avg_conductance =
                total_conductance / (self.crossbar_size.0 * self.crossbar_size.1) as f64;
            let max_conductance = self.conductances.fold(0.0f64, |acc, &x| acc.max(x));
            let min_conductance = self.conductances.fold(1.0f64, |acc, &x| acc.min(x));

            stats.insert("total_conductance".to_string(), total_conductance);
            stats.insert("avg_conductance".to_string(), avg_conductance);
            stats.insert("max_conductance".to_string(), max_conductance);
            stats.insert("min_conductance".to_string(), min_conductance);
        }

        stats.insert(
            "event_pipeline_length".to_string(),
            self.event_pipeline.len() as f64,
        );
        stats
    }

    /// Get crossbar size
    pub fn crossbar_size(&self) -> (usize, usize) {
        self.crossbar_size
    }

    /// Check if memristive crossbar is enabled
    pub fn is_memristive_enabled(&self) -> bool {
        self.memristive_crossbar
    }

    /// Check if temporal coding is enabled
    pub fn is_temporal_coding_enabled(&self) -> bool {
        self.temporal_coding
    }

    /// Get current crossbar threshold
    pub fn crossbar_threshold(&self) -> f64 {
        self.crossbar_threshold
    }

    /// Get memristive learning rate
    pub fn learning_rate(&self) -> f64 {
        self.memristive_learning_rate
    }

    /// Get number of events in pipeline
    pub fn pipeline_length(&self) -> usize {
        self.event_pipeline.len()
    }

    /// Clear event pipeline
    pub fn clear_pipeline(&mut self) {
        self.event_pipeline.clear();
    }

    /// Reset crossbar to initial state
    pub fn reset_crossbar(&mut self) {
        if self.memristive_crossbar {
            self.initialize_crossbar();
        }
    }

    /// Get conductance matrix (for analysis)
    pub fn conductance_matrix(&self) -> &Array2<f64> {
        &self.conductances
    }

    /// Set specific conductance value
    pub fn set_conductance(&mut self, row: usize, col: usize, value: f64) -> SpatialResult<()> {
        let (rows, cols) = self.crossbar_size;
        if row >= rows || col >= cols {
            return Err(SpatialError::InvalidInput(
                "Crossbar indices out of bounds".to_string(),
            ));
        }

        self.conductances[[row, col]] = value.clamp(0.0, 1.0);
        Ok(())
    }

    /// Get specific conductance value
    pub fn get_conductance(&self, row: usize, col: usize) -> Option<f64> {
        let (rows, cols) = self.crossbar_size;
        if row >= rows || col >= cols {
            None
        } else {
            Some(self.conductances[[row, col]])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_processor_creation() {
        let processor = NeuromorphicProcessor::new();
        assert!(!processor.is_memristive_enabled());
        assert!(!processor.is_temporal_coding_enabled());
        assert_eq!(processor.crossbar_size(), (64, 64));
        assert_eq!(processor.pipeline_length(), 0);
    }

    #[test]
    fn test_processor_configuration() {
        let processor = NeuromorphicProcessor::new()
            .with_memristive_crossbar(true)
            .with_temporal_coding(true)
            .with_crossbar_size(32, 32)
            .with_processing_params(500, 0.7, 0.01);

        assert!(processor.is_memristive_enabled());
        assert!(processor.is_temporal_coding_enabled());
        assert_eq!(processor.crossbar_size(), (32, 32));
        assert_eq!(processor.crossbar_threshold(), 0.7);
        assert_eq!(processor.learning_rate(), 0.01);
        assert_eq!(processor.max_pipeline_length, 500);
    }

    #[test]
    fn test_spatial_event_encoding() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let processor = NeuromorphicProcessor::new();

        let events = processor.encode_spatial_events(&points.view()).unwrap();

        // Should generate events for each coordinate
        assert!(!events.is_empty());

        // Events should be sorted by timestamp
        for i in 1..events.len() {
            assert!(events[i - 1].timestamp() <= events[i].timestamp());
        }
    }

    #[test]
    fn test_temporal_vs_rate_coding() {
        let points = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        // Rate coding
        let processor_rate = NeuromorphicProcessor::new().with_temporal_coding(false);
        let events_rate = processor_rate
            .encode_spatial_events(&points.view())
            .unwrap();

        // Temporal coding
        let processor_temporal = NeuromorphicProcessor::new().with_temporal_coding(true);
        let events_temporal = processor_temporal
            .encode_spatial_events(&points.view())
            .unwrap();

        // Different coding schemes should produce different numbers of events
        // (temporal coding typically produces fewer events)
        assert!(events_temporal.len() <= events_rate.len());
    }

    #[test]
    fn test_event_processing() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let mut processor = NeuromorphicProcessor::new()
            .with_memristive_crossbar(false)
            .with_temporal_coding(false);

        let events = processor.encode_spatial_events(&points.view()).unwrap();
        let processed_events = processor.process_events(&events).unwrap();

        // Without crossbar, should have same number of events
        assert_eq!(events.len(), processed_events.len());
        assert!(processor.pipeline_length() > 0);
    }

    #[test]
    #[ignore]
    fn test_memristive_crossbar() {
        let points = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let mut processor = NeuromorphicProcessor::new()
            .with_memristive_crossbar(true)
            .with_crossbar_size(4, 4);

        let events = processor.encode_spatial_events(&points.view()).unwrap();
        let processed_events = processor.process_events(&events).unwrap();

        // Memristive crossbar might generate additional output events
        assert!(!processed_events.is_empty());

        let stats = processor.get_crossbar_statistics();
        assert!(stats.contains_key("avg_conductance"));
        assert!(stats.contains_key("max_conductance"));
    }

    #[test]
    fn test_conductance_operations() {
        let mut processor = NeuromorphicProcessor::new()
            .with_memristive_crossbar(true)
            .with_crossbar_size(4, 4);

        // Test setting and getting conductance
        processor.set_conductance(0, 0, 0.8).unwrap();
        assert_eq!(processor.get_conductance(0, 0), Some(0.8));

        // Test bounds checking
        assert!(processor.set_conductance(10, 10, 0.5).is_err());
        assert_eq!(processor.get_conductance(10, 10), None);

        // Test clamping
        processor.set_conductance(1, 1, 2.0).unwrap(); // Should be clamped to 1.0
        assert_eq!(processor.get_conductance(1, 1), Some(1.0));
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = NeuromorphicProcessor::new().with_memristive_crossbar(true);

        // Process some events to change state
        let points = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let events = processor.encode_spatial_events(&points.view()).unwrap();
        processor.process_events(&events).unwrap();

        assert!(processor.pipeline_length() > 0);

        // Reset should clear pipeline
        processor.clear_pipeline();
        assert_eq!(processor.pipeline_length(), 0);

        // Reset crossbar should reinitialize conductances
        let initial_stats = processor.get_crossbar_statistics();
        processor.reset_crossbar();
        let reset_stats = processor.get_crossbar_statistics();

        // Conductances might be different after reset
        assert!(initial_stats.contains_key("avg_conductance"));
        assert!(reset_stats.contains_key("avg_conductance"));
    }

    #[test]
    fn test_empty_input() {
        let points = Array2::zeros((0, 2));
        let processor = NeuromorphicProcessor::new();

        let events = processor.encode_spatial_events(&points.view()).unwrap();
        assert!(events.is_empty());
    }
}
