//! Search space definitions for Neural Architecture Search

use crate::error::Result;
use std::collections::HashMap;
use std::fmt;
/// Represents a choice in the search space
#[derive(Debug, Clone)]
pub enum Choice {
    /// Discrete choice from a list of options
    Categorical(Vec<String>),
    /// Integer choice within a range
    Integer(i32, i32),
    /// Float choice within a range
    Float(f64, f64),
    /// Boolean choice
    Boolean,
}
/// Layer type choices
#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    Dense(usize),
    Conv2D {
        filters: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    Conv1D {
        kernel_size: usize,
        stride: usize,
    Conv3D {
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
    SeparableConv2D {
        depth_multiplier: usize,
    Conv2DTranspose {
        padding: (usize, usize),
    MaxPool2D {
        pool_size: (usize, usize),
    AvgPool2D {
    MaxPool1D {
        pool_size: usize,
    AvgPool1D {
    MaxPool3D {
        pool_size: (usize, usize, usize),
    AvgPool3D {
    GlobalMaxPool2D,
    GlobalAvgPool2D,
    GlobalMaxPool1D,
    GlobalAvgPool1D,
    GlobalMaxPool3D,
    GlobalAvgPool3D,
    UpSampling2D {
        size: (usize, usize),
    ZeroPadding2D {
    Cropping2D {
        cropping: (usize, usize),
    Concatenate {
        axis: i32,
    Add,
    Multiply,
    Dropout(f32),
    BatchNorm,
    LayerNorm,
    Activation(String),
    Residual,
    Attention {
        num_heads: usize,
        key_dim: usize,
    LSTM {
        units: usize,
        return_sequences: bool,
    GRU {
    Embedding {
        vocab_size: usize,
        embedding_dim: usize,
    Flatten,
    Reshape(Vec<i32>),
/// Search space configuration
pub struct SearchSpaceConfig {
    /// Available layer types
    pub layer_types: Vec<LayerType>,
    /// Minimum number of layers
    pub min_layers: usize,
    /// Maximum number of layers
    pub max_layers: usize,
    /// Available activation functions
    pub activations: Vec<String>,
    /// Width multiplier choices
    pub width_multipliers: Vec<f32>,
    /// Depth multiplier choices
    pub depth_multipliers: Vec<f32>,
    /// Skip connection probability
    pub skip_connection_prob: f32,
    /// Allow parallel branches
    pub allow_branches: bool,
    /// Maximum branches
    pub max_branches: usize,
    /// Custom choices
    pub custom_choices: HashMap<String, Choice>,
impl Default for SearchSpaceConfig {
    fn default() -> Self {
        Self {
            layer_types: vec![
                LayerType::Dense(64),
                LayerType::Dense(128),
                LayerType::Dense(256),
                LayerType::Conv2D {
                    filters: 32,
                    kernel_size: (3, 3),
                    stride: (1, 1),
                },
                    filters: 64,
                LayerType::MaxPool2D {
                    pool_size: (2, 2),
                    stride: (2, 2),
                LayerType::Dropout(0.2),
                LayerType::Dropout(0.5),
                LayerType::BatchNorm,
                LayerType::Activation("relu".to_string()),
                LayerType::Activation("swish".to_string()),
            ],
            min_layers: 3,
            max_layers: 20,
            activations: vec!["relu".to_string(), "swish".to_string(), "gelu".to_string()],
            width_multipliers: vec![0.5, 0.75, 1.0, 1.25, 1.5],
            depth_multipliers: vec![0.5, 0.75, 1.0, 1.25, 1.5],
            skip_connection_prob: 0.3,
            allow_branches: false,
            max_branches: 3,
            custom_choices: HashMap::new(),
        }
    }
/// Search space for neural architectures
pub struct SearchSpace {
    config: SearchSpaceConfig,
    layer_choices: Vec<LayerChoice>,
    connection_matrix: Option<ConnectionMatrix>,
/// Represents a choice of layer at a specific position
pub struct LayerChoice {
    /// Position in the network
    pub position: usize,
    /// Possible layer types at this position
    pub choices: Vec<LayerType>,
    /// Whether this layer is optional
    pub optional: bool,
/// Connection matrix for complex topologies
pub struct ConnectionMatrix {
    /// Number of layers
    pub num_layers: usize,
    /// Connection probabilities between layers
    pub connections: Vec<Vec<f32>>,
impl SearchSpace {
    /// Create a new search space
    pub fn new(config: SearchSpaceConfig) -> Result<Self> {
        let mut layer_choices = Vec::new();
        // Build layer choices based on _config
        for i in 0.._config.max_layers {
            let optional = i >= config.min_layers;
            layer_choices.push(LayerChoice {
                position: i,
                choices: config.layer_types.clone(),
                optional,
            });
        // Initialize connection matrix if branches are allowed
        let connection_matrix = if config.allow_branches {
            Some(ConnectionMatrix::new(
                config.max_layers,
                config.skip_connection_prob,
            ))
        } else {
            None
        };
        Ok(Self {
            config,
            layer_choices,
            connection_matrix,
        })
    /// Sample a random architecture from the search space
    pub fn sample(&self) -> Result<Architecture> {
        use rand::prelude::*;
        let mut rng = rng();
        let mut layers = Vec::new();
        let mut connections = Vec::new();
        // Sample number of layers
        let num_layers = rng.gen_range(self.config.min_layers..=self.config.max_layers);
        // Sample layers
        for i in 0..num_layers {
            if let Some(layer_choice) = self.layer_choices.get(i) {
                let idx = rng.gen_range(0..layer_choice.choices.len());
                layers.push(layer_choice.choices[idx].clone());
            }
        // Sample connections if branches are allowed
        if self.config.allow_branches {
            if let Some(matrix) = &self.connection_matrix {
                for i in 0..num_layers {
                    for j in (i + 1)..num_layers {
                        if rng.random::<f32>() < matrix.connections[i][j] {
                            connections.push((i..j));
                        }
                    }
                }
        // Sample width and depth multipliers
        let width_mult = self
            .config
            .width_multipliers
            .choose(&mut rng)
            .copied()
            .unwrap_or(1.0);
        let depth_mult = self
            .depth_multipliers
        Ok(Architecture {
            layers,
            connections,
            width_multiplier: width_mult,
            depth_multiplier: depth_mult,
    /// Get the size of the search space
    pub fn size(&self) -> f64 {
        let layer_combinations = self
            .layer_choices
            .iter()
            .take(self.config.max_layers)
            .map(|lc| lc.choices.len() as f64)
            .product::<f64>();
        let connection_combinations = if self.config.allow_branches {
            2f64.powf((self.config.max_layers * (self.config.max_layers - 1) / 2) as f64)
            1.0
        let multiplier_combinations =
            (self.config.width_multipliers.len() * self.config.depth_multipliers.len()) as f64;
        layer_combinations * connection_combinations * multiplier_combinations
    /// Mutate an architecture
    pub fn mutate(&self, architecture: &Architecture, mutationrate: f32) -> Result<Architecture> {
        let mut mutated = architecture.clone();
        // Mutate layers
        for (i, layer) in mutated.layers.iter_mut().enumerate() {
            if rng.random::<f32>() < mutation_rate {
                if let Some(layer_choice) = self.layer_choices.get(i) {
                    let idx = rng.gen_range(0..layer_choice.choices.len());
                    *layer = layer_choice.choices[idx].clone();
        // Add or remove layers
        if rng.random::<f32>() < mutation_rate {
            if mutated.layers.len() < self.config.max_layers && rng.gen_bool(0.5) {
                // Add a layer
                let pos = mutated.layers.len();
                if let Some(layer_choice) = self.layer_choices.get(pos) {
                    mutated.layers.push(layer_choice.choices[idx].clone());
            } else if mutated.layers.len() > self.config.min_layers {
                // Remove a layer
                let idx = rng.gen_range(0..mutated.layers.len());
                mutated.layers.remove(idx);
                // Update connections
                mutated.connections.retain(|(i..j)| *i != idx && *j != idx);
                for (i, j) in &mut mutated.connections {
                    if *i > idx {
                        *i -= 1;
                    if *j > idx {
                        *j -= 1;
        // Mutate connections
        if self.config.allow_branches && rng.random::<f32>() < mutation_rate {
            let num_layers = mutated.layers.len();
            for i in 0..num_layers {
                for j in (i + 1)..num_layers {
                    if rng.random::<f32>() < mutation_rate {
                        let has_connection = mutated.connections.contains(&(i, j));
                        if has_connection {
                            mutated.connections.retain(|(a, b)| !(*a == i && *b == j));
                        } else {
                            mutated.connections.push((i, j));
        // Mutate multipliers
            mutated.width_multiplier = *self
                .config
                .width_multipliers
                .choose(&mut rng)
                .unwrap_or(&1.0);
            mutated.depth_multiplier = *self
                .depth_multipliers
        Ok(mutated)
    /// Crossover two architectures
    pub fn crossover(
        &self,
        parent1: &Architecture,
        parent2: &Architecture,
    ) -> Result<Architecture> {
        // Determine child length
        let min_len = parent1.layers.len().min(parent2.layers.len());
        let max_len = parent1.layers.len().max(parent2.layers.len());
        let child_len = rng.gen_range(min_len..=max_len);
        let mut child_layers = Vec::new();
        let mut child_connections = Vec::new();
        // Crossover layers
        for i in 0..child_len {
            let layer = if i < parent1.layers.len() && i < parent2.layers.len() {
                // Both parents have this layer..choose randomly
                if rng.gen_bool(0.5) {
                    parent1.layers[i].clone()
                } else {
                    parent2.layers[i].clone()
            } else if i < parent1.layers.len() {
                parent1.layers[i].clone()
            } else {
                parent2.layers[i].clone()
            };
            child_layers.push(layer);
        // Crossover connections
            // Combine connections from both parents
            let mut all_connections = parent1.connections.clone();
            all_connections.extend(parent2.connections.clone());
            // Remove duplicates and invalid connections
            all_connections.sort_unstable();
            all_connections.dedup();
            child_connections = all_connections
                .into_iter()
                .filter(|(i, j)| *i < child_len && *j < child_len)
                .collect();
        // Crossover multipliers
        let width_multiplier = if rng.gen_bool(0.5) {
            parent1.width_multiplier
            parent2.width_multiplier
        let depth_multiplier = if rng.gen_bool(0.5) {
            parent1.depth_multiplier
            parent2.depth_multiplier
            layers: child_layers,
            connections: child_connections,
            width_multiplier,
            depth_multiplier,
impl ConnectionMatrix {
    /// Create a new connection matrix
    pub fn new(_num_layers: usize, skipprob: f32) -> Self {
        let mut connections = vec![vec![0.0; _num_layers]; _num_layers];
        // Initialize with skip connection probabilities
            for j in (i + 1).._num_layers {
                connections[i][j] = skip_prob;
            num_layers,
/// Represents a sampled architecture
pub struct Architecture {
    /// Layers in the architecture
    pub layers: Vec<LayerType>,
    /// Connections between layers (for non-sequential architectures)
    pub connections: Vec<(usize, usize)>,
    /// Width multiplier
    pub width_multiplier: f32,
    /// Depth multiplier  
    pub depth_multiplier: f32,
impl Architecture {
    /// Create a new architecture
    pub fn new(layers: Vec<LayerType>, connections: Vec<(usize, usize)>) -> Result<Self> {
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
    /// Create a new architecture with multipliers
    pub fn with_multipliers(
        layers: Vec<LayerType>,
        connections: Vec<(usize, usize)>,
        width_multiplier: f32,
        depth_multiplier: f32,
    ) -> Result<Self> {
impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Architecture:")?;
        writeln!(f, "  Layers:")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "    {}: {:?}", i, layer)?;
        if !self.connections.is_empty() {
            writeln!(f, "  Connections:")?;
            for (i, j) in &self.connections {
                writeln!(f, "    {} -> {}", i, j)?;
        writeln!(f, "  Width multiplier: {}", self.width_multiplier)?;
        writeln!(f, "  Depth multiplier: {}", self.depth_multiplier)?;
        Ok(())
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_search_space_creation() {
        let config = SearchSpaceConfig::default();
        let search_space = SearchSpace::new(config).unwrap();
        assert!(search_space.layer_choices.len() > 0);
    fn test_architecture_sampling() {
        let arch = search_space.sample().unwrap();
        assert!(arch.layers.len() >= search_space.config.min_layers);
        assert!(arch.layers.len() <= search_space.config.max_layers);
    fn test_architecture_mutation() {
        let mutated = search_space.mutate(&arch, 0.5).unwrap();
        // Mutation should produce a valid architecture
        assert!(mutated.layers.len() >= search_space.config.min_layers);
        assert!(mutated.layers.len() <= search_space.config.max_layers);
