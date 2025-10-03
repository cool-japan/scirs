//! Client selection and sampling strategies for federated learning

use crate::error::Result;
use std::collections::HashMap;
/// Client selection strategy trait
pub trait ClientSelection: Send + Sync {
    /// Select clients for a round
    fn select(
        &self,
        available_clients: &[usize],
        num_select: usize,
        client_info: &HashMap<usize, ClientInfo>,
    ) -> Result<Vec<usize>>;
    /// Get strategy name
    fn name(&self) -> &str;
}
/// Client information for selection
#[derive(Debug, Clone)]
pub struct ClientInfo {
    /// Number of data samples
    pub num_samples: usize,
    /// Device type (mobile, desktop, server)
    pub device_type: String,
    /// Available compute (relative scale)
    pub compute_capacity: f32,
    /// Network bandwidth (Mbps)
    pub bandwidth: f32,
    /// Battery level (0-1, None for plugged devices)
    pub battery_level: Option<f32>,
    /// Historical reliability (0-1)
    pub reliability: f32,
    /// Data distribution info
    pub label_distribution: Vec<f32>,
/// Sampling strategy trait
pub trait SamplingStrategy: Send + Sync {
    /// Sample data from local dataset
    fn sample(&self, datasize: usize, round: usize) -> Result<Vec<usize>>;
/// Random client selection
pub struct RandomSelection {
    seed: Option<u64>,
impl RandomSelection {
    /// Create new random selection
    pub fn new() -> Self {
        Self { seed: None }
    }
    /// Set seed for reproducibility
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(_seed) }
impl ClientSelection for RandomSelection {
        _client_info: &HashMap<usize, ClientInfo>,
    ) -> Result<Vec<usize>> {
        use scirs2_core::random::prelude::*;
use scirs2_core::random::seq::SliceRandom;
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut rng()).unwrap()
        };
        let mut clients = available_clients.to_vec();
        clients.shuffle(&mut rng);
        Ok(clients.into_iter().take(num_select).collect())
    fn name(&self) -> &str {
        "RandomSelection"
/// Importance-based client selection
pub struct ImportanceSelection {
    /// Weight for data size
    data_weight: f32,
    /// Weight for compute capacity
    compute_weight: f32,
    /// Weight for reliability
    reliability_weight: f32,
impl ImportanceSelection {
    /// Create new importance selection
        Self {
            data_weight: 0.5,
            compute_weight: 0.3,
            reliability_weight: 0.2,
        }
    /// Set weights
    pub fn with_weights(data: f32, compute: f32, reliability: f32) -> Self {
            data_weight: data,
            compute_weight: compute,
            reliability_weight: reliability,
    /// Calculate client importance score
    fn calculate_importance(&self, info: &ClientInfo) -> f32 {
        let data_score = (info.num_samples as f32).ln();
        let compute_score = info.compute_capacity;
        let reliability_score = info.reliability;
        self.data_weight * data_score
            + self.compute_weight * compute_score
            + self.reliability_weight * reliability_score
impl ClientSelection for ImportanceSelection {
        // Calculate importance scores
        let mut scored_clients: Vec<(usize, f32)> = available_clients
            .iter()
            .filter_map(|&client_id| {
                client_info
                    .get(&client_id)
                    .map(|info| (client_id, self.calculate_importance(info)))
            })
            .collect();
        // Sort by importance (descending)
        scored_clients.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // Select top clients
        Ok(scored_clients
            .into_iter()
            .take(num_select)
            .map(|(id_)| id)
            .collect())
        "ImportanceSelection"
/// Power-aware client selection
pub struct PowerAwareSelection {
    /// Minimum battery threshold
    min_battery: f32,
    /// Prefer plugged devices
    prefer_plugged: bool,
impl PowerAwareSelection {
    /// Create new power-aware selection
    pub fn new(_minbattery: f32) -> Self {
            min_battery,
            prefer_plugged: true,
impl ClientSelection for PowerAwareSelection {
        let mut eligible_clients: Vec<(usize, f32)> = available_clients
                client_info.get(&client_id).and_then(|info| {
                    match info.battery_level {
                        None => Some((client_id, 2.0)), // Plugged device, highest priority
                        Some(level) if level >= self._min_battery => Some((client_id, level), _ => None, // Battery too low
                    }
                })
        // Sort by power status (plugged first, then by battery level)
        eligible_clients.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // Randomly sample from eligible clients if we have more than needed
        if eligible_clients.len() > num_select {
            let mut rng = rng();
            eligible_clients.partial_shuffle(&mut rng, num_select);
        Ok(eligible_clients
        "PowerAwareSelection"
/// Diversity-aware client selection
pub struct DiversitySelection {
    /// Number of clusters
    num_clusters: usize,
impl DiversitySelection {
    /// Create new diversity selection
    pub fn new(_numclusters: usize) -> Self {
        Self { _num_clusters }
    /// Simple clustering based on label distribution
    fn cluster_clients(&self, clientinfo: &HashMap<usize, ClientInfo>) -> HashMap<usize, usize> {
        let mut clusters = HashMap::new();
        // Simple k-means style clustering on label distributions
        for (&client_id, info) in client_info {
            // Assign to cluster based on dominant label
            let cluster = info
                .label_distribution
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx_)| idx % self.num_clusters)
                .unwrap_or(0);
            clusters.insert(client_id, cluster);
        clusters
impl ClientSelection for DiversitySelection {
        let clusters = self.cluster_clients(client_info);
        let clients_per_cluster = num_select / self.num_clusters;
        let remainder = num_select % self.num_clusters;
        let mut selected = Vec::new();
        // Group clients by cluster
        let mut cluster_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for &client_id in available_clients {
            if let Some(&cluster) = clusters.get(&client_id) {
                cluster_groups
                    .entry(cluster)
                    .or_insert_with(Vec::new)
                    .push(client_id);
            }
        // Select from each cluster
        let mut rng = rng();
        for (cluster_id, mut clients) in cluster_groups {
            clients.shuffle(&mut rng);
            let take = if cluster_id < remainder {
                clients_per_cluster + 1
            } else {
                clients_per_cluster
            };
            selected.extend(clients.into_iter().take(take));
        Ok(selected)
        "DiversitySelection"
/// Uniform data sampling
pub struct UniformSampling;
impl SamplingStrategy for UniformSampling {
    fn sample(&self, data_size: usize, round: usize) -> Result<Vec<usize>> {
        Ok((0..data_size).collect())
        "UniformSampling"
/// Stratified sampling
pub struct StratifiedSampling {
    /// Fraction to sample per stratum
    sample_fraction: f32,
impl StratifiedSampling {
    /// Create new stratified sampling
    pub fn new(_samplefraction: f32) -> Self {
        Self { _sample_fraction }
impl SamplingStrategy for StratifiedSampling {
        let sample_size = (data_size as f32 * self.sample_fraction) as usize;
        let mut indices: Vec<usize> = (0..data_size).collect();
        indices.partial_shuffle(&mut rng, sample_size);
        Ok(indices.into_iter().take(sample_size).collect())
        "StratifiedSampling"
/// Cyclic sampling
pub struct CyclicSampling {
    /// Batch size
    batch_size: usize,
impl CyclicSampling {
    /// Create new cyclic sampling
    pub fn new(_batchsize: usize) -> Self {
        Self { _batch_size }
impl SamplingStrategy for CyclicSampling {
    fn sample(&self, datasize: usize, round: usize) -> Result<Vec<usize>> {
        let start = (round * self.batch_size) % data_size;
        let indices: Vec<usize> = (0..self.batch_size)
            .map(|i| (start + i) % data_size)
        Ok(indices)
        "CyclicSampling"
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_random_selection() {
        let selector = RandomSelection::with_seed(42);
        let clients = vec![0, 1, 2, 3, 4];
        let info = HashMap::new();
        let selected = selector.select(&clients, 3, &info).unwrap();
        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|id| clients.contains(id)));
    fn test_importance_selection() {
        let selector = ImportanceSelection::new();
        let clients = vec![0, 1];
        let mut info = HashMap::new();
        info.insert(
            0,
            ClientInfo {
                num_samples: 1000,
                device_type: "server".to_string(),
                compute_capacity: 1.0,
                bandwidth: 100.0,
                battery_level: None,
                reliability: 0.9,
                label_distribution: vec![0.5, 0.5],
            },
        );
            1,
                num_samples: 100,
                device_type: "mobile".to_string(),
                compute_capacity: 0.2,
                bandwidth: 10.0,
                battery_level: Some(0.5),
                reliability: 0.7,
                label_distribution: vec![0.8, 0.2],
        let selected = selector.select(&clients, 1, &info).unwrap();
        assert_eq!(selected, vec![0]); // Should select client 0 due to higher importance
