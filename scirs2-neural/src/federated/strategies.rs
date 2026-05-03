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
}

/// Sampling strategy trait
pub trait SamplingStrategy: Send + Sync {
    /// Sample data from local dataset
    fn sample(&self, data_size: usize, round: usize) -> Result<Vec<usize>>;
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Random client selection
pub struct RandomSelection {
    seed: Option<u64>,
}

impl RandomSelection {
    /// Create new random selection
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Set seed for reproducibility
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(seed) }
    }
}

impl Default for RandomSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientSelection for RandomSelection {
    fn select(
        &self,
        available_clients: &[usize],
        num_select: usize,
        _client_info: &HashMap<usize, ClientInfo>,
    ) -> Result<Vec<usize>> {
        use scirs2_core::random::rng;
        use scirs2_core::random::seq::SliceRandom;
        let mut clients = available_clients.to_vec();
        if let Some(seed) = self.seed {
            use scirs2_core::random::rngs::StdRng;
            use scirs2_core::random::SeedableRng;
            let mut rng_inst = StdRng::seed_from_u64(seed);
            clients.shuffle(&mut rng_inst);
        } else {
            let mut rng_inst = rng();
            clients.shuffle(&mut rng_inst);
        }
        Ok(clients.into_iter().take(num_select).collect())
    }

    fn name(&self) -> &str {
        "RandomSelection"
    }
}

/// Importance-based client selection
pub struct ImportanceSelection {
    /// Weight for data size
    data_weight: f32,
    /// Weight for compute capacity
    compute_weight: f32,
    /// Weight for reliability
    reliability_weight: f32,
}

impl ImportanceSelection {
    /// Create new importance selection with default weights
    pub fn new() -> Self {
        Self {
            data_weight: 0.5,
            compute_weight: 0.3,
            reliability_weight: 0.2,
        }
    }

    /// Set weights
    pub fn with_weights(data: f32, compute: f32, reliability: f32) -> Self {
        Self {
            data_weight: data,
            compute_weight: compute,
            reliability_weight: reliability,
        }
    }

    /// Calculate client importance score
    fn calculate_importance(&self, info: &ClientInfo) -> f32 {
        let data_score = (info.num_samples as f32).ln();
        let compute_score = info.compute_capacity;
        let reliability_score = info.reliability;
        self.data_weight * data_score
            + self.compute_weight * compute_score
            + self.reliability_weight * reliability_score
    }
}

impl Default for ImportanceSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientSelection for ImportanceSelection {
    fn select(
        &self,
        available_clients: &[usize],
        num_select: usize,
        client_info: &HashMap<usize, ClientInfo>,
    ) -> Result<Vec<usize>> {
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
        scored_clients.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored_clients
            .into_iter()
            .take(num_select)
            .map(|(id, _)| id)
            .collect())
    }

    fn name(&self) -> &str {
        "ImportanceSelection"
    }
}

/// Power-aware client selection
pub struct PowerAwareSelection {
    /// Minimum battery threshold
    min_battery: f32,
    /// Prefer plugged devices
    #[allow(dead_code)]
    prefer_plugged: bool,
}

impl PowerAwareSelection {
    /// Create new power-aware selection
    pub fn new(min_battery: f32) -> Self {
        Self {
            min_battery,
            prefer_plugged: true,
        }
    }
}

impl ClientSelection for PowerAwareSelection {
    fn select(
        &self,
        available_clients: &[usize],
        num_select: usize,
        client_info: &HashMap<usize, ClientInfo>,
    ) -> Result<Vec<usize>> {
        let mut eligible_clients: Vec<(usize, f32)> = available_clients
            .iter()
            .filter_map(|&client_id| {
                client_info.get(&client_id).and_then(|info| {
                    match info.battery_level {
                        None => Some((client_id, 2.0)), // Plugged device, highest priority
                        Some(level) if level >= self.min_battery => Some((client_id, level)),
                        _ => None, // Battery too low
                    }
                })
            })
            .collect();

        // Sort by power status (plugged first, then by battery level)
        eligible_clients.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(eligible_clients
            .into_iter()
            .take(num_select)
            .map(|(id, _)| id)
            .collect())
    }

    fn name(&self) -> &str {
        "PowerAwareSelection"
    }
}

/// Diversity-aware client selection
pub struct DiversitySelection {
    /// Number of clusters
    num_clusters: usize,
}

impl DiversitySelection {
    /// Create new diversity selection
    pub fn new(num_clusters: usize) -> Self {
        Self { num_clusters }
    }

    /// Simple clustering based on label distribution
    fn cluster_clients(&self, client_info: &HashMap<usize, ClientInfo>) -> HashMap<usize, usize> {
        let mut clusters = HashMap::new();
        for (&client_id, info) in client_info {
            // Assign to cluster based on dominant label index
            let cluster = info
                .label_distribution
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx % self.num_clusters)
                .unwrap_or(0);
            clusters.insert(client_id, cluster);
        }
        clusters
    }
}

impl ClientSelection for DiversitySelection {
    fn select(
        &self,
        available_clients: &[usize],
        num_select: usize,
        client_info: &HashMap<usize, ClientInfo>,
    ) -> Result<Vec<usize>> {
        let clusters = self.cluster_clients(client_info);
        let clients_per_cluster = num_select / self.num_clusters.max(1);
        let remainder = num_select % self.num_clusters.max(1);

        let mut selected = Vec::new();
        // Group clients by cluster
        let mut cluster_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for &client_id in available_clients {
            if let Some(&cluster) = clusters.get(&client_id) {
                cluster_groups.entry(cluster).or_default().push(client_id);
            }
        }

        // Select from each cluster
        use scirs2_core::random::rng;
        use scirs2_core::random::seq::SliceRandom;
        let mut rng_inst = rng();
        let mut cluster_ids: Vec<usize> = cluster_groups.keys().copied().collect();
        cluster_ids.sort();
        for (i, cluster_id) in cluster_ids.iter().enumerate() {
            if let Some(mut clients) = cluster_groups.remove(cluster_id) {
                clients.shuffle(&mut rng_inst);
                let take = if i < remainder {
                    clients_per_cluster + 1
                } else {
                    clients_per_cluster
                };
                selected.extend(clients.into_iter().take(take));
            }
        }
        Ok(selected)
    }

    fn name(&self) -> &str {
        "DiversitySelection"
    }
}

/// Uniform data sampling
pub struct UniformSampling;

impl UniformSampling {
    /// Create new uniform sampling
    pub fn new() -> Self {
        Self
    }
}

impl Default for UniformSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingStrategy for UniformSampling {
    fn sample(&self, data_size: usize, _round: usize) -> Result<Vec<usize>> {
        Ok((0..data_size).collect())
    }

    fn name(&self) -> &str {
        "UniformSampling"
    }
}

/// Stratified sampling
pub struct StratifiedSampling {
    /// Fraction to sample per stratum
    sample_fraction: f32,
}

impl StratifiedSampling {
    /// Create new stratified sampling
    pub fn new(sample_fraction: f32) -> Self {
        Self { sample_fraction }
    }
}

impl SamplingStrategy for StratifiedSampling {
    fn sample(&self, data_size: usize, _round: usize) -> Result<Vec<usize>> {
        use scirs2_core::random::rng;
        use scirs2_core::random::seq::SliceRandom;
        let sample_size = (data_size as f32 * self.sample_fraction) as usize;
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng_inst = rng();
        indices.shuffle(&mut rng_inst);
        Ok(indices.into_iter().take(sample_size).collect())
    }

    fn name(&self) -> &str {
        "StratifiedSampling"
    }
}

/// Cyclic sampling
pub struct CyclicSampling {
    /// Batch size
    batch_size: usize,
}

impl CyclicSampling {
    /// Create new cyclic sampling
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
}

impl SamplingStrategy for CyclicSampling {
    fn sample(&self, data_size: usize, round: usize) -> Result<Vec<usize>> {
        let start = (round * self.batch_size) % data_size.max(1);
        let indices: Vec<usize> = (0..self.batch_size)
            .map(|i| (start + i) % data_size.max(1))
            .collect();
        Ok(indices)
    }

    fn name(&self) -> &str {
        "CyclicSampling"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_selection() {
        let selector = RandomSelection::with_seed(42);
        let clients = vec![0, 1, 2, 3, 4];
        let info = HashMap::new();
        let selected = selector.select(&clients, 3, &info).expect("select failed");
        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|id| clients.contains(id)));
    }

    #[test]
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
        info.insert(
            1,
            ClientInfo {
                num_samples: 100,
                device_type: "mobile".to_string(),
                compute_capacity: 0.2,
                bandwidth: 10.0,
                battery_level: Some(0.5),
                reliability: 0.7,
                label_distribution: vec![0.8, 0.2],
            },
        );
        let selected = selector.select(&clients, 1, &info).expect("select failed");
        assert_eq!(selected, vec![0]); // Should select client 0 due to higher importance
    }

    #[test]
    fn test_uniform_sampling() {
        let sampler = UniformSampling::new();
        let indices = sampler.sample(5, 0).expect("sample failed");
        assert_eq!(indices.len(), 5);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_cyclic_sampling() {
        let sampler = CyclicSampling::new(3);
        let indices = sampler.sample(10, 0).expect("sample failed");
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
