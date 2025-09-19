//! Context for distributed linear algebra operations

use crate::distributed::{
    DistributedCommunicator, DistributedCoordinator, LoadBalancer,
    config::DistributedConfig, stats::DistributedStats
};
use crate::error::LinalgResult;

/// Context for distributed linear algebra operations
pub struct DistributedContext {
    /// Configuration
    pub config: DistributedConfig,

    /// Communicator
    pub communicator: DistributedCommunicator,

    /// Coordinator
    pub coordinator: DistributedCoordinator,

    /// Load balancer
    pub load_balancer: LoadBalancer,

    /// Statistics tracker
    pub stats: DistributedStats,
}

impl DistributedContext {
    /// Create new distributed context
    pub fn new(config: DistributedConfig) -> LinalgResult<Self> {
        let communicator = DistributedCommunicator::new(&config)?;
        let coordinator = DistributedCoordinator::new(&config)?;
        let load_balancer = LoadBalancer::new(&config)?;
        let stats = DistributedStats::new();

        Ok(Self {
            config,
            communicator,
            coordinator,
            load_balancer,
            stats,
        })
    }

    /// Finalize and return statistics
    pub fn finalize(mut self) -> LinalgResult<DistributedStats> {
        // Synchronize all nodes before shutdown
        self.coordinator.barrier()?;

        // Finalize communication
        self.communicator.finalize()?;

        Ok(self.stats)
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &DistributedStats {
        &self.stats
    }

    /// Update statistics
    pub fn update_stats(&mut self, update: impl FnOnce(&mut DistributedStats)) {
        update(&mut self.stats);
    }
}

/// Initialize distributed computing environment
#[allow(dead_code)]
pub fn initialize_distributed(config: DistributedConfig) -> LinalgResult<DistributedContext> {
    DistributedContext::new(config)
}

/// Shutdown distributed computing environment
#[allow(dead_code)]
pub fn finalize_distributed(context: DistributedContext) -> LinalgResult<DistributedStats> {
    context.finalize()
}