//! Resource Scheduling for TPU Pod Coordination
//!
//! This module provides comprehensive resource scheduling functionality for TPU pod coordination,
//! including device allocation, resource management, and scheduling optimization.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use num_traits::Float;

use super::{DeviceId, BatchId};
use super::load_balancing::{DeviceAvailability, LoadBalancer, LoadBalancingAlgorithm};
use crate::error::{OptimError, Result};

/// Resource requirements for batch execution
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirement (bytes)
    pub memory_bytes: usize,

    /// Compute requirement (FLOPS)
    pub compute_flops: u64,

    /// Communication bandwidth (GB/s)
    pub communication_bandwidth: f64,

    /// Preferred devices
    pub preferred_devices: Vec<DeviceId>,

    /// Minimum devices required
    pub min_devices: usize,

    /// Maximum devices that can be used
    pub max_devices: usize,

    /// Priority level
    pub priority: ResourcePriority,

    /// Deadline for allocation
    pub deadline: Option<Instant>,
}

/// Resource priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourcePriority {
    Low,
    Normal,
    High,
    Critical,
    Realtime,
}

/// Resource allocation result
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocated devices
    pub devices: Vec<DeviceId>,

    /// Memory allocation per device
    pub memory_allocation: HashMap<DeviceId, usize>,

    /// Allocation timestamp
    pub allocated_at: Instant,

    /// Allocation duration
    pub duration: Duration,

    /// Allocation quality score
    pub quality_score: f64,

    /// Resource efficiency
    pub efficiency: f64,
}

/// Scheduling request
#[derive(Debug, Clone)]
pub struct SchedulingRequest {
    /// Batch ID
    pub batch_id: BatchId,

    /// Resource requirements
    pub resource_requirements: ResourceRequirements,

    /// Submitted timestamp
    pub submitted_at: Instant,

    /// Request status
    pub status: RequestStatus,

    /// Scheduling attempts
    pub attempts: usize,
}

/// Request status
#[derive(Debug, Clone, Copy)]
pub enum RequestStatus {
    Pending,
    Scheduled,
    Allocated,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    FirstComeFirstServed,
    ShortestJobFirst,
    PriorityBased,
    RoundRobin,
    FairShare,
    Backfill,
    Gang,
    Adaptive,
}

/// Resource pool configuration
#[derive(Debug, Clone)]
pub struct ResourcePoolConfig {
    /// Total devices in pool
    pub total_devices: usize,

    /// Memory per device (bytes)
    pub memory_per_device: usize,

    /// Compute capacity per device (FLOPS)
    pub compute_per_device: u64,

    /// Communication bandwidth per device (GB/s)
    pub bandwidth_per_device: f64,

    /// Oversubscription ratio
    pub oversubscription_ratio: f64,

    /// Reservation policy
    pub reservation_policy: ReservationPolicy,
}

/// Reservation policy
#[derive(Debug, Clone, Copy)]
pub enum ReservationPolicy {
    NoReservation,
    StaticReservation,
    DynamicReservation,
    AdaptiveReservation,
}

/// Device reservation
#[derive(Debug, Clone)]
pub struct DeviceReservation {
    /// Reserved device
    pub device_id: DeviceId,

    /// Reservation start time
    pub start_time: Instant,

    /// Reservation duration
    pub duration: Duration,

    /// Reserved by batch
    pub batch_id: BatchId,

    /// Reservation type
    pub reservation_type: ReservationType,
}

/// Reservation types
#[derive(Debug, Clone, Copy)]
pub enum ReservationType {
    Exclusive,
    Shared,
    Preemptible,
    BestEffort,
}

/// Scheduling statistics
pub type SchedulingStatistics = HashMap<String, f64>;

/// Queue metrics
#[derive(Debug, Clone)]
pub struct QueueMetrics {
    /// Queue length
    pub queue_length: usize,

    /// Average wait time
    pub average_wait_time: Duration,

    /// Throughput (requests/second)
    pub throughput: f64,

    /// Success rate
    pub success_rate: f64,

    /// Resource utilization
    pub resource_utilization: f64,
}

/// Allocation metrics
#[derive(Debug, Clone)]
pub struct AllocationMetrics {
    /// Total allocations
    pub total_allocations: usize,

    /// Successful allocations
    pub successful_allocations: usize,

    /// Failed allocations
    pub failed_allocations: usize,

    /// Average allocation time
    pub average_allocation_time: Duration,

    /// Resource efficiency
    pub resource_efficiency: f64,

    /// Fragmentation level
    pub fragmentation_level: f64,
}

/// Resource scheduler for TPU coordination
#[derive(Debug)]
pub struct ResourceScheduler<T: Float> {
    /// Resource pool configuration
    pool_config: ResourcePoolConfig,

    /// Active allocations
    pub active_allocations: HashMap<BatchId, ResourceAllocation>,

    /// Device availability
    device_availability: HashMap<DeviceId, DeviceAvailability>,

    /// Scheduling queue
    scheduling_queue: VecDeque<SchedulingRequest>,

    /// Load balancer
    load_balancer: LoadBalancer,

    /// Scheduling strategy
    scheduling_strategy: SchedulingStrategy,

    /// Device reservations
    device_reservations: HashMap<DeviceId, Vec<DeviceReservation>>,

    /// Scheduling statistics
    statistics: SchedulingStatistics,

    /// Queue metrics
    queue_metrics: QueueMetrics,

    /// Allocation metrics
    allocation_metrics: AllocationMetrics,

    /// Phantom type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Send + Sync> ResourceScheduler<T> {
    /// Create a new resource scheduler
    pub fn new(pool_config: ResourcePoolConfig) -> Result<Self> {
        let mut device_availability = HashMap::new();

        // Initialize device availability for all devices in the pool
        for device_id in 0..pool_config.total_devices {
            device_availability.insert(
                DeviceId(device_id),
                DeviceAvailability {
                    available_memory: pool_config.memory_per_device,
                    compute_capacity: 1.0, // Normalized capacity
                    communication_bandwidth: pool_config.bandwidth_per_device,
                    current_load: 0.0,
                    reserved_until: None,
                },
            );
        }

        let load_balancer = LoadBalancer::new();

        Ok(Self {
            pool_config,
            active_allocations: HashMap::new(),
            device_availability,
            scheduling_queue: VecDeque::new(),
            load_balancer,
            scheduling_strategy: SchedulingStrategy::Adaptive,
            device_reservations: HashMap::new(),
            statistics: HashMap::new(),
            queue_metrics: QueueMetrics {
                queue_length: 0,
                average_wait_time: Duration::from_secs(0),
                throughput: 0.0,
                success_rate: 1.0,
                resource_utilization: 0.0,
            },
            allocation_metrics: AllocationMetrics {
                total_allocations: 0,
                successful_allocations: 0,
                failed_allocations: 0,
                average_allocation_time: Duration::from_secs(0),
                resource_efficiency: 0.0,
                fragmentation_level: 0.0,
            },
            _phantom: std::marker::PhantomData,
        })
    }

    /// Submit scheduling request
    pub fn submit_request(&mut self, mut request: SchedulingRequest) -> Result<()> {
        request.status = RequestStatus::Pending;
        request.submitted_at = Instant::now();
        self.scheduling_queue.push_back(request);
        self.update_queue_metrics();
        Ok(())
    }

    /// Allocate resources for a batch
    pub async fn allocate_resources(&mut self, batch_id: BatchId) -> Result<ResourceAllocation> {
        let allocation_start = Instant::now();

        // Find the request in the queue
        let request_index = self.scheduling_queue
            .iter()
            .position(|req| req.batch_id == batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Request not found in queue".to_string()))?;

        let mut request = self.scheduling_queue.remove(request_index).unwrap();
        request.status = RequestStatus::Scheduled;

        // Find suitable devices
        let allocation = self.find_allocation(&request).await?;

        // Update device availability
        for &device_id in &allocation.devices {
            if let Some(availability) = self.device_availability.get_mut(&device_id) {
                let memory_allocated = allocation.memory_allocation.get(&device_id).unwrap_or(&0);
                availability.available_memory = availability.available_memory.saturating_sub(*memory_allocated);
                availability.current_load += 1.0 / allocation.devices.len() as f64;
            }
        }

        request.status = RequestStatus::Allocated;
        self.active_allocations.insert(batch_id, allocation.clone());

        // Update metrics
        self.allocation_metrics.total_allocations += 1;
        self.allocation_metrics.successful_allocations += 1;
        let allocation_time = allocation_start.elapsed();
        self.allocation_metrics.average_allocation_time =
            (self.allocation_metrics.average_allocation_time + allocation_time) / 2;

        self.update_statistics();
        Ok(allocation)
    }

    /// Find suitable allocation for a request
    async fn find_allocation(&self, request: &SchedulingRequest) -> Result<ResourceAllocation> {
        let requirements = &request.resource_requirements;

        // Filter available devices
        let available_devices: Vec<DeviceId> = self.device_availability
            .iter()
            .filter(|(_, availability)| {
                availability.available_memory >= requirements.memory_bytes &&
                availability.current_load < 0.8 &&
                availability.reserved_until.map_or(true, |until| until < Instant::now())
            })
            .map(|(device_id, _)| *device_id)
            .collect();

        if available_devices.len() < requirements.min_devices {
            return Err(OptimError::ResourceUnavailable(
                format!("Insufficient devices available: {} required, {} available",
                    requirements.min_devices, available_devices.len())
            ));
        }

        // Select devices based on scheduling strategy
        let selected_devices = self.select_devices(&available_devices, requirements)?;

        // Calculate memory allocation per device
        let mut memory_allocation = HashMap::new();
        let memory_per_device = requirements.memory_bytes / selected_devices.len();

        for &device_id in &selected_devices {
            memory_allocation.insert(device_id, memory_per_device);
        }

        // Calculate allocation quality
        let quality_score = self.calculate_allocation_quality(&selected_devices, requirements);

        // Calculate resource efficiency
        let efficiency = self.calculate_resource_efficiency(&selected_devices, requirements);

        Ok(ResourceAllocation {
            devices: selected_devices,
            memory_allocation,
            allocated_at: Instant::now(),
            duration: Duration::from_secs(300), // Default 5 minutes
            quality_score,
            efficiency,
        })
    }

    /// Select devices based on scheduling strategy
    fn select_devices(
        &self,
        available_devices: &[DeviceId],
        requirements: &ResourceRequirements,
    ) -> Result<Vec<DeviceId>> {
        let target_count = requirements.max_devices.min(available_devices.len());

        match self.scheduling_strategy {
            SchedulingStrategy::FirstFit => {
                Ok(available_devices.iter().take(target_count).cloned().collect())
            }
            SchedulingStrategy::BestFit => {
                let mut device_scores: Vec<_> = available_devices
                    .iter()
                    .map(|&device_id| {
                        let score = self.calculate_device_fitness(device_id, requirements);
                        (device_id, score)
                    })
                    .collect();

                device_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                Ok(device_scores
                    .into_iter()
                    .take(target_count)
                    .map(|(device_id, _)| device_id)
                    .collect())
            }
            SchedulingStrategy::PriorityBased => {
                // Consider preferred devices first
                let mut selected = Vec::new();

                // Add preferred devices if available
                for &preferred in &requirements.preferred_devices {
                    if available_devices.contains(&preferred) && selected.len() < target_count {
                        selected.push(preferred);
                    }
                }

                // Fill remaining slots with other devices
                for &device_id in available_devices {
                    if !selected.contains(&device_id) && selected.len() < target_count {
                        selected.push(device_id);
                    }
                }

                Ok(selected)
            }
            SchedulingStrategy::Adaptive => {
                // Use load balancer for adaptive selection
                Ok(self.load_balancer.select_optimal_devices(
                    available_devices,
                    &self.device_availability,
                ).into_iter().take(target_count).collect())
            }
            _ => {
                // Default to first fit for other strategies
                Ok(available_devices.iter().take(target_count).cloned().collect())
            }
        }
    }

    /// Calculate device fitness for scheduling
    fn calculate_device_fitness(&self, device_id: DeviceId, requirements: &ResourceRequirements) -> f64 {
        if let Some(availability) = self.device_availability.get(&device_id) {
            let memory_ratio = availability.available_memory as f64 / requirements.memory_bytes as f64;
            let load_factor = 1.0 - availability.current_load;
            let bandwidth_ratio = availability.communication_bandwidth / requirements.communication_bandwidth;

            // Prefer devices with slightly more resources than needed (to avoid waste)
            let memory_score = if memory_ratio >= 1.0 && memory_ratio <= 2.0 {
                1.0 / memory_ratio // Prefer closer matches
            } else if memory_ratio > 2.0 {
                0.5 // Penalize excessive over-allocation
            } else {
                0.0 // Insufficient memory
            };

            (memory_score + load_factor + bandwidth_ratio.min(1.0)) / 3.0
        } else {
            0.0
        }
    }

    /// Calculate allocation quality score
    fn calculate_allocation_quality(&self, devices: &[DeviceId], requirements: &ResourceRequirements) -> f64 {
        let mut total_score = 0.0;

        for &device_id in devices {
            if let Some(availability) = self.device_availability.get(&device_id) {
                let memory_ratio = availability.available_memory as f64 / requirements.memory_bytes as f64;
                let load_score = 1.0 - availability.current_load;
                let bandwidth_score = (availability.communication_bandwidth / requirements.communication_bandwidth).min(1.0);

                let device_score = (memory_ratio.min(2.0) / 2.0 + load_score + bandwidth_score) / 3.0;
                total_score += device_score;
            }
        }

        if devices.is_empty() {
            0.0
        } else {
            total_score / devices.len() as f64
        }
    }

    /// Calculate resource efficiency
    fn calculate_resource_efficiency(&self, devices: &[DeviceId], requirements: &ResourceRequirements) -> f64 {
        let total_available_memory: usize = devices
            .iter()
            .filter_map(|&device_id| self.device_availability.get(&device_id))
            .map(|availability| availability.available_memory)
            .sum();

        let memory_efficiency = if total_available_memory == 0 {
            0.0
        } else {
            requirements.memory_bytes as f64 / total_available_memory as f64
        };

        let device_efficiency = requirements.min_devices as f64 / devices.len() as f64;

        (memory_efficiency + device_efficiency) / 2.0
    }

    /// Release resources for a batch
    pub fn release_resources(&mut self, batch_id: BatchId) -> Result<()> {
        if let Some(allocation) = self.active_allocations.remove(&batch_id) {
            // Release device resources
            for device_id in allocation.devices {
                if let Some(availability) = self.device_availability.get_mut(&device_id) {
                    let memory_to_release = allocation.memory_allocation.get(&device_id).unwrap_or(&0);
                    availability.available_memory += memory_to_release;
                    availability.current_load = (availability.current_load - 0.25).max(0.0);
                }
            }

            // Remove any reservations for this batch
            for reservations in self.device_reservations.values_mut() {
                reservations.retain(|reservation| reservation.batch_id != batch_id);
            }

            self.update_statistics();
        }
        Ok(())
    }

    /// Create device reservation
    pub fn create_reservation(
        &mut self,
        device_id: DeviceId,
        batch_id: BatchId,
        duration: Duration,
        reservation_type: ReservationType,
    ) -> Result<()> {
        let reservation = DeviceReservation {
            device_id,
            start_time: Instant::now(),
            duration,
            batch_id,
            reservation_type,
        };

        self.device_reservations
            .entry(device_id)
            .or_insert_with(Vec::new)
            .push(reservation);

        // Update device availability
        if let Some(availability) = self.device_availability.get_mut(&device_id) {
            availability.reserved_until = Some(Instant::now() + duration);
        }

        Ok(())
    }

    /// Cancel reservation
    pub fn cancel_reservation(&mut self, device_id: DeviceId, batch_id: BatchId) -> Result<()> {
        if let Some(reservations) = self.device_reservations.get_mut(&device_id) {
            reservations.retain(|reservation| reservation.batch_id != batch_id);

            // Update device availability if no more reservations
            if reservations.is_empty() {
                if let Some(availability) = self.device_availability.get_mut(&device_id) {
                    availability.reserved_until = None;
                }
            }
        }
        Ok(())
    }

    /// Process scheduling queue
    pub async fn process_queue(&mut self) -> Result<usize> {
        let mut processed = 0;
        let queue_snapshot: Vec<_> = self.scheduling_queue.iter().cloned().collect();

        for request in queue_snapshot {
            if request.status == RequestStatus::Pending {
                match self.allocate_resources(request.batch_id).await {
                    Ok(_) => processed += 1,
                    Err(_) => {
                        // Keep request in queue or mark as failed based on policy
                        if request.attempts >= 3 {
                            self.mark_request_failed(request.batch_id);
                        }
                    }
                }
            }
        }

        self.cleanup_expired_reservations();
        self.update_queue_metrics();
        Ok(processed)
    }

    /// Mark request as failed
    fn mark_request_failed(&mut self, batch_id: BatchId) {
        if let Some(request) = self.scheduling_queue.iter_mut().find(|req| req.batch_id == batch_id) {
            request.status = RequestStatus::Failed;
        }
        self.allocation_metrics.failed_allocations += 1;
    }

    /// Cleanup expired reservations
    fn cleanup_expired_reservations(&mut self) {
        let now = Instant::now();

        for (device_id, reservations) in &mut self.device_reservations {
            reservations.retain(|reservation| {
                now < reservation.start_time + reservation.duration
            });

            // Update device availability if all reservations expired
            if reservations.is_empty() {
                if let Some(availability) = self.device_availability.get_mut(device_id) {
                    availability.reserved_until = None;
                }
            }
        }
    }

    /// Update queue metrics
    fn update_queue_metrics(&mut self) {
        self.queue_metrics.queue_length = self.scheduling_queue.len();

        if !self.scheduling_queue.is_empty() {
            let total_wait_time: Duration = self.scheduling_queue
                .iter()
                .map(|req| req.submitted_at.elapsed())
                .sum();

            self.queue_metrics.average_wait_time = total_wait_time / self.scheduling_queue.len() as u32;
        }

        // Calculate resource utilization
        let total_devices = self.pool_config.total_devices as f64;
        let allocated_devices = self.active_allocations
            .values()
            .map(|allocation| allocation.devices.len())
            .sum::<usize>() as f64;

        self.queue_metrics.resource_utilization = allocated_devices / total_devices;

        // Calculate success rate
        let total_requests = self.allocation_metrics.total_allocations;
        if total_requests > 0 {
            self.queue_metrics.success_rate =
                self.allocation_metrics.successful_allocations as f64 / total_requests as f64;
        }
    }

    /// Update scheduling statistics
    fn update_statistics(&mut self) {
        self.statistics.insert("active_allocations".to_string(), self.active_allocations.len() as f64);
        self.statistics.insert("queue_length".to_string(), self.queue_metrics.queue_length as f64);
        self.statistics.insert("resource_utilization".to_string(), self.queue_metrics.resource_utilization);
        self.statistics.insert("success_rate".to_string(), self.queue_metrics.success_rate);
        self.statistics.insert("average_wait_time_ms".to_string(), self.queue_metrics.average_wait_time.as_millis() as f64);

        // Calculate fragmentation
        let available_devices = self.device_availability
            .values()
            .filter(|availability| availability.current_load < 0.8)
            .count() as f64;
        let total_devices = self.pool_config.total_devices as f64;
        let fragmentation = if total_devices > 0.0 {
            1.0 - (available_devices / total_devices)
        } else {
            0.0
        };
        self.allocation_metrics.fragmentation_level = fragmentation;
        self.statistics.insert("fragmentation_level".to_string(), fragmentation);
    }

    /// Get scheduling statistics
    pub fn get_statistics(&self) -> &SchedulingStatistics {
        &self.statistics
    }

    /// Get queue metrics
    pub fn get_queue_metrics(&self) -> &QueueMetrics {
        &self.queue_metrics
    }

    /// Get allocation metrics
    pub fn get_allocation_metrics(&self) -> &AllocationMetrics {
        &self.allocation_metrics
    }

    /// Set scheduling strategy
    pub fn set_scheduling_strategy(&mut self, strategy: SchedulingStrategy) {
        self.scheduling_strategy = strategy;
    }

    /// Get device availability
    pub fn get_device_availability(&self, device_id: DeviceId) -> Option<&DeviceAvailability> {
        self.device_availability.get(&device_id)
    }

    /// Get all device availability
    pub fn get_all_device_availability(&self) -> &HashMap<DeviceId, DeviceAvailability> {
        &self.device_availability
    }

    /// Get active allocations
    pub fn get_active_allocations(&self) -> &HashMap<BatchId, ResourceAllocation> {
        &self.active_allocations
    }

    /// Get device reservations
    pub fn get_device_reservations(&self, device_id: DeviceId) -> Option<&Vec<DeviceReservation>> {
        self.device_reservations.get(&device_id)
    }

    /// Update pool configuration
    pub fn update_pool_config(&mut self, config: ResourcePoolConfig) {
        self.pool_config = config;
        // Reinitialize device availability if pool size changed
        // Implementation would depend on specific requirements
    }

    /// Get scheduling queue status
    pub fn get_queue_status(&self) -> Vec<(BatchId, RequestStatus, Duration)> {
        self.scheduling_queue
            .iter()
            .map(|req| (req.batch_id, req.status, req.submitted_at.elapsed()))
            .collect()
    }
}

// Default implementations
impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            memory_bytes: 1024 * 1024 * 1024, // 1GB
            compute_flops: 1_000_000_000,      // 1 GFLOP
            communication_bandwidth: 10.0,     // 10 GB/s
            preferred_devices: Vec::new(),
            min_devices: 1,
            max_devices: 4,
            priority: ResourcePriority::Normal,
            deadline: None,
        }
    }
}

impl Default for ResourcePoolConfig {
    fn default() -> Self {
        Self {
            total_devices: 16,
            memory_per_device: 16 * 1024 * 1024 * 1024, // 16GB
            compute_per_device: 100_000_000_000,         // 100 GFLOPS
            bandwidth_per_device: 100.0,                 // 100 GB/s
            oversubscription_ratio: 1.2,
            reservation_policy: ReservationPolicy::DynamicReservation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_scheduler_creation() {
        let config = ResourcePoolConfig::default();
        let scheduler = ResourceScheduler::<f32>::new(config);
        assert!(scheduler.is_ok());
    }

    #[test]
    fn test_scheduling_request_submission() {
        let config = ResourcePoolConfig::default();
        let mut scheduler = ResourceScheduler::<f32>::new(config).unwrap();

        let request = SchedulingRequest {
            batch_id: BatchId(1),
            resource_requirements: ResourceRequirements::default(),
            submitted_at: Instant::now(),
            status: RequestStatus::Pending,
            attempts: 0,
        };

        let result = scheduler.submit_request(request);
        assert!(result.is_ok());
        assert_eq!(scheduler.get_queue_metrics().queue_length, 1);
    }

    #[tokio::test]
    async fn test_resource_allocation() {
        let config = ResourcePoolConfig::default();
        let mut scheduler = ResourceScheduler::<f32>::new(config).unwrap();

        let request = SchedulingRequest {
            batch_id: BatchId(1),
            resource_requirements: ResourceRequirements {
                min_devices: 2,
                max_devices: 4,
                ..Default::default()
            },
            submitted_at: Instant::now(),
            status: RequestStatus::Pending,
            attempts: 0,
        };

        scheduler.submit_request(request).unwrap();
        let allocation = scheduler.allocate_resources(BatchId(1)).await;
        assert!(allocation.is_ok());

        let allocation = allocation.unwrap();
        assert!(allocation.devices.len() >= 2);
        assert!(allocation.devices.len() <= 4);
    }

    #[test]
    fn test_device_reservation() {
        let config = ResourcePoolConfig::default();
        let mut scheduler = ResourceScheduler::<f32>::new(config).unwrap();

        let result = scheduler.create_reservation(
            DeviceId(0),
            BatchId(1),
            Duration::from_secs(300),
            ReservationType::Exclusive,
        );

        assert!(result.is_ok());

        let reservations = scheduler.get_device_reservations(DeviceId(0));
        assert!(reservations.is_some());
        assert_eq!(reservations.unwrap().len(), 1);
    }
}