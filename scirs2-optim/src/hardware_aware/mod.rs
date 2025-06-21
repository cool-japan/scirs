//! Hardware-aware optimization routines
//!
//! This module provides optimization strategies that adapt to different hardware configurations,
//! including CPUs, GPUs, TPUs, edge devices, and distributed systems.

use crate::error::Result;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Hardware platform types
#[derive(Debug, Clone, PartialEq)]
pub enum HardwarePlatform {
    /// CPU-based computation
    CPU {
        /// Number of cores
        cores: usize,
        /// Cache size in bytes
        cache_size: usize,
        /// SIMD instruction set availability
        simd_support: SIMDSupport,
    },
    /// GPU-based computation
    GPU {
        /// GPU memory in bytes
        memory: usize,
        /// Number of compute units/streaming multiprocessors
        compute_units: usize,
        /// Memory bandwidth in GB/s
        memory_bandwidth: f64,
        /// GPU architecture
        architecture: GPUArchitecture,
    },
    /// TPU (Tensor Processing Unit)
    TPU {
        /// TPU version
        version: TPUVersion,
        /// Matrix multiplication units
        matrix_units: usize,
        /// High bandwidth memory
        hbm_size: usize,
    },
    /// Edge/Mobile devices
    Edge {
        /// Power budget in watts
        power_budget: f64,
        /// Memory constraints
        memory_limit: usize,
        /// Quantization support
        quantization_support: QuantizationSupport,
    },
    /// Distributed system
    Distributed {
        /// Number of nodes
        num_nodes: usize,
        /// Network bandwidth between nodes
        network_bandwidth: f64,
        /// Node hardware type
        node_hardware: Box<HardwarePlatform>,
    },
}

/// SIMD instruction set support
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SIMDSupport {
    /// No SIMD support
    None,
    /// SSE (128-bit)
    SSE,
    /// AVX (256-bit)
    AVX,
    /// AVX-512 (512-bit)
    AVX512,
    /// ARM NEON
    NEON,
}

/// GPU architectures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GPUArchitecture {
    /// NVIDIA architectures
    Pascal,
    /// NVIDIA Volta architecture
    Volta,
    /// NVIDIA Turing architecture
    Turing,
    /// NVIDIA Ampere architecture
    Ampere,
    /// NVIDIA Hopper architecture
    Hopper,
    /// AMD architectures
    RDNA,
    /// AMD RDNA2 architecture
    RDNA2,
    /// AMD CDNA architecture
    CDNA,
    /// Intel architectures
    XeHPG,
    /// Intel Xe HPC architecture
    XeHPC,
}

/// TPU versions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TPUVersion {
    /// TPU v1
    V1,
    /// TPU v2
    V2,
    /// TPU v3
    V3,
    /// TPU v4
    V4,
    /// TPU v5
    V5,
}

/// Quantization support levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationSupport {
    /// No quantization support
    None,
    /// 8-bit integer quantization
    Int8,
    /// 16-bit floating point
    FP16,
    /// Brain floating point
    BF16,
    /// 4-bit quantization
    Int4,
    /// Mixed precision
    Mixed,
}

/// Hardware-specific optimization configuration
#[derive(Debug, Clone)]
pub struct HardwareOptimizationConfig<A: Float> {
    /// Optimized batch size for hardware
    pub batch_size: usize,
    /// Memory-efficient parameter update strategy
    pub memory_strategy: MemoryStrategy,
    /// Parallel computation strategy
    pub parallelization: ParallelizationStrategy,
    /// Precision strategy
    pub precision: PrecisionStrategy,
    /// Hardware-specific optimizer parameters
    pub optimizer_params: HashMap<String, A>,
    /// Communication strategy (for distributed)
    pub communication: Option<CommunicationStrategy>,
}

/// Memory optimization strategies
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Standard memory usage
    Standard,
    /// Gradient accumulation to reduce memory
    GradientAccumulation {
        /// Number of accumulation steps
        accumulation_steps: usize,
    },
    /// Gradient checkpointing
    GradientCheckpointing {
        /// Checkpoint ratio
        checkpoint_ratio: f64,
    },
    /// Parameter sharding
    ParameterSharding {
        /// Shard size
        shard_size: usize,
    },
    /// Offloading to CPU memory
    CPUOffloading {
        /// Offload ratio to CPU
        offload_ratio: f64,
    },
    /// Mixed memory strategies
    Mixed {
        /// List of memory strategies
        strategies: Vec<MemoryStrategy>,
        /// Weights for combining strategies
        strategy_weights: Vec<f64>,
    },
}

/// Parallelization strategies
#[derive(Debug, Clone)]
pub enum ParallelizationStrategy {
    /// Single-threaded execution
    SingleThread,
    /// Data parallelism
    DataParallel {
        /// Number of parallel workers
        num_workers: usize,
    },
    /// Model parallelism
    ModelParallel {
        /// Strategy for partitioning models
        partition_strategy: PartitionStrategy,
    },
    /// Pipeline parallelism
    Pipeline {
        /// Number of pipeline stages
        pipeline_stages: usize,
        /// Number of micro-batches
        micro_batches: usize,
    },
    /// Tensor parallelism
    TensorParallel {
        /// Size of tensor parallel group
        tensor_parallel_size: usize,
    },
    /// Hybrid parallelism
    Hybrid {
        /// Data parallel size
        data_parallel: usize,
        /// Model parallel size
        model_parallel: usize,
        /// Pipeline parallel size
        pipeline_parallel: usize,
    },
}

/// Model partitioning strategies
#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    /// Layer-wise partitioning
    LayerWise,
    /// Depth-wise partitioning
    DepthWise,
    /// Width-wise partitioning
    WidthWise,
    /// Custom partitioning
    Custom {
        /// Custom partition points
        partition_points: Vec<usize>,
    },
}

/// Precision strategies for different hardware
#[derive(Debug, Clone)]
pub enum PrecisionStrategy {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain floating point (BF16)
    BF16,
    /// Mixed precision training
    Mixed {
        /// Forward pass precision
        forward_precision: String,
        /// Backward pass precision
        backward_precision: String,
        /// Enable loss scaling
        loss_scaling: bool,
    },
    /// Integer quantization
    Quantized {
        /// Number of bits for weights
        weight_bits: u8,
        /// Number of bits for activations
        activation_bits: u8,
        /// Quantization method
        quantization_method: String,
    },
}

/// Communication strategies for distributed training
#[derive(Debug, Clone)]
pub enum CommunicationStrategy {
    /// All-reduce communication
    AllReduce {
        /// All-reduce algorithm
        algorithm: AllReduceAlgorithm,
        /// Enable gradient compression
        compression: bool,
    },
    /// Parameter server architecture
    ParameterServer {
        /// Number of parameter servers
        num_servers: usize,
        /// Update frequency
        update_frequency: usize,
    },
    /// Gossip protocols
    Gossip {
        /// Number of neighbors
        neighbors: usize,
        /// Gossip communication frequency
        gossip_frequency: usize,
    },
    /// Hierarchical communication
    Hierarchical {
        /// Number of local groups
        local_groups: usize,
        /// Inter-group communication strategy
        inter_group_strategy: Box<CommunicationStrategy>,
    },
}

/// All-reduce algorithms
#[derive(Debug, Clone)]
pub enum AllReduceAlgorithm {
    /// Ring all-reduce
    Ring,
    /// Tree all-reduce
    Tree,
    /// Butterfly all-reduce
    Butterfly,
    /// Halving-doubling
    HalvingDoubling,
}

/// Hardware-aware optimizer that adapts to different platforms
#[derive(Debug)]
pub struct HardwareAwareOptimizer<A: Float, D: Dimension> {
    /// Target hardware platform
    platform: HardwarePlatform,
    /// Hardware-specific configuration
    config: HardwareOptimizationConfig<A>,
    /// Performance profiler
    profiler: PerformanceProfiler<A>,
    /// Resource monitor
    resource_monitor: ResourceMonitor<A>,
    /// Adaptive tuning system
    adaptive_tuner: AdaptiveTuner<A>,
    /// Current optimization state
    current_state: OptimizationState<A, D>,
}

/// Performance profiler for hardware-specific metrics
#[derive(Debug)]
pub struct PerformanceProfiler<A: Float> {
    /// Computation time measurements
    computation_times: Vec<A>,
    /// Memory usage measurements
    memory_usage: Vec<usize>,
    /// Communication overhead (for distributed)
    #[allow(dead_code)]
    communication_overhead: Vec<A>,
    /// Energy consumption measurements
    energy_consumption: Vec<A>,
    /// Throughput measurements (samples/second)
    throughput: Vec<A>,
}

/// Resource monitor for real-time hardware monitoring
#[derive(Debug)]
pub struct ResourceMonitor<A: Float> {
    /// Current memory usage
    current_memory: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// CPU utilization
    cpu_utilization: A,
    /// GPU utilization (if applicable)
    #[allow(dead_code)]
    gpu_utilization: Option<A>,
    /// Power consumption
    power_consumption: A,
    /// Temperature readings
    temperature: A,
    /// Network utilization (for distributed)
    #[allow(dead_code)]
    network_utilization: Option<A>,
}

/// Adaptive tuner for dynamic optimization
#[derive(Debug)]
pub struct AdaptiveTuner<A: Float> {
    /// Tuning history
    #[allow(dead_code)]
    tuning_history: Vec<TuningRecord<A>>,
    /// Current tuning parameters
    #[allow(dead_code)]
    current_params: HashMap<String, A>,
    /// Performance target
    performance_target: A,
    /// Tuning strategy
    #[allow(dead_code)]
    strategy: TuningStrategy,
}

/// Tuning record for adaptive optimization
#[derive(Debug, Clone)]
pub struct TuningRecord<A: Float> {
    /// Tuning parameters used
    pub parameters: HashMap<String, A>,
    /// Performance achieved
    pub performance: A,
    /// Resource consumption
    pub resource_usage: A,
    /// Timestamp
    pub timestamp: u64,
}

/// Tuning strategies
#[derive(Debug, Clone)]
pub enum TuningStrategy {
    /// Grid search over parameter space
    GridSearch {
        /// Grid search resolution
        resolution: usize,
    },
    /// Bayesian optimization
    BayesianOptimization {
        /// Number of samples
        num_samples: usize,
    },
    /// Genetic algorithm
    GeneticAlgorithm {
        /// Population size
        population_size: usize,
        /// Number of generations
        generations: usize,
    },
    /// Reinforcement learning based
    ReinforcementLearning {
        /// Exploration rate
        exploration_rate: f64,
    },
}

/// Current optimization state
#[derive(Debug)]
pub struct OptimizationState<A: Float, D: Dimension> {
    /// Current parameters
    parameters: Array<A, D>,
    /// Gradient accumulator
    #[allow(dead_code)]
    gradient_accumulator: Option<Array<A, D>>,
    /// Optimizer state (momentum, etc.)
    #[allow(dead_code)]
    optimizer_state: HashMap<String, Array<A, D>>,
    /// Step count
    #[allow(dead_code)]
    step_count: usize,
    /// Learning rate schedule state
    #[allow(dead_code)]
    lr_schedule_state: A,
}

impl<
        A: Float + ScalarOperand + Debug + std::iter::Sum + for<'a> std::iter::Sum<&'a A>,
        D: Dimension,
    > HardwareAwareOptimizer<A, D>
{
    /// Create a new hardware-aware optimizer
    pub fn new(platform: HardwarePlatform, initial_parameters: Array<A, D>) -> Self {
        let config = Self::default_config_for_platform(&platform);
        let profiler = PerformanceProfiler::new();
        let resource_monitor = ResourceMonitor::new();
        let adaptive_tuner = AdaptiveTuner::new();

        let current_state = OptimizationState {
            parameters: initial_parameters,
            gradient_accumulator: None,
            optimizer_state: HashMap::new(),
            step_count: 0,
            lr_schedule_state: A::from(0.001).unwrap(),
        };

        Self {
            platform,
            config,
            profiler,
            resource_monitor,
            adaptive_tuner,
            current_state,
        }
    }

    /// Optimize configuration for target hardware
    pub fn optimize_for_hardware(&mut self) -> Result<()> {
        match self.platform.clone() {
            HardwarePlatform::CPU {
                cores,
                cache_size,
                simd_support,
            } => {
                self.optimize_for_cpu(cores, cache_size, simd_support)?;
            }
            HardwarePlatform::GPU {
                memory,
                compute_units,
                memory_bandwidth,
                architecture,
            } => {
                self.optimize_for_gpu(memory, compute_units, memory_bandwidth, architecture)?;
            }
            HardwarePlatform::TPU {
                version,
                matrix_units,
                hbm_size,
            } => {
                self.optimize_for_tpu(version, matrix_units, hbm_size)?;
            }
            HardwarePlatform::Edge {
                power_budget,
                memory_limit,
                quantization_support,
            } => {
                self.optimize_for_edge(power_budget, memory_limit, quantization_support)?;
            }
            HardwarePlatform::Distributed {
                num_nodes,
                network_bandwidth,
                node_hardware,
            } => {
                self.optimize_for_distributed(num_nodes, network_bandwidth, &node_hardware)?;
            }
        }
        Ok(())
    }

    /// CPU-specific optimizations
    fn optimize_for_cpu(
        &mut self,
        cores: usize,
        cache_size: usize,
        simd_support: SIMDSupport,
    ) -> Result<()> {
        // Optimize batch size for cache efficiency
        let cache_friendly_batch_size = (cache_size / 4) / self.current_state.parameters.len(); // Rough estimate
        self.config.batch_size = cache_friendly_batch_size.clamp(16, 512);

        // Configure parallelization based on cores
        self.config.parallelization = ParallelizationStrategy::DataParallel {
            num_workers: cores.min(8), // Don't over-parallelize
        };

        // SIMD-specific optimizations
        match simd_support {
            SIMDSupport::AVX512 => {
                self.config
                    .optimizer_params
                    .insert("vectorized_ops".to_string(), A::from(512.0).unwrap());
            }
            SIMDSupport::AVX => {
                self.config
                    .optimizer_params
                    .insert("vectorized_ops".to_string(), A::from(256.0).unwrap());
            }
            SIMDSupport::SSE => {
                self.config
                    .optimizer_params
                    .insert("vectorized_ops".to_string(), A::from(128.0).unwrap());
            }
            SIMDSupport::NEON => {
                self.config
                    .optimizer_params
                    .insert("vectorized_ops".to_string(), A::from(128.0).unwrap());
            }
            SIMDSupport::None => {
                self.config
                    .optimizer_params
                    .insert("vectorized_ops".to_string(), A::from(32.0).unwrap());
            }
        }

        // Use full precision for CPU
        self.config.precision = PrecisionStrategy::FP32;

        Ok(())
    }

    /// GPU-specific optimizations
    fn optimize_for_gpu(
        &mut self,
        memory: usize,
        compute_units: usize,
        memory_bandwidth: f64,
        architecture: GPUArchitecture,
    ) -> Result<()> {
        // Optimize batch size for GPU memory
        let gpu_memory_gb = memory as f64 / (1024.0 * 1024.0 * 1024.0);
        let optimal_batch_size = if gpu_memory_gb >= 32.0 {
            256
        } else if gpu_memory_gb >= 16.0 {
            128
        } else if gpu_memory_gb >= 8.0 {
            64
        } else {
            32
        };
        self.config.batch_size = optimal_batch_size;

        // Configure parallelization for GPU
        self.config.parallelization = ParallelizationStrategy::DataParallel {
            num_workers: compute_units.min(16),
        };

        // Architecture-specific optimizations
        match architecture {
            GPUArchitecture::Ampere | GPUArchitecture::Hopper => {
                // Use mixed precision for modern architectures
                self.config.precision = PrecisionStrategy::Mixed {
                    forward_precision: "fp16".to_string(),
                    backward_precision: "fp32".to_string(),
                    loss_scaling: true,
                };
                self.config
                    .optimizer_params
                    .insert("tensor_cores".to_string(), A::from(1.0).unwrap());
            }
            GPUArchitecture::Volta | GPUArchitecture::Turing => {
                self.config.precision = PrecisionStrategy::FP16;
                self.config
                    .optimizer_params
                    .insert("tensor_cores".to_string(), A::from(1.0).unwrap());
            }
            _ => {
                self.config.precision = PrecisionStrategy::FP32;
            }
        }

        // Memory bandwidth optimizations
        if memory_bandwidth < 500.0 {
            // Low bandwidth
            self.config.memory_strategy = MemoryStrategy::GradientAccumulation {
                accumulation_steps: 4,
            };
        } else {
            self.config.memory_strategy = MemoryStrategy::Standard;
        }

        Ok(())
    }

    /// TPU-specific optimizations
    fn optimize_for_tpu(
        &mut self,
        version: TPUVersion,
        matrix_units: usize,
        hbm_size: usize,
    ) -> Result<()> {
        // TPUs work best with large batch sizes
        let tpu_batch_size = match version {
            TPUVersion::V1 | TPUVersion::V2 => 128,
            TPUVersion::V3 => 256,
            TPUVersion::V4 | TPUVersion::V5 => 512,
        };
        self.config.batch_size = tpu_batch_size;

        // TPUs prefer BF16 precision
        self.config.precision = PrecisionStrategy::BF16;

        // Configure for matrix operations
        self.config.optimizer_params.insert(
            "matrix_units".to_string(),
            A::from(matrix_units as f64).unwrap(),
        );

        // Use all available matrix units
        self.config.parallelization = ParallelizationStrategy::TensorParallel {
            tensor_parallel_size: matrix_units.min(8),
        };

        // HBM-specific optimizations
        if hbm_size > 32 * 1024 * 1024 * 1024 {
            // 32GB+
            self.config.memory_strategy = MemoryStrategy::Standard;
        } else {
            self.config.memory_strategy = MemoryStrategy::GradientCheckpointing {
                checkpoint_ratio: 0.5,
            };
        }

        Ok(())
    }

    /// Edge device optimizations
    fn optimize_for_edge(
        &mut self,
        power_budget: f64,
        memory_limit: usize,
        quantization_support: QuantizationSupport,
    ) -> Result<()> {
        // Small batch sizes for memory constraints
        let edge_batch_size = (memory_limit / (4 * 1024 * 1024)).clamp(1, 32); // Very conservative
        self.config.batch_size = edge_batch_size;

        // Single-threaded for power efficiency
        self.config.parallelization = ParallelizationStrategy::SingleThread;

        // Aggressive quantization for edge devices
        match quantization_support {
            QuantizationSupport::Int4 => {
                self.config.precision = PrecisionStrategy::Quantized {
                    weight_bits: 4,
                    activation_bits: 8,
                    quantization_method: "dynamic".to_string(),
                };
            }
            QuantizationSupport::Int8 => {
                self.config.precision = PrecisionStrategy::Quantized {
                    weight_bits: 8,
                    activation_bits: 8,
                    quantization_method: "static".to_string(),
                };
            }
            QuantizationSupport::FP16 => {
                self.config.precision = PrecisionStrategy::FP16;
            }
            _ => {
                self.config.precision = PrecisionStrategy::FP32;
            }
        }

        // Power-aware optimizations
        if power_budget < 5.0 {
            // Very low power
            self.config
                .optimizer_params
                .insert("update_frequency".to_string(), A::from(10.0).unwrap());
            self.config.memory_strategy = MemoryStrategy::CPUOffloading { offload_ratio: 0.8 };
        }

        Ok(())
    }

    /// Distributed system optimizations
    fn optimize_for_distributed(
        &mut self,
        num_nodes: usize,
        network_bandwidth: f64,
        node_hardware: &HardwarePlatform,
    ) -> Result<()> {
        // Scale batch size with number of nodes
        let base_batch_size = match node_hardware {
            HardwarePlatform::GPU { .. } => 128,
            HardwarePlatform::CPU { .. } => 64,
            _ => 32,
        };
        self.config.batch_size = base_batch_size * num_nodes;

        // Configure communication strategy based on network bandwidth
        let communication = if network_bandwidth >= 100.0 {
            // High bandwidth (100 Gbps+)
            CommunicationStrategy::AllReduce {
                algorithm: AllReduceAlgorithm::Ring,
                compression: false,
            }
        } else if network_bandwidth >= 10.0 {
            // Medium bandwidth (10 Gbps+)
            CommunicationStrategy::AllReduce {
                algorithm: AllReduceAlgorithm::Tree,
                compression: true,
            }
        } else {
            // Low bandwidth
            CommunicationStrategy::ParameterServer {
                num_servers: (num_nodes / 4).max(1),
                update_frequency: 10,
            }
        };
        self.config.communication = Some(communication);

        // Configure parallelization strategy
        if num_nodes >= 64 {
            self.config.parallelization = ParallelizationStrategy::Hybrid {
                data_parallel: 8,
                model_parallel: 4,
                pipeline_parallel: num_nodes / 32,
            };
        } else if num_nodes >= 16 {
            self.config.parallelization = ParallelizationStrategy::Pipeline {
                pipeline_stages: 4,
                micro_batches: 8,
            };
        } else {
            self.config.parallelization = ParallelizationStrategy::DataParallel {
                num_workers: num_nodes,
            };
        }

        Ok(())
    }

    /// Profile current performance
    pub fn profile_performance(&mut self, computation_time: A, memory_used: usize, energy: A) {
        self.profiler.computation_times.push(computation_time);
        self.profiler.memory_usage.push(memory_used);
        self.profiler.energy_consumption.push(energy);

        // Calculate throughput (simplified)
        let throughput = A::from(self.config.batch_size as f64).unwrap() / computation_time;
        self.profiler.throughput.push(throughput);

        // Keep history bounded
        const MAX_HISTORY: usize = 1000;
        if self.profiler.computation_times.len() > MAX_HISTORY {
            self.profiler.computation_times.remove(0);
            self.profiler.memory_usage.remove(0);
            self.profiler.energy_consumption.remove(0);
            self.profiler.throughput.remove(0);
        }
    }

    /// Update resource monitoring
    pub fn update_resource_monitor(&mut self, memory: usize, cpu_util: A, power: A, temp: A) {
        self.resource_monitor.current_memory = memory;
        self.resource_monitor.peak_memory = self.resource_monitor.peak_memory.max(memory);
        self.resource_monitor.cpu_utilization = cpu_util;
        self.resource_monitor.power_consumption = power;
        self.resource_monitor.temperature = temp;
    }

    /// Adaptive tuning based on performance feedback
    pub fn adaptive_tune(&mut self, target_performance: A) -> Result<()> {
        self.adaptive_tuner.performance_target = target_performance;

        // Simple adaptive tuning logic
        let current_performance = self.get_average_performance();

        if current_performance < target_performance {
            // Need to improve performance
            self.tune_for_performance()?;
        } else {
            // Can optimize for efficiency
            self.tune_for_efficiency()?;
        }

        Ok(())
    }

    /// Tune for better performance
    fn tune_for_performance(&mut self) -> Result<()> {
        // Increase batch size if memory allows
        if self.resource_monitor.current_memory < self.resource_monitor.peak_memory * 8 / 10 {
            self.config.batch_size = (self.config.batch_size * 12 / 10).min(1024);
        }

        // Reduce precision for speed
        match self.config.precision {
            PrecisionStrategy::FP32 => {
                self.config.precision = PrecisionStrategy::FP16;
            }
            PrecisionStrategy::FP16 => {
                self.config.precision = PrecisionStrategy::Mixed {
                    forward_precision: "fp16".to_string(),
                    backward_precision: "fp32".to_string(),
                    loss_scaling: true,
                };
            }
            _ => {}
        }

        Ok(())
    }

    /// Tune for better efficiency
    fn tune_for_efficiency(&mut self) -> Result<()> {
        // Reduce batch size to save memory
        self.config.batch_size = (self.config.batch_size * 9 / 10).max(1);

        // Enable gradient accumulation to maintain effective batch size
        self.config.memory_strategy = MemoryStrategy::GradientAccumulation {
            accumulation_steps: 2,
        };

        Ok(())
    }

    /// Get average performance from recent measurements
    fn get_average_performance(&self) -> A {
        if self.profiler.throughput.is_empty() {
            A::zero()
        } else {
            let recent_throughput =
                &self.profiler.throughput[self.profiler.throughput.len().saturating_sub(10)..];
            recent_throughput.iter().copied().sum::<A>() / A::from(recent_throughput.len()).unwrap()
        }
    }

    /// Get current configuration
    pub fn get_config(&self) -> &HardwareOptimizationConfig<A> {
        &self.config
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HardwarePerformanceStats<A> {
        let avg_computation_time = if self.profiler.computation_times.is_empty() {
            A::zero()
        } else {
            self.profiler.computation_times.iter().sum::<A>()
                / A::from(self.profiler.computation_times.len()).unwrap()
        };

        let avg_throughput = if self.profiler.throughput.is_empty() {
            A::zero()
        } else {
            self.profiler.throughput.iter().sum::<A>()
                / A::from(self.profiler.throughput.len()).unwrap()
        };

        let avg_energy = if self.profiler.energy_consumption.is_empty() {
            A::zero()
        } else {
            self.profiler.energy_consumption.iter().copied().sum::<A>()
                / A::from(self.profiler.energy_consumption.len()).unwrap()
        };

        HardwarePerformanceStats {
            average_computation_time: avg_computation_time,
            average_throughput: avg_throughput,
            peak_memory_usage: self.resource_monitor.peak_memory,
            average_energy_consumption: avg_energy,
            hardware_utilization: self.resource_monitor.cpu_utilization,
            efficiency_score: avg_throughput / (avg_energy + A::from(1e-8).unwrap()), // Avoid division by zero
        }
    }

    /// Create default configuration for platform
    fn default_config_for_platform(platform: &HardwarePlatform) -> HardwareOptimizationConfig<A> {
        match platform {
            HardwarePlatform::CPU { .. } => HardwareOptimizationConfig {
                batch_size: 64,
                memory_strategy: MemoryStrategy::Standard,
                parallelization: ParallelizationStrategy::DataParallel { num_workers: 4 },
                precision: PrecisionStrategy::FP32,
                optimizer_params: HashMap::new(),
                communication: None,
            },
            HardwarePlatform::GPU { .. } => HardwareOptimizationConfig {
                batch_size: 128,
                memory_strategy: MemoryStrategy::Standard,
                parallelization: ParallelizationStrategy::DataParallel { num_workers: 1 },
                precision: PrecisionStrategy::FP16,
                optimizer_params: HashMap::new(),
                communication: None,
            },
            HardwarePlatform::TPU { .. } => HardwareOptimizationConfig {
                batch_size: 256,
                memory_strategy: MemoryStrategy::Standard,
                parallelization: ParallelizationStrategy::TensorParallel {
                    tensor_parallel_size: 8,
                },
                precision: PrecisionStrategy::BF16,
                optimizer_params: HashMap::new(),
                communication: None,
            },
            HardwarePlatform::Edge { .. } => HardwareOptimizationConfig {
                batch_size: 16,
                memory_strategy: MemoryStrategy::GradientCheckpointing {
                    checkpoint_ratio: 0.5,
                },
                parallelization: ParallelizationStrategy::SingleThread,
                precision: PrecisionStrategy::Quantized {
                    weight_bits: 8,
                    activation_bits: 8,
                    quantization_method: "dynamic".to_string(),
                },
                optimizer_params: HashMap::new(),
                communication: None,
            },
            HardwarePlatform::Distributed { .. } => HardwareOptimizationConfig {
                batch_size: 512,
                memory_strategy: MemoryStrategy::Standard,
                parallelization: ParallelizationStrategy::DataParallel { num_workers: 8 },
                precision: PrecisionStrategy::FP16,
                optimizer_params: HashMap::new(),
                communication: Some(CommunicationStrategy::AllReduce {
                    algorithm: AllReduceAlgorithm::Ring,
                    compression: false,
                }),
            },
        }
    }
}

impl<A: Float> Default for PerformanceProfiler<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float> PerformanceProfiler<A> {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            computation_times: Vec::new(),
            memory_usage: Vec::new(),
            communication_overhead: Vec::new(),
            energy_consumption: Vec::new(),
            throughput: Vec::new(),
        }
    }
}

impl<A: Float> Default for ResourceMonitor<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float> ResourceMonitor<A> {
    /// Create a new resource monitor
    pub fn new() -> Self {
        Self {
            current_memory: 0,
            peak_memory: 0,
            cpu_utilization: A::zero(),
            gpu_utilization: None,
            power_consumption: A::zero(),
            temperature: A::zero(),
            network_utilization: None,
        }
    }
}

impl<A: Float> Default for AdaptiveTuner<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float> AdaptiveTuner<A> {
    /// Create a new adaptive tuner
    pub fn new() -> Self {
        Self {
            tuning_history: Vec::new(),
            current_params: HashMap::new(),
            performance_target: A::from(100.0).unwrap(),
            strategy: TuningStrategy::BayesianOptimization { num_samples: 50 },
        }
    }
}

/// Hardware performance statistics
#[derive(Debug, Clone)]
pub struct HardwarePerformanceStats<A: Float> {
    /// Average computation time per step
    pub average_computation_time: A,
    /// Average throughput (samples/second)
    pub average_throughput: A,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Average energy consumption
    pub average_energy_consumption: A,
    /// Hardware utilization percentage
    pub hardware_utilization: A,
    /// Efficiency score (throughput/energy)
    pub efficiency_score: A,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_cpu_optimization() {
        let platform = HardwarePlatform::CPU {
            cores: 8,
            cache_size: 32 * 1024 * 1024, // 32MB cache
            simd_support: SIMDSupport::AVX,
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        optimizer.optimize_for_hardware().unwrap();

        // Check CPU-specific optimizations
        assert!(optimizer.config.batch_size <= 512);
        assert!(matches!(
            optimizer.config.parallelization,
            ParallelizationStrategy::DataParallel { .. }
        ));
        assert!(matches!(
            optimizer.config.precision,
            PrecisionStrategy::FP32
        ));
        assert!(optimizer
            .config
            .optimizer_params
            .contains_key("vectorized_ops"));
    }

    #[test]
    fn test_gpu_optimization() {
        let platform = HardwarePlatform::GPU {
            memory: 16 * 1024 * 1024 * 1024, // 16GB
            compute_units: 80,
            memory_bandwidth: 900.0,
            architecture: GPUArchitecture::Ampere,
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        optimizer.optimize_for_hardware().unwrap();

        // Check GPU-specific optimizations
        assert_eq!(optimizer.config.batch_size, 128);
        assert!(matches!(
            optimizer.config.precision,
            PrecisionStrategy::Mixed { .. }
        ));
        assert!(optimizer
            .config
            .optimizer_params
            .contains_key("tensor_cores"));
    }

    #[test]
    fn test_tpu_optimization() {
        let platform = HardwarePlatform::TPU {
            version: TPUVersion::V4,
            matrix_units: 8,
            hbm_size: 32 * 1024 * 1024 * 1024, // 32GB HBM
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        optimizer.optimize_for_hardware().unwrap();

        // Check TPU-specific optimizations
        assert_eq!(optimizer.config.batch_size, 512);
        assert!(matches!(
            optimizer.config.precision,
            PrecisionStrategy::BF16
        ));
        assert!(matches!(
            optimizer.config.parallelization,
            ParallelizationStrategy::TensorParallel { .. }
        ));
    }

    #[test]
    fn test_edge_optimization() {
        let platform = HardwarePlatform::Edge {
            power_budget: 3.0,               // 3 watts
            memory_limit: 512 * 1024 * 1024, // 512MB
            quantization_support: QuantizationSupport::Int8,
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        optimizer.optimize_for_hardware().unwrap();

        // Check edge-specific optimizations
        assert!(optimizer.config.batch_size <= 32);
        assert!(matches!(
            optimizer.config.parallelization,
            ParallelizationStrategy::SingleThread
        ));
        assert!(matches!(
            optimizer.config.precision,
            PrecisionStrategy::Quantized { .. }
        ));
    }

    #[test]
    fn test_distributed_optimization() {
        let node_hardware = HardwarePlatform::GPU {
            memory: 8 * 1024 * 1024 * 1024, // 8GB per node
            compute_units: 40,
            memory_bandwidth: 500.0,
            architecture: GPUArchitecture::Volta,
        };

        let platform = HardwarePlatform::Distributed {
            num_nodes: 16,
            network_bandwidth: 50.0, // 50 Gbps
            node_hardware: Box::new(node_hardware),
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        optimizer.optimize_for_hardware().unwrap();

        // Check distributed-specific optimizations
        assert_eq!(optimizer.config.batch_size, 128 * 16); // Scaled by number of nodes
        assert!(optimizer.config.communication.is_some());
        assert!(matches!(
            optimizer.config.parallelization,
            ParallelizationStrategy::Pipeline { .. }
        ));
    }

    #[test]
    fn test_performance_profiling() {
        let platform = HardwarePlatform::CPU {
            cores: 4,
            cache_size: 8 * 1024 * 1024,
            simd_support: SIMDSupport::SSE,
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        // Add some performance measurements
        optimizer.profile_performance(0.1, 1000000, 5.0);
        optimizer.profile_performance(0.12, 1100000, 5.2);
        optimizer.profile_performance(0.09, 950000, 4.8);

        let stats = optimizer.get_performance_stats();

        assert!(stats.average_computation_time > 0.0);
        assert!(stats.average_throughput > 0.0);
        assert_eq!(stats.peak_memory_usage, 0); // Not updated in this test
    }

    #[test]
    fn test_adaptive_tuning() {
        let platform = HardwarePlatform::GPU {
            memory: 8 * 1024 * 1024 * 1024,
            compute_units: 20,
            memory_bandwidth: 300.0,
            architecture: GPUArchitecture::Turing,
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

        // Simulate low performance
        optimizer.profiler.throughput.push(50.0);
        optimizer.resource_monitor.current_memory = 1_000_000_000; // 1GB
        optimizer.resource_monitor.peak_memory = 4_000_000_000; // 4GB

        let initial_batch_size = optimizer.config.batch_size;
        optimizer.adaptive_tune(100.0).unwrap(); // Target 100 samples/sec

        // Should have tuned for better performance
        assert!(optimizer.config.batch_size >= initial_batch_size);
    }

    #[test]
    fn test_hardware_platform_matching() {
        let platforms = vec![
            HardwarePlatform::CPU {
                cores: 8,
                cache_size: 16_000_000,
                simd_support: SIMDSupport::AVX,
            },
            HardwarePlatform::GPU {
                memory: 12_000_000_000,
                compute_units: 60,
                memory_bandwidth: 600.0,
                architecture: GPUArchitecture::Ampere,
            },
            HardwarePlatform::TPU {
                version: TPUVersion::V3,
                matrix_units: 8,
                hbm_size: 16_000_000_000,
            },
            HardwarePlatform::Edge {
                power_budget: 2.0,
                memory_limit: 256_000_000,
                quantization_support: QuantizationSupport::Int4,
            },
        ];

        for platform in platforms {
            let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let mut optimizer = HardwareAwareOptimizer::new(platform, initial_params);

            // Should not panic and should complete successfully
            let result = optimizer.optimize_for_hardware();
            assert!(result.is_ok());

            // Each platform should have different configurations
            let config = optimizer.get_config();
            assert!(config.batch_size > 0);
        }
    }
}
