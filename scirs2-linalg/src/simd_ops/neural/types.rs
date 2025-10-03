//! Common types for neural memory optimization module.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use std::collections::HashMap;
use std::fmt::Debug;

/// Workload characteristics for optimization
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Types of operations being performed
    pub operation_types: Vec<MemoryOperationType>,
    /// Data sizes and shapes
    pub datasizes: Vec<TensorShape>,
    /// Computation intensity (operations per byte)
    pub computation_intensity: f64,
    /// Memory intensity (bytes accessed per operation)
    pub memory_intensity: f64,
}

/// Tensor shape information
#[derive(Debug, Clone)]
pub struct TensorShape {
    /// Tensor dimensions
    pub dimensions: Vec<usize>,
    /// Element data type
    pub element_type: ElementType,
    /// Memory layout
    pub memory_layout: MemoryLayout,
}

/// Element types
#[derive(Debug, Clone, PartialEq)]
pub enum ElementType {
    F32,
    F64,
    I32,
    I64,
    Complex32,
    Complex64,
}

/// Memory layout types
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked,
}

/// Memory operation types
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOperationType {
    MatrixMultiplication,
    MatrixAddition,
    MatrixTranspose,
    VectorOperation,
    Reduction,
    Broadcasting,
    Convolution,
    ElementwiseOperation,
    Copy,
    Streaming,
}

/// Data type classification
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    F16,
    BF16,
    Complex32,
    Complex64,
}

/// Context information for memory access
#[derive(Debug, Clone)]
pub struct AccessContext<T> {
    /// Matrix dimensions being processed
    pub matrix_dimensions: Vec<(usize, usize)>,
    /// Operation type
    pub operation_type: MemoryOperationType,
    /// Thread count
    pub thread_count: usize,
    /// NUMA node
    pub numa_node: usize,
    /// Available cache sizes
    pub cachesizes: CacheSizes,
    /// Memory pressure
    pub memory_pressure: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Ambient parameters
    pub ambient_params: AmbientParameters<T>,
}

/// Cache size hierarchy
#[derive(Debug, Clone)]
pub struct CacheSizes {
    /// L1 data cache size
    pub l1_data: usize,
    /// L1 instruction cache size
    pub l1_instruction: usize,
    /// L2 cache size
    pub l2: usize,
    /// L3 cache size
    pub l3: usize,
    /// Cache line size
    pub cache_linesize: usize,
    /// Translation lookaside buffer entries
    pub tlb_entries: usize,
}

/// Ambient parameters affecting memory performance
#[derive(Debug, Clone)]
pub struct AmbientParameters<T> {
    /// Temperature (affects memory timing)
    pub temperature: f64,
    /// Power state
    pub power_state: PowerState,
    /// Memory frequency
    pub memory_frequency: f64,
    /// Memory voltage
    pub memory_voltage: f64,
    /// Thermal throttling active
    pub thermal_throttling: bool,
    /// Background memory traffic
    pub background_traffic: f64,
    /// Compiler optimization level
    pub optimization_level: OptimizationLevel,
    /// Custom parameters
    pub custom_params: HashMap<String, T>,
}

/// Power state enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum PowerState {
    MaxPerformance,
    Balanced,
    PowerSaver,
    Adaptive,
    Custom(f64),
}

/// Compiler optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    Debug,
    Release,
    RelWithDebInfo,
    MinSizeRel,
    Custom(String),
}

/// Activation functions for neural networks
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU(f64),
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    Mish,
    Identity,
}

/// Memory access types
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessType {
    Read,
    Write,
    ReadModifyWrite,
    Prefetch,
    Writeback,
}

/// Types of memory layouts
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutType {
    Linear,
    Blocked,
    Hierarchical,
    ZOrder,
    Hilbert,
    Custom(String),
}

/// Padding strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PaddingStrategy {
    None,
    CacheLinePadding,
    PagePadding,
    Optimal,
    Custom(Vec<usize>),
}

/// Data ordering strategies
#[derive(Debug, Clone, PartialEq)]
pub enum DataOrdering {
    Sequential,
    Strided,
    Random,
    Optimal,
    CacheFriendly,
    NumaAware,
}

/// Pattern identifier
pub type PatternId = u64;

// Default implementations
impl Default for CacheSizes {
    fn default() -> Self {
        Self {
            l1_data: 32 * 1024,
            l1_instruction: 32 * 1024,
            l2: 256 * 1024,
            l3: 8 * 1024 * 1024,
            cache_linesize: 64,
            tlb_entries: 512,
        }
    }
}

impl<T> Default for AccessContext<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn default() -> Self {
        Self {
            matrix_dimensions: Vec::new(),
            operation_type: MemoryOperationType::MatrixMultiplication,
            thread_count: 1,
            numa_node: 0,
            cachesizes: CacheSizes::default(),
            memory_pressure: 0.0,
            cpu_utilization: 0.0,
            ambient_params: AmbientParameters::default(),
        }
    }
}

impl<T> Default for AmbientParameters<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn default() -> Self {
        Self {
            temperature: 25.0,
            power_state: PowerState::Balanced,
            memory_frequency: 3200.0,
            memory_voltage: 1.35,
            thermal_throttling: false,
            background_traffic: 0.1,
            optimization_level: OptimizationLevel::Release,
            custom_params: HashMap::new(),
        }
    }
}
