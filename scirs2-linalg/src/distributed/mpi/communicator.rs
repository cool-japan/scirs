//! MPI Communicator Management
//!
//! This module provides MPI communicator functionality including process management,
//! communication operations, and request handling for both blocking and non-blocking operations.

use crate::error::{LinalgError, LinalgResult};
use super::{MPIConfig, MPIDatatype, MPIReduceOp};
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::{Arc, Mutex, RwLock};

/// MPI communicator wrapper with advanced features
#[derive(Debug)]
pub struct MPICommunicator {
    /// Base MPI communicator handle
    comm_handle: MPICommHandle,
    /// Rank of this process
    rank: i32,
    /// Total number of processes
    size: i32,
    /// Derived datatypes for efficient communication
    derived_types: HashMap<String, super::datatypes::MPIDatatype>,
    /// Persistent request pool
    persistent_requests: HashMap<String, MPIPersistentRequest>,
    /// Active non-blocking operations
    active_operations: Arc<RwLock<HashMap<String, MPIRequest>>>,
    /// Communication statistics
    comm_stats: Arc<Mutex<MPICommStats>>,
}

/// MPI communicator handle (opaque type for FFI)
#[derive(Debug)]
pub struct MPICommHandle {
    handle: *mut c_void,
}

unsafe impl Send for MPICommHandle {}
unsafe impl Sync for MPICommHandle {}

/// Persistent MPI request for repeated communication patterns
#[derive(Debug)]
pub struct MPIPersistentRequest {
    request_handle: *mut c_void,
    operation_type: PersistentOperationType,
    buffer_info: BufferInfo,
    is_active: bool,
}

/// Types of persistent operations
#[derive(Debug, Clone, Copy)]
pub enum PersistentOperationType {
    Send,
    Recv,
    Bcast,
    Allreduce,
    Allgather,
    Scatter,
    Gather,
}

/// Buffer information for MPI operations
#[derive(Debug, Clone)]
pub struct BufferInfo {
    buffer_ptr: *mut c_void,
    buffersize: usize,
    element_count: usize,
    datatype: String,
}

/// MPI request for non-blocking operations
#[derive(Debug)]
pub struct MPIRequest {
    request_handle: *mut c_void,
    operation_id: String,
    start_time: std::time::Instant,
    expected_bytes: usize,
    operation_type: RequestOperationType,
}

/// Types of MPI request operations
#[derive(Debug, Clone, Copy)]
pub enum RequestOperationType {
    PointToPoint,
    Collective,
    RMA,
    IO,
}

/// Communication statistics for MPI
#[derive(Debug, Default, Clone)]
pub struct MPICommStats {
    /// Total messages sent
    pub messages_sent: usize,
    /// Total messages received
    pub messages_received: usize,
    /// Total bytes sent
    pub bytes_sent: usize,
    /// Total bytes received
    pub bytes_received: usize,
    /// Average message latency
    pub avg_latency: f64,
    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,
    /// Communication efficiency
    pub efficiency: f64,
    /// Error count
    pub error_count: usize,
}

/// MPI status structure
#[derive(Debug, Default, Clone)]
pub struct MPIStatus {
    pub source: i32,
    pub tag: i32,
    pub error: i32,
    pub count: usize,
}

// FFI declarations for MPI (simplified)
extern "C" {
    fn mpi_init(argc: *mut c_int, argv: *mut *mut *mut i8) -> c_int;
    fn mpi_initialized(flag: *mut c_int) -> c_int;
    fn mpi_comm_world() -> *mut c_void;
    fn mpi_comm_rank(comm: *mut c_void) -> c_int;
    fn mpi_commsize(comm: *mut c_void) -> c_int;
    fn mpi_isend(buf: *const c_void, count: usize, datatype: c_int, dest: c_int, tag: c_int, comm: *mut c_void) -> *mut c_void;
    fn mpi_irecv(buf: *mut c_void, count: usize, datatype: c_int, source: c_int, tag: c_int, comm: *mut c_void) -> *mut c_void;
    fn mpi_wait(request: *mut c_void, status: *mut c_void) -> c_int;
    fn mpi_bcast(buffer: *mut c_void, count: usize, datatype: c_int, root: c_int, comm: *mut c_void) -> c_int;
    fn mpi_allreduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: usize, datatype: c_int, op: c_int, comm: *mut c_void) -> c_int;
    fn mpi_gather(sendbuf: *const c_void, sendcount: usize, sendtype: c_int, recvbuf: *mut c_void, recvcount: usize, recvtype: c_int, root: c_int, comm: *mut c_void) -> c_int;
    fn mpi_scatter(sendbuf: *const c_void, sendcount: usize, sendtype: c_int, recvbuf: *mut c_void, recvcount: usize, recvtype: c_int, root: c_int, comm: *mut c_void) -> c_int;
    fn mpi_barrier(comm: *mut c_void) -> c_int;
    fn mpi_finalize() -> c_int;
}

impl MPICommunicator {
    /// Create a new MPI communicator
    pub fn new(config: &MPIConfig) -> LinalgResult<Self> {
        // Initialize MPI if not already done
        unsafe {
            let mut flag: c_int = 0;
            if mpi_initialized(&mut flag) != 0 || flag == 0 {
                let mut argc = 0;
                let argv: *mut *mut i8 = std::ptr::null_mut();
                if mpi_init(&mut argc, &mut argv) != 0 {
                    return Err(LinalgError::InitializationError(
                        "Failed to initialize MPI".to_string()
                    ));
                }
            }
        }

        // Get communicator handle
        let comm_handle = MPICommHandle {
            handle: unsafe { mpi_comm_world() },
        };

        // Get rank and size
        let rank = unsafe { mpi_comm_rank(comm_handle.handle) };
        let size = unsafe { mpi_commsize(comm_handle.handle) };

        if rank < 0 || size <= 0 {
            return Err(LinalgError::InitializationError(
                "Invalid MPI rank or size".to_string()
            ));
        }

        Ok(Self {
            comm_handle,
            rank,
            size,
            derived_types: HashMap::new(),
            persistent_requests: HashMap::new(),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            comm_stats: Arc::new(Mutex::new(MPICommStats::default())),
        })
    }

    /// Get the rank of this process
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get the total number of processes
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Get the communicator handle
    pub fn handle(&self) -> &MPICommHandle {
        &self.comm_handle
    }

    /// Create a barrier synchronization point
    pub fn barrier(&self) -> LinalgResult<()> {
        unsafe {
            let result = mpi_barrier(self.comm_handle.handle);
            if result != 0 {
                return Err(LinalgError::CommunicationError(
                    format!("MPI barrier failed with code {}", result)
                ));
            }
        }
        Ok(())
    }

    /// Get communication statistics
    pub fn get_stats(&self) -> MPICommStats {
        self.comm_stats.lock().unwrap().clone()
    }

    /// Reset communication statistics
    pub fn reset_stats(&self) {
        let mut stats = self.comm_stats.lock().unwrap();
        *stats = MPICommStats::default();
    }

    /// Get active operations count
    pub fn active_operations_count(&self) -> usize {
        self.active_operations.read().unwrap().len()
    }

    /// Check if any operations are active
    pub fn has_active_operations(&self) -> bool {
        !self.active_operations.read().unwrap().is_empty()
    }

    /// Register a derived datatype
    pub fn register_datatype(&mut self, name: String, datatype: super::datatypes::MPIDatatype) {
        self.derived_types.insert(name, datatype);
    }

    /// Get a registered datatype
    pub fn get_datatype(&self, name: &str) -> Option<&super::datatypes::MPIDatatype> {
        self.derived_types.get(name)
    }

    /// Add a persistent request
    pub fn add_persistent_request(&mut self, name: String, request: MPIPersistentRequest) {
        self.persistent_requests.insert(name, request);
    }

    /// Get a persistent request
    pub fn get_persistent_request(&self, name: &str) -> Option<&MPIPersistentRequest> {
        self.persistent_requests.get(name)
    }

    /// Update statistics after an operation
    fn update_stats(&self, operation_type: RequestOperationType, bytes: usize, elapsed: f64) {
        let mut stats = self.comm_stats.lock().unwrap();

        match operation_type {
            RequestOperationType::PointToPoint => {
                stats.messages_sent += 1;
                stats.bytes_sent += bytes;
            },
            RequestOperationType::Collective => {
                stats.messages_sent += 1;
                stats.bytes_sent += bytes;
            },
            _ => {}
        }

        // Update average latency
        let total_messages = stats.messages_sent + stats.messages_received;
        if total_messages > 0 {
            stats.avg_latency = (stats.avg_latency * (total_messages - 1) as f64 + elapsed) / total_messages as f64;
        } else {
            stats.avg_latency = elapsed;
        }

        // Update peak bandwidth
        let bandwidth = bytes as f64 / elapsed;
        if bandwidth > stats.peak_bandwidth {
            stats.peak_bandwidth = bandwidth;
        }
    }
}

impl MPICommHandle {
    /// Get the raw handle pointer
    pub fn raw_handle(&self) -> *mut c_void {
        self.handle
    }

    /// Check if the handle is valid
    pub fn is_valid(&self) -> bool {
        !self.handle.is_null()
    }
}

impl MPIPersistentRequest {
    /// Create a new persistent request
    pub fn new(
        request_handle: *mut c_void,
        operation_type: PersistentOperationType,
        buffer_info: BufferInfo,
    ) -> Self {
        Self {
            request_handle,
            operation_type,
            buffer_info,
            is_active: false,
        }
    }

    /// Check if the request is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Activate the persistent request
    pub fn activate(&mut self) {
        self.is_active = true;
    }

    /// Deactivate the persistent request
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Get the operation type
    pub fn operation_type(&self) -> PersistentOperationType {
        self.operation_type
    }

    /// Get the buffer info
    pub fn buffer_info(&self) -> &BufferInfo {
        &self.buffer_info
    }
}

impl BufferInfo {
    /// Create new buffer information
    pub fn new(buffer_ptr: *mut c_void, buffersize: usize, element_count: usize, datatype: String) -> Self {
        Self {
            buffer_ptr,
            buffersize,
            element_count,
            datatype,
        }
    }

    /// Get buffer size
    pub fn buffersize(&self) -> usize {
        self.buffersize
    }

    /// Get element count
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Get datatype name
    pub fn datatype(&self) -> &str {
        &self.datatype
    }

    /// Get buffer pointer
    pub fn buffer_ptr(&self) -> *mut c_void {
        self.buffer_ptr
    }
}

impl MPIRequest {
    /// Create a new MPI request
    pub fn new(
        request_handle: *mut c_void,
        operation_id: String,
        expected_bytes: usize,
        operation_type: RequestOperationType,
    ) -> Self {
        Self {
            request_handle,
            operation_id,
            start_time: std::time::Instant::now(),
            expected_bytes,
            operation_type,
        }
    }

    /// Get the operation ID
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    /// Get the start time
    pub fn start_time(&self) -> std::time::Instant {
        self.start_time
    }

    /// Get expected bytes
    pub fn expected_bytes(&self) -> usize {
        self.expected_bytes
    }

    /// Get operation type
    pub fn operation_type(&self) -> RequestOperationType {
        self.operation_type
    }

    /// Get the request handle
    pub fn request_handle(&self) -> *mut c_void {
        self.request_handle
    }
}

unsafe impl Send for MPIPersistentRequest {}
unsafe impl Sync for MPIPersistentRequest {}

unsafe impl Send for MPIRequest {}
unsafe impl Sync for MPIRequest {}

unsafe impl Send for BufferInfo {}
unsafe impl Sync for BufferInfo {}