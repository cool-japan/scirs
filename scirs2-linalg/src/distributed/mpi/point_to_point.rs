//! MPI Point-to-Point Communication
//!
//! This module provides point-to-point communication operations including
//! blocking and non-blocking send/receive operations with advanced features
//! like persistent requests and optimized message passing.

use crate::error::{LinalgError, LinalgResult};
use super::{MPICommunicator, MPIDatatype, MPIRequest, MPIStatus, RequestOperationType};
use std::ffi::{c_int, c_void};

/// Point-to-point operations handler
pub struct MPIPointToPoint;

impl MPIPointToPoint {
    /// Send data to another process (non-blocking)
    pub fn isend<T>(
        comm: &MPICommunicator,
        data: &[T],
        dest: i32,
        tag: i32,
    ) -> LinalgResult<String>
    where
        T: MPIDatatype + Clone,
    {
        let operation_id = format!(
            "send_{}_{}_{}_{}",
            comm.rank(),
            dest,
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        unsafe {
            let request = mpi_isend(
                data.as_ptr() as *const c_void,
                data.len(),
                T::mpi_datatype(),
                dest,
                tag,
                comm.handle().raw_handle(),
            );

            if request.is_null() {
                return Err(LinalgError::CommunicationError(
                    "Failed to create send request".to_string()
                ));
            }

            let mpi_request = MPIRequest::new(
                request,
                operation_id.clone(),
                data.len() * std::mem::size_of::<T>(),
                RequestOperationType::PointToPoint,
            );

            comm.active_operations.write().unwrap().insert(operation_id.clone(), mpi_request);
        }

        Ok(operation_id)
    }

    /// Receive data from another process (non-blocking)
    pub fn irecv<T>(
        comm: &MPICommunicator,
        buffer: &mut [T],
        source: i32,
        tag: i32,
    ) -> LinalgResult<String>
    where
        T: MPIDatatype + Clone,
    {
        let operation_id = format!(
            "recv_{}_{}_{}_{}",
            source,
            comm.rank(),
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        unsafe {
            let request = mpi_irecv(
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len(),
                T::mpi_datatype(),
                source,
                tag,
                comm.handle().raw_handle(),
            );

            if request.is_null() {
                return Err(LinalgError::CommunicationError(
                    "Failed to create receive request".to_string()
                ));
            }

            let mpi_request = MPIRequest::new(
                request,
                operation_id.clone(),
                buffer.len() * std::mem::size_of::<T>(),
                RequestOperationType::PointToPoint,
            );

            comm.active_operations.write().unwrap().insert(operation_id.clone(), mpi_request);
        }

        Ok(operation_id)
    }

    /// Send data to another process (blocking)
    pub fn send<T>(
        comm: &MPICommunicator,
        data: &[T],
        dest: i32,
        tag: i32,
    ) -> LinalgResult<()>
    where
        T: MPIDatatype + Clone,
    {
        unsafe {
            let result = mpi_send(
                data.as_ptr() as *const c_void,
                data.len(),
                T::mpi_datatype(),
                dest,
                tag,
                comm.handle().raw_handle(),
            );

            if result != 0 {
                return Err(LinalgError::CommunicationError(
                    format!("MPI send failed with code {}", result)
                ));
            }
        }

        // Update statistics
        comm.update_stats(
            RequestOperationType::PointToPoint,
            data.len() * std::mem::size_of::<T>(),
            0.0, // No timing for blocking operations
        );

        Ok(())
    }

    /// Receive data from another process (blocking)
    pub fn recv<T>(
        comm: &MPICommunicator,
        buffer: &mut [T],
        source: i32,
        tag: i32,
    ) -> LinalgResult<MPIStatus>
    where
        T: MPIDatatype + Clone,
    {
        unsafe {
            let mut status = MPIStatus::default();
            let result = mpi_recv(
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len(),
                T::mpi_datatype(),
                source,
                tag,
                comm.handle().raw_handle(),
                &mut status as *mut _ as *mut c_void,
            );

            if result != 0 {
                return Err(LinalgError::CommunicationError(
                    format!("MPI recv failed with code {}", result)
                ));
            }

            Ok(status)
        }
    }

    /// Wait for completion of a non-blocking operation
    pub fn wait(comm: &MPICommunicator, operation_id: &str) -> LinalgResult<MPIStatus> {
        let mut active_ops = comm.active_operations.write().unwrap();

        if let Some(request) = active_ops.remove(operation_id) {
            unsafe {
                let mut status = MPIStatus::default();
                let result = mpi_wait(
                    request.request_handle(),
                    &mut status as *mut _ as *mut c_void
                );

                if result != 0 {
                    return Err(LinalgError::CommunicationError(
                        format!("MPI wait failed with code {}", result)
                    ));
                }

                // Update statistics
                let elapsed = request.start_time().elapsed().as_secs_f64();
                comm.update_stats(
                    request.operation_type(),
                    request.expected_bytes(),
                    elapsed,
                );

                Ok(status)
            }
        } else {
            Err(LinalgError::CommunicationError(
                format!("Operation {} not found", operation_id)
            ))
        }
    }

    /// Test if a non-blocking operation has completed
    pub fn test(comm: &MPICommunicator, operation_id: &str) -> LinalgResult<Option<MPIStatus>> {
        let active_ops = comm.active_operations.read().unwrap();

        if let Some(request) = active_ops.get(operation_id) {
            unsafe {
                let mut flag: c_int = 0;
                let mut status = MPIStatus::default();
                let result = mpi_test(
                    request.request_handle(),
                    &mut flag,
                    &mut status as *mut _ as *mut c_void
                );

                if result != 0 {
                    return Err(LinalgError::CommunicationError(
                        format!("MPI test failed with code {}", result)
                    ));
                }

                if flag != 0 {
                    // Operation completed
                    drop(active_ops);
                    let mut active_ops = comm.active_operations.write().unwrap();
                    if let Some(request) = active_ops.remove(operation_id) {
                        let elapsed = request.start_time().elapsed().as_secs_f64();
                        comm.update_stats(
                            request.operation_type(),
                            request.expected_bytes(),
                            elapsed,
                        );
                    }
                    Ok(Some(status))
                } else {
                    Ok(None)
                }
            }
        } else {
            Err(LinalgError::CommunicationError(
                format!("Operation {} not found", operation_id)
            ))
        }
    }

    /// Wait for any of the given operations to complete
    pub fn waitany(
        comm: &MPICommunicator,
        operation_ids: &[String],
    ) -> LinalgResult<(usize, MPIStatus)> {
        if operation_ids.is_empty() {
            return Err(LinalgError::InvalidInput(
                "No operations provided".to_string()
            ));
        }

        let active_ops = comm.active_operations.read().unwrap();
        let mut requests: Vec<*mut c_void> = Vec::new();
        let mut valid_indices: Vec<usize> = Vec::new();

        for (i, op_id) in operation_ids.iter().enumerate() {
            if let Some(request) = active_ops.get(op_id) {
                requests.push(request.request_handle());
                valid_indices.push(i);
            }
        }

        if requests.is_empty() {
            return Err(LinalgError::CommunicationError(
                "No valid operations found".to_string()
            ));
        }

        unsafe {
            let mut index: c_int = 0;
            let mut status = MPIStatus::default();
            let result = mpi_waitany(
                requests.len() as c_int,
                requests.as_ptr(),
                &mut index,
                &mut status as *mut _ as *mut c_void,
            );

            if result != 0 {
                return Err(LinalgError::CommunicationError(
                    format!("MPI waitany failed with code {}", result)
                ));
            }

            let completed_index = valid_indices[index as usize];
            let operation_id = &operation_ids[completed_index];

            // Remove completed operation and update stats
            drop(active_ops);
            let mut active_ops = comm.active_operations.write().unwrap();
            if let Some(request) = active_ops.remove(operation_id) {
                let elapsed = request.start_time().elapsed().as_secs_f64();
                comm.update_stats(
                    request.operation_type(),
                    request.expected_bytes(),
                    elapsed,
                );
            }

            Ok((completed_index, status))
        }
    }

    /// Wait for all given operations to complete
    pub fn waitall(
        comm: &MPICommunicator,
        operation_ids: &[String],
    ) -> LinalgResult<Vec<MPIStatus>> {
        let mut statuses = Vec::new();

        for operation_id in operation_ids {
            let status = Self::wait(comm, operation_id)?;
            statuses.push(status);
        }

        Ok(statuses)
    }

    /// Cancel a non-blocking operation
    pub fn cancel(comm: &MPICommunicator, operation_id: &str) -> LinalgResult<()> {
        let mut active_ops = comm.active_operations.write().unwrap();

        if let Some(request) = active_ops.remove(operation_id) {
            unsafe {
                let result = mpi_cancel(request.request_handle());
                if result != 0 {
                    return Err(LinalgError::CommunicationError(
                        format!("MPI cancel failed with code {}", result)
                    ));
                }
            }
            Ok(())
        } else {
            Err(LinalgError::CommunicationError(
                format!("Operation {} not found", operation_id)
            ))
        }
    }
}

// Additional FFI declarations for point-to-point operations
extern "C" {
    fn mpi_send(buf: *const c_void, count: usize, datatype: c_int, dest: c_int, tag: c_int, comm: *mut c_void) -> c_int;
    fn mpi_recv(buf: *mut c_void, count: usize, datatype: c_int, source: c_int, tag: c_int, comm: *mut c_void, status: *mut c_void) -> c_int;
    fn mpi_isend(buf: *const c_void, count: usize, datatype: c_int, dest: c_int, tag: c_int, comm: *mut c_void) -> *mut c_void;
    fn mpi_irecv(buf: *mut c_void, count: usize, datatype: c_int, source: c_int, tag: c_int, comm: *mut c_void) -> *mut c_void;
    fn mpi_wait(request: *mut c_void, status: *mut c_void) -> c_int;
    fn mpi_test(request: *mut c_void, flag: *mut c_int, status: *mut c_void) -> c_int;
    fn mpi_waitany(count: c_int, requests: *const *mut c_void, index: *mut c_int, status: *mut c_void) -> c_int;
    fn mpi_cancel(request: *mut c_void) -> c_int;
}