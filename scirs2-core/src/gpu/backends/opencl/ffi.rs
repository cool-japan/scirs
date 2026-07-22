//! Pure-Rust runtime loader for the OpenCL ICD.
//!
//! This module declares the small slice of the OpenCL C ABI that the
//! `scirs2-core` OpenCL backend actually uses and resolves it at runtime from
//! the vendor-installed ICD (`libOpenCL.so.1` on Linux, `OpenCL.dll` on
//! Windows, the `OpenCL` framework on macOS) through the pure-Rust
//! [`libloading`] wrapper.
//!
//! Nothing here links against an OpenCL development library: there is no
//! `#[link]`, no `build.rs`, and no `-lOpenCL`. The crate therefore builds on
//! machines that only ship the versioned runtime library (or none at all).
//! When the ICD is missing at runtime the loader returns `None` and the
//! backend reports itself unavailable — never a panic — exactly like the CUDA
//! driver probe in `simd_ops/gpu_detection.rs` and like wgpu loading Vulkan/GL
//! ICDs at runtime.
//!
//! The resolved function table is cached process-wide in a
//! [`std::sync::OnceLock`], so the ICD is loaded at most once.

use std::ffi::{c_char, c_void, CString};
use std::ptr::{null, null_mut};
use std::sync::OnceLock;

use libloading::Library;

use crate::gpu::GpuError;

// ---------------------------------------------------------------------------
// C ABI scalar types (verified against opencl-sys 0.6.1).
// ---------------------------------------------------------------------------

/// `cl_int` — 32-bit signed integer used for status codes.
#[allow(non_camel_case_types)]
pub type cl_int = i32;
/// `cl_uint` — 32-bit unsigned integer used for counts and indices.
#[allow(non_camel_case_types)]
pub type cl_uint = u32;
/// `cl_ulong` — 64-bit unsigned integer.
#[allow(non_camel_case_types)]
pub type cl_ulong = u64;
/// `cl_bool` — OpenCL boolean (`CL_TRUE`/`CL_FALSE`).
#[allow(non_camel_case_types)]
pub type cl_bool = cl_uint;
/// `cl_bitfield` — 64-bit bitfield used for flag sets.
#[allow(non_camel_case_types)]
pub type cl_bitfield = cl_ulong;
/// `size_t` — matches the platform pointer width.
#[allow(non_camel_case_types)]
pub type size_t = usize;

// ---------------------------------------------------------------------------
// Opaque handle types (every OpenCL handle is a `void*`).
// ---------------------------------------------------------------------------

/// `cl_platform_id` handle.
#[allow(non_camel_case_types)]
pub type cl_platform_id = *mut c_void;
/// `cl_device_id` handle.
#[allow(non_camel_case_types)]
pub type cl_device_id = *mut c_void;
/// `cl_context` handle.
#[allow(non_camel_case_types)]
pub type cl_context = *mut c_void;
/// `cl_command_queue` handle.
#[allow(non_camel_case_types)]
pub type cl_command_queue = *mut c_void;
/// `cl_mem` handle.
#[allow(non_camel_case_types)]
pub type cl_mem = *mut c_void;
/// `cl_program` handle.
#[allow(non_camel_case_types)]
pub type cl_program = *mut c_void;
/// `cl_kernel` handle.
#[allow(non_camel_case_types)]
pub type cl_kernel = *mut c_void;
/// `cl_event` handle.
#[allow(non_camel_case_types)]
pub type cl_event = *mut c_void;

/// `cl_device_type` bitfield.
#[allow(non_camel_case_types)]
pub type cl_device_type = cl_bitfield;
/// `cl_mem_flags` bitfield.
#[allow(non_camel_case_types)]
pub type cl_mem_flags = cl_bitfield;
/// `cl_command_queue_properties` bitfield.
#[allow(non_camel_case_types)]
pub type cl_command_queue_properties = cl_bitfield;
/// `cl_queue_properties` list element (OpenCL 2.0+ properties array).
#[allow(non_camel_case_types)]
pub type cl_queue_properties = cl_ulong;
/// `cl_context_properties` list element.
#[allow(non_camel_case_types)]
pub type cl_context_properties = isize;
/// `cl_mem_info` query selector.
#[allow(non_camel_case_types)]
pub type cl_mem_info = cl_uint;
/// `cl_program_build_info` query selector.
#[allow(non_camel_case_types)]
pub type cl_program_build_info = cl_uint;

// ---------------------------------------------------------------------------
// Constants (verified against opencl-sys 0.6.1).
// ---------------------------------------------------------------------------

/// `CL_SUCCESS` status code.
pub const CL_SUCCESS: cl_int = 0;
/// `CL_TRUE` boolean value.
pub const CL_TRUE: cl_bool = 1;
/// `CL_BLOCKING` — blocking enqueue flag (`CL_TRUE`).
pub const CL_BLOCKING: cl_bool = 1;
/// `CL_DEVICE_TYPE_GPU` — select GPU devices.
pub const CL_DEVICE_TYPE_GPU: cl_device_type = 1 << 2;
/// `CL_QUEUE_PROFILING_ENABLE` — enable command-queue profiling.
pub const CL_QUEUE_PROFILING_ENABLE: cl_command_queue_properties = 1 << 1;
/// `CL_QUEUE_PROPERTIES` — property name for the 2.0+ properties array.
pub const CL_QUEUE_PROPERTIES: cl_uint = 0x1093;
/// `CL_MEM_READ_WRITE` — read/write memory-object access flag.
pub const CL_MEM_READ_WRITE: cl_mem_flags = 1 << 0;
/// `CL_MEM_SIZE` — query a memory object's byte size.
pub const CL_MEM_SIZE: cl_mem_info = 0x1102;
/// `CL_PROGRAM_BUILD_LOG` — query a program's build log.
pub const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183;

// ---------------------------------------------------------------------------
// Function-pointer types.
//
// OpenCL uses the plain C calling convention (cdecl) on every supported
// target — including 32-bit Windows, where (unlike the CUDA driver) it is
// still `__cdecl`. Therefore these are `extern "C"`, NOT `extern "system"`.
// ---------------------------------------------------------------------------

type FnGetPlatformIDs = unsafe extern "C" fn(cl_uint, *mut cl_platform_id, *mut cl_uint) -> cl_int;
type FnGetDeviceIDs = unsafe extern "C" fn(
    cl_platform_id,
    cl_device_type,
    cl_uint,
    *mut cl_device_id,
    *mut cl_uint,
) -> cl_int;
type FnCreateContext = unsafe extern "C" fn(
    *const cl_context_properties,
    cl_uint,
    *const cl_device_id,
    Option<unsafe extern "C" fn(*const c_char, *const c_void, size_t, *mut c_void)>,
    *mut c_void,
    *mut cl_int,
) -> cl_context;
/// OpenCL 2.0+ command-queue constructor.
type FnCreateCommandQueueWithProperties = unsafe extern "C" fn(
    cl_context,
    cl_device_id,
    *const cl_queue_properties,
    *mut cl_int,
) -> cl_command_queue;
/// OpenCL 1.x command-queue constructor (fallback for pre-2.0 ICDs).
type FnCreateCommandQueue = unsafe extern "C" fn(
    cl_context,
    cl_device_id,
    cl_command_queue_properties,
    *mut cl_int,
) -> cl_command_queue;
type FnCreateProgramWithSource = unsafe extern "C" fn(
    cl_context,
    cl_uint,
    *const *const c_char,
    *const size_t,
    *mut cl_int,
) -> cl_program;
type FnBuildProgram = unsafe extern "C" fn(
    cl_program,
    cl_uint,
    *const cl_device_id,
    *const c_char,
    Option<unsafe extern "C" fn(cl_program, *mut c_void)>,
    *mut c_void,
) -> cl_int;
type FnGetProgramBuildInfo = unsafe extern "C" fn(
    cl_program,
    cl_device_id,
    cl_program_build_info,
    size_t,
    *mut c_void,
    *mut size_t,
) -> cl_int;
type FnCreateKernel = unsafe extern "C" fn(cl_program, *const c_char, *mut cl_int) -> cl_kernel;
type FnCreateBuffer =
    unsafe extern "C" fn(cl_context, cl_mem_flags, size_t, *mut c_void, *mut cl_int) -> cl_mem;
type FnEnqueueWriteBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    cl_bool,
    size_t,
    size_t,
    *const c_void,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnEnqueueReadBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    cl_bool,
    size_t,
    size_t,
    *mut c_void,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnSetKernelArg = unsafe extern "C" fn(cl_kernel, cl_uint, size_t, *const c_void) -> cl_int;
type FnEnqueueNDRangeKernel = unsafe extern "C" fn(
    cl_command_queue,
    cl_kernel,
    cl_uint,
    *const size_t,
    *const size_t,
    *const size_t,
    cl_uint,
    *const cl_event,
    *mut cl_event,
) -> cl_int;
type FnGetMemObjectInfo =
    unsafe extern "C" fn(cl_mem, cl_mem_info, size_t, *mut c_void, *mut size_t) -> cl_int;
type FnFinish = unsafe extern "C" fn(cl_command_queue) -> cl_int;
/// Shared signature for the five `clRelease*` entry points — each takes a
/// single opaque handle and returns a status code.
type FnRelease = unsafe extern "C" fn(*mut c_void) -> cl_int;

/// The command-queue constructor actually resolved from the ICD.
///
/// OpenCL 2.0 replaced `clCreateCommandQueue` with
/// `clCreateCommandQueueWithProperties`; 1.x-only ICDs export only the former.
/// We prefer the 2.0+ entry point and transparently fall back to the 1.x one.
enum QueueCreator {
    /// OpenCL 2.0+ `clCreateCommandQueueWithProperties`.
    WithProperties(FnCreateCommandQueueWithProperties),
    /// OpenCL 1.x `clCreateCommandQueue`.
    Legacy(FnCreateCommandQueue),
}

/// Resolved OpenCL C ABI entry points plus the resident library handle.
pub struct OpenClApi {
    // The loaded ICD. Keeping it inside the struct guarantees the mapping
    // outlives every resolved function pointer.
    _lib: Library,
    cl_get_platform_ids: FnGetPlatformIDs,
    cl_get_device_ids: FnGetDeviceIDs,
    cl_create_context: FnCreateContext,
    queue_creator: QueueCreator,
    cl_create_program_with_source: FnCreateProgramWithSource,
    cl_build_program: FnBuildProgram,
    cl_get_program_build_info: FnGetProgramBuildInfo,
    cl_create_kernel: FnCreateKernel,
    cl_create_buffer: FnCreateBuffer,
    cl_enqueue_write_buffer: FnEnqueueWriteBuffer,
    cl_enqueue_read_buffer: FnEnqueueReadBuffer,
    cl_set_kernel_arg: FnSetKernelArg,
    cl_enqueue_nd_range_kernel: FnEnqueueNDRangeKernel,
    cl_get_mem_object_info: FnGetMemObjectInfo,
    cl_finish: FnFinish,
    cl_release_context: FnRelease,
    cl_release_command_queue: FnRelease,
    cl_release_program: FnRelease,
    cl_release_kernel: FnRelease,
    cl_release_mem_object: FnRelease,
}

// SAFETY: `OpenClApi` holds raw C function pointers (which are `Copy`, `Send`
// and `Sync`) plus a `libloading::Library` (itself `Send + Sync`). The OpenCL
// ICD entry points are re-entrant and safe to call from multiple threads; all
// state that requires external synchronisation (command queues) is guarded by
// `Arc<Mutex<..>>` at the call sites. Keeping `_lib` resident means the
// pointers never dangle.
unsafe impl Send for OpenClApi {}
unsafe impl Sync for OpenClApi {}

/// Error produced while loading and binding the OpenCL ICD.
#[derive(Debug)]
pub enum OpenClLoadError {
    /// No OpenCL ICD library could be opened.
    LibraryNotFound(String),
    /// A required OpenCL symbol was absent from the loaded ICD.
    SymbolMissing(&'static str),
}

impl std::fmt::Display for OpenClLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenClLoadError::LibraryNotFound(msg) => {
                write!(f, "OpenCL ICD library could not be loaded: {msg}")
            }
            OpenClLoadError::SymbolMissing(name) => {
                write!(f, "OpenCL ICD is missing required symbol `{name}`")
            }
        }
    }
}

impl std::error::Error for OpenClLoadError {}

impl OpenClApi {
    /// Load the OpenCL ICD and resolve every entry point used by the backend.
    ///
    /// Tries the versioned Linux runtime name first, then the unversioned
    /// name, then the platform-neutral `OpenCL` name (Windows DLL / macOS
    /// framework). Returns [`OpenClLoadError`] when no library can be opened
    /// or a required symbol is absent; never panics.
    pub fn load() -> Result<Self, OpenClLoadError> {
        // SAFETY: opening a vendor-provided shared library. A missing or
        // broken library yields `Err`; running the library initialisers of an
        // OpenCL ICD under `dlopen`/`LoadLibrary` is the supported usage.
        let lib = unsafe {
            Library::new("libOpenCL.so.1")
                .or_else(|_| Library::new("libOpenCL.so"))
                .or_else(|_| Library::new("OpenCL"))
        }
        .map_err(|e| OpenClLoadError::LibraryNotFound(e.to_string()))?;

        // SAFETY: each symbol is looked up by its documented OpenCL name and
        // bound to a function-pointer type whose signature matches the OpenCL
        // C ABI. The `Symbol` temporaries are dropped at the end of each `let`
        // (they only copy out `'static` function pointers), so `lib` is free
        // to move into the struct afterwards.
        unsafe {
            macro_rules! sym {
                ($bytes:expr, $name:expr) => {
                    *lib.get($bytes)
                        .map_err(|_| OpenClLoadError::SymbolMissing($name))?
                };
            }

            // Prefer the OpenCL 2.0+ constructor; fall back to the 1.x one so
            // pre-2.0 ICDs remain usable rather than reported unavailable.
            let queue_creator = if let Ok(f) = lib
                .get::<FnCreateCommandQueueWithProperties>(b"clCreateCommandQueueWithProperties\0")
            {
                QueueCreator::WithProperties(*f)
            } else if let Ok(f) = lib.get::<FnCreateCommandQueue>(b"clCreateCommandQueue\0") {
                QueueCreator::Legacy(*f)
            } else {
                return Err(OpenClLoadError::SymbolMissing(
                    "clCreateCommandQueueWithProperties/clCreateCommandQueue",
                ));
            };

            let cl_get_platform_ids: FnGetPlatformIDs =
                sym!(b"clGetPlatformIDs\0", "clGetPlatformIDs");
            let cl_get_device_ids: FnGetDeviceIDs = sym!(b"clGetDeviceIDs\0", "clGetDeviceIDs");
            let cl_create_context: FnCreateContext = sym!(b"clCreateContext\0", "clCreateContext");
            let cl_create_program_with_source: FnCreateProgramWithSource =
                sym!(b"clCreateProgramWithSource\0", "clCreateProgramWithSource");
            let cl_build_program: FnBuildProgram = sym!(b"clBuildProgram\0", "clBuildProgram");
            let cl_get_program_build_info: FnGetProgramBuildInfo =
                sym!(b"clGetProgramBuildInfo\0", "clGetProgramBuildInfo");
            let cl_create_kernel: FnCreateKernel = sym!(b"clCreateKernel\0", "clCreateKernel");
            let cl_create_buffer: FnCreateBuffer = sym!(b"clCreateBuffer\0", "clCreateBuffer");
            let cl_enqueue_write_buffer: FnEnqueueWriteBuffer =
                sym!(b"clEnqueueWriteBuffer\0", "clEnqueueWriteBuffer");
            let cl_enqueue_read_buffer: FnEnqueueReadBuffer =
                sym!(b"clEnqueueReadBuffer\0", "clEnqueueReadBuffer");
            let cl_set_kernel_arg: FnSetKernelArg = sym!(b"clSetKernelArg\0", "clSetKernelArg");
            let cl_enqueue_nd_range_kernel: FnEnqueueNDRangeKernel =
                sym!(b"clEnqueueNDRangeKernel\0", "clEnqueueNDRangeKernel");
            let cl_get_mem_object_info: FnGetMemObjectInfo =
                sym!(b"clGetMemObjectInfo\0", "clGetMemObjectInfo");
            let cl_finish: FnFinish = sym!(b"clFinish\0", "clFinish");
            let cl_release_context: FnRelease = sym!(b"clReleaseContext\0", "clReleaseContext");
            let cl_release_command_queue: FnRelease =
                sym!(b"clReleaseCommandQueue\0", "clReleaseCommandQueue");
            let cl_release_program: FnRelease = sym!(b"clReleaseProgram\0", "clReleaseProgram");
            let cl_release_kernel: FnRelease = sym!(b"clReleaseKernel\0", "clReleaseKernel");
            let cl_release_mem_object: FnRelease =
                sym!(b"clReleaseMemObject\0", "clReleaseMemObject");

            Ok(Self {
                cl_get_platform_ids,
                cl_get_device_ids,
                cl_create_context,
                queue_creator,
                cl_create_program_with_source,
                cl_build_program,
                cl_get_program_build_info,
                cl_create_kernel,
                cl_create_buffer,
                cl_enqueue_write_buffer,
                cl_enqueue_read_buffer,
                cl_set_kernel_arg,
                cl_enqueue_nd_range_kernel,
                cl_get_mem_object_info,
                cl_finish,
                cl_release_context,
                cl_release_command_queue,
                cl_release_program,
                cl_release_kernel,
                cl_release_mem_object,
                _lib: lib,
            })
        }
    }

    /// Enumerate all OpenCL platform ids.
    pub fn platform_ids(&self) -> Result<Vec<cl_platform_id>, GpuError> {
        let mut count: cl_uint = 0;
        // SAFETY: querying the platform count with a null output buffer, as
        // documented for `clGetPlatformIDs`.
        let status = unsafe { (self.cl_get_platform_ids)(0, null_mut(), &mut count) };
        if status != CL_SUCCESS {
            return Err(GpuError::Other(format!(
                "clGetPlatformIDs (count) failed with status {status}"
            )));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut ids: Vec<cl_platform_id> = vec![null_mut(); count as usize];
        // SAFETY: `ids` has capacity for `count` platform ids.
        let status = unsafe { (self.cl_get_platform_ids)(count, ids.as_mut_ptr(), null_mut()) };
        if status != CL_SUCCESS {
            return Err(GpuError::Other(format!(
                "clGetPlatformIDs failed with status {status}"
            )));
        }
        Ok(ids)
    }

    /// Enumerate device ids of the requested type across every platform.
    pub fn device_ids(&self, device_type: cl_device_type) -> Result<Vec<cl_device_id>, GpuError> {
        let platforms = self.platform_ids()?;
        let mut all: Vec<cl_device_id> = Vec::new();
        for platform in platforms {
            let mut count: cl_uint = 0;
            // SAFETY: querying the device count for a valid platform; a
            // platform with no matching device returns a non-success status
            // (e.g. CL_DEVICE_NOT_FOUND), which we simply skip.
            let status = unsafe {
                (self.cl_get_device_ids)(platform, device_type, 0, null_mut(), &mut count)
            };
            if status != CL_SUCCESS || count == 0 {
                continue;
            }
            let mut ids: Vec<cl_device_id> = vec![null_mut(); count as usize];
            // SAFETY: `ids` has capacity for `count` device ids.
            let status = unsafe {
                (self.cl_get_device_ids)(platform, device_type, count, ids.as_mut_ptr(), null_mut())
            };
            if status == CL_SUCCESS {
                all.extend_from_slice(&ids);
            }
        }
        Ok(all)
    }

    /// Create a single-device context.
    pub fn create_context(&self, device: cl_device_id) -> Result<cl_context, GpuError> {
        let devices = [device];
        let mut err: cl_int = 0;
        // SAFETY: null context properties, one valid device, no notification
        // callback — the documented minimal `clCreateContext` invocation.
        let ctx = unsafe {
            (self.cl_create_context)(null(), 1, devices.as_ptr(), None, null_mut(), &mut err)
        };
        if err != CL_SUCCESS || ctx.is_null() {
            return Err(GpuError::Other(format!(
                "clCreateContext failed with status {err}"
            )));
        }
        Ok(ctx)
    }

    /// Create a profiling-enabled command queue for `device` within `ctx`.
    ///
    /// Uses the OpenCL 2.0+ properties-array constructor when available and
    /// the 1.x constructor otherwise.
    pub fn create_command_queue(
        &self,
        ctx: cl_context,
        device: cl_device_id,
    ) -> Result<cl_command_queue, GpuError> {
        let mut err: cl_int = 0;
        let queue = match self.queue_creator {
            QueueCreator::WithProperties(create) => {
                let props: [cl_queue_properties; 3] = [
                    CL_QUEUE_PROPERTIES as cl_queue_properties,
                    CL_QUEUE_PROFILING_ENABLE,
                    0,
                ];
                // SAFETY: null-terminated properties array with a single
                // property, matching `clCreateCommandQueueWithProperties`.
                unsafe { create(ctx, device, props.as_ptr(), &mut err) }
            }
            QueueCreator::Legacy(create) => {
                // SAFETY: 1.x `clCreateCommandQueue` with a plain bitfield of
                // queue properties.
                unsafe { create(ctx, device, CL_QUEUE_PROFILING_ENABLE, &mut err) }
            }
        };
        if err != CL_SUCCESS || queue.is_null() {
            return Err(GpuError::Other(format!(
                "clCreateCommandQueue failed with status {err}"
            )));
        }
        Ok(queue)
    }

    /// Build an OpenCL program from source for a single device.
    ///
    /// On a build failure the program's build log is fetched and included in
    /// the returned error, and the partially-created program is released.
    pub fn build_program(
        &self,
        ctx: cl_context,
        device: cl_device_id,
        source: &str,
    ) -> Result<cl_program, GpuError> {
        let src_bytes = source.as_bytes();
        let strings: [*const c_char; 1] = [src_bytes.as_ptr() as *const c_char];
        let lengths: [size_t; 1] = [src_bytes.len()];
        let mut err: cl_int = 0;
        // SAFETY: one source string with an explicit length, as required by
        // `clCreateProgramWithSource`.
        let program = unsafe {
            (self.cl_create_program_with_source)(
                ctx,
                1,
                strings.as_ptr(),
                lengths.as_ptr(),
                &mut err,
            )
        };
        if err != CL_SUCCESS || program.is_null() {
            return Err(GpuError::KernelCompilationError(format!(
                "clCreateProgramWithSource failed with status {err}"
            )));
        }

        let devices = [device];
        // SAFETY: building for a single valid device, no options, no build
        // callback — the documented synchronous `clBuildProgram` invocation.
        let build_status = unsafe {
            (self.cl_build_program)(program, 1, devices.as_ptr(), null(), None, null_mut())
        };
        if build_status != CL_SUCCESS {
            let log = self.program_build_log(program, device);
            // SAFETY: `program` is a valid handle we own and are discarding.
            unsafe {
                (self.cl_release_program)(program);
            }
            return Err(GpuError::KernelCompilationError(format!(
                "clBuildProgram failed with status {build_status}: {log}"
            )));
        }
        Ok(program)
    }

    /// Fetch the build log for `program` on `device` (empty on any error).
    fn program_build_log(&self, program: cl_program, device: cl_device_id) -> String {
        let mut size: size_t = 0;
        // SAFETY: querying the log size with a null output buffer.
        let status = unsafe {
            (self.cl_get_program_build_info)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                null_mut(),
                &mut size,
            )
        };
        if status != CL_SUCCESS || size == 0 {
            return String::new();
        }
        let mut buffer = vec![0u8; size];
        // SAFETY: `buffer` has capacity for `size` bytes.
        let status = unsafe {
            (self.cl_get_program_build_info)(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                size,
                buffer.as_mut_ptr() as *mut c_void,
                null_mut(),
            )
        };
        if status != CL_SUCCESS {
            return String::new();
        }
        while buffer.last() == Some(&0) {
            buffer.pop();
        }
        String::from_utf8_lossy(&buffer).into_owned()
    }

    /// Create a kernel object for the named entry point in `program`.
    pub fn create_kernel(&self, program: cl_program, name: &str) -> Result<cl_kernel, GpuError> {
        let c_name = CString::new(name)
            .map_err(|_| GpuError::Other(format!("invalid OpenCL kernel name `{name}`")))?;
        let mut err: cl_int = 0;
        // SAFETY: `c_name` is a valid null-terminated string that outlives the
        // call, matching `clCreateKernel`.
        let kernel = unsafe { (self.cl_create_kernel)(program, c_name.as_ptr(), &mut err) };
        if err != CL_SUCCESS || kernel.is_null() {
            return Err(GpuError::Other(format!(
                "clCreateKernel(`{name}`) failed with status {err}"
            )));
        }
        Ok(kernel)
    }

    /// Allocate a device buffer of `size` bytes.
    pub fn create_buffer(
        &self,
        ctx: cl_context,
        flags: cl_mem_flags,
        size: usize,
    ) -> Result<cl_mem, GpuError> {
        let mut err: cl_int = 0;
        // SAFETY: no host pointer provided; the runtime allocates `size` bytes.
        let mem = unsafe { (self.cl_create_buffer)(ctx, flags, size, null_mut(), &mut err) };
        if err != CL_SUCCESS || mem.is_null() {
            return Err(GpuError::OutOfMemory(format!(
                "clCreateBuffer({size} bytes) failed with status {err}"
            )));
        }
        Ok(mem)
    }

    /// Blocking write of `data` into `mem` at `offset`.
    pub fn enqueue_write(
        &self,
        queue: cl_command_queue,
        mem: cl_mem,
        offset: usize,
        data: &[u8],
    ) -> Result<(), GpuError> {
        // SAFETY: blocking write of exactly `data.len()` bytes from a valid
        // host slice into a valid device buffer, no wait list.
        let status = unsafe {
            (self.cl_enqueue_write_buffer)(
                queue,
                mem,
                CL_BLOCKING,
                offset,
                data.len(),
                data.as_ptr() as *const c_void,
                0,
                null(),
                null_mut(),
            )
        };
        check(status, "clEnqueueWriteBuffer")
    }

    /// Blocking read of `data.len()` bytes from `mem` at `offset` into `data`.
    pub fn enqueue_read(
        &self,
        queue: cl_command_queue,
        mem: cl_mem,
        offset: usize,
        data: &mut [u8],
    ) -> Result<(), GpuError> {
        // SAFETY: blocking read of exactly `data.len()` bytes from a valid
        // device buffer into a valid, writable host slice, no wait list.
        let status = unsafe {
            (self.cl_enqueue_read_buffer)(
                queue,
                mem,
                CL_BLOCKING,
                offset,
                data.len(),
                data.as_mut_ptr() as *mut c_void,
                0,
                null(),
                null_mut(),
            )
        };
        check(status, "clEnqueueReadBuffer")
    }

    /// Bind a by-value scalar argument (raw bytes) at `index`.
    pub fn set_arg_bytes(
        &self,
        kernel: cl_kernel,
        index: cl_uint,
        bytes: &[u8],
    ) -> Result<(), GpuError> {
        // SAFETY: `bytes` points to `bytes.len()` readable bytes that OpenCL
        // copies synchronously into the kernel's argument slot.
        let status = unsafe {
            (self.cl_set_kernel_arg)(kernel, index, bytes.len(), bytes.as_ptr() as *const c_void)
        };
        check(status, "clSetKernelArg")
    }

    /// Bind a `cl_mem` buffer argument at `index`.
    ///
    /// A buffer argument is passed as the address of the `cl_mem` handle with
    /// size `size_of::<cl_mem>()`, per the OpenCL specification.
    pub fn set_arg_mem(
        &self,
        kernel: cl_kernel,
        index: cl_uint,
        mem: &cl_mem,
    ) -> Result<(), GpuError> {
        // SAFETY: `mem` is the address of a live `cl_mem` handle; OpenCL reads
        // `size_of::<cl_mem>()` bytes from it synchronously.
        let status = unsafe {
            (self.cl_set_kernel_arg)(
                kernel,
                index,
                std::mem::size_of::<cl_mem>(),
                mem as *const cl_mem as *const c_void,
            )
        };
        check(status, "clSetKernelArg")
    }

    /// Enqueue an N-D range kernel launch.
    pub fn enqueue_nd_range(
        &self,
        queue: cl_command_queue,
        kernel: cl_kernel,
        global: &[usize],
        local: Option<&[usize]>,
    ) -> Result<(), GpuError> {
        let work_dim = global.len() as cl_uint;
        let local_ptr = match local {
            Some(l) => l.as_ptr(),
            None => null(),
        };
        // SAFETY: `work_dim` matches the length of the `global` (and, when
        // provided, `local`) work-size arrays; no global offset and no wait
        // list are used.
        let status = unsafe {
            (self.cl_enqueue_nd_range_kernel)(
                queue,
                kernel,
                work_dim,
                null(),
                global.as_ptr(),
                local_ptr,
                0,
                null(),
                null_mut(),
            )
        };
        check(status, "clEnqueueNDRangeKernel")
    }

    /// Block until all previously enqueued commands on `queue` complete.
    pub fn finish(&self, queue: cl_command_queue) -> Result<(), GpuError> {
        // SAFETY: `queue` is a valid command queue handle.
        let status = unsafe { (self.cl_finish)(queue) };
        check(status, "clFinish")
    }

    /// Query the byte size of a memory object.
    #[allow(dead_code)]
    pub fn mem_size(&self, mem: cl_mem) -> Result<usize, GpuError> {
        let mut size: size_t = 0;
        // SAFETY: `CL_MEM_SIZE` writes a single `size_t` into `size`.
        let status = unsafe {
            (self.cl_get_mem_object_info)(
                mem,
                CL_MEM_SIZE,
                std::mem::size_of::<size_t>(),
                &mut size as *mut size_t as *mut c_void,
                null_mut(),
            )
        };
        if status != CL_SUCCESS {
            return Err(GpuError::Other(format!(
                "clGetMemObjectInfo(CL_MEM_SIZE) failed with status {status}"
            )));
        }
        Ok(size)
    }

    /// Release a context handle (used by [`ClContext`]'s `Drop`).
    fn release_context(&self, ctx: cl_context) {
        // SAFETY: `ctx` is a valid context handle we own.
        unsafe {
            (self.cl_release_context)(ctx);
        }
    }

    /// Release a command-queue handle (used by [`ClQueue`]'s `Drop`).
    fn release_command_queue(&self, queue: cl_command_queue) {
        // SAFETY: `queue` is a valid command-queue handle we own.
        unsafe {
            (self.cl_release_command_queue)(queue);
        }
    }

    /// Release a program handle (used by [`ClProgram`]'s `Drop`).
    fn release_program(&self, program: cl_program) {
        // SAFETY: `program` is a valid program handle we own.
        unsafe {
            (self.cl_release_program)(program);
        }
    }

    /// Release a kernel handle (used by [`ClKernel`]'s `Drop`).
    fn release_kernel(&self, kernel: cl_kernel) {
        // SAFETY: `kernel` is a valid kernel handle we own.
        unsafe {
            (self.cl_release_kernel)(kernel);
        }
    }

    /// Release a memory-object handle (used by [`ClBuffer`]'s `Drop`).
    fn release_mem_object(&self, mem: cl_mem) {
        // SAFETY: `mem` is a valid memory-object handle we own.
        unsafe {
            (self.cl_release_mem_object)(mem);
        }
    }
}

/// Convert an OpenCL status code into a `Result`.
fn check(status: cl_int, op: &str) -> Result<(), GpuError> {
    if status == CL_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::Other(format!("{op} failed with status {status}")))
    }
}

/// Process-wide cached ICD function table (`None` when no ICD is present).
static OPENCL_API: OnceLock<Option<OpenClApi>> = OnceLock::new();

/// Return the resolved OpenCL API, loading it once, or `None` when no ICD is
/// available. Never panics; safe to call from any thread and hot paths.
pub fn api() -> Option<&'static OpenClApi> {
    OPENCL_API.get_or_init(|| OpenClApi::load().ok()).as_ref()
}

// ---------------------------------------------------------------------------
// RAII handle newtypes. Each releases its handle on `Drop` through the cached
// API, and each is a no-op when the API is unavailable (which cannot happen
// while a live handle exists, but is guarded to avoid panicking in `Drop`).
// ---------------------------------------------------------------------------

/// Owned OpenCL context handle.
pub struct ClContext(pub cl_context);

impl Drop for ClContext {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        if let Some(a) = api() {
            a.release_context(self.0);
        }
    }
}

/// Owned OpenCL command-queue handle.
pub struct ClQueue(pub cl_command_queue);

impl Drop for ClQueue {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        if let Some(a) = api() {
            a.release_command_queue(self.0);
        }
    }
}

/// Owned OpenCL program handle.
pub struct ClProgram(pub cl_program);

impl Drop for ClProgram {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        if let Some(a) = api() {
            a.release_program(self.0);
        }
    }
}

/// Owned OpenCL kernel handle.
pub struct ClKernel(pub cl_kernel);

impl Drop for ClKernel {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        if let Some(a) = api() {
            a.release_kernel(self.0);
        }
    }
}

/// Owned OpenCL memory-object handle together with its tracked byte size.
pub struct ClBuffer {
    /// The underlying `cl_mem` handle.
    pub mem: cl_mem,
    /// The requested byte size of the allocation.
    pub size: usize,
}

impl Drop for ClBuffer {
    fn drop(&mut self) {
        if self.mem.is_null() {
            return;
        }
        if let Some(a) = api() {
            a.release_mem_object(self.mem);
        }
    }
}

// SAFETY: these newtypes wrap OpenCL handles (raw `void*`). OpenCL runtime
// objects are internally reference-counted and thread-safe to reference from
// multiple threads; the backend only mutates queue state under `Arc<Mutex<..>>`.
// Marking the handles `Send + Sync` lets the wrapping backend types derive the
// same automatically.
unsafe impl Send for ClContext {}
unsafe impl Sync for ClContext {}
unsafe impl Send for ClQueue {}
unsafe impl Sync for ClQueue {}
unsafe impl Send for ClProgram {}
unsafe impl Sync for ClProgram {}
unsafe impl Send for ClKernel {}
unsafe impl Sync for ClKernel {}
unsafe impl Send for ClBuffer {}
unsafe impl Sync for ClBuffer {}
