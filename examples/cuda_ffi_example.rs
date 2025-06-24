// Full CUDA + Rust example with kernel execution
use std::{ffi::{CString, c_void}, mem, ptr};

#[link(name = "cuda")]
extern "C" {
    // Module management
    fn cuModuleLoadData(module: *mut CUmodule, image: *const i8) -> i32;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> i32;
    
    // Kernel launch
    fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    
    // Context management
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate(pctx: *mut CUcontext, flags: u32, dev: i32) -> i32;
    fn cuCtxSynchronize() -> i32;
}

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
}

type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUcontext = *mut c_void;
type CUstream = *mut c_void;

const CUDA_SUCCESS: i32 = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

fn check_cuda(code: i32, msg: &str) {
    if code != CUDA_SUCCESS {
        unsafe {
            let err_str = cudaGetErrorString(code);
            let err_msg = std::ffi::CStr::from_ptr(err_str).to_string_lossy();
            panic!("CUDA error {}: {} - {}", code, msg, err_msg);
        }
    }
}

fn main() {
    unsafe {
        println!("=== Full CUDA + Rust Example ===\n");
        
        // Initialize CUDA
        check_cuda(cuInit(0), "Failed to initialize CUDA");
        
        // Get device and create context
        let mut device = 0;
        check_cuda(cuDeviceGet(&mut device, 0), "Failed to get device");
        
        let mut context: CUcontext = ptr::null_mut();
        check_cuda(cuCtxCreate(&mut context, 0, device), "Failed to create context");
        
        // Load PTX module
        let ptx_data = include_str!("cuda_kernel.ptx");
        let ptx_cstring = CString::new(ptx_data).unwrap();
        let mut module: CUmodule = ptr::null_mut();
        check_cuda(
            cuModuleLoadData(&mut module, ptx_cstring.as_ptr()),
            "Failed to load PTX module"
        );
        
        // Get kernel functions
        let kernel_name = CString::new("vector_add").unwrap();
        let mut kernel: CUfunction = ptr::null_mut();
        check_cuda(
            cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr()),
            "Failed to get kernel function"
        );
        
        // Test 1: Vector Addition
        println!("Test 1: Vector Addition");
        let n = 1024 * 1024; // 1M elements
        let size = n * mem::size_of::<f32>();
        
        // Host data
        let h_a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let h_b: Vec<f32> = (0..n).map(|i| i as f32 * 0.002).collect();
        let mut h_c: Vec<f32> = vec![0.0; n];
        
        // Device memory
        let mut d_a: *mut c_void = ptr::null_mut();
        let mut d_b: *mut c_void = ptr::null_mut();
        let mut d_c: *mut c_void = ptr::null_mut();
        
        check_cuda(cudaMalloc(&mut d_a, size), "Failed to allocate d_a");
        check_cuda(cudaMalloc(&mut d_b, size), "Failed to allocate d_b");
        check_cuda(cudaMalloc(&mut d_c, size), "Failed to allocate d_c");
        
        // Copy to device
        check_cuda(
            cudaMemcpy(d_a, h_a.as_ptr() as *const c_void, size, CUDA_MEMCPY_HOST_TO_DEVICE),
            "Failed to copy to d_a"
        );
        check_cuda(
            cudaMemcpy(d_b, h_b.as_ptr() as *const c_void, size, CUDA_MEMCPY_HOST_TO_DEVICE),
            "Failed to copy to d_b"
        );
        
        // Launch kernel
        let block_size = 256;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        
        let mut args: [*mut c_void; 4] = [
            &mut d_c as *mut _ as *mut c_void,
            &mut d_a as *mut _ as *mut c_void,
            &mut d_b as *mut _ as *mut c_void,
            &mut (n as i32) as *mut _ as *mut c_void,
        ];
        
        println!("Launching kernel with {} blocks of {} threads", grid_size, block_size);
        let start = std::time::Instant::now();
        
        check_cuda(
            cuLaunchKernel(
                kernel,
                grid_size, 1, 1,
                block_size, 1, 1,
                0,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut()
            ),
            "Failed to launch kernel"
        );
        
        check_cuda(cuCtxSynchronize(), "Failed to synchronize");
        
        let elapsed = start.elapsed();
        println!("Kernel execution time: {:?}", elapsed);
        
        // Copy result back
        check_cuda(
            cudaMemcpy(h_c.as_mut_ptr() as *mut c_void, d_c, size, CUDA_MEMCPY_DEVICE_TO_HOST),
            "Failed to copy result"
        );
        
        // Verify results
        println!("Verifying results...");
        let mut correct = true;
        for i in 0..5 {
            let expected = h_a[i] + h_b[i];
            if (h_c[i] - expected).abs() > 0.0001 {
                correct = false;
                println!("Mismatch at {}: {} + {} = {} (expected {})", 
                         i, h_a[i], h_b[i], h_c[i], expected);
            }
        }
        if correct {
            println!("✓ Results verified!");
        }
        
        // Test 2: Stress Test
        println!("\nTest 2: GPU Stress Test");
        let stress_kernel_name = CString::new("stress_kernel").unwrap();
        let mut stress_kernel: CUfunction = ptr::null_mut();
        check_cuda(
            cuModuleGetFunction(&mut stress_kernel, module, stress_kernel_name.as_ptr()),
            "Failed to get stress kernel"
        );
        
        let iterations = 1000;
        let mut stress_args: [*mut c_void; 3] = [
            &mut d_a as *mut _ as *mut c_void,
            &mut (n as i32) as *mut _ as *mut c_void,
            &mut (iterations as i32) as *mut _ as *mut c_void,
        ];
        
        println!("Running stress test with {} iterations...", iterations);
        let stress_start = std::time::Instant::now();
        
        for run in 0..10 {
            check_cuda(
                cuLaunchKernel(
                    stress_kernel,
                    grid_size, 1, 1,
                    block_size, 1, 1,
                    0,
                    ptr::null_mut(),
                    stress_args.as_mut_ptr(),
                    ptr::null_mut()
                ),
                "Failed to launch stress kernel"
            );
            
            check_cuda(cuCtxSynchronize(), "Failed to synchronize");
            println!("Stress run {} completed", run + 1);
        }
        
        let stress_elapsed = stress_start.elapsed();
        println!("Total stress test time: {:?}", stress_elapsed);
        println!("This should have caused significant GPU utilization!");
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        println!("\n✓ Full CUDA + Rust example completed successfully!");
    }
}