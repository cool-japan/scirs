#![feature(test)]
extern crate test;

use ndarray::{Array, Array2};
use scirs2_core::array_protocol::{
    self, add, matmul, transpose, GPUBackend, GPUConfig, GPUNdarray, NdarrayWrapper,
};
use test::Bencher;

/// Benchmark matrix multiplication with regular ndarray
#[bench]
fn bench_ndarray_matmul(b: &mut Bencher) {
    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    b.iter(|| {
        let result = a.dot(&b);
        test::black_box(result);
    });
}

/// Benchmark matrix multiplication with array protocol (NdarrayWrapper)
#[bench]
fn bench_array_protocol_matmul(b: &mut Bencher) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    let wrapped_a = NdarrayWrapper::new(a);
    let wrapped_b = NdarrayWrapper::new(b);

    b.iter(|| {
        let result = matmul(&wrapped_a, &wrapped_b).unwrap();
        test::black_box(result);
    });
}

/// Benchmark matrix multiplication with GPU array
#[bench]
fn bench_gpu_array_matmul(b: &mut Bencher) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };

    let gpu_a = GPUNdarray::new(a, gpu_config.clone());
    let gpu_b = GPUNdarray::new(b, gpu_config);

    b.iter(|| {
        let result = matmul(&gpu_a, &gpu_b).unwrap();
        test::black_box(result);
    });
}

/// Benchmark element-wise addition with regular ndarray
#[bench]
fn bench_ndarray_add(b: &mut Bencher) {
    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    b.iter(|| {
        let result = &a + &b;
        test::black_box(result);
    });
}

/// Benchmark element-wise addition with array protocol (NdarrayWrapper)
#[bench]
fn bench_array_protocol_add(b: &mut Bencher) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    let wrapped_a = NdarrayWrapper::new(a);
    let wrapped_b = NdarrayWrapper::new(b);

    b.iter(|| {
        let result = add(&wrapped_a, &wrapped_b).unwrap();
        test::black_box(result);
    });
}

/// Benchmark transpose with regular ndarray
#[bench]
fn bench_ndarray_transpose(b: &mut Bencher) {
    let a = Array2::<f64>::ones((100, 100));

    b.iter(|| {
        let result = a.t().to_owned();
        test::black_box(result);
    });
}

/// Benchmark transpose with array protocol (NdarrayWrapper)
#[bench]
fn bench_array_protocol_transpose(b: &mut Bencher) {
    array_protocol::init();

    let a = Array2::<f64>::ones((100, 100));
    let wrapped_a = NdarrayWrapper::new(a);

    b.iter(|| {
        let result = transpose(&wrapped_a).unwrap();
        test::black_box(result);
    });
}
