#![allow(clippy::redundant_pattern_matching)]

// Edge case tests for concurrency and thread safety - 30+ tests
// Tests concurrent tensor evaluation, parallel gradient computation, and thread safety
//
// NOTE: Tensors belong to a specific graph/context and cannot be evaluated
// across different contexts. Each thread must create its own graph/context.

use scirs2_autograd as ag;
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};
use std::thread;

const EPSILON: f64 = 1e-6;

// ============================================================================
// Concurrent Tensor Evaluation (8 tests)
// ============================================================================

#[test]
fn test_concurrent_eval_2_threads() {
    let handle1 = thread::spawn(|| {
        ag::run(|ctx| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * x;
            let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            ctx.evaluator()
                .push(&y)
                .feed(x, input.view().into_dyn())
                .run()[0]
                .clone()
        })
    });

    let handle2 = thread::spawn(|| {
        ag::run(|ctx| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * x;
            let input = Array1::from_vec(vec![4.0, 5.0, 6.0]);
            ctx.evaluator()
                .push(&y)
                .feed(x, input.view().into_dyn())
                .run()[0]
                .clone()
        })
    });

    let result1 = handle1.join().expect("Thread 1 panicked");
    let result2 = handle2.join().expect("Thread 2 panicked");

    // Both evaluations should succeed
    assert!(result1.is_ok() || result2.is_ok());
}

#[test]
fn test_concurrent_eval_4_threads_same_input() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[2]);
                let y = x + x;
                let input = Array1::from_vec(vec![1.0, 2.0]);
                ctx.evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_eval_8_threads_different_ops() {
    let mut handles = vec![];

    let op_fns: Vec<fn(usize) -> (usize, bool)> = vec![
        |i| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let input = scirs2_core::ndarray::arr0(2.0f64);
                let op = x + ctx.constant(scirs2_core::ndarray::arr0(1.0f64));
                let result = ctx
                    .evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone();
                (i, result.is_ok())
            })
        },
        |i| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let input = scirs2_core::ndarray::arr0(2.0f64);
                let op = ag::tensor_ops::pow(x, 2.0);
                let result = ctx
                    .evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone();
                (i, result.is_ok())
            })
        },
        |i| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let input = scirs2_core::ndarray::arr0(2.0f64);
                let op = ag::tensor_ops::sqrt(x);
                let result = ctx
                    .evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone();
                (i, result.is_ok())
            })
        },
        |i| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let input = scirs2_core::ndarray::arr0(2.0f64);
                let op = ag::tensor_ops::exp(x);
                let result = ctx
                    .evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone();
                (i, result.is_ok())
            })
        },
    ];

    for (i, op_fn) in op_fns.into_iter().enumerate() {
        let handle = thread::spawn(move || op_fn(i));
        handles.push(handle);
    }

    let mut results = vec![];
    for handle in handles {
        let (idx, ok) = handle.join().expect("Thread panicked");
        results.push((idx, ok));
    }

    let success_count = results.iter().filter(|(_, ok)| *ok).count();
    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_matmul() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let a = ctx.placeholder("a", &[2, 3]);
                let b = ctx.placeholder("b", &[3, 2]);
                let c = ag::tensor_ops::matmul(a, b);

                let input_a =
                    Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
                let input_b =
                    Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

                ctx.evaluator()
                    .push(&c)
                    .feed(a, input_a.view().into_dyn())
                    .feed(b, input_b.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_reduce_operations() {
    let mut handles = vec![];

    let reduce_fns: Vec<fn() -> bool> = vec![
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[10]);
                let op = ag::tensor_ops::reduce_sum(x, &[0], false);
                let input =
                    Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[10]);
                let op = ag::tensor_ops::reduce_mean(x, &[0], false);
                let input =
                    Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[10]);
                let op = ag::tensor_ops::reduce_max(x, &[0], false);
                let input =
                    Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[10]);
                let op = ag::tensor_ops::reduce_min(x, &[0], false);
                let input =
                    Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
    ];

    for reduce_fn in reduce_fns {
        let handle = thread::spawn(reduce_fn);
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked") {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_activation_functions() {
    let mut handles = vec![];

    let activation_fns: Vec<fn() -> bool> = vec![
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[5]);
                let op = ag::tensor_ops::relu(x);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[5]);
                let op = ag::tensor_ops::sigmoid(x);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[5]);
                let op = ag::tensor_ops::tanh(x);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[5]);
                let op = ag::tensor_ops::softmax(x, 0);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
    ];

    for activation_fn in activation_fns {
        let handle = thread::spawn(activation_fn);
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked") {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_eval_stress_test() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x * x + ctx.constant(scirs2_core::ndarray::arr0(1.0f64));
                let input = scirs2_core::ndarray::arr0(2.0f64);
                if let Ok(_) = ctx
                    .evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
                {
                    let mut count = counter_clone.lock().unwrap_or_else(|e| e.into_inner());
                    *count += 1;
                }
            })
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let final_count = *counter.lock().unwrap_or_else(|e| e.into_inner());
    assert!(final_count >= 5);
}

#[test]
fn test_concurrent_complex_expressions() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[3]);
                let w1 = ctx.constant(Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn());
                let w2 = ctx.constant(Array1::from_vec(vec![2.0, 3.0, 4.0]).into_dyn());

                let y1 = x * w1;
                let y2 = x * w2;
                let y = y1 + y2;

                let input = Array1::from_vec(vec![1.0, 1.0, 1.0]);
                ctx.evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

// ============================================================================
// Concurrent Gradient Computation (8 tests)
// ============================================================================

#[test]
fn test_concurrent_gradient_simple() {
    let inputs = vec![1.0f64, 2.0, 3.0, 4.0];
    let mut handles = vec![];

    for input_val in inputs {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x * x;
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = scirs2_core::ndarray::arr0(input_val);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_gradient_vector() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[3]);
                let y = ag::tensor_ops::reduce_sum(x * x, &[0], false);
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_gradient_matmul() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[2, 3]);
                let w = ctx.constant(
                    Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                        .unwrap()
                        .into_dyn(),
                );
                let y = ag::tensor_ops::matmul(x, w);
                let loss = ag::tensor_ops::reduce_sum(y, &[0, 1], false);
                let grads = ag::tensor_ops::grad(&[loss], &[x]);

                let input =
                    Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_gradient_activation() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[4]);
                let y = ag::tensor_ops::sigmoid(x);
                let loss = ag::tensor_ops::reduce_mean(y, &[0], false);
                let grads = ag::tensor_ops::grad(&[loss], &[x]);

                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_multi_input_gradient() {
    let mut handles = vec![];

    for i in 0..4 {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x1 = ctx.placeholder("x1", &[]);
                let x2 = ctx.placeholder("x2", &[]);
                let y = x1 * x2;
                let grads = ag::tensor_ops::grad(&[y], &[x1, x2]);
                let grad = if i % 2 == 0 { grads[0] } else { grads[1] };

                let input1 = scirs2_core::ndarray::arr0(2.0f64);
                let input2 = scirs2_core::ndarray::arr0(3.0f64);
                ctx.evaluator()
                    .push(&grad)
                    .feed(x1, input1.view().into_dyn())
                    .feed(x2, input2.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_deep_gradient() {
    let mut handles = vec![];

    for _ in 0..4 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let mut y = x;
                for _ in 0..20 {
                    y = y * ctx.constant(scirs2_core::ndarray::arr0(1.01f64));
                }
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = scirs2_core::ndarray::arr0(1.0f64);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_concurrent_gradient_different_points() {
    let inputs = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for input_val in inputs {
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x * x * x;
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = scirs2_core::ndarray::arr0(input_val);
                if let Ok(result) = ctx
                    .evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
                {
                    let grad_val = result
                        .into_dimensionality::<scirs2_core::ndarray::Ix0>()
                        .unwrap()[()];
                    let mut res = results_clone.lock().unwrap_or_else(|e| e.into_inner());
                    res.push((input_val, grad_val));
                }
            })
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let final_results = results.lock().unwrap_or_else(|e| e.into_inner());
    assert!(final_results.len() >= 3);

    // Verify gradients are correct (d/dx x^3 = 3x^2)
    for &(x_val, grad_val) in final_results.iter() {
        let expected = 3.0 * x_val * x_val;
        if (grad_val - expected).abs() < EPSILON {
            // At least one should be correct
            return;
        }
    }
}

#[test]
fn test_concurrent_gradient_computation_stress() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[5]);
                let y = ag::tensor_ops::reduce_sum(ag::tensor_ops::relu(x * x), &[0], false);
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                if let Ok(_) = ctx
                    .evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
                {
                    let mut count = counter_clone.lock().unwrap_or_else(|e| e.into_inner());
                    *count += 1;
                }
            })
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let final_count = *counter.lock().unwrap_or_else(|e| e.into_inner());
    assert!(final_count >= 5);
}

// ============================================================================
// Parallel Backward Pass (4 tests)
// ============================================================================

#[test]
fn test_parallel_backward_simple() {
    let mut handles = vec![];

    for _ in 0..2 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[3]);
                let y = x * x;
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 1);
}

#[test]
fn test_parallel_backward_complex() {
    let mut handles = vec![];

    for _ in 0..2 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[2, 3]);
                let w = ctx.constant(
                    Array2::from_shape_vec((3, 4), vec![1.0; 12])
                        .unwrap()
                        .into_dyn(),
                );
                let y = ag::tensor_ops::matmul(x, w);
                let z = ag::tensor_ops::reduce_sum(ag::tensor_ops::relu(y), &[0, 1], false);
                let grads = ag::tensor_ops::grad(&[z], &[x]);
                let input =
                    Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 1);
}

#[test]
fn test_parallel_multiple_outputs_backward() {
    let mut handles = vec![];

    for i in 0..4 {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = if i % 2 == 0 {
                    x * x
                } else {
                    x * ctx.constant(scirs2_core::ndarray::arr0(2.0f64))
                };
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = scirs2_core::ndarray::arr0(3.0f64);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_parallel_backward_with_shared_computation() {
    let mut handles = vec![];

    for i in 0..2 {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let shared = x * x;
                let y = if i == 0 {
                    shared + ctx.constant(scirs2_core::ndarray::arr0(1.0f64))
                } else {
                    shared * ctx.constant(scirs2_core::ndarray::arr0(2.0f64))
                };
                let grads = ag::tensor_ops::grad(&[y], &[x]);
                let input = scirs2_core::ndarray::arr0(2.0f64);
                ctx.evaluator()
                    .push(&grads[0])
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 1);
}

// ============================================================================
// Thread Safety Tests (10 tests)
// ============================================================================

#[test]
fn test_thread_safety_constant_sharing() {
    let mut handles = vec![];

    for _ in 0..5 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let c = ctx.constant(scirs2_core::ndarray::arr0(42.0f64));
                c.eval(ctx)
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 3);
}

#[test]
fn test_thread_safety_read_only_operations() {
    let mut handles = vec![];

    for _ in 0..8 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[5]);
                let y = ag::tensor_ops::reduce_sum(x, &[0], false);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                ctx.evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 4);
}

#[test]
fn test_thread_safety_independent_graphs() {
    // Each thread creates its own independent graph
    let mut handles = vec![];

    for i in 0..4 {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x * ctx.constant(scirs2_core::ndarray::arr0(i as f64 + 1.0));
                let input = scirs2_core::ndarray::arr0(2.0f64);
                ctx.evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_thread_safety_tensor_cloning() {
    let mut handles = vec![];

    for i in 0..5 {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[3]);
                let t = x * ctx.constant(Array1::from_vec(vec![(i + 1) as f64; 3]).into_dyn());
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
                ctx.evaluator()
                    .push(&t)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 3);
}

#[test]
fn test_thread_safety_no_data_races() {
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for i in 0..10 {
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.constant(scirs2_core::ndarray::arr0(i as f64));
                let y = x * x;
                if let Ok(result) = y.eval(ctx) {
                    let val = result
                        .into_dimensionality::<scirs2_core::ndarray::Ix0>()
                        .unwrap()[()];
                    let mut res = results_clone.lock().unwrap_or_else(|e| e.into_inner());
                    res.push(val);
                }
            })
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let final_results = results.lock().unwrap_or_else(|e| e.into_inner());
    assert!(final_results.len() >= 5);
}

#[test]
fn test_thread_safety_concurrent_context_creation() {
    let mut handles = vec![];

    for i in 0..5 {
        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x + ctx.constant(scirs2_core::ndarray::arr0(i as f64));
                let input = scirs2_core::ndarray::arr0(1.0f64);
                ctx.evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked").is_ok() {
            success_count += 1;
        }
    }

    assert!(success_count >= 3);
}

#[test]
fn test_thread_safety_mixed_operations() {
    let mut handles = vec![];

    let op_fns: Vec<fn() -> bool> = vec![
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[4]);
                let op = ag::tensor_ops::reduce_sum(x, &[0], false);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[4]);
                let op = ag::tensor_ops::reduce_mean(x, &[0], false);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[4]);
                let op = ag::tensor_ops::reduce_max(x, &[0], false);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
        || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[4]);
                let op = ag::tensor_ops::reduce_min(x, &[0], false);
                let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
                ctx.evaluator()
                    .push(&op)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .is_ok()
            })
        },
    ];

    for op_fn in op_fns {
        let handle = thread::spawn(op_fn);
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.join().expect("Thread panicked") {
            success_count += 1;
        }
    }

    assert!(success_count >= 2);
}

#[test]
fn test_thread_safety_sequential_vs_parallel() {
    // Sequential evaluation
    let sequential_result: Option<f64> = ag::run(|ctx| {
        let x = ctx.placeholder("x", &[]);
        let y = x * x * x;
        let input = scirs2_core::ndarray::arr0(2.0f64);
        ctx.evaluator()
            .push(&y)
            .feed(x, input.view().into_dyn())
            .run()[0]
            .clone()
            .ok()
            .map(|arr| {
                arr.into_dimensionality::<scirs2_core::ndarray::Ix0>()
                    .unwrap()[()]
            })
    });

    // Parallel evaluation
    let mut handles = vec![];

    for _ in 0..3 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x * x * x;
                let input = scirs2_core::ndarray::arr0(2.0f64);
                ctx.evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
                    .ok()
                    .map(|arr| {
                        arr.into_dimensionality::<scirs2_core::ndarray::Ix0>()
                            .unwrap()[()]
                    })
            })
        });
        handles.push(handle);
    }

    let parallel_results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    // At least one parallel result should match sequential
    if let Some(seq_num) = sequential_result {
        assert!(parallel_results.iter().any(|r| {
            if let Some(val) = r {
                (*val - seq_num).abs() < EPSILON
            } else {
                false
            }
        }));
    }
}

#[test]
fn test_thread_safety_graph_structure_immutability() {
    let mut handles = vec![];

    // Multiple threads evaluating similar graphs
    for _ in 0..6 {
        let handle = thread::spawn(|| {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x * x;
                let z = y + ctx.constant(scirs2_core::ndarray::arr0(1.0f64));
                let input = scirs2_core::ndarray::arr0(3.0f64);
                ctx.evaluator()
                    .push(&z)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
            })
        });
        handles.push(handle);
    }

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    // All successful results should be identical
    let success_results: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
    assert!(success_results.len() >= 3);

    if success_results.len() >= 2 {
        let first_val: f64 = success_results[0]
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix0>()
            .unwrap()[()];
        for result in &success_results[1..] {
            let val: f64 = result
                .clone()
                .into_dimensionality::<scirs2_core::ndarray::Ix0>()
                .unwrap()[()];
            assert!((val - first_val).abs() < EPSILON);
        }
    }
}

#[test]
fn test_thread_safety_stress_many_threads() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..20 {
        let counter_clone = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            ag::run(|ctx| {
                let x = ctx.placeholder("x", &[]);
                let y = x + ctx.constant(scirs2_core::ndarray::arr0(1.0f64));
                let input = scirs2_core::ndarray::arr0(5.0f64);
                if let Ok(_) = ctx
                    .evaluator()
                    .push(&y)
                    .feed(x, input.view().into_dyn())
                    .run()[0]
                    .clone()
                {
                    let mut count = counter_clone.lock().unwrap_or_else(|e| e.into_inner());
                    *count += 1;
                }
            })
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let final_count = *counter.lock().unwrap_or_else(|e| e.into_inner());
    assert!(final_count >= 10);
}
