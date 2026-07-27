#![allow(clippy::non_canonical_partial_ord_impl)]
#![allow(clippy::module_inception)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::let_and_return)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_vec)]
//! # SciRS2 Autograd - Automatic Differentiation for Rust
#![recursion_limit = "1024"]
//!
//! **scirs2-autograd** provides PyTorch-style automatic differentiation with lazy tensor evaluation,
//! enabling efficient gradient computation for scientific computing and deep learning.
//!
//! ## 🎯 Key Features
//!
//! - **Reverse-mode Autodiff**: Efficient backpropagation for neural networks
//! - **Lazy Evaluation**: Build computation graphs, evaluate only when needed
//! - **Higher-order Gradients**: Compute derivatives of derivatives
//! - **Neural Network Ops**: Optimized operations for deep learning
//! - **Optimizers**: Adam, SGD, RMSprop with state management
//! - **Model Persistence**: Save and load trained models
//! - **Variable Management**: Namespace-based variable organization
//!
//! ## 📦 Installation
//!
//! ```toml
//! [dependencies]
//! scirs2-autograd = { version = "0.6.3", features = ["blas"] }
//! ```
//!
//! ### BLAS Acceleration (Recommended)
//!
//! For fast matrix operations, enable BLAS (uses OxiBLAS - pure Rust):
//!
//! ```toml
//! [dependencies]
//! scirs2-autograd = { version = "0.6.3", features = ["blas"] }
//! ```
//!
//! ## 🚀 Quick Start
//!
//! ### Basic Differentiation
//!
//! Compute gradients of a simple function:
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     // Define variables
//!     let x = ctx.placeholder("x", &[]);
//!     let y = ctx.placeholder("y", &[]);
//!
//!     // Build computation graph: z = 2x² + 3y + 1
//!     let z = 2.0 * x * x + 3.0 * y + 1.0;
//!
//!     // Compute dz/dy
//!     let dz_dy = &T::grad(&[z], &[y])[0];
//!     println!("dz/dy = {:?}", dz_dy.eval(ctx));  // => 3.0
//!
//!     // Compute dz/dx (feed x=2)
//!     let dz_dx = &T::grad(&[z], &[x])[0];
//!     let x_val = scirs2_core::ndarray::arr0(2.0);
//!     let result = ctx.evaluator()
//!         .push(dz_dx)
//!         .feed(x, x_val.view().into_dyn())
//!         .run()[0].clone();
//!     println!("dz/dx at x=2: {:?}", result);  // => 8.0
//!
//!     // Higher-order: d²z/dx²
//!     let d2z_dx2 = &T::grad(&[dz_dx], &[x])[0];
//!     println!("d²z/dx² = {:?}", d2z_dx2.eval(ctx));  // => 4.0
//! });
//! ```
//!
//! ### Neural Network Training
//!
//! Train a multi-layer perceptron for MNIST:
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::optimizers::adam::Adam;
//! use ag::tensor_ops::*;
//! use ag::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create variable environment
//! let mut env = ag::VariableEnvironment::new();
//! let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
//!
//! // Initialize network weights
//! env.name("w1").set(rng.glorot_uniform(&[784, 256]));
//! env.name("b1").set(ag::ndarray_ext::zeros(&[1, 256]));
//! env.name("w2").set(rng.glorot_uniform(&[256, 10]));
//! env.name("b2").set(ag::ndarray_ext::zeros(&[1, 10]));
//!
//! // Create Adam optimizer
//! let var_ids = env.default_namespace().current_var_ids();
//! let adam = Adam::default("adam", var_ids, &mut env);
//!
//! // Training loop
//! for epoch in 0..10 {
//!     env.run(|ctx| {
//!         // Define computation graph
//!         let x = ctx.placeholder("x", &[-1, 784]);
//!         let y = ctx.placeholder("y", &[-1]);
//!
//!         let w1 = ctx.variable("w1");
//!         let b1 = ctx.variable("b1");
//!         let w2 = ctx.variable("w2");
//!         let b2 = ctx.variable("b2");
//!
//!         // Forward pass: x -> hidden -> output
//!         let hidden = relu(matmul(x, w1) + b1);
//!         let logits = matmul(hidden, w2) + b2;
//!
//!         // Loss: cross-entropy
//!         let loss = reduce_mean(
//!             sparse_softmax_cross_entropy(logits, &y),
//!             &[0],
//!             false
//!         );
//!
//!         // Backpropagation
//!         let params = &[w1, b1, w2, b2];
//!         let grads = &grad(&[loss], params);
//!
//!         // Update weights (requires actual data feeding)
//!         // let mut feeder = ag::Feeder::new();
//!         // feeder.push(x, x_batch).push(y, y_batch);
//!         // adam.update(params, grads, ctx, feeder);
//!     });
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Operations
//!
//! Define custom differentiable operations:
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops::*;
//!
//! ag::run::<f64, _, _>(|ctx| {
//!     let x = ones(&[3, 4], ctx);
//!
//!     // Apply custom transformations using tensor.map()
//!     let y = x.map(|arr| arr.mapv(|v: f64| v * 2.0 + 1.0));
//!
//!     // Hooks for debugging
//!     let z = x.showshape();  // Print shape
//!     let w = x.raw_hook(|arr| println!("Tensor value: {}", arr));
//! });
//! ```
//!
//! ## 🧠 Core Concepts
//!
//! ### Tensors
//!
//! Lazy-evaluated multi-dimensional arrays with automatic gradient tracking:
//!
//! ```rust,no_run
//! use scirs2_autograd as ag;
//! use ag::tensor_ops::*;
//! use ag::prelude::*;
//!
//! ag::run::<f64, _, _>(|ctx| {
//!     // Create tensors
//!     let a = zeros(&[2, 3], ctx);        // All zeros
//!     let b = ones(&[2, 3], ctx);         // All ones
//!     let c = ctx.placeholder("c", &[2, 3]);  // Placeholder (fill later)
//!     let d = ctx.variable("d");          // Trainable variable
//! });
//! ```
//!
//! ### Computation Graphs
//!
//! Build graphs of operations, evaluate lazily:
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run::<f64, _, _>(|ctx| {
//!     let x = ctx.placeholder("x", &[2, 2]);
//!     let y = ctx.placeholder("y", &[2, 2]);
//!
//!     // Build graph (no computation yet)
//!     let z = T::matmul(x, y);
//!     let w = T::sigmoid(z);
//!
//!     // Evaluate when needed
//!     // let result = w.eval(ctx);
//! });
//! ```
//!
//! ### Gradient Computation
//!
//! Reverse-mode automatic differentiation:
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run(|ctx| {
//!     let x = ctx.placeholder("x", &[]);
//!     let y = x * x * x;  // y = x³
//!
//!     // Compute dy/dx = 3x²
//!     let dy_dx = &T::grad(&[y], &[x])[0];
//!
//!     // Evaluate at x=2: 3(2²) = 12
//!     let x_val = scirs2_core::ndarray::arr0(2.0);
//!     let grad_val = ctx.evaluator()
//!         .push(dy_dx)
//!         .feed(x, x_val.view().into_dyn())
//!         .run()[0].clone();
//! });
//! ```
//!
//! ## 🎨 Available Operations
//!
//! ### Basic Math
//!
//! - Arithmetic: `+`, `-`, `*`, `/`, `pow`
//! - Comparison: `equal`, `not_equal`, `greater`, `less`
//! - Reduction: `sum`, `mean`, `max`, `min`
//!
//! ### Neural Network Ops
//!
//! - Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`
//! - Pooling: `max_pool2d`, `avg_pool2d`
//! - Convolution: `conv2d`, `conv2d_transpose`
//! - Normalization: `batch_norm`, `layer_norm`
//! - Dropout: `dropout`
//!
//! ### Matrix Operations
//!
//! - `matmul` - Matrix multiplication
//! - `transpose` - Matrix transpose
//! - `reshape` - Change tensor shape
//! - `concat` - Concatenate tensors
//! - `split` - Split tensor
//!
//! ### Loss Functions
//!
//! - `sparse_softmax_cross_entropy` - Classification loss
//! - `sigmoid_cross_entropy` - Binary classification
//! - `softmax_cross_entropy` - Multi-class loss
//!
//! ## 🔧 Optimizers
//!
//! Built-in optimization algorithms:
//!
//! - **Adam**: Adaptive moment estimation (recommended)
//! - **SGD**: Stochastic gradient descent with momentum
//! - **RMSprop**: Root mean square propagation
//! - **Adagrad**: Adaptive learning rates
//!
//! ## 💾 Model Persistence
//!
//! Save and load trained models:
//!
//! ```rust,no_run
//! use scirs2_autograd as ag;
//!
//! let mut env = ag::VariableEnvironment::<f64>::new();
//!
//! // After training...
//! env.save("model.safetensors")?;
//!
//! // Later, load the model
//! let env = ag::VariableEnvironment::<f64>::load("model.safetensors")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## 📊 Performance
//!
//! scirs2-autograd is designed for efficiency:
//!
//! - **Lazy Evaluation**: Build graphs without computation overhead
//! - **Minimal Allocations**: Reuse memory where possible
//! - **BLAS Integration**: Fast matrix operations via OxiBLAS (pure Rust)
//! - **Zero-copy**: Efficient data handling with ndarray views
//!
//! Typical training speed: **0.11 sec/epoch** for MNIST MLP (2.7GHz Intel Core i5)
//!
//! ## 🔗 Integration
//!
//! - **scirs2-neural**: High-level neural network layers
//! - **scirs2-linalg**: Matrix operations
//! - **scirs2-optimize**: Optimization algorithms
//! - **ndarray**: Core array library (re-exported)
//!
//! ## 📚 Comparison with PyTorch
//!
//! | Feature | PyTorch | scirs2-autograd |
//! |---------|---------|-----------------|
//! | Autodiff | ✅ | ✅ |
//! | Dynamic Graphs | ✅ | ✅ |
//! | GPU Support | ✅ | ✅ (v0.5.0) |
//! | Type Safety | ❌ | ✅ |
//! | Memory Safety | ⚠️ | ✅ |
//! | Pure Rust | ❌ | ✅ |
//!
//! ## 🚀 v0.5.0 Features
//!
//! - **GPU Acceleration**: CUDA, Metal, OpenCL, WebGPU backends
//! - **Higher-Order Derivatives**: Hessian-vector products, full Jacobians
//! - **Memory Optimization**: Advanced checkpointing, memory pooling
//! - **Graph Optimization**: CSE, operation fusion
//! - **Distributed Training**: Data/model parallelism
//! - **Symbolic Differentiation**: Analytical derivatives
//!
//! ## 🔒 Version
//!
//! Current version: **0.6.3**

#[allow(unused_imports)]
// Re-export from scirs2-core for POLICY compliance
pub use scirs2_core::ndarray;
pub use scirs2_core::random as rand;

// BLAS dependencies now handled through scirs2-core

extern crate approx;
extern crate libc;
extern crate matrixmultiply;
// extern crate rayon;  // Now use scirs2-core parallel abstractions
extern crate rustc_hash;
extern crate serde;
extern crate serde_json;
extern crate special;
extern crate uuid;

pub mod error;
pub mod error_helpers;
pub mod evaluation;
mod gradient;
pub mod gradient_clipping;
pub mod graph;
pub mod high_performance;
pub mod hooks;
pub mod integration;
pub mod memory_pool;
pub mod ndarray_ext;
pub mod op;
pub mod optimization;
pub mod optimizers;
pub mod parallel;
pub mod prelude;
pub mod schedulers;
pub mod tensor;
pub mod tensor_ops;
pub mod test_helper;
pub mod testing;
pub mod tracing;
pub mod validation;
pub mod variable;
pub mod visualization;

// v0.2.0 modules
pub mod distributed;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod higher_order;
pub mod sparse;
pub mod symbolic;
pub mod tape;

// Symbolic backend: EML expression → autograd Tensor bridge
#[cfg(feature = "symbolic")]
pub mod symbolic_backend;
#[cfg(feature = "symbolic")]
pub use symbolic_backend::eml_scalar_op;

// EML tape extensions: element-wise, Jacobian, Hessian ops + dispatch
#[cfg(feature = "symbolic")]
pub use tape::dispatch::{extract_lowered_op, is_eml_backed, try_build_symbolic_jacobian};
#[cfg(feature = "symbolic")]
pub use tape::eml_tape::{eml_elementwise, eml_hessian, eml_jacobian};

// v0.3.0 modules
pub mod forward_mode;
pub mod onnx;
pub mod transforms;

// v0.3.0 additional modules
pub mod custom_gradient;
pub mod gradient_accumulation;
pub mod jacobian_ops;

// v0.3.0 compiler-style optimisations and scheduling
pub mod autodiff_enhanced;
pub mod graph_transforms;
pub mod jit_fusion;
pub mod profiling;
pub mod scheduling;

use rustc_hash::{FxHashMap, FxHashSet};
use std::any::TypeId;
use std::fmt;

/// A primitive type in this crate, which is actually a decorated `scirs2_core::numeric::Float`.
pub trait Float:
    scirs2_core::numeric::Float
    + scirs2_core::numeric::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + serde::Serialize
    + serde::de::DeserializeOwned
    + 'static
{
}

#[doc(hidden)]
/// Internal trait.
pub trait Int:
    num_integer::Integer
    + scirs2_core::numeric::NumAssignOps
    + scirs2_core::numeric::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + Sized
    + serde::Serialize
    + serde::de::DeserializeOwned
    + 'static
{
}

impl<T> Float for T where
    T: num_traits::Float
        + scirs2_core::numeric::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static
{
}

impl<T> Int for T where
    T: num_integer::Integer
        + scirs2_core::numeric::NumAssignOps
        + scirs2_core::numeric::ToPrimitive
        + Copy
        + Send
        + Sync
        + fmt::Display
        + Sized
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static
{
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub(crate) fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use crate::ndarray_ext::array_gen;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::evaluation::{Evaluator, Feeder};

pub use crate::tensor::Tensor;

pub(crate) use graph::Graph;

pub use crate::error::{AutogradError, EvalError, OpError, Result};
pub use crate::graph::{run, Context};
pub use crate::high_performance::{
    memory_efficient_grad_accumulation, parallel_gradient_computation, simd_backward_pass,
    ultra_backward_pass,
};
pub use crate::variable::{
    AutogradTensor, SafeVariable, SafeVariableEnvironment, VariableEnvironment,
};

// Re-export functional optimizers and training utilities (Issue #94)
pub use crate::optimizers::{FunctionalAdam, FunctionalOptimizer, FunctionalSGD};

// Forward-mode AD public API (v0.3.0)
pub use crate::forward_mode::{
    gradient_forward, hessian, hessian_vector_product, jacobian_forward, jvp, DualNumber,
};

// JAX-inspired function transformations (v0.3.0)
pub use crate::transforms::{
    batched_value_and_grad, check_grad, compose, iterate, numerical_jacobian, pmap, scan,
    stop_gradient, stop_gradient_1d, stop_gradient_dual, vmap, Checkpoint, JitHint,
};
// Custom gradient rules (v0.3.0)
pub use crate::custom_gradient::{
    custom_op, custom_unary_op, detach, gradient_reversal, scale_gradient, selective_stop_gradient,
    CustomGradientOp, ScaleGradient, SelectiveStopGradient,
};

// Gradient accumulation (v0.3.0)
pub use crate::gradient_accumulation::{
    GradientAccumulator, GradientStats, VirtualBatchAccumulator,
};

// Higher-order derivative extensions (v0.3.0)
pub use crate::higher_order::extensions::{
    efficient_second_order_grad, fisher_diagonal, fisher_information, fisher_information_forward,
    hessian_diagonal, hessian_diagonal_forward, laplacian, laplacian_forward,
};

// Jacobian operations (v0.3.0)
pub use crate::jacobian_ops::{
    batch_jacobian, jacobian_auto, jacobian_check, jacobian_diagonal, jacobian_reverse,
    jvp_forward, jvp_graph, numerical_jacobian as numerical_jacobian_fd, vjp_multi, vjp_reverse,
};

// Graph visualization (v0.3.0)
pub use crate::visualization::{
    graph_summary, graph_to_dot, graph_to_json, graph_to_mermaid, GraphStats,
};

// Scheduling (v0.3.0)
pub use crate::scheduling::{
    build_memory_plan, critical_path, forward_schedule, level_decomposition,
    memory_optimal_schedule, parallel_analysis, reverse_schedule, validate_schedule,
    work_stealing_schedule, CriticalPath, MemoryPlan, ParallelAnalysis, Schedule,
    ScheduleDirection, WorkStealingSchedule,
};

// Graph transforms (v0.3.0)
pub use crate::graph_transforms::{
    analyse_graph, detect_algebraic_simplifications, detect_cse, detect_fusions, find_dead_nodes,
    find_foldable_constants, infer_shapes, AlgebraicSimplification, FusionGroup, FusionKind,
    SimplificationRule, TransformReport,
};

// Autodiff enhancements (v0.3.0)
pub use crate::autodiff_enhanced::{
    binomial_checkpoint_plan, build_rematerialization_plan, plan_jacobian_computation,
    select_jacobian_mode, solve_implicit_diff, sqrt_checkpoint_plan, uniform_checkpoint_plan,
    CheckpointPlan, CheckpointStrategy, DiffRuleRegistry, ImplicitDiffConfig, ImplicitDiffResult,
    JacobianMode, MixedModeJacobianPlan, RematerializationDecision, RematerializationPolicy,
};

// Profiling (v0.3.0)
pub use crate::profiling::{
    analyse_gradient_flow, classify_gradient, count_ops, estimate_bandwidth, estimate_flops,
    graph_complexity, has_gradient_issues, profile_graph, total_flops, BandwidthEstimate,
    EstimateConfidence, FlopEstimate, GradientFlowStats, GradientHealth, GradientThresholds,
    GraphComplexity, OpCounts, OpTiming, OperationProfiler, ProfilingReport,
};

// Re-export transforms::grad and transforms::value_and_grad under qualified path
// to avoid ambiguity with tensor_ops::grad
pub mod jax {
    //! JAX-style functional transformations re-exported for convenience.
    //!
    //! Use `scirs2_autograd::jax::grad` to get the JAX-style `grad` transform,
    //! distinct from `tensor_ops::grad` which operates on the computation graph.
    pub use crate::transforms::{
        batched_value_and_grad, check_grad, compose, grad, grad_grad, iterate, jacobian,
        numerical_jacobian, pmap, scan, stop_gradient, stop_gradient_1d, stop_gradient_dual,
        value_and_grad, vmap, Checkpoint, JitHint,
    };
}
