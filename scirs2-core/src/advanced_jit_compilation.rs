//! Advanced JIT Compilation Framework
//!
//! This module provides a comprehensive Just-In-Time (JIT) compilation framework
//! with LLVM integration for runtime optimization in Advanced mode. It enables
//! dynamic code generation, runtime optimization, and adaptive compilation strategies
//! to maximize performance for scientific computing workloads.
//!
//! # Features
//!
//! - **LLVM-based Code Generation**: Advanced optimization through LLVM infrastructure
//! - **Runtime Kernel Compilation**: JIT compilation of computational kernels
//! - **Adaptive Optimization**: Dynamic optimization based on runtime characteristics
//! - **Cross-platform Support**: Native code generation for multiple architectures
//! - **Intelligent Caching**: Smart caching of compiled code with automatic invalidation
//! - **Performance Profiling**: Integrated profiling for continuous optimization
//! - **Template-based Specialization**: Automatic code specialization for specific data types
//! - **Vectorization**: Automatic SIMD optimization for mathematical operations

// This module has been refactored into a modular structure.
// All functionality is now available through the advanced_jit_compilation_impl submodule.

#[path = "advanced_jit_compilation_impl/mod.rs"]
mod advanced_jit_compilation_impl;
pub use advanced_jit_compilation_impl::*;
