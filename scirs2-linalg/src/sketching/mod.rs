//! Sketch matrix types and random embedding transforms for large-scale linear algebra.
//!
//! This module provides randomised linear maps (sketches) used to compress tall or
//! wide matrices while approximately preserving geometric structure.

pub mod sketch_matrix;

pub use sketch_matrix::{
    jl_embed_points, CountSketchMatrix, GaussianSketch, JLTransform, SRHTTransform,
};
