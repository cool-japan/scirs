//! Complete ndarray re-export for SciRS2 ecosystem
//!
//! This module provides a single, unified access point for ALL ndarray functionality,
//! ensuring SciRS2 POLICY compliance across the entire ecosystem.
//!
//! ## Design Philosophy
//!
//! 1. **Complete Feature Parity**: All ndarray functionality available through scirs2-core
//! 2. **Zero Breaking Changes**: Existing ndarray_ext continues to work
//! 3. **Policy Compliance**: No need for direct ndarray imports anywhere
//! 4. **Single Source of Truth**: One place for all array operations
//!
//! ## Usage
//!
//! ```rust
//! // Instead of:
//! use ndarray::{Array, array, s, Axis};  // ❌ POLICY violation
//!
//! // Use:
//! use scirs2_core::ndarray::*;  // ✅ POLICY compliant
//!
//! let arr = array![[1, 2], [3, 4]];
//! let slice = arr.slice(s![.., 0]);
//! ```

// ========================================
// COMPLETE NDARRAY RE-EXPORT
// ========================================

/// Re-export everything from ndarray crate
pub use ndarray::*;

// Ensure specific items are definitely available (for clarity)
pub use ndarray::{
    ArcArray,
    ArcArray1,
    ArcArray2,
    // Core array types
    Array,
    Array0,
    Array1,
    Array2,
    Array3,
    Array4,
    Array5,
    Array6,
    ArrayBase,
    // Builder patterns
    ArrayBase as NdArray,

    ArrayD,
    // View types
    ArrayView,
    ArrayView0,
    ArrayView1,
    ArrayView2,
    ArrayView3,
    ArrayView4,
    ArrayView5,
    ArrayView6,
    ArrayViewD,
    ArrayViewMut,
    ArrayViewMut0,
    ArrayViewMut1,
    ArrayViewMut2,
    ArrayViewMut3,
    ArrayViewMut4,
    ArrayViewMut5,
    ArrayViewMut6,
    ArrayViewMutD,

    AsArray,

    // Axis and indexing
    Axis,
    AxisDescription,
    CowArray,
    // Essential traits
    Data,
    DataMut,
    DataOwned,
    DataShared,
    // Dimension types
    Dim,
    Dimension,
    // Utility types
    ErrorKind,
    FoldWhile,

    Ix0,
    Ix1,
    Ix2,
    Ix3,
    Ix4,
    Ix5,
    Ix6,
    IxDyn,
    IxDynImpl,
    // Mathematical operations
    LinalgScalar,
    NdFloat,

    NdProducer,
    NewAxis,

    Order,
    OwnedArcRepr,
    // Storage
    OwnedRepr,
    RawArrayView,
    RawArrayViewMut,

    RawData,
    RawDataClone,
    RawDataMut,
    RawViewRepr,

    RemoveAxis,

    ScalarOperand,
    // Shape and strides
    Shape,
    ShapeBuilder,
    ShapeError,

    Slice,
    SliceArg,
    SliceInfo,
    SliceInfoElem,
    StrideShape,
    ViewRepr,
    // Iteration
    Zip,
};

// ========================================
// ESSENTIAL MACROS
// ========================================

/// Re-export essential macros that were missing in some modules
pub use ndarray::{
    arr0, // Creates 0-dimensional arrays
    arr1, // Creates 1-dimensional arrays
    arr2, // Creates 2-dimensional arrays
    arr3, // Creates 3-dimensional arrays
    // Array creation macros
    array, // Creates arrays with literal syntax
    azip,  // Zip iterator macro
    s,     // Slicing macro - CRITICAL for ToRSh
    stack, // Stack arrays macro
};

// ========================================
// NDARRAY-RELATED CRATE RE-EXPORTS
// ========================================

#[cfg(feature = "random")]
pub use ndarray_rand::{rand_distr as distributions, RandomExt, SamplingStrategy};

#[cfg(feature = "linalg")]
pub use ndarray_linalg;

#[cfg(feature = "array_stats")]
pub use ndarray_stats::{
    errors as stats_errors, interpolate, CorrelationExt, DeviationExt, MaybeNan, QuantileExt,
    Sort1dExt, SummaryStatisticsExt,
};

#[cfg(feature = "array_io")]
pub use ndarray_npy::{
    NpzReader, NpzWriter, ReadNpyExt, ReadNpzError, ViewMutNpyExt, ViewNpyExt, WriteNpyError,
    WriteNpyExt,
};

// ========================================
// ENHANCED FUNCTIONALITY
// ========================================

/// Additional utilities for SciRS2 ecosystem
pub mod utils {
    use super::*;

    /// Create an identity matrix
    pub fn eye<A>(n: usize) -> Array2<A>
    where
        A: Clone + num_traits::Zero + num_traits::One,
    {
        let mut arr = Array2::zeros((n, n));
        for i in 0..n {
            arr[[i, i]] = A::one();
        }
        arr
    }

    /// Create a diagonal matrix from a vector
    pub fn diag<A>(v: &Array1<A>) -> Array2<A>
    where
        A: Clone + num_traits::Zero,
    {
        let n = v.len();
        let mut arr = Array2::zeros((n, n));
        for i in 0..n {
            arr[[i, i]] = v[i].clone();
        }
        arr
    }

    /// Check if arrays are approximately equal
    pub fn allclose<A, D>(
        a: &ArrayBase<impl Data<Elem = A>, D>,
        b: &ArrayBase<impl Data<Elem = A>, D>,
        rtol: A,
        atol: A,
    ) -> bool
    where
        A: PartialOrd
            + std::ops::Sub<Output = A>
            + std::ops::Mul<Output = A>
            + std::ops::Add<Output = A>
            + Clone,
        D: Dimension,
    {
        if a.shape() != b.shape() {
            return false;
        }

        a.iter().zip(b.iter()).all(|(a_val, b_val)| {
            let diff = if a_val > b_val {
                a_val.clone() - b_val.clone()
            } else {
                b_val.clone() - a_val.clone()
            };

            let threshold = atol.clone()
                + rtol.clone()
                    * (if a_val > b_val {
                        a_val.clone()
                    } else {
                        b_val.clone()
                    });

            diff <= threshold
        })
    }

    /// Concatenate arrays along an axis
    pub fn concatenate<A, D>(
        axis: Axis,
        arrays: &[ArrayView<A, D>],
    ) -> Result<Array<A, D>, ShapeError>
    where
        A: Clone,
        D: Dimension + RemoveAxis,
    {
        ndarray::concatenate(axis, arrays)
    }

    /// Stack arrays along a new axis
    pub fn stack<A, D>(
        axis: Axis,
        arrays: &[ArrayView<A, D>],
    ) -> Result<Array<A, D::Larger>, ShapeError>
    where
        A: Clone,
        D: Dimension,
        D::Larger: RemoveAxis,
    {
        ndarray::stack(axis, arrays)
    }
}

// ========================================
// COMPATIBILITY LAYER
// ========================================

/// Compatibility module for smooth migration from fragmented imports
pub mod compat {
    pub use super::*;

    /// Alias for commonly used types to match existing usage patterns
    pub type DynArray<T> = ArrayD<T>;
    pub type Matrix<T> = Array2<T>;
    pub type Vector<T> = Array1<T>;
    pub type Tensor3<T> = Array3<T>;
    pub type Tensor4<T> = Array4<T>;

    /// Re-export from ndarray_ext for backward compatibility
    pub use crate::ndarray_ext::{
        broadcast_1d_to_2d,
        broadcast_apply,
        fancy_index_2d,
        // Keep existing extended functionality
        indexing,
        is_broadcast_compatible,
        manipulation,
        mask_select,
        matrix,
        reshape_2d,
        split_2d,
        stack_2d,
        stats,
        take_2d,
        transpose_2d,
        where_condition,
    };
}

// ========================================
// PRELUDE MODULE
// ========================================

/// Prelude module with most commonly used items
pub mod prelude {
    pub use super::{
        arr1,
        arr2,
        // Essential macros
        array,
        azip,
        // Utilities
        concatenate,
        s,
        stack,

        stack as stack_fn,
        // Essential types
        Array,
        Array0,
        Array1,
        Array2,
        Array3,
        ArrayD,
        ArrayView,
        ArrayView1,
        ArrayView2,
        ArrayViewMut,

        // Common operations
        Axis,
        // Essential traits
        Dimension,
        Ix1,
        Ix2,
        Ix3,
        IxDyn,
        ScalarOperand,
        ShapeBuilder,

        Zip,
    };

    #[cfg(feature = "random")]
    pub use super::RandomExt;

    // Useful type aliases
    pub type Matrix<T> = super::Array2<T>;
    pub type Vector<T> = super::Array1<T>;
}

// ========================================
// EXAMPLES MODULE
// ========================================

#[cfg(test)]
pub mod examples {
    //! Examples demonstrating unified ndarray access through scirs2-core

    use super::*;

    /// Example: Using all essential ndarray features through scirs2-core
    ///
    /// ```
    /// use scirs2_core::ndarray::*;
    ///
    /// // Create arrays using the array! macro
    /// let a = array![[1, 2, 3], [4, 5, 6]];
    ///
    /// // Use the s! macro for slicing
    /// let row = a.slice(s![0, ..]);
    /// let col = a.slice(s![.., 1]);
    ///
    /// // Use Axis for operations
    /// let sum_axis0 = a.sum_axis(Axis(0));
    /// let mean_axis1 = a.mean_axis(Axis(1));
    ///
    /// // Stack and concatenate
    /// let b = array![[7, 8, 9], [10, 11, 12]];
    /// let stacked = stack![Axis(0), a, b];
    ///
    /// // Views and iteration
    /// for row in a.axis_iter(Axis(0)) {
    ///     println!("Row: {:?}", row);
    /// }
    /// ```
    #[test]
    fn test_complete_functionality() {
        // Array creation
        let a = array![[1., 2.], [3., 4.]];
        assert_eq!(a.shape(), &[2, 2]);

        // Slicing with s! macro
        let slice = a.slice(s![.., 0]);
        assert_eq!(slice.len(), 2);

        // Mathematical operations
        let b = &a + &a;
        assert_eq!(b[[0, 0]], 2.);

        // Axis operations
        let sum = a.sum_axis(Axis(0));
        assert_eq!(sum.len(), 2);

        // Broadcasting
        let c = array![1., 2.];
        let d = &a + &c;
        assert_eq!(d[[0, 0]], 2.);
    }
}

// ========================================
// MIGRATION GUIDE
// ========================================

pub mod migration_guide {
    //! # Migration Guide: From Fragmented to Unified ndarray Access
    //!
    //! ## Before (Fragmented, Policy-Violating)
    //!
    //! ```rust,ignore
    //! // Different files used different imports
    //! use scirs2_autograd::ndarray::{Array1, array};
    //! use scirs2_core::ndarray_ext::{ArrayView};
    //! use ndarray::{s!, Axis};  // POLICY VIOLATION!
    //! ```
    //!
    //! ## After (Unified, Policy-Compliant)
    //!
    //! ```rust,ignore
    //! // Single, consistent import
    //! use scirs2_core::ndarray::*;
    //!
    //! // Everything works:
    //! let arr = array![[1, 2], [3, 4]];
    //! let slice = arr.slice(s![.., 0]);
    //! let view: ArrayView<_, _> = arr.view();
    //! let sum = arr.sum_axis(Axis(0));
    //! ```
    //!
    //! ## Benefits
    //!
    //! 1. **Single Import Path**: No more confusion about where to import from
    //! 2. **Complete Functionality**: All ndarray features available
    //! 3. **Policy Compliance**: No direct ndarray imports needed
    //! 4. **Future-Proof**: Centralized control over array functionality
    //!
    //! ## Quick Reference
    //!
    //! | Old Import | New Import |
    //! |------------|------------|
    //! | `use ndarray::{Array, array}` | `use scirs2_core::ndarray::{Array, array}` |
    //! | `use scirs2_autograd::ndarray::*` | `use scirs2_core::ndarray::*` |
    //! | `use scirs2_core::ndarray_ext::*` | `use scirs2_core::ndarray::*` |
    //! | `use ndarray::{s!, Axis}` | `use scirs2_core::ndarray::{s, Axis}` |
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_macro_available() {
        let arr = array![[1, 2], [3, 4]];
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[0, 0]], 1);
    }

    #[test]
    fn test_s_macro_available() {
        let arr = array![[1, 2, 3], [4, 5, 6]];
        let slice = arr.slice(s![.., 1..]);
        assert_eq!(slice.shape(), &[2, 2]);
    }

    #[test]
    fn test_axis_operations() {
        let arr = array![[1., 2.], [3., 4.]];
        let sum = arr.sum_axis(Axis(0));
        assert_eq!(sum, array![4., 6.]);
    }

    #[test]
    fn test_views_and_iteration() {
        let mut arr = array![[1, 2], [3, 4]];

        // Test immutable view first
        {
            let view: ArrayView<_, _> = arr.view();
            for val in view.iter() {
                assert!(*val > 0);
            }
        }

        // Test mutable view after immutable view is dropped
        {
            let mut view_mut: ArrayViewMut<_, _> = arr.view_mut();
            for val in view_mut.iter_mut() {
                *val *= 2;
            }
        }

        assert_eq!(arr[[0, 0]], 2);
    }

    #[test]
    fn test_concatenate_and_stack() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 6], [7, 8]];

        // Concatenate along axis 0
        let concat = concatenate(Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(concat.shape(), &[4, 2]);

        // Stack along new axis
        let stacked = ndarray::stack(Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_zip_operations() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];
        let mut c = array![0, 0, 0];

        azip!((a in &a, b in &b, c in &mut c) {
            *c = a + b;
        });

        assert_eq!(c, array![5, 7, 9]);
    }
}
