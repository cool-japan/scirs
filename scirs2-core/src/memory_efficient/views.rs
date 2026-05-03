use super::validation;
use crate::error::{CoreError, ErrorContext, ErrorLocation};
use ::ndarray::{
    Array, ArrayBase, ArrayView as NdArrayView, ArrayViewMut as NdArrayViewMut, Data, Dimension,
    Ix1, Ix2, RawDataMut,
};

/// A type alias for ndarray's ArrayView with additional functionality
pub type ArrayView<'a, A, D> = NdArrayView<'a, A, D>;

/// A type alias for ndarray's ArrayViewMut with additional functionality
pub type ViewMut<'a, A, D> = NdArrayViewMut<'a, A, D>;

/// Create a view of an array with a different element type
///
/// This function creates a view of the given array interpreting its elements
/// as a different type. This is useful for viewing binary data as different
/// types without copying.
///
/// # Safety
///
/// This function is unsafe because it does not check that the memory layout
/// of the source type is compatible with the destination type.
///
/// # Arguments
///
/// * `array` - The array to view
///
/// # Returns
///
/// A view of the array with elements interpreted as the new type
pub unsafe fn view_as<A, B, S, D>(array: &ArrayBase<S, D>) -> Result<ArrayView<'_, B, D>, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    validation::check_not_empty(array)?;

    // Calculate new shape based on type sizes
    let a_size = std::mem::size_of::<A>();
    let b_size = std::mem::size_of::<B>();

    if a_size == 0 || b_size == 0 {
        return Err(CoreError::ValidationError(
            ErrorContext::new("Cannot reinterpret view of zero-sized type".to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    if a_size != b_size {
        // Reinterpreting between differently-sized element types requires a shape change
        // that cannot be expressed within the generic `D` dimension parameter; use a
        // 1-D-specific helper or reshape the array to Ix1 before calling view_as.
        return Err(CoreError::NotImplementedError(
            ErrorContext::new(format!(
                "view_as with differing element sizes ({a_size} vs {b_size}) requires a shape \
                 change not representable in the generic dimension D; \
                 reshape to Ix1 first or use a dedicated 1-D helper"
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // Alignment check: the data pointer must be aligned for B.
    let ptr = array.as_ptr() as *const B;
    if (ptr as usize) % std::mem::align_of::<B>() != 0 {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Data pointer is not aligned for the target type (required alignment: {})",
                std::mem::align_of::<B>()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // SAFETY: We have verified:
    //   1. The array is non-empty (checked by check_not_empty above).
    //   2. Both element types have the same size, so the stride values remain valid
    //      for B (ndarray's cast asserts size equality internally as well).
    //   3. The pointer is properly aligned for B (alignment check above).
    //   4. The caller, by using this `unsafe fn`, guarantees that the byte
    //      representation of every A element is a valid B value.
    //   5. The lifetime 'a of the returned view is tied to the lifetime of
    //      `array`, so no use-after-free is possible.
    let raw_view = array.raw_view().cast::<B>();
    Ok(raw_view.deref_into_view())
}

/// Create a mutable view of an array with a different element type
///
/// # Safety
///
/// This function is unsafe because it does not check that the memory layout
/// of the source type is compatible with the destination type.
///
/// # Arguments
///
/// * `array` - The array to view
///
/// # Returns
///
/// A mutable view of the array with elements interpreted as the new type
pub unsafe fn view_mut_as<A, B, S, D>(
    array: &mut ArrayBase<S, D>,
) -> Result<ViewMut<'_, B, D>, CoreError>
where
    A: Clone,
    S: Data<Elem = A> + RawDataMut,
    D: Dimension,
{
    validation::check_not_empty(array)?;

    // Calculate new shape based on type sizes
    let a_size = std::mem::size_of::<A>();
    let b_size = std::mem::size_of::<B>();

    if a_size == 0 || b_size == 0 {
        return Err(CoreError::ValidationError(
            ErrorContext::new("Cannot reinterpret view of zero-sized type".to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    if a_size != b_size {
        // Reinterpreting between differently-sized element types requires a shape change
        // that cannot be expressed within the generic `D` dimension parameter; use a
        // 1-D-specific helper or reshape the array to Ix1 before calling view_mut_as.
        return Err(CoreError::NotImplementedError(
            ErrorContext::new(format!(
                "view_mut_as with differing element sizes ({a_size} vs {b_size}) requires a \
                 shape change not representable in the generic dimension D; \
                 reshape to Ix1 first or use a dedicated 1-D helper"
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // Alignment check: the data pointer must be aligned for B.
    let ptr = array.as_ptr() as *const B;
    if (ptr as usize) % std::mem::align_of::<B>() != 0 {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Data pointer is not aligned for the target type (required alignment: {})",
                std::mem::align_of::<B>()
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // SAFETY: We have verified:
    //   1. The array is non-empty (checked by check_not_empty above).
    //   2. Both element types have the same size, so strides remain valid for B.
    //   3. The pointer is properly aligned for B (alignment check above).
    //   4. The caller, by using this `unsafe fn`, guarantees that the byte
    //      representation of every A element is a valid B value, and that
    //      writing valid B values leaves valid A byte patterns (since sizes
    //      are equal, the allocation remains properly sized).
    //   5. The lifetime 'a of the returned view is tied to the mutable borrow
    //      of `array`, enforcing exclusive access for the view's lifetime.
    let raw_view_mut = array.raw_view_mut().cast::<B>();
    Ok(raw_view_mut.deref_into_view_mut())
}

/// Create a transposed copy of a 2D array
///
/// # Arguments
///
/// * `array` - The array to transpose
///
/// # Returns
///
/// A transposed copy of the array
#[allow(dead_code)]
pub fn transpose_view<A, S>(array: &ArrayBase<S, Ix2>) -> Result<Array<A, Ix2>, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
{
    validation::check_not_empty(array)?;

    // Create a transposed owned copy
    Ok(array.to_owned().t().to_owned())
}

/// Create a copy of the diagonal of a 2D array
///
/// # Arguments
///
/// * `array` - The array to view
///
/// # Returns
///
/// A copy of the diagonal of the array
#[allow(dead_code)]
pub fn diagonal_view<A, S>(array: &ArrayBase<S, Ix2>) -> Result<Array<A, Ix1>, CoreError>
where
    A: Clone,
    S: Data<Elem = A>,
{
    validation::check_not_empty(array)?;
    validation::check_square(array)?;

    // Create a diagonal copy
    Ok(array.diag().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// `f32` and `u32` share the same size (4 bytes) and same alignment (4 bytes).
    /// Viewing an f32 array as u32 reads the raw IEEE-754 bit pattern.
    #[test]
    fn test_view_as_same_size_f32_as_u32() {
        // Positive zero: bit pattern 0x0000_0000
        let arr = array![0.0_f32, 1.0_f32];
        // SAFETY: u32 is valid for any 4-byte bit pattern; f32 values we use are
        // well-defined IEEE-754 values, so reading them as u32 is well-defined.
        let view = unsafe { view_as::<f32, u32, _, _>(&arr) };
        assert!(view.is_ok(), "view_as should succeed for same-size types");
        let view = view.expect("view_as returned Err unexpectedly");
        // 0.0f32 → 0x0000_0000, 1.0f32 → 0x3F80_0000
        assert_eq!(view[0], 0x0000_0000_u32);
        assert_eq!(view[1], 0x3F80_0000_u32);
    }

    /// `u32` and `i32` share the same size and alignment; reinterpreting should
    /// succeed and expose the two's-complement representation.
    #[test]
    fn test_view_as_u32_as_i32_round_trip() {
        let arr = array![0_u32, 0xFFFF_FFFF_u32];
        // SAFETY: every u32 bit pattern is a valid i32 (two's complement).
        let view = unsafe { view_as::<u32, i32, _, _>(&arr) };
        assert!(view.is_ok());
        let view = view.expect("view_as returned Err unexpectedly");
        assert_eq!(view[0], 0_i32);
        assert_eq!(view[1], -1_i32);
    }

    /// Attempting to view a `u8` array as `u32` (different sizes: 1 vs 4) must
    /// return a `NotImplementedError` because the shape would need to change.
    #[test]
    fn test_view_as_different_sizes_returns_error() {
        let arr = array![0_u8, 1_u8, 2_u8, 3_u8];
        // SAFETY: irrelevant — we expect an Err before any unsafe memory access.
        let result = unsafe { view_as::<u8, u32, _, _>(&arr) };
        assert!(
            result.is_err(),
            "view_as with different element sizes must fail"
        );
        match result.expect_err("expected an error") {
            CoreError::NotImplementedError(_) => {}
            other => panic!("Expected NotImplementedError, got: {other:?}"),
        }
    }

    /// `view_mut_as` should succeed for same-size types, and mutations through
    /// the reinterpreted view must be visible in the original array.
    #[test]
    fn test_view_mut_as_same_size_mutates_original() {
        let mut arr = array![0_u32, 1_u32];
        {
            // SAFETY: i32 is valid for any u32 bit pattern we set below.
            let mut view = unsafe { view_mut_as::<u32, i32, _, _>(&mut arr) }
                .expect("view_mut_as returned Err unexpectedly");
            view[0] = -1_i32; // sets bit pattern 0xFFFF_FFFF
        }
        // After the view is dropped, inspect via u32 lens
        assert_eq!(arr[0], 0xFFFF_FFFF_u32);
        assert_eq!(arr[1], 1_u32); // unchanged
    }

    /// Attempting to view an empty array must return a `ValidationError`.
    #[test]
    fn test_view_as_empty_array_returns_error() {
        let arr = ndarray::Array1::<f32>::zeros(0);
        // SAFETY: no memory is accessed; we expect an Err for the empty check.
        let result = unsafe { view_as::<f32, u32, _, _>(&arr) };
        assert!(result.is_err(), "view_as on empty array must fail");
        match result.expect_err("expected an error") {
            CoreError::ValidationError(_) => {}
            other => panic!("Expected ValidationError, got: {other:?}"),
        }
    }
}
