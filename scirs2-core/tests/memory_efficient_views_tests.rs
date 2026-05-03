#[cfg(feature = "memory_efficient")]
mod tests {
    use scirs2_core::error::CoreError;
    use scirs2_core::memory_efficient::{diagonal_view, transpose_view, ArrayView};
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_transpose_view() {
        let data = Array2::from_shape_fn((3, 4), |(i, j)| i * 10 + j);

        // Create a transpose view
        let view = transpose_view(&data).expect("Test: operation failed");

        // Check that the shape is transposed
        assert_eq!(view.shape(), &[4, 3]);

        // Check that the elements are correctly transposed
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(view[[j, i]], data[[i, j]]);
            }
        }

        // Since we're returning copies rather than views, changes to the original
        // data won't affect the result. Just check the original transpose was correct.
        assert_eq!(view[[2, 1]], data[[1, 2]]);
    }

    #[test]
    fn test_diagonal_view() {
        let data = Array2::from_shape_fn((3, 3), |(i, j)| if i == j { i + 1 } else { 0 });

        // Create a diagonal view
        let view = diagonal_view(&data).expect("Test: operation failed");

        // Check that the shape is correct
        assert_eq!(view.shape(), &[3]);

        // Check that the elements are the diagonal
        assert_eq!(view[0], 1);
        assert_eq!(view[1], 2);
        assert_eq!(view[2], 3);
    }

    #[test]
    fn test_diagonal_view_non_square() {
        let data = Array2::from_shape_fn((3, 4), |(i, j)| i * 10 + j);

        // Try to create a diagonal view of a non-square matrix
        let result = diagonal_view(&data);

        // This should fail
        assert!(result.is_err());

        // Check the error type
        match result {
            Err(CoreError::ValidationError(ctx)) => {
                assert!(ctx.message.contains("square"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[test]
    fn test_empty_array_views() {
        let empty: Array2<f64> = Array2::from_shape_fn((0, 0), |_| 0.0);

        // Try to create views of an empty array
        let transpose_result = transpose_view(&empty);
        let diagonal_result = diagonal_view(&empty);

        // This should fail with a validation error
        assert!(transpose_result.is_err());
        assert!(diagonal_result.is_err());

        match transpose_result {
            Err(CoreError::ValidationError(ctx)) => {
                assert!(ctx.message.contains("empty"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    // view_as with differing element sizes (f64 → u8, 8 vs 1 byte) cannot be represented
    // within the same generic dimension D and must return NotImplementedError.
    #[test]
    fn test_view_as_differing_sizes_not_implemented() {
        use scirs2_core::memory_efficient::view_as;

        let data = Array2::from_shape_fn((3, 4), |(i, j)| i as f64 * 10.0 + j as f64);

        // SAFETY: We expect an Err before any memory is accessed.
        let result = unsafe { view_as::<f64, u8, _, _>(&data) };

        assert!(
            result.is_err(),
            "view_as with different element sizes must fail"
        );
        match result.expect_err("expected an error") {
            CoreError::NotImplementedError(_) => {}
            other => panic!("Expected NotImplementedError for differing sizes, got: {other:?}"),
        }
    }

    // view_as with same-size types (f64 → i64, both 8 bytes) must succeed.
    #[test]
    fn test_view_as_same_size_succeeds() {
        use scirs2_core::memory_efficient::view_as;

        let data = Array2::from_shape_fn((2, 2), |(i, j)| (i * 2 + j) as f64);

        // SAFETY: i64 is valid for any 8-byte bit pattern produced by f64.
        let result = unsafe { view_as::<f64, i64, _, _>(&data) };
        assert!(
            result.is_ok(),
            "view_as should succeed for f64 → i64 (same size)"
        );
        let view = result.expect("view_as returned Err unexpectedly");
        // Shape must be preserved.
        assert_eq!(view.shape(), data.shape());
    }
}
