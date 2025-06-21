use super::validation;
use crate::error::{CoreError, ErrorContext, ErrorLocation};
use ndarray::{
    Array, ArrayBase, ArrayView as NdArrayView, ArrayViewMut as NdArrayViewMut, Data, Dimension,
    Ix1, Ix2,
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

    if a_size % b_size != 0 && b_size % a_size != 0 {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Type sizes are not compatible: {} is not divisible by or a divisor of {}",
                a_size, b_size
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // This is a placeholder - the actual implementation would need to create a view
    // with the appropriate type conversion
    Err(CoreError::ImplementationError(
        ErrorContext::new("view_as is not yet implemented".to_string())
            .with_location(ErrorLocation::new(file!(), line!())),
    ))
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

    if a_size % b_size != 0 && b_size % a_size != 0 {
        return Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Type sizes are not compatible: {} is not divisible by or a divisor of {}",
                a_size, b_size
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }

    // This is a placeholder - the actual implementation would need to create a mutable view
    // with the appropriate type conversion
    Err(CoreError::ImplementationError(
        ErrorContext::new("view_mut_as is not yet implemented".to_string())
            .with_location(ErrorLocation::new(file!(), line!())),
    ))
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
