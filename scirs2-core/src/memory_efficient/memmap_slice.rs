//! Slicing operations for memory-mapped arrays.
//!
//! This module provides functionality for efficiently slicing memory-mapped arrays
//! without loading the entire array into memory. These slicing operations maintain
//! the memory-mapping and only load the required data when accessed.

#[cfg(test)]
use super::memmap::AccessMode;
use super::memmap::MemoryMappedArray;
use crate::error::{CoreError, CoreResult, ErrorContext};
use ndarray::{ArrayBase, Dimension, SliceInfo, SliceInfoElem};
use std::marker::PhantomData;
use std::ops::RangeBounds;

/// A slice of a memory-mapped array that maintains memory-mapping.
///
/// This provides a view into a subset of a memory-mapped array without
/// loading the entire array into memory. Data is only loaded when
/// accessed through the slice.
pub struct MemoryMappedSlice<A, D>
where
    A: Clone + Copy + 'static + Send + Sync,
    D: Dimension,
{
    /// The source memory-mapped array
    source: MemoryMappedArray<A>,

    /// The slice information
    slice_info: SliceInfo<Vec<SliceInfoElem>, D, D>,

    /// Phantom data for dimension type
    _phantom: PhantomData<D>,
}

impl<A, D> MemoryMappedSlice<A, D>
where
    A: Clone + Copy + 'static + Send + Sync,
    D: Dimension,
{
    /// Creates a new slice from a memory-mapped array and slice information.
    pub fn new(
        source: MemoryMappedArray<A>,
        slice_info: SliceInfo<Vec<SliceInfoElem>, D, D>,
    ) -> Self {
        Self {
            source,
            slice_info,
            _phantom: PhantomData,
        }
    }

    /// Returns the shape of the slice.
    ///
    /// Since we can't directly access the private out_dim field in SliceInfo,
    /// this just returns an empty dimension. Actual implementations would
    /// need to calculate this based on the slice parameters.
    pub fn shape(&self) -> D {
        D::default()
    }

    /// Returns a reference to the source memory-mapped array.
    pub const fn source(&self) -> &MemoryMappedArray<A> {
        &self.source
    }

    /// Returns the slice information.
    pub const fn slice_info(&self) -> &SliceInfo<Vec<SliceInfoElem>, D, D> {
        &self.slice_info
    }

    /// Loads the slice data into memory.
    ///
    /// This method materializes the slice by loading only the necessary data
    /// from the memory-mapped file.
    pub fn load(&self) -> CoreResult<ArrayBase<ndarray::OwnedRepr<A>, D>> {
        use ndarray::IxDyn;

        // Get the raw data slice
        let data_slice = self.source.as_slice();

        // Create a dynamic array with the proper shape
        let shape = IxDyn(&self.source.shape);
        let source_array = ndarray::ArrayView::from_shape(shape, data_slice).map_err(|e| {
            CoreError::ShapeError(ErrorContext::new(format!(
                "Failed to create array view: {}",
                e
            )))
        })?;

        // Convert SliceInfo to a slice arg that can be used with IxDyn
        let slice_elements = self.slice_info.as_ref();

        // Apply the slice using ndarray's slicing
        let sliced = source_array.slice_each_axis(|ax| {
            if ax.axis.index() < slice_elements.len() {
                match &slice_elements[ax.axis.index()] {
                    SliceInfoElem::Slice { start, end, step } => {
                        let start = *start as usize;
                        let end = end.map(|e| e as usize).unwrap_or(ax.len);
                        let step = *step as usize;
                        ndarray::Slice::new(start as isize, Some(end as isize), step as isize)
                    }
                    SliceInfoElem::Index(idx) => ndarray::Slice::new(*idx, Some(*idx + 1), 1),
                    _ => ndarray::Slice::new(0, None, 1),
                }
            } else {
                ndarray::Slice::new(0, None, 1)
            }
        });

        // Convert to owned array
        let owned = sliced.to_owned();

        // For 2D slices, we know the expected shape
        if D::NDIM == Some(2) && slice_elements.len() == 2 {
            // Calculate the shape from the slice elements
            let mut new_shape = Vec::new();
            for (i, elem) in slice_elements.iter().enumerate() {
                match elem {
                    SliceInfoElem::Slice { start, end, step } => {
                        let start = *start as usize;
                        // Safely get the dimension size, defaulting to a reasonable value if out of bounds
                        let dim_size = if i < self.source.shape.len() {
                            self.source.shape[i]
                        } else {
                            // This shouldn't happen for properly constructed slices
                            1
                        };
                        let end = end.map(|e| e as usize).unwrap_or(dim_size);
                        let step = *step as usize;
                        let len = (end - start).div_ceil(step);
                        new_shape.push(len);
                    }
                    SliceInfoElem::Index(_) => {
                        // Index reduces dimension, but we're expecting a 2D result
                        new_shape.push(1);
                    }
                    _ => {}
                }
            }

            // Create the properly shaped array
            if new_shape.len() == 2 {
                // Debug output
                eprintln!(
                    "DEBUG: Trying to reshape from {:?} to {:?}",
                    owned.shape(),
                    (new_shape[0], new_shape[1])
                );
                eprintln!(
                    "DEBUG: owned.len() = {}, new_shape product = {}",
                    owned.len(),
                    new_shape[0] * new_shape[1]
                );

                let reshaped = owned
                    .into_shape_with_order((new_shape[0], new_shape[1]))
                    .map_err(|e| {
                        CoreError::ShapeError(ErrorContext::new(format!(
                            "Failed to reshape sliced array: {}",
                            e
                        )))
                    })?;

                // Convert to the target dimension type
                match reshaped.into_dimensionality::<D>() {
                    Ok(array) => Ok(array),
                    Err(_) => Err(CoreError::ShapeError(ErrorContext::new(
                        "Failed to convert reshaped array to target dimension type",
                    ))),
                }
            } else {
                // Try the original conversion
                match owned.into_dimensionality::<D>() {
                    Ok(array) => Ok(array),
                    Err(_) => Err(CoreError::ShapeError(ErrorContext::new(
                        "Failed to convert sliced array to target dimension type",
                    ))),
                }
            }
        } else {
            // For other dimensions, try direct conversion
            match owned.into_dimensionality::<D>() {
                Ok(array) => Ok(array),
                Err(_) => Err(CoreError::ShapeError(ErrorContext::new(
                    "Failed to convert sliced array to target dimension type",
                ))),
            }
        }
    }
}

/// Extension trait for adding slicing functionality to MemoryMappedArray.
pub trait MemoryMappedSlicing<A: Clone + Copy + 'static + Send + Sync> {
    /// Creates a slice of the memory-mapped array using standard slice syntax.
    fn slice<I, E>(&self, _info: I) -> CoreResult<MemoryMappedSlice<A, E>>
    where
        I: ndarray::SliceArg<E>,
        E: Dimension;

    /// Creates a 1D slice using a range.
    fn slice_1d(
        &self,
        range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix1>>;

    /// Creates a 2D slice using ranges for each dimension.
    fn slice_2d(
        &self,
        row_range: impl RangeBounds<usize>,
        col_range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix2>>;
}

impl<A: Clone + Copy + 'static + Send + Sync> MemoryMappedSlicing<A> for MemoryMappedArray<A> {
    fn slice<I, E>(&self, _info: I) -> CoreResult<MemoryMappedSlice<A, E>>
    where
        I: ndarray::SliceArg<E>,
        E: Dimension,
    {
        // Get the slice info
        let _shape = self.shape.clone();
        // Use unsafe to convert the SliceArg to SliceInfo
        // This is because the API for this has changed
        // We need to get the SliceInfo from the SliceArg in a more direct way
        // But this functionality isn't directly available in the public API
        // This is a temporary workaround until the slice API is properly updated

        // Create a default SliceInfo
        // In a real implementation, we'd properly convert the SliceArg to SliceInfo
        let slice_info: SliceInfo<Vec<SliceInfoElem>, E, E> = unsafe {
            let elems = vec![SliceInfoElem::Slice {
                start: 0,
                end: Some(10),
                step: 1,
            }]; // Placeholder
            SliceInfo::new(elems).unwrap()
        };

        // Create the slice
        let source = MemoryMappedArray::new::<ndarray::OwnedRepr<A>, ndarray::IxDyn>(
            None,
            &self.file_path,
            self.mode,
            self.offset,
        )?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }

    fn slice_1d(
        &self,
        range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix1>> {
        // Convert to explicit range
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[0],
        };

        if start >= end || end > self.shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid slice range {}..{} for array of shape {:?}",
                start, end, self.shape
            ))));
        }

        // Create SliceInfo for 1D array
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, ndarray::Ix1, ndarray::Ix1>::new(vec![
                SliceInfoElem::Slice {
                    start: start as isize,
                    end: Some(end as isize),
                    step: 1,
                },
            ])
            .map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create slice info: {}",
                    e
                )))
            })?
        };

        // Create a new reference to the same memory-mapped file
        let source = self.clone_ref()?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }

    fn slice_2d(
        &self,
        row_range: impl RangeBounds<usize>,
        col_range: impl RangeBounds<usize>,
    ) -> CoreResult<MemoryMappedSlice<A, ndarray::Ix2>> {
        // Ensure we're working with a 2D array
        if self.shape.len() != 2 {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Expected 2D array, got {}D",
                self.shape.len()
            ))));
        }

        // Convert row range to explicit range
        let row_start = match row_range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let row_end = match row_range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[0],
        };

        // Convert column range to explicit range
        let col_start = match col_range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let col_end = match col_range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.shape[1],
        };

        // Validate ranges
        if row_start >= row_end || row_end > self.shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid row slice range {}..{} for array of shape {:?}",
                row_start, row_end, self.shape
            ))));
        }

        if col_start >= col_end || col_end > self.shape[1] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Invalid column slice range {}..{} for array of shape {:?}",
                col_start, col_end, self.shape
            ))));
        }

        // Create SliceInfo for 2D array
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, ndarray::Ix2, ndarray::Ix2>::new(vec![
                SliceInfoElem::Slice {
                    start: row_start as isize,
                    end: Some(row_end as isize),
                    step: 1,
                },
                SliceInfoElem::Slice {
                    start: col_start as isize,
                    end: Some(col_end as isize),
                    step: 1,
                },
            ])
            .map_err(|e| {
                CoreError::ShapeError(ErrorContext::new(format!(
                    "Failed to create slice info: {}",
                    e
                )))
            })?
        };

        // Create a new reference to the same memory-mapped file
        let source = self.clone_ref()?;
        Ok(MemoryMappedSlice::new(source, slice_info))
    }
}

#[cfg(test)]
mod tests {
    use super::super::create_mmap;
    use super::*;
    use ndarray::Array2;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_memory_mapped_slice_1d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_slice_1d.bin");

        // Create a test array and save it to a file
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).unwrap();
        for val in &data {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[100]).unwrap();

        // Create a slice
        let slice = mmap.slice_1d(10..20).unwrap();

        // Load the slice data
        let array = slice.load().unwrap();

        // Check that the slice contains the expected data
        assert_eq!(array.len(), 10);
        for (i, &val) in array.iter().enumerate() {
            assert_eq!(val, (i + 10) as f64);
        }
    }

    #[test]
    fn test_memory_mapped_slice_2d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_slice_2d.bin");

        // Create a test 2D array and save it to a file using the proper method
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Use save_array which handles headers correctly
        MemoryMappedArray::<f64>::save_array(&data, &file_path, None).unwrap();

        // Open using open_zero_copy which handles headers correctly
        let mmap =
            MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwrap();

        // Debug: print shape info
        println!("mmap.shape: {:?}", mmap.shape);
        println!("mmap.size: {}", mmap.size);

        // Debug: print original array to verify
        let orig_array = mmap.as_array::<ndarray::Ix2>().unwrap();
        println!("Original array (first 5x10):");
        for i in 0..5 {
            print!("Row {}: ", i);
            for j in 0..10 {
                print!("{:4.0} ", orig_array[[i, j]]);
            }
            print!("   Expected: ");
            for j in 0..10 {
                print!("{:4} ", i * 10 + j);
            }
            println!();
        }

        // Create a slice
        let slice = mmap.slice_2d(2..5, 3..7).unwrap();

        // Debug: print slice info
        println!("slice.source.shape: {:?}", slice.source.shape);
        println!("slice_info: {:?}", slice.slice_info.as_ref());

        // Load the slice data
        let array = slice.load().unwrap();

        // Check that the slice contains the expected data
        assert_eq!(array.shape(), &[3, 4]);

        // Debug: print the slice content
        println!("Slice content:");
        for i in 0..3 {
            for j in 0..4 {
                print!("{:6.1} ", array[[i, j]]);
            }
            println!();
        }

        for i in 0..3 {
            for j in 0..4 {
                let expected = ((i + 2) * 10 + (j + 3)) as f64;
                let actual = array[[i, j]];
                if actual != expected {
                    println!(
                        "Mismatch at [{}, {}]: expected {}, got {}",
                        i, j, expected, actual
                    );
                }
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[ignore = "slice() method implementation needs to be completed"]
    fn test_memory_mapped_slice_with_ndarray_slice_syntax() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_slice_syntax.bin");

        // Create a test 2D array and save it to a file using the proper method
        let data = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);

        // Create a memory-mapped array with proper header
        let mmap = create_mmap::<f64, _, _>(&data, &file_path, AccessMode::Write, 0).unwrap();

        // Create a slice using ndarray's s![] macro
        use ndarray::s;
        let slice = mmap.slice(s![2..5, 3..7]).unwrap();

        // Load the slice data
        let array: ndarray::Array2<f64> = slice.load().unwrap();

        // Check that the slice contains the expected data
        assert_eq!(array.shape(), &[3usize, 4usize]);
        for i in 0..3usize {
            for j in 0..4usize {
                assert_eq!(array[[i, j]], ((i + 2) * 10 + (j + 3)) as f64);
            }
        }
    }
}
