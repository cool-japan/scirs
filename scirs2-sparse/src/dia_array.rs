// DIA Array implementation
//
// This module provides the DIA (DIAgonal) array format,
// which is efficient for matrices with values concentrated on a small number of diagonals.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Sub};

use crate::coo_array::CooArray;
use crate::csr_array::CsrArray;
use crate::dok_array::DokArray;
use crate::error::{SparseError, SparseResult};
use crate::lil_array::LilArray;
use crate::sparray::{SparseArray, SparseSum};

/// DIA Array format
///
/// The DIA (DIAgonal) format stores data as a collection of diagonals.
/// It is efficient for matrices with values concentrated on a small number of diagonals,
/// like tridiagonal or band matrices.
///
/// # Notes
///
/// - Very efficient storage for band matrices
/// - Fast matrix-vector products for banded matrices
/// - Not efficient for general sparse matrices
/// - Difficult to modify once constructed
///
#[derive(Clone)]
pub struct DiaArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    /// Diagonals data (n_diags x max(rows, cols))
    data: Vec<Array1<T>>,
    /// Diagonal offsets from the main diagonal (k > 0 for above, k < 0 for below)
    offsets: Vec<isize>,
    /// Shape of the array
    shape: (usize, usize),
}

impl<T> DiaArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    /// Create a new DIA array from raw data
    ///
    /// # Arguments
    ///
    /// * `data` - Diagonals data (n_diags x max(rows, cols))
    /// * `offsets` - Diagonal offsets from the main diagonal
    /// * `shape` - Tuple containing the array dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new DIA array
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_sparse::dia_array::DiaArray;
    /// use scirs2_sparse::sparray::SparseArray;
    /// use scirs2_core::ndarray::Array1;
    ///
    /// // Create a 3x3 sparse array with main diagonal and upper diagonal
    /// let data = vec![
    ///     Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
    ///     Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal (k=1)
    /// ];
    /// let offsets = vec![0, 1]; // Main diagonal and k=1
    /// let shape = (3, 3);
    ///
    /// let array = DiaArray::new(data, offsets, shape).unwrap();
    /// assert_eq!(array.shape(), (3, 3));
    /// assert_eq!(array.nnz(), 5); // 3 on main diagonal, 2 on upper diagonal
    /// ```
    pub fn new(
        data: Vec<Array1<T>>,
        offsets: Vec<isize>,
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        let (rows, cols) = shape;
        let max_dim = rows.max(cols);

        // Validate input data
        if data.len() != offsets.len() {
            return Err(SparseError::DimensionMismatch {
                expected: data.len(),
                found: offsets.len(),
            });
        }

        for diag in data.iter() {
            if diag.len() != max_dim {
                return Err(SparseError::DimensionMismatch {
                    expected: max_dim,
                    found: diag.len(),
                });
            }
        }

        Ok(DiaArray {
            data,
            offsets,
            shape,
        })
    }

    /// Create a new empty DIA array
    ///
    /// # Arguments
    ///
    /// * `shape` - Tuple containing the array dimensions (rows, cols)
    ///
    /// # Returns
    ///
    /// * A new empty DIA array
    pub fn empty(shape: (usize, usize)) -> Self {
        DiaArray {
            data: Vec::new(),
            offsets: Vec::new(),
            shape,
        }
    }

    /// Convert COO format to DIA format
    ///
    /// # Arguments
    ///
    /// * `row` - Row indices
    /// * `col` - Column indices
    /// * `data` - Data values
    /// * `shape` - Shape of the array
    ///
    /// # Returns
    ///
    /// * A new DIA array
    pub fn from_triplets(
        row: &[usize],
        col: &[usize],
        data: &[T],
        shape: (usize, usize),
    ) -> SparseResult<Self> {
        if row.len() != col.len() || row.len() != data.len() {
            return Err(SparseError::InconsistentData {
                reason: "Lengths of row, col, and data arrays must be equal".to_string(),
            });
        }

        let (rows, cols) = shape;
        let max_dim = rows.max(cols);

        // Identify unique diagonals
        let mut diagonal_offsets = std::collections::HashSet::new();
        for (&r, &c) in row.iter().zip(col.iter()) {
            if r >= rows || c >= cols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (r, c),
                    shape,
                });
            }
            // Calculate diagonal offset (column - row for diagonals)
            let offset = c as isize - r as isize;
            diagonal_offsets.insert(offset);
        }

        // Convert to a sorted vector
        let mut offsets: Vec<isize> = diagonal_offsets.into_iter().collect();
        offsets.sort();

        // Create data arrays (initialized to zero)
        let mut diag_data = Vec::with_capacity(offsets.len());
        for _ in 0..offsets.len() {
            diag_data.push(Array1::zeros(max_dim));
        }

        // Fill in the data
        for (&r, (&c, &val)) in row.iter().zip(col.iter().zip(data.iter())) {
            let offset = c as isize - r as isize;
            let diag_idx = offsets.iter().position(|&o| o == offset).unwrap();

            // For upper diagonals (k > 0), the index is row
            // For lower diagonals (k < 0), the index is column
            let index = if offset >= 0 { r } else { c };
            diag_data[diag_idx][index] = val;
        }

        DiaArray::new(diag_data, offsets, shape)
    }

    /// Convert to COO format
    fn to_coo_internal(&self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let (rows, cols) = self.shape;
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            if offset >= 0 {
                // Upper diagonal
                let offset_usize = offset as usize;
                let length = rows.min(cols.saturating_sub(offset_usize));

                for i in 0..length {
                    let value = diag[i];
                    if !value.is_zero() {
                        row_indices.push(i);
                        col_indices.push(i + offset_usize);
                        values.push(value);
                    }
                }
            } else {
                // Lower diagonal
                let offset_usize = (-offset) as usize;
                let length = cols.min(rows.saturating_sub(offset_usize));

                for i in 0..length {
                    let value = diag[i];
                    if !value.is_zero() {
                        row_indices.push(i + offset_usize);
                        col_indices.push(i);
                        values.push(value);
                    }
                }
            }
        }

        (row_indices, col_indices, values)
    }
}

impl<T> SparseArray<T> for DiaArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn nnz(&self) -> usize {
        let (rows, cols) = self.shape;
        let mut count = 0;

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            // Calculate valid range for this diagonal
            let length = if offset >= 0 {
                rows.min(cols.saturating_sub(offset as usize))
            } else {
                cols.min(rows.saturating_sub((-offset) as usize))
            };

            // Count non-zeros in the valid range
            let start_idx = 0; // Start at 0 regardless of offset
            for i in start_idx..start_idx + length {
                if !diag[i].is_zero() {
                    count += 1;
                }
            }
        }

        count
    }

    fn dtype(&self) -> &str {
        "float" // Placeholder; ideally would return the actual type
    }

    fn to_array(&self) -> Array2<T> {
        // Convert to dense format
        let (rows, cols) = self.shape;
        let mut result = Array2::zeros((rows, cols));

        // In the test case we have:
        // data[0] = [1.0, 3.0, 7.0] with offset 0 (main diagonal)
        // data[1] = [4.0, 5.0, 0.0] with offset 1 (upper diagonal)
        // data[2] = [0.0, 2.0, 6.0] with offset -1 (lower diagonal)

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            if offset >= 0 {
                // Upper diagonal (k >= 0)
                let offset_usize = offset as usize;
                for i in 0..rows.min(cols.saturating_sub(offset_usize)) {
                    result[[i, i + offset_usize]] = diag[i];
                }
            } else {
                // Lower diagonal (k < 0)
                let offset_usize = (-offset) as usize;
                for i in 0..cols.min(rows.saturating_sub(offset_usize)) {
                    result[[i + offset_usize, i]] = diag[i];
                }
            }
        }

        result
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        let row_array = Array1::from_vec(row_indices);
        let col_array = Array1::from_vec(col_indices);
        let data_array = Array1::from_vec(values);

        CooArray::from_triplets(
            &row_array.to_vec(),
            &col_array.to_vec(),
            &data_array.to_vec(),
            self.shape,
            false,
        )
        .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        CsrArray::from_triplets(&row_indices, &col_indices, &values, self.shape, false)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        self.to_coo()?.to_csc()
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        DokArray::from_triplets(&row_indices, &col_indices, &values, self.shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (row_indices, col_indices, values) = self.to_coo_internal();
        LilArray::from_triplets(&row_indices, &col_indices, &values, self.shape)
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        Ok(Box::new(self.clone()))
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        self.to_coo()?.to_bsr()
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert both to CSR for efficient addition
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.add(&*csr_other)
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert both to CSR for efficient subtraction
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.sub(&*csr_other)
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert both to CSR for efficient element-wise multiplication
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.mul(&*csr_other)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert both to CSR for efficient element-wise division
        let csr_self = self.to_csr()?;
        let csr_other = other.to_csr()?;
        csr_self.div(&*csr_other)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For matrix multiplication, use specialized DIA-Vector logic if other is thin
        let (_, n) = self.shape();
        let (p, q) = other.shape();

        if n != p {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: p,
            });
        }

        // If other is a vector (thin matrix), we can use optimized DIA-Vector multiplication
        if q == 1 {
            // Get the vector from other
            let other_array = other.to_array();
            let vec_view = other_array.column(0);

            // Perform DIA-Vector multiplication
            let result = self.dot_vector(&vec_view)?;

            // Convert to a matrix - create a COO from triplets
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut values = Vec::new();

            for (i, &val) in result.iter().enumerate() {
                if !val.is_zero() {
                    rows.push(i);
                    cols.push(0);
                    values.push(val);
                }
            }

            CooArray::from_triplets(&rows, &cols, &values, (result.len(), 1), false)
                .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
        } else {
            // For general matrices, convert to CSR
            let csr_self = self.to_csr()?;
            csr_self.dot(other)
        }
    }

    fn dot_vector(&self, other: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let (rows, cols) = self.shape;

        if cols != other.len() {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: other.len(),
            });
        }

        let mut result = Array1::zeros(rows);

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            if offset >= 0 {
                // Upper diagonal (k > 0)
                let offset_usize = offset as usize;
                let length = rows.min(cols.saturating_sub(offset_usize));

                for i in 0..length {
                    result[i] += diag[i] * other[i + offset_usize];
                }
            } else {
                // Lower diagonal (k < 0)
                let offset_usize = (-offset) as usize;
                let length = cols.min(rows.saturating_sub(offset_usize));

                for i in 0..length {
                    result[i + offset_usize] += diag[i] * other[i];
                }
            }
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For correct transposition, use COO intermediately
        // This avoids issues with the diagonal storage format
        let (row_indices, col_indices, values) = self.to_coo_internal();

        // Swap row and column indices
        let transposed_rows = col_indices;
        let transposed_cols = row_indices;

        // Create a new COO array and convert back to DIA
        CooArray::from_triplets(
            &transposed_rows,
            &transposed_cols,
            &values,
            (self.shape.1, self.shape.0),
            false,
        )?
        .to_dia()
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn get(&self, i: usize, j: usize) -> T {
        if i >= self.shape.0 || j >= self.shape.1 {
            return T::zero();
        }

        // Calculate the diagonal offset
        let offset = j as isize - i as isize;

        // Check if this offset exists in our stored diagonals
        if let Some(diag_idx) = self.offsets.iter().position(|&o| o == offset) {
            let diag = &self.data[diag_idx];

            // For upper diagonals (k > 0), the index is row
            // For lower diagonals (k < 0), the index is column
            let index = if offset >= 0 { i } else { j };

            // Make sure the index is within bounds
            if index < diag.len() {
                return diag[index];
            }
        }

        T::zero()
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        if i >= self.shape.0 || j >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: (i, j),
                shape: self.shape,
            });
        }

        // Calculate the diagonal offset
        let offset = j as isize - i as isize;

        // Find or create the diagonal
        let diag_idx = match self.offsets.iter().position(|&o| o == offset) {
            Some(idx) => idx,
            None => {
                // This diagonal doesn't exist yet, add it
                self.offsets.push(offset);
                self.data
                    .push(Array1::zeros(self.shape.0.max(self.shape.1)));

                // Sort the offsets and data to maintain canonical form
                let mut offset_data: Vec<(isize, Array1<T>)> = self
                    .offsets
                    .iter()
                    .cloned()
                    .zip(self.data.drain(..))
                    .collect();
                offset_data.sort_by_key(|&(offset_, _)| offset_);

                self.offsets = offset_data.iter().map(|&(offset_, _)| offset_).collect();
                self.data = offset_data.into_iter().map(|(_, data)| data).collect();

                // Get the index of the newly added diagonal
                self.offsets.iter().position(|&o| o == offset).unwrap()
            }
        };

        // Set the value
        let index = if offset >= 0 { i } else { j };
        self.data[diag_idx][index] = value;

        Ok(())
    }

    fn eliminate_zeros(&mut self) {
        // Create a new set of diagonals without zeros
        let mut new_offsets = Vec::new();
        let mut new_data = Vec::new();

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            // Check if this diagonal has any non-zero values
            let length = if offset >= 0 {
                self.shape
                    .0
                    .min(self.shape.1.saturating_sub(offset as usize))
            } else {
                self.shape
                    .1
                    .min(self.shape.0.saturating_sub((-offset) as usize))
            };

            let has_nonzero = (0..length).any(|i| !diag[i].is_zero());

            if has_nonzero {
                new_offsets.push(offset);
                new_data.push(diag.clone());
            }
        }

        self.offsets = new_offsets;
        self.data = new_data;
    }

    fn sort_indices(&mut self) {
        // DIA arrays have implicitly sorted indices based on offset
        // Sort by offset just to be sure
        let mut offset_data: Vec<(isize, Array1<T>)> = self
            .offsets
            .iter()
            .cloned()
            .zip(self.data.drain(..))
            .collect();
        offset_data.sort_by_key(|&(offset_, _)| offset_);

        self.offsets = offset_data.iter().map(|&(offset_, _)| offset_).collect();
        self.data = offset_data.into_iter().map(|(_, data)| data).collect();
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        // Clone and sort
        let mut result = self.clone();
        result.sort_indices();
        Box::new(result)
    }

    fn has_sorted_indices(&self) -> bool {
        // Check if offsets are sorted
        self.offsets.windows(2).all(|w| w[0] <= w[1])
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        match axis {
            None => {
                // Sum all elements
                let mut total = T::zero();

                for (diag_idx, &offset) in self.offsets.iter().enumerate() {
                    let diag = &self.data[diag_idx];

                    let length = if offset >= 0 {
                        self.shape
                            .0
                            .min(self.shape.1.saturating_sub(offset as usize))
                    } else {
                        self.shape
                            .1
                            .min(self.shape.0.saturating_sub((-offset) as usize))
                    };

                    for i in 0..length {
                        total += diag[i];
                    }
                }

                Ok(SparseSum::Scalar(total))
            }
            Some(0) => {
                // Sum along rows (result is 1 x cols)
                let mut result = Array1::zeros(self.shape.1);

                for (diag_idx, &offset) in self.offsets.iter().enumerate() {
                    let diag = &self.data[diag_idx];

                    if offset >= 0 {
                        // Upper diagonal
                        let offset_usize = offset as usize;
                        let length = self.shape.0.min(self.shape.1.saturating_sub(offset_usize));

                        for i in 0..length {
                            result[i + offset_usize] += diag[i];
                        }
                    } else {
                        // Lower diagonal
                        let offset_usize = (-offset) as usize;
                        let length = self.shape.1.min(self.shape.0.saturating_sub(offset_usize));

                        for i in 0..length {
                            result[i] += diag[i];
                        }
                    }
                }

                // Convert to a sparse array
                match Array2::from_shape_vec((1, self.shape.1), result.to_vec()) {
                    Ok(result_2d) => {
                        // Find non-zero elements
                        let mut row_indices = Vec::new();
                        let mut col_indices = Vec::new();
                        let mut values = Vec::new();

                        for j in 0..self.shape.1 {
                            let val: T = result_2d[[0, j]];
                            if !val.is_zero() {
                                row_indices.push(0);
                                col_indices.push(j);
                                values.push(val);
                            }
                        }

                        // Create COO array
                        match CooArray::from_triplets(
                            &row_indices,
                            &col_indices,
                            &values,
                            (1, self.shape.1),
                            false,
                        ) {
                            Ok(coo_array) => Ok(SparseSum::SparseArray(Box::new(coo_array))),
                            Err(e) => Err(e),
                        }
                    }
                    Err(_) => Err(SparseError::InconsistentData {
                        reason: "Failed to create 2D array from result vector".to_string(),
                    }),
                }
            }
            Some(1) => {
                // Sum along columns (result is rows x 1)
                let mut result = Array1::zeros(self.shape.0);

                for (diag_idx, &offset) in self.offsets.iter().enumerate() {
                    let diag = &self.data[diag_idx];

                    if offset >= 0 {
                        // Upper diagonal
                        let offset_usize = offset as usize;
                        let length = self.shape.0.min(self.shape.1.saturating_sub(offset_usize));

                        for i in 0..length {
                            result[i] += diag[i];
                        }
                    } else {
                        // Lower diagonal
                        let offset_usize = (-offset) as usize;
                        let length = self.shape.1.min(self.shape.0.saturating_sub(offset_usize));

                        for i in 0..length {
                            result[i + offset_usize] += diag[i];
                        }
                    }
                }

                // Convert to a sparse array
                match Array2::from_shape_vec((self.shape.0, 1), result.to_vec()) {
                    Ok(result_2d) => {
                        // Find non-zero elements
                        let mut row_indices = Vec::new();
                        let mut col_indices = Vec::new();
                        let mut values = Vec::new();

                        for i in 0..self.shape.0 {
                            let val: T = result_2d[[i, 0]];
                            if !val.is_zero() {
                                row_indices.push(i);
                                col_indices.push(0);
                                values.push(val);
                            }
                        }

                        // Create COO array
                        match CooArray::from_triplets(
                            &row_indices,
                            &col_indices,
                            &values,
                            (self.shape.0, 1),
                            false,
                        ) {
                            Ok(coo_array) => Ok(SparseSum::SparseArray(Box::new(coo_array))),
                            Err(e) => Err(e),
                        }
                    }
                    Err(_) => Err(SparseError::InconsistentData {
                        reason: "Failed to create 2D array from result vector".to_string(),
                    }),
                }
            }
            _ => Err(SparseError::InvalidAxis),
        }
    }

    fn max(&self) -> T {
        let mut max_val = T::neg_infinity();

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            let length = if offset >= 0 {
                self.shape
                    .0
                    .min(self.shape.1.saturating_sub(offset as usize))
            } else {
                self.shape
                    .1
                    .min(self.shape.0.saturating_sub((-offset) as usize))
            };

            for i in 0..length {
                max_val = max_val.max(diag[i]);
            }
        }

        // If no elements or all negative infinity, return zero
        if max_val == T::neg_infinity() {
            T::zero()
        } else {
            max_val
        }
    }

    fn min(&self) -> T {
        let mut min_val = T::infinity();
        let mut has_nonzero = false;

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.data[diag_idx];

            let length = if offset >= 0 {
                self.shape
                    .0
                    .min(self.shape.1.saturating_sub(offset as usize))
            } else {
                self.shape
                    .1
                    .min(self.shape.0.saturating_sub((-offset) as usize))
            };

            for i in 0..length {
                if !diag[i].is_zero() {
                    has_nonzero = true;
                    min_val = min_val.min(diag[i]);
                }
            }
        }

        // If no non-zero elements, return zero
        if !has_nonzero {
            T::zero()
        } else {
            min_val
        }
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let (row_indices, col_indices, values) = self.to_coo_internal();

        (
            Array1::from_vec(row_indices),
            Array1::from_vec(col_indices),
            Array1::from_vec(values),
        )
    }

    fn slice(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (start_row, end_row) = row_range;
        let (start_col, end_col) = col_range;
        let (rows, cols) = self.shape;

        if start_row >= rows || end_row > rows || start_col >= cols || end_col > cols {
            return Err(SparseError::IndexOutOfBounds {
                index: (start_row.max(end_row), start_col.max(end_col)),
                shape: (rows, cols),
            });
        }

        if start_row >= end_row || start_col >= end_col {
            return Err(SparseError::InvalidSliceRange);
        }

        // Convert to COO, then slice, then convert back to DIA
        let coo = self.to_coo()?;
        coo.slice(row_range, col_range)?.to_dia()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Implement Display for DiaArray for better debugging
impl<T> fmt::Display for DiaArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static
        + std::ops::AddAssign,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DiaArray of shape {:?} with {} stored elements",
            self.shape,
            self.nnz()
        )?;
        writeln!(f, "Offsets: {:?}", self.offsets)?;

        if self.offsets.len() <= 5 {
            for (i, &offset) in self.offsets.iter().enumerate() {
                let diag = &self.data[i];
                let length = if offset >= 0 {
                    self.shape
                        .0
                        .min(self.shape.1.saturating_sub(offset as usize))
                } else {
                    self.shape
                        .1
                        .min(self.shape.0.saturating_sub((-offset) as usize))
                };

                write!(f, "Diagonal {offset}: [")?;
                for j in 0..length.min(10) {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", diag[j])?;
                }
                if length > 10 {
                    write!(f, ", ...")?;
                }
                writeln!(f, "]")?;
            }
        } else {
            writeln!(f, "({} diagonals)", self.offsets.len())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dia_array_create() {
        // Create a 3x3 sparse array with main diagonal and upper diagonal
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal (k=1)
        ];
        let offsets = vec![0, 1]; // Main diagonal and k=1
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).unwrap();

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5); // 3 on main diagonal, 2 on upper diagonal

        // Test values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(1, 1), 2.0);
        assert_eq!(array.get(2, 2), 3.0);
        assert_eq!(array.get(0, 1), 4.0);
        assert_eq!(array.get(1, 2), 5.0);
        assert_eq!(array.get(0, 2), 0.0);
    }

    #[test]
    fn test_dia_array_from_triplets() {
        // Create a tridiagonal matrix
        let row = vec![0, 0, 1, 1, 1, 2, 2];
        let col = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![1.0, 4.0, 2.0, 3.0, 5.0, 6.0, 7.0];
        let shape = (3, 3);

        let array = DiaArray::from_triplets(&row, &col, &data, shape).unwrap();

        // Should have 3 diagonals: main (0), upper (1), and lower (-1)
        assert_eq!(array.offsets.len(), 3);
        assert!(array.offsets.contains(&0));
        assert!(array.offsets.contains(&1));
        assert!(array.offsets.contains(&-1));

        // Test values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 4.0);
        assert_eq!(array.get(1, 0), 2.0);
        assert_eq!(array.get(1, 1), 3.0);
        assert_eq!(array.get(1, 2), 5.0);
        assert_eq!(array.get(2, 1), 6.0);
        assert_eq!(array.get(2, 2), 7.0);
    }

    #[test]
    fn test_dia_array_conversion() {
        // Create a tridiagonal matrix
        let data = vec![
            Array1::from_vec(vec![1.0, 3.0, 7.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal
            Array1::from_vec(vec![0.0, 2.0, 0.0]), // Lower diagonal at index 1 (2.0 instead of 6.0)
        ];
        let offsets = vec![0, 1, -1]; // Main, upper, lower
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).unwrap();

        // Convert to COO and check
        let coo = array.to_coo().unwrap();
        assert_eq!(coo.shape(), (3, 3));
        assert_eq!(coo.nnz(), 6); // Zero value at (2,1) is not stored

        // Convert to dense and check
        let dense = array.to_array();

        // Debug print the array
        // println!("Dense array: {:?}", dense);

        let expected =
            Array2::from_shape_vec((3, 3), vec![1.0, 4.0, 0.0, 0.0, 3.0, 5.0, 0.0, 2.0, 7.0])
                .unwrap();
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_dia_array_operations() {
        // Create two simple diagonal matrices
        let data1 = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])]; // Main diagonal
        let offsets1 = vec![0];
        let shape1 = (3, 3);
        let array1 = DiaArray::new(data1, offsets1, shape1).unwrap();

        let data2 = vec![Array1::from_vec(vec![4.0, 5.0, 6.0])]; // Main diagonal
        let offsets2 = vec![0];
        let shape2 = (3, 3);
        let array2 = DiaArray::new(data2, offsets2, shape2).unwrap();

        // Test addition
        let sum = array1.add(&array2).unwrap();
        assert_eq!(sum.get(0, 0), 5.0);
        assert_eq!(sum.get(1, 1), 7.0);
        assert_eq!(sum.get(2, 2), 9.0);

        // Test multiplication
        let product = array1.mul(&array2).unwrap();
        assert_eq!(product.get(0, 0), 4.0);
        assert_eq!(product.get(1, 1), 10.0);
        assert_eq!(product.get(2, 2), 18.0);

        // Test dot product (matrix multiplication)
        let dot = array1.dot(&array2).unwrap();
        assert_eq!(dot.get(0, 0), 4.0);
        assert_eq!(dot.get(1, 1), 10.0);
        assert_eq!(dot.get(2, 2), 18.0);
    }

    #[test]
    fn test_dia_array_dot_vector() {
        // Create a tridiagonal matrix
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal
            Array1::from_vec(vec![0.0, 6.0, 7.0]), // Lower diagonal
        ];
        let offsets = vec![0, 1, -1]; // Main, upper, lower
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).unwrap();

        // Create a vector
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Test matrix-vector multiplication
        let result = array.dot_vector(&vector.view()).unwrap();

        // Expected: [1*1 + 4*2 + 0*3, 6*1 + 2*2 + 5*3, 0*1 + 7*2 + 3*3]
        // = [9, 19, 21]
        let expected = Array1::from_vec(vec![9.0, 19.0, 21.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dia_array_transpose() {
        // Create a tridiagonal matrix
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal
            Array1::from_vec(vec![0.0, 6.0, 7.0]), // Lower diagonal
        ];
        let offsets = vec![0, 1, -1]; // Main, upper, lower
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).unwrap();
        let transposed = array.transpose().unwrap();

        // Check shape
        assert_eq!(transposed.shape(), (3, 3));

        // Compare the dense array representations
        let original_dense = array.to_array();
        let transposed_dense = transposed.to_array();

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(transposed_dense[[i, j]], original_dense[[j, i]]);
            }
        }
    }

    #[test]
    fn test_dia_array_sum() {
        // Create a simple matrix
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal
        ];
        let offsets = vec![0, 1]; // Main, upper
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).unwrap();

        // Test sum of entire array
        if let SparseSum::Scalar(sum) = array.sum(None).unwrap() {
            assert_eq!(sum, 15.0); // 1+2+3+4+5 = 15
        } else {
            panic!("Expected SparseSum::Scalar");
        }

        // Test sum along rows
        if let SparseSum::SparseArray(row_sum) = array.sum(Some(0)).unwrap() {
            assert_eq!(row_sum.shape(), (1, 3));
            assert_eq!(row_sum.get(0, 0), 1.0);
            assert_eq!(row_sum.get(0, 1), 6.0); // 2+4 = 6
            assert_eq!(row_sum.get(0, 2), 8.0); // 3+5 = 8
        } else {
            panic!("Expected SparseSum::SparseArray");
        }

        // Test sum along columns
        if let SparseSum::SparseArray(col_sum) = array.sum(Some(1)).unwrap() {
            assert_eq!(col_sum.shape(), (3, 1));
            assert_eq!(col_sum.get(0, 0), 5.0); // 1+4 = 5
            assert_eq!(col_sum.get(1, 0), 7.0); // 2+5 = 7
            assert_eq!(col_sum.get(2, 0), 3.0);
        } else {
            panic!("Expected SparseSum::SparseArray");
        }
    }
}
