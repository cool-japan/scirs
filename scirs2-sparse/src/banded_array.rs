//! Banded matrix format for sparse matrices
//!
//! Banded matrices are matrices where all non-zero elements are within a band
//! around the main diagonal. This format is highly efficient for matrices with
//! this structure, especially for solving linear systems.

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::{Debug, Display};

/// Banded array format for sparse matrices
///
/// The BandedArray format stores only the non-zero bands of a matrix.
/// The data is stored in a 2D array where each row represents a diagonal
/// and each column represents the matrix row.
///
/// For a matrix with lower bandwidth `kl` and upper bandwidth `ku`,
/// the data array has shape `(kl + ku + 1, n)` where `n` is the number
/// of matrix rows.
#[derive(Debug, Clone)]
pub struct BandedArray<T>
where
    T: std::ops::AddAssign + std::fmt::Display,
{
    /// Band data stored as (kl + ku + 1, n) array
    data: Array2<T>,
    /// Lower bandwidth (number of subdiagonals)
    kl: usize,
    /// Upper bandwidth (number of superdiagonals)
    ku: usize,
    /// Matrix shape
    shape: (usize, usize),
}

impl<T> BandedArray<T>
where
    T: Float + Debug + Display + Copy + Zero + One + Send + Sync + 'static + std::ops::AddAssign,
{
    /// Create a new banded array
    pub fn new(data: Array2<T>, kl: usize, ku: usize, shape: (usize, usize)) -> SparseResult<Self> {
        let expected_bands = kl + ku + 1;
        let (bands, cols) = data.dim();

        if bands != expected_bands {
            return Err(SparseError::ValueError(format!(
                "Data array should have {expected_bands} bands, got {bands}"
            )));
        }

        if cols != shape.0 {
            return Err(SparseError::ValueError(format!(
                "Data array columns {} should match matrix rows {}",
                cols, shape.0
            )));
        }

        Ok(Self {
            data,
            kl,
            ku,
            shape,
        })
    }

    /// Create a new zero banded array
    pub fn zeros(shape: (usize, usize), kl: usize, ku: usize) -> Self {
        let bands = kl + ku + 1;
        let data = Array2::zeros((bands, shape.0));

        Self {
            data,
            kl,
            ku,
            shape,
        }
    }

    /// Create a new identity banded array
    pub fn eye(n: usize, kl: usize, ku: usize) -> Self {
        let mut result = Self::zeros((n, n), kl, ku);

        // Set main diagonal to 1
        for i in 0..n {
            result.set_unchecked(i, i, T::one());
        }

        result
    }

    /// Create from triplet format (row, col, data)
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        data: &[T],
        shape: (usize, usize),
        kl: usize,
        ku: usize,
    ) -> SparseResult<Self> {
        let mut result = Self::zeros(shape, kl, ku);

        for (&row, (&col, &value)) in rows.iter().zip(cols.iter().zip(data.iter())) {
            if row >= shape.0 || col >= shape.1 {
                return Err(SparseError::ValueError("Index out of bounds".to_string()));
            }

            if result.is_in_band(row, col) {
                result.set_unchecked(row, col, value);
            } else if !value.is_zero() {
                return Err(SparseError::ValueError(format!(
                    "Non-zero element at ({row}, {col}) is outside band structure"
                )));
            }
        }

        Ok(result)
    }

    /// Create tridiagonal matrix
    pub fn tridiagonal(diag: &[T], lower: &[T], upper: &[T]) -> SparseResult<Self> {
        let n = diag.len();

        if lower.len() != n - 1 || upper.len() != n - 1 {
            return Err(SparseError::ValueError(
                "Off-diagonal arrays must have length n-1".to_string(),
            ));
        }

        let mut result = Self::zeros((n, n), 1, 1);

        // Main diagonal
        for (i, &val) in diag.iter().enumerate() {
            result.set_unchecked(i, i, val);
        }

        // Lower diagonal
        for (i, &val) in lower.iter().enumerate() {
            result.set_unchecked(i + 1, i, val);
        }

        // Upper diagonal
        for (i, &val) in upper.iter().enumerate() {
            result.set_unchecked(i, i + 1, val);
        }

        Ok(result)
    }

    /// Check if an element is within the band structure
    pub fn is_in_band(&self, row: usize, col: usize) -> bool {
        if row >= self.shape.0 || col >= self.shape.1 {
            return false;
        }

        let diff = col as isize - row as isize;
        diff >= -(self.kl as isize) && diff <= self.ku as isize
    }

    /// Set an element (unchecked for performance)
    pub fn set_unchecked(&mut self, row: usize, col: usize, value: T) {
        if let Some(band_idx) = self
            .ku
            .checked_add(row)
            .and_then(|sum| sum.checked_sub(col))
        {
            if band_idx < self.data.nrows() {
                self.data[[band_idx, col]] = value;
            }
        }
    }

    /// Set an element with bounds and band checking
    pub fn set_direct(&mut self, row: usize, col: usize, value: T) -> SparseResult<()> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::ValueError(format!(
                "Index ({}, {}) out of bounds for shape {:?}",
                row, col, self.shape
            )));
        }

        if !self.is_in_band(row, col) {
            if !value.is_zero() {
                return Err(SparseError::ValueError(format!(
                    "Cannot set non-zero value {value} at ({row}, {col}) - outside band structure"
                )));
            }
            // For zero values outside the band, just ignore (they're implicitly zero)
            return Ok(());
        }

        self.set_unchecked(row, col, value);
        Ok(())
    }

    /// Get the raw band data
    pub fn data(&self) -> &Array2<T> {
        &self.data
    }

    /// Get mutable reference to the raw band data
    pub fn data_mut(&mut self) -> &mut Array2<T> {
        &mut self.data
    }

    /// Get lower bandwidth
    pub fn kl(&self) -> usize {
        self.kl
    }

    /// Get upper bandwidth
    pub fn ku(&self) -> usize {
        self.ku
    }

    /// Solve a banded linear system using LU decomposition
    pub fn solve(&self, b: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        if self.shape.0 != self.shape.1 {
            return Err(SparseError::ValueError(
                "Matrix must be square for solving".to_string(),
            ));
        }

        if b.len() != self.shape.0 {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.0,
                found: b.len(),
            });
        }

        // Perform banded LU decomposition
        let (l, u, p) = self.lu_decomposition()?;

        // Solve L * U * x = P * b
        let pb = apply_permutation(&p, b);
        let y = l.forward_substitution(&pb.view())?;
        let x = u.back_substitution(&y.view())?;

        Ok(x)
    }

    /// LU decomposition for banded matrices
    pub fn lu_decomposition(&self) -> SparseResult<(BandedArray<T>, BandedArray<T>, Vec<usize>)> {
        let n = self.shape.0;
        let mut l = BandedArray::zeros((n, n), self.kl, 0); // Lower triangular
        let mut u = self.clone(); // Will become upper triangular
        let mut p: Vec<usize> = (0..n).collect(); // Permutation vector

        // Gaussian elimination with partial pivoting
        for k in 0..(n - 1) {
            // Find pivot within the band
            let mut pivot_row = k;
            let mut max_val = u.get(k, k).abs();

            for i in (k + 1)..(k + 1 + self.kl).min(n) {
                let val = u.get(i, k).abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
            }

            // Swap rows if needed
            if pivot_row != k {
                u.swap_rows(k, pivot_row);
                l.swap_rows(k, pivot_row);
                p.swap(k, pivot_row);
            }

            let pivot = u.get(k, k);
            if pivot.is_zero() {
                return Err(SparseError::ValueError("Matrix is singular".to_string()));
            }

            // Eliminate column
            for i in (k + 1)..(k + 1 + self.kl).min(n) {
                let factor = u.get(i, k) / pivot;
                l.set_unchecked(i, k, factor);

                for j in k..(k + 1 + self.ku).min(n) {
                    let val = u.get(i, j) - factor * u.get(k, j);
                    if u.is_in_band(i, j) {
                        u.set_unchecked(i, j, val);
                    }
                }
            }
        }

        // Set L diagonal to 1
        for i in 0..n {
            l.set_unchecked(i, i, T::one());
        }

        Ok((l, u, p))
    }

    /// Forward substitution for lower triangular banded matrix
    pub fn forward_substitution(&self, b: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let n = self.shape.0;
        let mut x = Array1::zeros(n);

        for i in 0..n {
            let mut sum = T::zero();
            let start = i.saturating_sub(self.kl);

            for j in start..i {
                sum += self.get(i, j) * x[j];
            }

            x[i] = (b[i] - sum) / self.get(i, i);
        }

        Ok(x)
    }

    /// Back substitution for upper triangular banded matrix
    pub fn back_substitution(&self, b: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        let n = self.shape.0;
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = T::zero();
            let end = (i + self.ku + 1).min(n);

            for j in (i + 1)..end {
                sum += self.get(i, j) * x[j];
            }

            x[i] = (b[i] - sum) / self.get(i, i);
        }

        Ok(x)
    }

    /// Swap two rows in the banded matrix
    fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }

        // Determine the range of columns to swap
        let min_col = i.saturating_sub(self.kl).max(j.saturating_sub(self.kl));
        let max_col = (i + self.ku).min(j + self.ku).min(self.shape.1 - 1);

        for col in min_col..=max_col {
            if self.is_in_band(i, col) && self.is_in_band(j, col) {
                let temp = self.get(i, col);
                self.set_unchecked(i, col, self.get(j, col));
                self.set_unchecked(j, col, temp);
            }
        }
    }

    /// Matrix-vector multiplication optimized for banded structure
    pub fn matvec(&self, x: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        if x.len() != self.shape.1 {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.1,
                found: x.len(),
            });
        }

        let mut y = Array1::zeros(self.shape.0);

        for i in 0..self.shape.0 {
            let start_col = i.saturating_sub(self.kl);
            let end_col = (i + self.ku + 1).min(self.shape.1);

            for j in start_col..end_col {
                y[i] += self.get(i, j) * x[j];
            }
        }

        Ok(y)
    }
}

impl<T> SparseArray<T> for BandedArray<T>
where
    T: Float + Debug + Display + Copy + Zero + One + Send + Sync + 'static + std::ops::AddAssign,
{
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn nnz(&self) -> usize {
        let mut count = 0;
        for band in 0..(self.kl + self.ku + 1) {
            for col in 0..self.shape.0 {
                if !self.data[[band, col]].is_zero() {
                    count += 1;
                }
            }
        }
        count
    }

    fn get(&self, row: usize, col: usize) -> T {
        if !self.is_in_band(row, col) {
            return T::zero();
        }

        if let Some(band_idx) = self
            .ku
            .checked_add(row)
            .and_then(|sum| sum.checked_sub(col))
        {
            if band_idx < self.kl + self.ku + 1 && col < self.shape.1 {
                self.data[[band_idx, col]]
            } else {
                T::zero()
            }
        } else {
            T::zero()
        }
    }

    fn find(&self) -> (Array1<usize>, Array1<usize>, Array1<T>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for i in 0..self.shape.0 {
            let start_col = i.saturating_sub(self.kl);
            let end_col = (i + self.ku + 1).min(self.shape.1);

            for j in start_col..end_col {
                let val = self.get(i, j);
                if !val.is_zero() {
                    rows.push(i);
                    cols.push(j);
                    data.push(val);
                }
            }
        }

        (
            Array1::from_vec(rows),
            Array1::from_vec(cols),
            Array1::from_vec(data),
        )
    }

    fn to_array(&self) -> Array2<T> {
        let mut result = Array2::zeros(self.shape);

        for i in 0..self.shape.0 {
            let start_col = i.saturating_sub(self.kl);
            let end_col = (i + self.ku + 1).min(self.shape.1);

            for j in start_col..end_col {
                result[[i, j]] = self.get(i, j);
            }
        }

        result
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For now, convert to dense and multiply
        let a_dense = self.to_array();
        let b_dense = other.to_array();

        if a_dense.ncols() != b_dense.nrows() {
            return Err(SparseError::DimensionMismatch {
                expected: a_dense.ncols(),
                found: b_dense.nrows(),
            });
        }

        let result = a_dense.dot(&b_dense);

        // Try to convert back to banded format if possible
        // For simplicity, convert to CSR for now
        let (rows, cols, data) = array_to_triplets(&result);
        let csr =
            crate::csr_array::CsrArray::from_triplets(&rows, &cols, &data, result.dim(), false)?;

        Ok(Box::new(csr))
    }

    fn dtype(&self) -> &str {
        std::any::type_name::<T>()
    }

    fn toarray(&self) -> Array2<T> {
        self.to_array()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (rows, cols, data) = self.find();
        let coo = crate::coo_array::CooArray::from_triplets(
            rows.as_slice().unwrap(),
            cols.as_slice().unwrap(),
            data.as_slice().unwrap(),
            self.shape,
            false,
        )?;
        Ok(Box::new(coo))
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (rows, cols, data) = self.find();
        let csr = crate::csr_array::CsrArray::from_triplets(
            rows.as_slice().unwrap(),
            cols.as_slice().unwrap(),
            data.as_slice().unwrap(),
            self.shape,
            false,
        )?;
        Ok(Box::new(csr))
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (rows, cols, data) = self.find();
        let csc = crate::csc_array::CscArray::from_triplets(
            rows.as_slice().unwrap(),
            cols.as_slice().unwrap(),
            data.as_slice().unwrap(),
            self.shape,
            false,
        )?;
        Ok(Box::new(csc))
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (rows, cols, data) = self.find();
        let mut dok = crate::dok_array::DokArray::new(self.shape);
        for ((row, col), &val) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            dok.set(*row, *col, val)?;
        }
        Ok(Box::new(dok))
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let mut lil = crate::lil_array::LilArray::new(self.shape);
        for i in 0..self.shape.0 {
            let start_col = i.saturating_sub(self.kl);
            let end_col = (i + self.ku + 1).min(self.shape.1);

            for j in start_col..end_col {
                let val = self.get(i, j);
                if !val.is_zero() {
                    lil.set(i, j, val)?;
                }
            }
        }
        Ok(Box::new(lil))
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert banded to diagonal format
        let mut diagonals = Vec::new();
        let mut offsets = Vec::new();

        for band in 0..(self.kl + self.ku + 1) {
            let offset = (band as isize) - (self.ku as isize);
            let mut diagonal = Vec::new();

            for row in 0..self.shape.0 {
                if row < self.shape.0 && band < self.data.dim().0 {
                    diagonal.push(self.data[[band, row]]);
                }
            }

            if diagonal.iter().any(|&x| !x.is_zero()) {
                diagonals.push(Array1::from_vec(diagonal));
                offsets.push(offset);
            }
        }

        let dia = crate::dia_array::DiaArray::new(diagonals, offsets, self.shape)?;
        Ok(Box::new(dia))
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR first, then to BSR
        let csr = self.to_csr()?;
        csr.to_bsr()
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.0 * self.shape.1,
                found: other.shape().0 * other.shape().1,
            });
        }

        let a_dense = self.to_array();
        let b_dense = other.to_array();
        let result = a_dense + b_dense;

        let (rows, cols, data) = array_to_triplets(&result);
        let csr =
            crate::csr_array::CsrArray::from_triplets(&rows, &cols, &data, result.dim(), false)?;
        Ok(Box::new(csr))
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.0 * self.shape.1,
                found: other.shape().0 * other.shape().1,
            });
        }

        let a_dense = self.to_array();
        let b_dense = other.to_array();
        let result = a_dense - b_dense;

        let (rows, cols, data) = array_to_triplets(&result);
        let csr =
            crate::csr_array::CsrArray::from_triplets(&rows, &cols, &data, result.dim(), false)?;
        Ok(Box::new(csr))
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.0 * self.shape.1,
                found: other.shape().0 * other.shape().1,
            });
        }

        let a_dense = self.to_array();
        let b_dense = other.to_array();
        let result = a_dense * b_dense;

        let (rows, cols, data) = array_to_triplets(&result);
        let csr =
            crate::csr_array::CsrArray::from_triplets(&rows, &cols, &data, result.dim(), false)?;
        Ok(Box::new(csr))
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        if self.shape != other.shape() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.0 * self.shape.1,
                found: other.shape().0 * other.shape().1,
            });
        }

        let a_dense = self.to_array();
        let b_dense = other.to_array();
        let result = a_dense / b_dense;

        let (rows, cols, data) = array_to_triplets(&result);
        let csr =
            crate::csr_array::CsrArray::from_triplets(&rows, &cols, &data, result.dim(), false)?;
        Ok(Box::new(csr))
    }

    fn dot_vector(&self, other: &ArrayView1<T>) -> SparseResult<Array1<T>> {
        if self.shape.1 != other.len() {
            return Err(SparseError::DimensionMismatch {
                expected: self.shape.1,
                found: other.len(),
            });
        }

        let mut result = Array1::zeros(self.shape.0);

        for i in 0..self.shape.0 {
            let start_col = i.saturating_sub(self.kl);
            let end_col = (i + self.ku + 1).min(self.shape.1);

            for j in start_col..end_col {
                let val = self.get(i, j);
                if !val.is_zero() {
                    result[i] += val * other[j];
                }
            }
        }

        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let mut transposed = BandedArray::zeros((self.shape.1, self.shape.0), self.ku, self.kl);

        for i in 0..self.shape.0 {
            let start_col = i.saturating_sub(self.kl);
            let end_col = (i + self.ku + 1).min(self.shape.1);

            for j in start_col..end_col {
                let val = self.get(i, j);
                if !val.is_zero() {
                    transposed.set_direct(j, i, val)?;
                }
            }
        }

        Ok(Box::new(transposed))
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn set(&mut self, i: usize, j: usize, value: T) -> SparseResult<()> {
        self.set_direct(i, j, value)
    }

    fn eliminate_zeros(&mut self) {
        // For banded arrays, we typically don't eliminate structural zeros
        // as they maintain the band structure
    }

    fn sort_indices(&mut self) {
        // Banded arrays maintain sorted indices by structure
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        self.copy()
    }

    fn has_sorted_indices(&self) -> bool {
        true // Banded arrays always have sorted indices by structure
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<crate::sparray::SparseSum<T>> {
        match axis {
            None => {
                // Sum all elements
                let total = self.data.iter().fold(T::zero(), |acc, &x| acc + x);
                Ok(crate::sparray::SparseSum::Scalar(total))
            }
            Some(0) => {
                // Sum along rows (result is column vector)
                let mut result: Array1<T> = Array1::zeros(self.shape.1);
                for i in 0..self.shape.0 {
                    let start_col = i.saturating_sub(self.kl);
                    let end_col = (i + self.ku + 1).min(self.shape.1);

                    for j in start_col..end_col {
                        let val = self.get(i, j);
                        result[j] += val;
                    }
                }
                // Convert to CSR format
                let mut data = Vec::new();
                let mut indices = Vec::new();
                let mut indptr = vec![0];

                for (col, &val) in result.iter().enumerate() {
                    if !val.is_zero() {
                        data.push(val);
                        indices.push(col);
                    }
                }
                indptr.push(data.len());

                let result_array = crate::csr_array::CsrArray::new(
                    Array1::from_vec(data),
                    Array1::from_vec(indices),
                    Array1::from_vec(indptr),
                    (1, self.shape.1),
                )?;

                Ok(crate::sparray::SparseSum::SparseArray(Box::new(
                    result_array,
                )))
            }
            Some(1) => {
                // Sum along columns (result is column vector)
                let mut result: Array1<T> = Array1::zeros(self.shape.0);
                for i in 0..self.shape.0 {
                    let start_col = i.saturating_sub(self.kl);
                    let end_col = (i + self.ku + 1).min(self.shape.1);

                    for j in start_col..end_col {
                        let val = self.get(i, j);
                        result[i] += val;
                    }
                }
                // Convert to CSR format (column vector)
                let mut data = Vec::new();
                let mut indices = Vec::new();
                let mut indptr = vec![0];

                for &val in result.iter() {
                    if !val.is_zero() {
                        data.push(val);
                        indices.push(0); // All values are in column 0
                    }
                    indptr.push(data.len());
                }

                let result_array = crate::csr_array::CsrArray::new(
                    Array1::from_vec(data),
                    Array1::from_vec(indices),
                    Array1::from_vec(indptr),
                    (self.shape.0, 1),
                )?;

                Ok(crate::sparray::SparseSum::SparseArray(Box::new(
                    result_array,
                )))
            }
            Some(_) => Err(SparseError::ValueError("Invalid axis".to_string())),
        }
    }

    fn max(&self) -> T {
        self.data
            .iter()
            .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b })
    }

    fn min(&self) -> T {
        self.data
            .iter()
            .fold(T::infinity(), |a, &b| if a < b { a } else { b })
    }

    fn slice(
        &self,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        let (start_row, end_row) = row_range;
        let (start_col, end_col) = col_range;

        if end_row > self.shape.0 || end_col > self.shape.1 {
            return Err(SparseError::ValueError(
                "Slice bounds exceed matrix dimensions".to_string(),
            ));
        }

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for i in start_row..end_row {
            let band_start_col = i.saturating_sub(self.kl).max(start_col);
            let band_end_col = (i + self.ku + 1).min(self.shape.1).min(end_col);

            for j in band_start_col..band_end_col {
                let val = self.get(i, j);
                if !val.is_zero() {
                    rows.push(i - start_row);
                    cols.push(j - start_col);
                    data.push(val);
                }
            }
        }

        let shape = (end_row - start_row, end_col - start_col);
        let csr = crate::csr_array::CsrArray::from_triplets(&rows, &cols, &data, shape, false)?;
        Ok(Box::new(csr))
    }
}

/// Apply permutation to a vector
#[allow(dead_code)]
fn apply_permutation<T: Copy + Zero>(p: &[usize], v: &ArrayView1<T>) -> Array1<T> {
    let mut result = Array1::zeros(v.len());
    for (i, &pi) in p.iter().enumerate() {
        result[i] = v[pi];
    }
    result
}

/// Convert dense array to triplet format
#[allow(dead_code)]
fn array_to_triplets<T: Float + Debug + Copy + Zero>(
    array: &Array2<T>,
) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for ((i, j), &val) in array.indexed_iter() {
        if !val.is_zero() {
            rows.push(i);
            cols.push(j);
            data.push(val);
        }
    }

    (rows, cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_banded_array_creation() {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 1.0, 2.0, 3.0, // Upper diagonal
                4.0, 5.0, 6.0, 7.0, // Main diagonal
                8.0, 9.0, 10.0, 0.0, // Lower diagonal
            ],
        )
        .unwrap();

        let banded = BandedArray::new(data, 1, 1, (4, 4)).unwrap();

        assert_eq!(banded.shape(), (4, 4));
        assert_eq!(banded.kl(), 1);
        assert_eq!(banded.ku(), 1);

        // Check main diagonal
        assert_eq!(banded.get(0, 0), 4.0);
        assert_eq!(banded.get(1, 1), 5.0);
        assert_eq!(banded.get(2, 2), 6.0);
        assert_eq!(banded.get(3, 3), 7.0);

        // Check upper diagonal
        assert_eq!(banded.get(0, 1), 1.0);
        assert_eq!(banded.get(1, 2), 2.0);
        assert_eq!(banded.get(2, 3), 3.0);

        // Check lower diagonal
        assert_eq!(banded.get(1, 0), 8.0);
        assert_eq!(banded.get(2, 1), 9.0);
        assert_eq!(banded.get(3, 2), 10.0);

        // Check out-of-band elements
        assert_eq!(banded.get(0, 2), 0.0);
        assert_eq!(banded.get(2, 0), 0.0);
    }

    #[test]
    fn test_tridiagonal_matrix() {
        let diag = vec![2.0, 3.0, 4.0];
        let lower = vec![1.0, 1.0];
        let upper = vec![5.0, 6.0];

        let banded = BandedArray::tridiagonal(&diag, &lower, &upper).unwrap();

        assert_eq!(banded.shape(), (3, 3));
        assert_eq!(banded.get(0, 0), 2.0);
        assert_eq!(banded.get(1, 1), 3.0);
        assert_eq!(banded.get(2, 2), 4.0);
        assert_eq!(banded.get(1, 0), 1.0);
        assert_eq!(banded.get(2, 1), 1.0);
        assert_eq!(banded.get(0, 1), 5.0);
        assert_eq!(banded.get(1, 2), 6.0);
    }

    #[test]
    fn test_banded_matvec() {
        let diag = vec![2.0, 3.0, 4.0];
        let lower = vec![1.0, 1.0];
        let upper = vec![5.0, 6.0];

        let banded = BandedArray::tridiagonal(&diag, &lower, &upper).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let y = banded.matvec(&x.view()).unwrap();

        // Manual calculation:
        // [2 5 0] [1]   [2*1 + 5*2 + 0*3] = [12]
        // [1 3 6] [2] = [1*1 + 3*2 + 6*3] = [25]
        // [0 1 4] [3]   [0*1 + 1*2 + 4*3] = [14]

        assert_relative_eq!(y[0], 12.0);
        assert_relative_eq!(y[1], 25.0);
        assert_relative_eq!(y[2], 14.0);
    }

    #[test]
    fn test_banded_solve() {
        // Create a simple tridiagonal system
        let diag = vec![2.0, 2.0, 2.0];
        let lower = vec![-1.0, -1.0];
        let upper = vec![-1.0, -1.0];

        let banded = BandedArray::tridiagonal(&diag, &lower, &upper).unwrap();
        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let x = banded.solve(&b.view()).unwrap();

        // Verify solution by computing A*x
        let ax = banded.matvec(&x.view()).unwrap();

        for i in 0..3 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_is_in_band() {
        let banded = BandedArray::<f64>::zeros((5, 5), 2, 1);

        // Main diagonal should be in band
        assert!(banded.is_in_band(2, 2));

        // One position above main diagonal
        assert!(banded.is_in_band(2, 3));

        // Two positions below main diagonal
        assert!(banded.is_in_band(2, 0));

        // Outside band
        assert!(!banded.is_in_band(0, 2));
        assert!(!banded.is_in_band(4, 0));
    }

    #[test]
    fn test_eye_matrix() {
        let eye = BandedArray::<f64>::eye(3, 1, 1);

        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(2, 2), 1.0);
        assert_eq!(eye.get(0, 1), 0.0);
        assert_eq!(eye.get(1, 0), 0.0);

        assert_eq!(eye.nnz(), 3);
    }
}
