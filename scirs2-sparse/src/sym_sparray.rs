//! Symmetric Sparse Array trait
//!
//! This module defines a trait for symmetric sparse array implementations.
//! It extends the base SparseArray trait with methods specific to symmetric
//! matrices.

use crate::coo_array::CooArray;
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::{SparseArray, SparseSum};
use crate::sym_coo::SymCooArray;
use crate::sym_csr::SymCsrArray;
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Trait for symmetric sparse arrays
///
/// Extends the base SparseArray trait with methods specific to
/// symmetric matrices. Implementations guarantee that the matrix
/// is kept symmetric throughout all operations.
pub trait SymSparseArray<T>: SparseArray<T>
where
    T: Float + Debug + Copy + 'static,
{
    /// Get the number of stored non-zero elements
    ///
    /// For symmetric formats, this returns the number of elements
    /// actually stored (typically lower or upper triangular part).
    ///
    /// # Returns
    ///
    /// Number of stored non-zero elements
    fn nnz_stored(&self) -> usize;

    /// Check if the matrix is guaranteed to be symmetric
    ///
    /// For implementations of this trait, this should always return true.
    /// It's included for consistency with checking functions.
    ///
    /// # Returns
    ///
    /// Always true for SymSparseArray implementations
    fn is_symmetric(&self) -> bool {
        true
    }

    /// Convert to standard CSR array
    ///
    /// # Returns
    ///
    /// A standard CSR array with the full symmetric matrix
    fn to_csr(&self) -> SparseResult<CsrArray<T>>;

    /// Convert to standard COO array
    ///
    /// # Returns
    ///
    /// A standard COO array with the full symmetric matrix
    fn to_coo(&self) -> SparseResult<CooArray<T>>;

    /// Convert to symmetric CSR array
    ///
    /// # Returns
    ///
    /// A symmetric CSR array
    fn to_sym_csr(&self) -> SparseResult<SymCsrArray<T>>;

    /// Convert to symmetric COO array
    ///
    /// # Returns
    ///
    /// A symmetric COO array
    fn to_sym_coo(&self) -> SparseResult<SymCooArray<T>>;
}

/// Implementation of SymSparseArray for SymCsrArray
impl<T> SymSparseArray<T> for SymCsrArray<T>
where
    T: Float
        + Debug
        + Copy
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    fn nnz_stored(&self) -> usize {
        self.inner().nnz_stored()
    }

    fn to_csr(&self) -> SparseResult<CsrArray<T>> {
        self.to_csr_array()
    }

    fn to_coo(&self) -> SparseResult<CooArray<T>> {
        // Extract matrix data
        let csr_inner = self.inner();
        let shape = csr_inner.shape;

        // Convert to triplets format for full symmetric matrix
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for i in 0..shape.0 {
            for j_ptr in csr_inner.indptr[i]..csr_inner.indptr[i + 1] {
                let j = csr_inner.indices[j_ptr];
                let val = csr_inner.data[j_ptr];

                // Add the element itself
                rows.push(i);
                cols.push(j);
                data.push(val);

                // Add its symmetric pair (if not diagonal)
                if i != j {
                    rows.push(j);
                    cols.push(i);
                    data.push(val);
                }
            }
        }

        // Create a COO array from the triplets
        CooArray::from_triplets(&rows, &cols, &data, shape, false)
    }

    fn to_sym_csr(&self) -> SparseResult<SymCsrArray<T>> {
        // Already a SymCsrArray, return a clone
        Ok(self.clone())
    }

    fn to_sym_coo(&self) -> SparseResult<SymCooArray<T>> {
        // Convert the internal SymCsrMatrix to SymCooMatrix
        let csr_inner = self.inner();

        // Extract data from CSR format into COO format (lower triangular part only)
        let mut data = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        for i in 0..csr_inner.shape.0 {
            for j in csr_inner.indptr[i]..csr_inner.indptr[i + 1] {
                let col = csr_inner.indices[j];
                let val = csr_inner.data[j];

                data.push(val);
                rows.push(i);
                cols.push(col);
            }
        }

        use crate::sym_coo::SymCooMatrix;
        let sym_coo = SymCooMatrix::new(data, rows, cols, csr_inner.shape)?;

        Ok(SymCooArray::new(sym_coo))
    }
}

/// Implementation of SymSparseArray for SymCooArray
impl<T> SymSparseArray<T> for SymCooArray<T>
where
    T: Float
        + Debug
        + Copy
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    fn nnz_stored(&self) -> usize {
        self.inner().nnz_stored()
    }

    fn to_csr(&self) -> SparseResult<CsrArray<T>> {
        // Convert to full COO, then to CSR
        let coo = self.to_coo_array()?;
        match coo.to_csr() {
            Ok(boxed_csr) => {
                // Convert Box<dyn SparseArray<T>> to CsrArray<T>
                match boxed_csr.as_any().downcast_ref::<CsrArray<T>>() {
                    Some(csr_array) => Ok(csr_array.clone()),
                    None => Err(SparseError::ConversionError(
                        "Failed to downcast to CsrArray".to_string(),
                    )),
                }
            }
            Err(e) => Err(e),
        }
    }

    fn to_coo(&self) -> SparseResult<CooArray<T>> {
        self.to_coo_array()
    }

    fn to_sym_csr(&self) -> SparseResult<SymCsrArray<T>> {
        // We already have a symmetric COO matrix with only the lower triangular part
        // Let's create a CSR matrix directly from it
        let coo_inner = self.inner();

        // Extract the triplets data
        let data = coo_inner.data.clone();
        let rows = coo_inner.rows.clone();
        let cols = coo_inner.cols.clone();
        let shape = coo_inner.shape;

        // Create a new CsrMatrix from these triplets
        let csr = crate::csr::CsrMatrix::new(data, rows, cols, shape)?;

        // Create a symmetric CSR matrix without checking symmetry (we know it's symmetric)
        use crate::sym_csr::SymCsrMatrix;

        // Extract the lower triangular part (already in the correct format)
        let mut sym_data = Vec::new();
        let mut sym_indices = Vec::new();
        let mut sym_indptr = vec![0];
        let n = shape.0;

        for i in 0..n {
            for j_ptr in csr.indptr[i]..csr.indptr[i + 1] {
                let j = csr.indices[j_ptr];
                let val = csr.data[j_ptr];

                // Only include elements in lower triangular part (including diagonal)
                if j <= i {
                    sym_data.push(val);
                    sym_indices.push(j);
                }
            }

            sym_indptr.push(sym_data.len());
        }

        let sym_csr = SymCsrMatrix::new(sym_data, sym_indptr, sym_indices, shape)?;

        Ok(SymCsrArray::new(sym_csr))
    }

    fn to_sym_coo(&self) -> SparseResult<SymCooArray<T>> {
        // Already a SymCooArray, return a clone
        Ok(self.clone())
    }
}

/// Implementation of SparseArray for SymCsrArray
impl<T> SparseArray<T> for SymCsrArray<T>
where
    T: Float
        + Debug
        + Copy
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let coo_array = <Self as SymSparseArray<T>>::to_coo(self)?;
        Ok(Box::new(coo_array))
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CsrArray (full matrix)
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        Ok(Box::new(csr))
    }
    fn shape(&self) -> (usize, usize) {
        self.inner().shape()
    }

    fn nnz(&self) -> usize {
        self.inner().nnz()
    }

    fn dtype(&self) -> &str {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            "f32"
        } else {
            "f64"
        }
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.inner().get(row, col)
    }

    fn to_array(&self) -> ndarray::Array2<T> {
        // Convert to dense vector of vectors, then to ndarray
        let dense = self.inner().to_dense();
        let mut array = ndarray::Array2::zeros(self.shape());

        for i in 0..dense.len() {
            for j in 0..dense[i].len() {
                array[[i, j]] = dense[i][j];
            }
        }

        array
    }

    fn toarray(&self) -> ndarray::Array2<T> {
        self.to_array()
    }

    fn set(&mut self, _i: usize, _j: usize, _value: T) -> SparseResult<()> {
        Err(SparseError::NotImplemented(
            "Setting individual elements in SymCsrArray is not supported. Convert to another format first.".to_string()
        ))
    }

    fn dot_vector(&self, other: &ndarray::ArrayView1<T>) -> SparseResult<ndarray::Array1<T>> {
        // Use optimized symmetric matrix-vector product
        crate::sym_ops::sym_csr_matvec(self.inner(), other)
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Subtraction will preserve symmetry if other is also symmetric
        // For simplicity, we'll convert to CSR, perform subtraction, and convert back
        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let result = self_csr.sub(other)?;

        // For now, we return the CSR result without converting back
        Ok(result)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Division may not preserve symmetry, so we'll just return a CSR result
        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        self_csr.div(other)
    }

    fn eliminate_zeros(&mut self) {
        // No-op for SymCsrArray as it already maintains minimal storage
    }

    fn sort_indices(&mut self) {
        // No-op for SymCsrArray as indices are already sorted
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        // CSR format typically maintains sorted indices, so return a clone
        Box::new(self.clone())
    }

    fn has_sorted_indices(&self) -> bool {
        true
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        // Convert to CSR and use its implementation
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        SparseArray::<T>::sum(&csr, axis)
    }

    fn max(&self) -> T {
        // Convert to CSR and find the maximum value
        match <Self as SymSparseArray<T>>::to_csr(self) {
            Ok(csr) => SparseArray::<T>::max(&csr),
            Err(_) => T::nan(), // Return NaN if conversion fails
        }
    }

    fn min(&self) -> T {
        // Convert to CSR and find the minimum value
        match <Self as SymSparseArray<T>>::to_csr(self) {
            Ok(csr) => SparseArray::<T>::min(&csr),
            Err(_) => T::nan(), // Return NaN if conversion fails
        }
    }

    fn slice(
        &self,
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Slicing a symmetric matrix may not preserve symmetry
        // Convert to CSR and use its implementation
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        csr.slice(rows, cols)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn find(
        &self,
    ) -> (
        ndarray::Array1<usize>,
        ndarray::Array1<usize>,
        ndarray::Array1<T>,
    ) {
        // To get the full matrix coordinates and values, we need to convert to a full CSR matrix
        match <Self as SymSparseArray<T>>::to_csr(self) {
            Ok(csr) => csr.find(),
            Err(_) => (
                ndarray::Array1::from_vec(Vec::new()),
                ndarray::Array1::from_vec(Vec::new()),
                ndarray::Array1::from_vec(Vec::new()),
            ),
        }
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // First check shapes are compatible
        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape != other_shape {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self_shape.0,
                found: other_shape.0,
            });
        }

        // Convert both to CSR to perform addition
        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let other_csr_box = other.to_csr()?;

        // We need to unbox the other_csr to use it directly
        match other_csr_box.as_any().downcast_ref::<CsrArray<T>>() {
            Some(other_csr) => {
                // Add as standard CSR arrays
                let result = self_csr.add(other_csr)?;

                // Since we expect result to be symmetric (if the other was symmetric),
                // we can try to convert it back to a SymCsrArray

                // First convert the result to CSR Matrix
                use crate::csr::CsrMatrix;

                // We need to get the data from the result
                let (rows, cols, data) = result.find();
                let csr_matrix =
                    CsrMatrix::new(data.to_vec(), rows.to_vec(), cols.to_vec(), result.shape())?;

                // Convert to SymCsrMatrix
                use crate::sym_csr::SymCsrMatrix;
                let sym_csr = SymCsrMatrix::from_csr(&csr_matrix)?;

                // Create and return SymCsrArray
                let sym_csr_array = SymCsrArray::new(sym_csr);
                Ok(Box::new(sym_csr_array) as Box<dyn SparseArray<T>>)
            }
            None => {
                // If we can't downcast, convert both to dense arrays and add them
                let self_dense = self.to_array();
                let other_dense = other.to_array();

                // Create the result dense array
                let mut result_dense = ndarray::Array2::zeros(self_shape);
                for i in 0..self_shape.0 {
                    for j in 0..self_shape.1 {
                        result_dense[[i, j]] = self_dense[[i, j]] + other_dense[[i, j]];
                    }
                }

                // Convert back to CSR
                // Convert result_dense to triplets
                let mut rows = Vec::new();
                let mut cols = Vec::new();
                let mut values = Vec::new();

                for i in 0..self_shape.0 {
                    for j in 0..self_shape.1 {
                        let val = result_dense[[i, j]];
                        if val != T::zero() {
                            rows.push(i);
                            cols.push(j);
                            values.push(val);
                        }
                    }
                }

                let csr = CsrArray::from_triplets(&rows, &cols, &values, self_shape, false)?;
                Ok(Box::new(csr) as Box<dyn SparseArray<T>>)
            }
        }
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // First check shapes are compatible
        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape != other_shape {
            return Err(crate::error::SparseError::DimensionMismatch {
                expected: self_shape.0,
                found: other_shape.0,
            });
        }

        // Convert both to CSR to perform multiplication
        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let other_csr_box = other.to_csr()?;

        // We need to unbox the other_csr to use it directly
        match other_csr_box.as_any().downcast_ref::<CsrArray<T>>() {
            Some(other_csr) => {
                // Multiply as standard CSR arrays (element-wise)
                let result = self_csr.mul(other_csr)?;

                // Element-wise multiplication of symmetric matrices preserves symmetry,
                // so we can convert back to SymCsrArray

                // We need to get the data from the result
                let (rows, cols, data) = result.find();

                // Convert to CsrMatrix
                use crate::csr::CsrMatrix;
                let csr_matrix =
                    CsrMatrix::new(data.to_vec(), rows.to_vec(), cols.to_vec(), result.shape())?;

                // Convert to SymCsrMatrix
                use crate::sym_csr::SymCsrMatrix;
                let sym_csr = SymCsrMatrix::from_csr(&csr_matrix)?;

                // Create and return SymCsrArray
                let sym_csr_array = SymCsrArray::new(sym_csr);
                Ok(Box::new(sym_csr_array) as Box<dyn SparseArray<T>>)
            }
            None => {
                // If we can't downcast, convert both to dense arrays and multiply them
                let self_dense = self.to_array();
                let other_dense = other.to_array();

                // Create the result dense array
                let mut result_dense = ndarray::Array2::zeros(self_shape);
                for i in 0..self_shape.0 {
                    for j in 0..self_shape.1 {
                        result_dense[[i, j]] = self_dense[[i, j]] * other_dense[[i, j]];
                    }
                }

                // Convert back to CSR
                // Convert result_dense to triplets
                let mut rows = Vec::new();
                let mut cols = Vec::new();
                let mut values = Vec::new();

                for i in 0..self_shape.0 {
                    for j in 0..self_shape.1 {
                        let val = result_dense[[i, j]];
                        if val != T::zero() {
                            rows.push(i);
                            cols.push(j);
                            values.push(val);
                        }
                    }
                }

                let csr = CsrArray::from_triplets(&rows, &cols, &values, self_shape, false)?;
                Ok(Box::new(csr) as Box<dyn SparseArray<T>>)
            }
        }
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For matrix multiplication, symmetry is not generally preserved
        // So we just convert to standard CSR, perform the operation, and return a CSR array

        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let result = self_csr.dot(other)?;

        // For dot product of a symmetric matrix with itself, the result is symmetric
        // We could check for this case and return a SymCsrArray, but for now we'll
        // keep it simple and just return the CSR array
        Ok(result)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For symmetric matrices, transpose is the same as the original
        Ok(Box::new(self.clone()))
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to CSC
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_csc()
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to DIA
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_dia()
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to DOK
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_dok()
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to LIL
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_lil()
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to BSR
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_bsr()
    }
}

/// Implementation of SparseArray for SymCooArray
impl<T> SparseArray<T> for SymCooArray<T>
where
    T: Float
        + Debug
        + Copy
        + 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    fn to_coo(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        let coo_array = <Self as SymSparseArray<T>>::to_coo(self)?;
        Ok(Box::new(coo_array))
    }

    fn to_csr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CsrArray (full matrix)
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        Ok(Box::new(csr))
    }
    fn shape(&self) -> (usize, usize) {
        self.inner().shape()
    }

    fn nnz(&self) -> usize {
        self.inner().nnz()
    }

    fn dtype(&self) -> &str {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            "f32"
        } else {
            "f64"
        }
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.inner().get(row, col)
    }

    fn to_array(&self) -> ndarray::Array2<T> {
        // Convert to dense vector of vectors, then to ndarray
        let dense = self.inner().to_dense();
        let mut array = ndarray::Array2::zeros(self.shape());

        for i in 0..dense.len() {
            for j in 0..dense[i].len() {
                array[[i, j]] = dense[i][j];
            }
        }

        array
    }

    fn toarray(&self) -> ndarray::Array2<T> {
        self.to_array()
    }

    fn set(&mut self, _i: usize, _j: usize, _value: T) -> SparseResult<()> {
        Err(SparseError::NotImplemented(
            "Setting individual elements in SymCooArray is not supported. Convert to another format first.".to_string()
        ))
    }

    fn dot_vector(&self, other: &ndarray::ArrayView1<T>) -> SparseResult<ndarray::Array1<T>> {
        // Use optimized symmetric matrix-vector product
        crate::sym_ops::sym_coo_matvec(self.inner(), other)
    }

    fn copy(&self) -> Box<dyn SparseArray<T>> {
        Box::new(self.clone())
    }

    fn sub(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For simplicity, we'll use the CSR implementation
        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        self_csr.sub(other)
    }

    fn div(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For simplicity, we'll use the CSR implementation
        let self_csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        self_csr.div(other)
    }

    fn eliminate_zeros(&mut self) {
        // Not implemented for SymCooArray
        // Could be implemented by filtering out zero values in the future
    }

    fn sort_indices(&mut self) {
        // Not implemented for SymCooArray
        // Could be implemented by sorting indices by row, then column in the future
    }

    fn sorted_indices(&self) -> Box<dyn SparseArray<T>> {
        // Convert to SymCsrArray which has sorted indices
        match self.to_sym_csr() {
            Ok(csr) => Box::new(csr),
            Err(_) => Box::new(self.clone()), // Return self if conversion fails
        }
    }

    fn has_sorted_indices(&self) -> bool {
        false
    }

    fn sum(&self, axis: Option<usize>) -> SparseResult<SparseSum<T>> {
        // Convert to CSR and use its implementation
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        SparseArray::<T>::sum(&csr, axis)
    }

    fn max(&self) -> T {
        // Convert to CSR and find the maximum value
        match <Self as SymSparseArray<T>>::to_csr(self) {
            Ok(csr) => SparseArray::<T>::max(&csr),
            Err(_) => T::nan(), // Return NaN if conversion fails
        }
    }

    fn min(&self) -> T {
        // Convert to CSR and find the minimum value
        match <Self as SymSparseArray<T>>::to_csr(self) {
            Ok(csr) => SparseArray::<T>::min(&csr),
            Err(_) => T::nan(), // Return NaN if conversion fails
        }
    }

    fn slice(
        &self,
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR and use its implementation
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        csr.slice(rows, cols)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn find(
        &self,
    ) -> (
        ndarray::Array1<usize>,
        ndarray::Array1<usize>,
        ndarray::Array1<T>,
    ) {
        // To get the full matrix coordinates and values, we need to convert to a full COO matrix
        match <Self as SymSparseArray<T>>::to_coo(self) {
            Ok(coo) => coo.find(),
            Err(_) => (
                ndarray::Array1::from_vec(Vec::new()),
                ndarray::Array1::from_vec(Vec::new()),
                ndarray::Array1::from_vec(Vec::new()),
            ),
        }
    }

    fn add(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to SymCsrArray and use its implementation
        self.to_sym_csr()?.add(other)
    }

    fn mul(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to SymCsrArray and use its implementation
        self.to_sym_csr()?.mul(other)
    }

    fn dot(&self, other: &dyn SparseArray<T>) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR for dot product
        <Self as SymSparseArray<T>>::to_csr(self)?.dot(other)
    }

    fn transpose(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // For symmetric matrices, transpose is the same as the original
        Ok(Box::new(self.clone()))
    }

    fn to_csc(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to CSC
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_csc()
    }

    fn to_dia(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to DIA
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_dia()
    }

    fn to_dok(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to DOK
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_dok()
    }

    fn to_lil(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to LIL
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_lil()
    }

    fn to_bsr(&self) -> SparseResult<Box<dyn SparseArray<T>>> {
        // Convert to CSR, then to BSR
        let csr = <Self as SymSparseArray<T>>::to_csr(self)?;
        let csr_box: Box<dyn SparseArray<T>> = Box::new(csr);
        csr_box.to_bsr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sym_csr::{SymCsrArray, SymCsrMatrix};

    // Create a simple symmetric matrix for testing
    fn create_test_sym_csr() -> SymCsrArray<f64> {
        // Create a simple symmetric matrix in CSR format
        // [2 1 0]
        // [1 2 3]
        // [0 3 1]

        // Lower triangular part only:
        // [2 0 0]
        // [1 2 0]
        // [0 3 1]

        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let indices = vec![0, 0, 1, 1, 2];
        let indptr = vec![0, 1, 3, 5];

        let sym_matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
        SymCsrArray::new(sym_matrix)
    }

    #[test]
    fn test_sym_sparse_array_trait() {
        let sym_csr = create_test_sym_csr();

        // Test basic properties
        assert_eq!(sym_csr.shape(), (3, 3));
        assert!(sym_csr.is_symmetric());

        // Test SparseArray methods
        assert_eq!(sym_csr.get(0, 0), 2.0);
        assert_eq!(sym_csr.get(0, 1), 1.0);
        assert_eq!(sym_csr.get(1, 0), 1.0); // Through symmetry

        // Test nnz_stored vs nnz
        assert_eq!(sym_csr.nnz_stored(), 5); // Only stored elements
        assert_eq!(sym_csr.nnz(), 7); // Including symmetric pairs

        // Test conversion between formats
        let sym_coo = sym_csr.to_sym_coo().unwrap();
        assert_eq!(sym_coo.shape(), (3, 3));
        assert!(sym_coo.is_symmetric());

        let csr = SymSparseArray::<f64>::to_csr(&sym_csr).unwrap();
        assert_eq!(csr.shape(), (3, 3));

        let coo = SymSparseArray::<f64>::to_coo(&sym_csr).unwrap();
        assert_eq!(coo.shape(), (3, 3));

        // Test that find() returns the full matrix elements
        let (rows, _cols, _data) = sym_csr.find();
        assert!(rows.len() > sym_csr.nnz_stored()); // Should include symmetric pairs
    }

    #[test]
    fn test_sym_sparse_array_operations() {
        let sym_csr = create_test_sym_csr();

        // Create another symmetric matrix for testing operations
        let sym_csr2 = create_test_sym_csr();

        // Test addition
        let sum = sym_csr.add(&sym_csr2).unwrap();
        assert_eq!(sum.shape(), (3, 3));
        assert_eq!(sum.get(0, 0), 4.0); // 2 + 2
        assert_eq!(sum.get(0, 1), 2.0); // 1 + 1
        assert_eq!(sum.get(1, 0), 2.0); // 1 + 1 (symmetric)

        // Test element-wise multiplication
        let prod = sym_csr.mul(&sym_csr2).unwrap();
        assert_eq!(prod.shape(), (3, 3));
        assert_eq!(prod.get(0, 0), 4.0); // 2 * 2
        assert_eq!(prod.get(0, 1), 1.0); // 1 * 1
        assert_eq!(prod.get(1, 0), 1.0); // 1 * 1 (symmetric)

        // Test matrix multiplication
        let dot = sym_csr.dot(&sym_csr2).unwrap();
        assert_eq!(dot.shape(), (3, 3));

        // Test transpose (should be no change for symmetric matrices)
        let trans = sym_csr.transpose().unwrap();
        assert_eq!(trans.shape(), sym_csr.shape());
        assert_eq!(SparseArray::get(&*trans, 0, 1), sym_csr.get(0, 1));
        assert_eq!(SparseArray::get(&*trans, 1, 0), sym_csr.get(1, 0));
    }
}
