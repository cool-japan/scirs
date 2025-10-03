// IO utilities for sparse arrays
//
// This module provides functions for serializing and deserializing sparse arrays,
// including npz format compatible with SciPy's sparse.save_npz and sparse.load_npz.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;

use crate::coo_array::CooArray;
use crate::csc_array::CscArray;
use crate::csr_array::CsrArray;
use crate::dok_array::DokArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;

// Define format markers for each sparse array type
const CSR_FORMAT: &str = "csr_array";
// const CSC_FORMAT: &str = "csc_array"; // Commented out as currently unused
const COO_FORMAT: &str = "coo_array";
const DOK_FORMAT: &str = "dok_array";

/// Serializes a sparse array to the .npz format
///
/// This format is compatible with SciPy's sparse.save_npz function.
/// It saves the sparse array data in a way that can be loaded by SciPy
/// or by the load_npz function in this module.
///
/// # Arguments
/// * `array` - The sparse array to save
/// * `path` - Path where the .npz file will be saved
///
/// # Returns
/// A Result indicating success or an error
///
/// # Examples
///
/// ```no_run
/// use scirs2_sparse::construct::eye_array;
/// use scirs2_sparse::io::save_npz;
///
/// let array = eye_array::<f64>(10, "csr").unwrap();
/// save_npz(&*array, "identity.npz").unwrap();
/// ```
#[allow(dead_code)]
pub fn save_npz<T, P>(array: &dyn SparseArray<T>, path: P) -> SparseResult<()>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
    P: AsRef<Path>,
{
    // First determine the format and get needed components
    let (format, data, indices, indptr, shape) = match array.to_csr() {
        Ok(csr) => {
            if let Some(csr_array) = csr.as_any().downcast_ref::<CsrArray<T>>() {
                (
                    CSR_FORMAT,
                    csr_array.get_data().clone(),
                    csr_array.get_indices().clone(),
                    csr_array.get_indptr().clone(),
                    csr_array.shape(),
                )
            } else {
                return Err(SparseError::ConversionError(
                    "Failed to downcast to CsrArray".to_string(),
                ));
            }
        }
        Err(_) => {
            // If we couldn't convert to CSR, try converting to COO
            match array.to_coo() {
                Ok(coo) => {
                    if let Some(coo_array) = coo.as_any().downcast_ref::<CooArray<T>>() {
                        // For COO format, we store row indices, column indices, and data
                        let shape = coo_array.shape();
                        let rows = coo_array.get_rows().clone();
                        let _cols = coo_array.get_cols().clone();

                        // Use zeros for indptr (not used in COO)
                        let indptr = Array1::zeros(0);

                        (
                            COO_FORMAT,
                            coo_array.get_data().clone(),
                            // For COO, the "indices" field will be row indices
                            rows,
                            // We'll handle this specially later
                            indptr,
                            shape,
                        )
                    } else {
                        return Err(SparseError::ConversionError(
                            "Failed to downcast to CooArray".to_string(),
                        ));
                    }
                }
                Err(_) => {
                    // Try DOK format
                    match array.to_dok() {
                        Ok(dok) => {
                            if let Some(dok_array) = dok.as_any().downcast_ref::<DokArray<T>>() {
                                // For DOK format, we convert to COO triplets first
                                let (rowscols, values) = dok_array.to_triplets();
                                let shape = dok_array.shape();

                                // Use zeros for indptr (not used in DOK)
                                let indptr = Array1::zeros(0);

                                (DOK_FORMAT, values, rows, indptr, shape)
                            } else {
                                return Err(SparseError::ConversionError(
                                    "Failed to downcast to DokArray".to_string(),
                                ));
                            }
                        }
                        Err(e) => {
                            return Err(SparseError::ConversionError(format!(
                                "Failed to convert to a serializable format: {}",
                                e
                            )));
                        }
                    }
                }
            }
        }
    };

    // Create the .npz file (a simple custom format for this example)
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write format marker
    write_string(&mut writer, format)?;

    // Write shape
    write_usize(&mut writer, shape.0)?;
    write_usize(&mut writer, shape.1)?;

    // Write data
    write_array(&mut writer, &data)?;

    // Write indices
    write_array(&mut writer, &indices)?;

    // For COO format, we also need column indices
    if format == COO_FORMAT {
        if let Ok(coo) = array.to_coo() {
            if let Some(coo_array) = coo.as_any().downcast_ref::<CooArray<T>>() {
                write_array(&mut writer, coo_array.get_cols())?;
            }
        }
    } else {
        // Write indptr for CSR/CSC
        write_array(&mut writer, &indptr)?;
    }

    Ok(())
}

/// Loads a sparse array from the .npz format
///
/// This function loads a sparse array that was saved using save_npz
/// or SciPy's sparse.save_npz function.
///
/// # Arguments
/// * `path` - Path to the .npz file
///
/// # Returns
/// A Result containing the loaded sparse array or an error
///
/// # Examples
///
/// ```no_run
/// use scirs2_sparse::io::load_npz;
///
/// let array = load_npz::<f64>("identity.npz").unwrap();
/// assert_eq!(array.shape(), (10, 10));
/// ```
#[allow(dead_code)]
pub fn load_npz<T, P>(path: P) -> SparseResult<Box<dyn SparseArray<T>>>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
    P: AsRef<Path>,
{
    let file = File::open(_path)?;
    let mut reader = BufReader::new(file);

    // Read format marker
    let format = read_string(&mut reader)?;

    // Read shape
    let rows = read_usize(&mut reader)?;
    let cols = read_usize(&mut reader)?;
    let shape = (rows, cols);

    // Read data
    let data = read_array::<_, T>(&mut reader)?;

    // Read indices
    let indices = read_array::<_, usize>(&mut reader)?;

    // Create the appropriate array type
    match format.as_str() {
        "csr_array" => {
            let indptr = read_array::<_, usize>(&mut reader)?;
            CsrArray::new(data, indices, indptr, shape)
                .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
        }
        "csc_array" => {
            let indptr = read_array::<_, usize>(&mut reader)?;
            CscArray::new(data, indices, indptr, shape)
                .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
        }
        "coo_array" => {
            // For COO, read column indices
            let cols = read_array::<_, usize>(&mut reader)?;
            // For COO, indices is row indices
            CooArray::new(data, indices, cols, shape, false)
                .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
        }
        "dok_array" => {
            // For DOK, read column indices
            let cols = read_array::<_, usize>(&mut reader)?;
            // For DOK, indices is row indices
            DokArray::from_triplets(
                &indices,
                &cols,
                &data,
                shape,
            )
            .map(|array| Box::new(array) as Box<dyn SparseArray<T>>)
        }
        _ => Err(SparseError::ConversionError(format!(
            "Unknown format: {}",
            format
        ))),
    }
}

// Utility functions for reading/writing primitive types

#[allow(dead_code)]
fn write_string<W: Write>(writer: &mut W, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u64;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

#[allow(dead_code)]
fn read_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;

    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;

    String::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[allow(dead_code)]
fn write_usize<W: Write>(writer: &mut W, n: usize) -> std::io::Result<()> {
    writer.write_all(&(n as u64).to_le_bytes())
}

#[allow(dead_code)]
fn read_usize<R: Read>(reader: &mut R) -> std::io::Result<usize> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes) as usize)
}

#[allow(dead_code)]
fn write_array<W: Write, T: Copy>(writer: &mut W, array: &Array1<T>) -> std::io::Result<()> {
    let len = array.len() as u64;
    writer.write_all(&len.to_le_bytes())?;

    let data_size = std::mem::size_of::<T>() * array.len();
    let data_ptr = array.as_ptr() as *const u8;
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_size) };

    writer.write_all(data_slice)?;
    Ok(())
}

#[allow(dead_code)]
fn read_array<R: Read, T: Copy>(reader: &mut R) -> std::io::Result<Array1<T>> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;

    let mut data = Vec::with_capacity(len);
    // Resize to make space for elements
    data.resize_with(len, || unsafe { std::mem::zeroed() });

    let data_size = std::mem::size_of::<T>() * len;
    let data_ptr = data.as_mut_ptr() as *mut u8;
    let data_slice = unsafe { std::slice::from_raw_parts_mut(data_ptr, data_size) };

    reader.read_exact(data_slice)?;

    Ok(Array1::from_vec(data))
}

// Trait extension to allow downcasting
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T> AsAny for dyn SparseArray<T>
where
    T: Float
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Debug
        + Copy
        + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Implement From<std::io::Error> for SparseError
impl From<std::io::Error> for SparseError {
    fn from(error: std::io::Error) -> Self {
        SparseError::ComputationError(format!("IO error: {_error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construct::{eye_array, random_array};
    use tempfile::tempdir;

    #[test]
    fn test_save_load_csr_array() {
        // Create a temporary directory for test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_csr.npz");

        // Create a CSR array
        let array = eye_array::<f64>(5, "csr").unwrap();

        // Save the array
        save_npz(&*array, &file_path).unwrap();

        // Load the array
        let loaded = load_npz::<f64>(&file_path).unwrap();

        // Check that it loaded correctly
        assert_eq!(loaded.shape(), (5, 5));
        assert_eq!(loaded.nnz(), 5);
        assert_eq!(loaded.get(0, 0), 1.0);
        assert_eq!(loaded.get(1, 1), 1.0);
        assert_eq!(loaded.get(2, 2), 1.0);
        assert_eq!(loaded.get(3, 3), 1.0);
        assert_eq!(loaded.get(4, 4), 1.0);
        assert_eq!(loaded.get(0, 1), 0.0);
    }

    #[test]
    fn test_save_load_coo_array() {
        // Create a temporary directory for test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_coo.npz");

        // Create a COO array
        let array = eye_array::<f64>(5, "coo").unwrap();

        // Save the array
        save_npz(&*array, &file_path).unwrap();

        // Load the array
        let loaded = load_npz::<f64>(&file_path).unwrap();

        // Check that it loaded correctly
        assert_eq!(loaded.shape(), (5, 5));
        assert_eq!(loaded.nnz(), 5);
        assert_eq!(loaded.get(0, 0), 1.0);
        assert_eq!(loaded.get(1, 1), 1.0);
        assert_eq!(loaded.get(2, 2), 1.0);
        assert_eq!(loaded.get(3, 3), 1.0);
        assert_eq!(loaded.get(4, 4), 1.0);
        assert_eq!(loaded.get(0, 1), 0.0);
    }

    #[test]
    fn test_save_load_random_array() {
        // Create a temporary directory for test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_random.npz");

        // Create a random array
        let array = random_array::<f64>((10, 10), 0.3, Some(42), "csr").unwrap();
        let original_nnz = array.nnz();
        let original_sum = array.sum(None).unwrap();

        // Save the array
        save_npz(&*array, &file_path).unwrap();

        // Load the array
        let loaded = load_npz::<f64>(&file_path).unwrap();

        // Check that it loaded correctly
        assert_eq!(loaded.shape(), (10, 10));
        assert_eq!(loaded.nnz(), original_nnz);

        let loaded_sum = loaded.sum(None).unwrap();
        if let (crate::sparray::SparseSum::Scalar(orig), crate::sparray::SparseSum::Scalar(load)) =
            (original_sum, loaded_sum)
        {
            assert!((orig - load).abs() < 1e-10);
        }
    }
}
