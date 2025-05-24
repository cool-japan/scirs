//! Data serialization utilities
//!
//! This module provides functionality for serializing and deserializing
//! scientific data structures including arrays, matrices, and structured data.
//!
//! Features:
//! - Binary serialization of ndarray arrays
//! - JSON serialization for structured data
//! - Enhanced sparse matrix serialization with multiple formats (COO, CSR, CSC)
//! - Matrix Market format integration
//! - Compression support for sparse matrices
//! - Memory-efficient sparse matrix operations

use ndarray::{Array, Array2, ArrayBase, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// Format for data serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Binary format (compact, not human-readable)
    Binary,
    /// JSON format (human-readable)
    JSON,
    /// MessagePack format (compact binary, cross-language)
    MessagePack,
}

/// Serialize an ndarray to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `array` - The ndarray to serialize
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array2, IxDyn};
/// use scirs2_io::serialize::{serialize_array, SerializationFormat};
///
/// let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let array_dyn = array.into_dyn();
///
/// // Binary serialization (compact)
/// serialize_array("data.bin", &array_dyn, SerializationFormat::Binary).unwrap();
///
/// // JSON serialization (human-readable)
/// serialize_array("data.json", &array_dyn, SerializationFormat::JSON).unwrap();
/// ```
pub fn serialize_array<P, A, S>(
    path: P,
    array: &ArrayBase<S, IxDyn>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize + Clone,
    S: ndarray::Data<Elem = A>,
{
    // Create a serializable representation
    let shape = array.shape().to_vec();
    let data: Vec<A> = array.iter().cloned().collect();

    // Create a serializable struct
    let serializable = SerializedArray {
        metadata: ArrayMetadata {
            shape,
            dtype: std::any::type_name::<A>().to_string(),
            order: 'C',
            metadata: std::collections::HashMap::new(),
        },
        data,
    };

    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    match format {
        SerializationFormat::Binary => {
            bincode::serialize_into(&mut writer, &serializable)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::JSON => {
            serde_json::to_writer_pretty(&mut writer, &serializable)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::MessagePack => {
            rmp_serde::encode::write(&mut writer, &serializable)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    Ok(())
}

/// Deserialize an ndarray from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<Array<A, D>>` - Deserialized array or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array, IxDyn};
/// use scirs2_io::serialize::{deserialize_array, SerializationFormat};
///
/// // Binary deserialization
/// let array = deserialize_array::<_, f64>("data.bin", SerializationFormat::Binary).unwrap();
/// println!("Deserialized array shape: {:?}", array.shape());
///
/// // JSON deserialization
/// let array = deserialize_array::<_, f64>("data.json", SerializationFormat::JSON).unwrap();
/// ```
pub fn deserialize_array<P, A>(path: P, format: SerializationFormat) -> Result<Array<A, IxDyn>>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Clone,
{
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let serialized: SerializedArray<A> = match format {
        SerializationFormat::Binary => bincode::deserialize_from(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::JSON => serde_json::from_reader(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::MessagePack => rmp_serde::from_read(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
    };

    // Convert back to ndarray
    let array = Array::from_shape_vec(IxDyn(&serialized.metadata.shape), serialized.data)
        .map_err(|e| IoError::FormatError(format!("Failed to reconstruct array: {}", e)))?;

    Ok(array)
}

/// Array metadata for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadata {
    /// Array shape
    pub shape: Vec<usize>,
    /// Array element type
    pub dtype: String,
    /// Array order ('C' for row-major, 'F' for column-major)
    pub order: char,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Serialized array data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedArray<A> {
    /// Array metadata
    pub metadata: ArrayMetadata,
    /// Flattened array data
    pub data: Vec<A>,
}

/// Serialize an ndarray with metadata
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `array` - The ndarray to serialize
/// * `metadata` - Additional metadata
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_io::serialize::{serialize_array_with_metadata, SerializationFormat};
/// use std::collections::HashMap;
///
/// let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let array_dyn = array.into_dyn();
///
/// // Create custom metadata
/// let mut metadata = HashMap::new();
/// metadata.insert("description".to_string(), "Test array".to_string());
/// metadata.insert("units".to_string(), "meters".to_string());
///
/// serialize_array_with_metadata(
///     "data_with_metadata.json",
///     &array_dyn,
///     metadata,
///     SerializationFormat::JSON
/// ).unwrap();
/// ```
pub fn serialize_array_with_metadata<P, A, S>(
    path: P,
    array: &ArrayBase<S, IxDyn>,
    metadata: std::collections::HashMap<String, String>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize + Clone,
    S: ndarray::Data<Elem = A>,
{
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Create array metadata
    let shape = array.shape().to_vec();
    let dtype = std::any::type_name::<A>().to_string();

    let array_metadata = ArrayMetadata {
        shape,
        dtype,
        order: 'C', // Rust ndarray uses row-major (C) order
        metadata,
    };

    // Create serialized array
    let serialized = SerializedArray {
        metadata: array_metadata,
        data: array.iter().cloned().collect(),
    };

    // Serialize
    match format {
        SerializationFormat::Binary => {
            bincode::serialize_into(&mut writer, &serialized)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::JSON => {
            serde_json::to_writer_pretty(&mut writer, &serialized)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::MessagePack => {
            rmp_serde::encode::write(&mut writer, &serialized)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    Ok(())
}

/// Deserialize an ndarray with metadata from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<(Array<A, IxDyn>, std::collections::HashMap<String, String>)>` - Deserialized array and metadata
///
/// # Examples
///
/// ```no_run
/// use ndarray::Array2;
/// use scirs2_io::serialize::{deserialize_array_with_metadata, SerializationFormat};
///
/// // Deserialize array with metadata
/// let (array, metadata) = deserialize_array_with_metadata::<_, f64>(
///     "data_with_metadata.json",
///     SerializationFormat::JSON
/// ).unwrap();
///
/// println!("Deserialized array shape: {:?}", array.shape());
/// println!("Metadata: {:?}", metadata);
/// ```
pub fn deserialize_array_with_metadata<P, A>(
    path: P,
    format: SerializationFormat,
) -> Result<(Array<A, IxDyn>, std::collections::HashMap<String, String>)>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Clone,
{
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let serialized: SerializedArray<A> = match format {
        SerializationFormat::Binary => bincode::deserialize_from(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::JSON => serde_json::from_reader(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
        SerializationFormat::MessagePack => rmp_serde::from_read(reader)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?,
    };

    // Extract shape and data
    let shape = serialized.metadata.shape;
    let data = serialized.data;

    // Create ndarray
    let array = Array::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| IoError::FormatError(format!("Invalid shape: {:?}", e)))?;

    Ok((array, serialized.metadata.metadata))
}

/// Serialize a struct to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `data` - The struct to serialize
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use serde::{Serialize, Deserialize};
/// use scirs2_io::serialize::{serialize_struct, SerializationFormat};
///
/// #[derive(Serialize, Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
///     height: f64,
/// }
///
/// let person = Person {
///     name: "Alice".to_string(),
///     age: 30,
///     height: 175.5,
/// };
///
/// // JSON serialization
/// serialize_struct("person.json", &person, SerializationFormat::JSON).unwrap();
/// ```
pub fn serialize_struct<P, T>(path: P, data: &T, format: SerializationFormat) -> Result<()>
where
    P: AsRef<Path>,
    T: Serialize,
{
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    match format {
        SerializationFormat::Binary => {
            bincode::serialize_into(&mut writer, data)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::JSON => {
            serde_json::to_writer_pretty(&mut writer, data)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        SerializationFormat::MessagePack => {
            rmp_serde::encode::write(&mut writer, data)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    Ok(())
}

/// Deserialize a struct from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<T>` - Deserialized struct or error
///
/// # Examples
///
/// ```no_run
/// use serde::{Serialize, Deserialize};
/// use scirs2_io::serialize::{deserialize_struct, SerializationFormat};
///
/// #[derive(Serialize, Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
///     height: f64,
/// }
///
/// // JSON deserialization
/// let person: Person = deserialize_struct("person.json", SerializationFormat::JSON).unwrap();
/// println!("Name: {}, Age: {}, Height: {}", person.name, person.age, person.height);
/// ```
pub fn deserialize_struct<P, T>(path: P, format: SerializationFormat) -> Result<T>
where
    P: AsRef<Path>,
    T: for<'de> Deserialize<'de>,
{
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    match format {
        SerializationFormat::Binary => {
            let data = bincode::deserialize_from(reader)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            Ok(data)
        }
        SerializationFormat::JSON => {
            let data = serde_json::from_reader(reader)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            Ok(data)
        }
        SerializationFormat::MessagePack => {
            let data = rmp_serde::from_read(reader)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            Ok(data)
        }
    }
}

/// Type representing a sparse matrix in COO (Coordinate) format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrixCOO<A> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Values
    pub values: Vec<A>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl<A> SparseMatrixCOO<A> {
    /// Create a new sparse matrix in COO format
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a value to the sparse matrix
    pub fn push(&mut self, row: usize, col: usize, value: A) {
        if row < self.rows && col < self.cols {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Serialize a sparse matrix to a file
///
/// # Arguments
///
/// * `path` - Path to the output file
/// * `matrix` - The sparse matrix to serialize
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::serialize::{serialize_sparse_matrix, SparseMatrixCOO, SerializationFormat};
///
/// // Create a sparse matrix
/// let mut sparse = SparseMatrixCOO::<f64>::new(100, 100);
/// sparse.push(0, 0, 1.0);
/// sparse.push(10, 10, 2.0);
/// sparse.push(50, 50, 3.0);
///
/// // Add metadata
/// sparse.metadata.insert("description".to_string(), "Sparse test matrix".to_string());
///
/// // Serialize to file
/// serialize_sparse_matrix("sparse.json", &sparse, SerializationFormat::JSON).unwrap();
/// ```
pub fn serialize_sparse_matrix<P, A>(
    path: P,
    matrix: &SparseMatrixCOO<A>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize,
{
    serialize_struct(path, matrix, format)
}

/// Deserialize a sparse matrix from a file
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `format` - Serialization format
///
/// # Returns
///
/// * `Result<SparseMatrixCOO<A>>` - Deserialized sparse matrix or error
///
/// # Examples
///
/// ```no_run
/// use scirs2_io::serialize::{deserialize_sparse_matrix, SerializationFormat};
///
/// // Deserialize sparse matrix
/// let sparse = deserialize_sparse_matrix::<_, f64>("sparse.json", SerializationFormat::JSON).unwrap();
/// println!("Sparse matrix: {}x{} with {} non-zero elements", sparse.rows, sparse.cols, sparse.nnz());
/// ```
pub fn deserialize_sparse_matrix<P, A>(
    path: P,
    format: SerializationFormat,
) -> Result<SparseMatrixCOO<A>>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de>,
{
    deserialize_struct(path, format)
}

/// Sparse matrix format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparseFormat {
    /// Coordinate format (COO) - triplet format with (row, col, value)
    COO,
    /// Compressed Sparse Row (CSR) format
    CSR,
    /// Compressed Sparse Column (CSC) format  
    CSC,
}

/// Enhanced sparse matrix with multiple format support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrix<A> {
    /// Matrix dimensions
    pub shape: (usize, usize),
    /// Current storage format
    pub format: SparseFormat,
    /// COO format data (always maintained for compatibility)
    pub coo_data: SparseMatrixCOO<A>,
    /// CSR format data (computed on demand)
    #[serde(skip)]
    pub csr_data: Option<SparseMatrixCSR<A>>,
    /// CSC format data (computed on demand)
    #[serde(skip)]
    pub csc_data: Option<SparseMatrixCSC<A>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Compressed Sparse Row (CSR) format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrixCSR<A> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Row pointers (indptr)
    pub row_ptrs: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Values
    pub values: Vec<A>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Compressed Sparse Column (CSC) format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrixCSC<A> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Column pointers (indptr)
    pub col_ptrs: Vec<usize>,
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Values
    pub values: Vec<A>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl<A: Clone> SparseMatrix<A> {
    /// Create a new sparse matrix from COO data
    pub fn from_coo(coo: SparseMatrixCOO<A>) -> Self {
        let shape = (coo.rows, coo.cols);
        Self {
            shape,
            format: SparseFormat::COO,
            coo_data: coo,
            csr_data: None,
            csc_data: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new sparse matrix with specified dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: (rows, cols),
            format: SparseFormat::COO,
            coo_data: SparseMatrixCOO::new(rows, cols),
            csr_data: None,
            csc_data: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a value to the sparse matrix (always updates COO format)
    pub fn insert(&mut self, row: usize, col: usize, value: A) {
        self.coo_data.push(row, col, value);
        // Invalidate cached formats
        self.csr_data = None;
        self.csc_data = None;
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.coo_data.nnz()
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Convert to CSR format (cached)
    pub fn to_csr(&mut self) -> Result<&SparseMatrixCSR<A>>
    where
        A: Clone + Default + PartialEq,
    {
        if self.csr_data.is_none() {
            self.csr_data = Some(self.convert_to_csr()?);
        }
        Ok(self.csr_data.as_ref().unwrap())
    }

    /// Convert to CSC format (cached)
    pub fn to_csc(&mut self) -> Result<&SparseMatrixCSC<A>>
    where
        A: Clone + Default + PartialEq,
    {
        if self.csc_data.is_none() {
            self.csc_data = Some(self.convert_to_csc()?);
        }
        Ok(self.csc_data.as_ref().unwrap())
    }

    /// Convert COO to CSR format
    fn convert_to_csr(&self) -> Result<SparseMatrixCSR<A>>
    where
        A: Clone + Default,
    {
        let nnz = self.coo_data.nnz();
        let rows = self.shape.0;

        if nnz == 0 {
            return Ok(SparseMatrixCSR {
                rows,
                cols: self.shape.1,
                row_ptrs: vec![0; rows + 1],
                col_indices: Vec::new(),
                values: Vec::new(),
                metadata: self.metadata.clone(),
            });
        }

        // Sort by row then column
        let mut triplets: Vec<(usize, usize, A)> = self
            .coo_data
            .row_indices
            .iter()
            .zip(self.coo_data.col_indices.iter())
            .zip(self.coo_data.values.iter())
            .map(|((&r, &c), v)| (r, c, v.clone()))
            .collect();

        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Build CSR structure
        let mut row_ptrs = vec![0; rows + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        let mut current_row = 0;

        for (i, (row, col, val)) in triplets.iter().enumerate() {
            // Fill row_ptrs for empty rows
            while current_row < *row {
                current_row += 1;
                row_ptrs[current_row] = i;
            }

            col_indices.push(*col);
            values.push(val.clone());
        }

        // Fill remaining row_ptrs
        while current_row < rows {
            current_row += 1;
            row_ptrs[current_row] = nnz;
        }

        Ok(SparseMatrixCSR {
            rows,
            cols: self.shape.1,
            row_ptrs,
            col_indices,
            values,
            metadata: self.metadata.clone(),
        })
    }

    /// Convert COO to CSC format
    fn convert_to_csc(&self) -> Result<SparseMatrixCSC<A>>
    where
        A: Clone + Default,
    {
        let nnz = self.coo_data.nnz();
        let cols = self.shape.1;

        if nnz == 0 {
            return Ok(SparseMatrixCSC {
                rows: self.shape.0,
                cols,
                col_ptrs: vec![0; cols + 1],
                row_indices: Vec::new(),
                values: Vec::new(),
                metadata: self.metadata.clone(),
            });
        }

        // Sort by column then row
        let mut triplets: Vec<(usize, usize, A)> = self
            .coo_data
            .row_indices
            .iter()
            .zip(self.coo_data.col_indices.iter())
            .zip(self.coo_data.values.iter())
            .map(|((&r, &c), v)| (r, c, v.clone()))
            .collect();

        triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Build CSC structure
        let mut col_ptrs = vec![0; cols + 1];
        let mut row_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        let mut current_col = 0;

        for (i, (row, col, val)) in triplets.iter().enumerate() {
            // Fill col_ptrs for empty columns
            while current_col < *col {
                current_col += 1;
                col_ptrs[current_col] = i;
            }

            row_indices.push(*row);
            values.push(val.clone());
        }

        // Fill remaining col_ptrs
        while current_col < cols {
            current_col += 1;
            col_ptrs[current_col] = nnz;
        }

        Ok(SparseMatrixCSC {
            rows: self.shape.0,
            cols,
            col_ptrs,
            row_indices,
            values,
            metadata: self.metadata.clone(),
        })
    }

    /// Convert to dense matrix representation
    pub fn to_dense(&self) -> Array2<A>
    where
        A: Clone + Default,
    {
        let mut dense = Array2::default(self.shape);

        for ((row, col), value) in self
            .coo_data
            .row_indices
            .iter()
            .zip(self.coo_data.col_indices.iter())
            .zip(self.coo_data.values.iter())
        {
            dense[[*row, *col]] = value.clone();
        }

        dense
    }

    /// Calculate sparsity ratio (percentage of zero elements)
    pub fn sparsity(&self) -> f64 {
        let total_elements = self.shape.0 * self.shape.1;
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let coo_size = self.coo_data.values.len()
            * (std::mem::size_of::<A>() + 2 * std::mem::size_of::<usize>());

        let csr_size = if let Some(ref csr) = self.csr_data {
            csr.values.len() * std::mem::size_of::<A>()
                + csr.col_indices.len() * std::mem::size_of::<usize>()
                + csr.row_ptrs.len() * std::mem::size_of::<usize>()
        } else {
            0
        };

        let csc_size = if let Some(ref csc) = self.csc_data {
            csc.values.len() * std::mem::size_of::<A>()
                + csc.row_indices.len() * std::mem::size_of::<usize>()
                + csc.col_ptrs.len() * std::mem::size_of::<usize>()
        } else {
            0
        };

        coo_size + csr_size + csc_size
    }
}

impl<A: Clone> SparseMatrixCSR<A> {
    /// Create a new CSR sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptrs: vec![0; rows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get a row slice
    pub fn row(&self, row: usize) -> Option<(&[usize], &[A])> {
        if row >= self.rows {
            return None;
        }

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        Some((&self.col_indices[start..end], &self.values[start..end]))
    }
}

impl<A: Clone> SparseMatrixCSC<A> {
    /// Create a new CSC sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptrs: vec![0; cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get a column slice
    pub fn column(&self, col: usize) -> Option<(&[usize], &[A])> {
        if col >= self.cols {
            return None;
        }

        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];

        Some((&self.row_indices[start..end], &self.values[start..end]))
    }
}

/// Enhanced sparse matrix serialization with format conversion
pub fn serialize_enhanced_sparse_matrix<P, A>(
    path: P,
    matrix: &SparseMatrix<A>,
    format: SerializationFormat,
) -> Result<()>
where
    P: AsRef<Path>,
    A: Serialize,
{
    serialize_struct(path, matrix, format)
}

/// Enhanced sparse matrix deserialization
pub fn deserialize_enhanced_sparse_matrix<P, A>(
    path: P,
    format: SerializationFormat,
) -> Result<SparseMatrix<A>>
where
    P: AsRef<Path>,
    A: for<'de> Deserialize<'de> + Default,
{
    deserialize_struct(path, format)
}

/// Convert Matrix Market format to enhanced sparse matrix
pub fn from_matrix_market<A>(mm_matrix: &crate::matrix_market::MMSparseMatrix<A>) -> SparseMatrix<A>
where
    A: Clone,
{
    let mut coo = SparseMatrixCOO::new(mm_matrix.rows, mm_matrix.cols);

    for entry in &mm_matrix.entries {
        coo.push(entry.row, entry.col, entry.value.clone());
    }

    let mut sparse = SparseMatrix::from_coo(coo);
    sparse
        .metadata
        .insert("source".to_string(), "Matrix Market".to_string());
    sparse.metadata.insert(
        "format".to_string(),
        format!("{:?}", mm_matrix.header.format),
    );
    sparse.metadata.insert(
        "data_type".to_string(),
        format!("{:?}", mm_matrix.header.data_type),
    );
    sparse.metadata.insert(
        "symmetry".to_string(),
        format!("{:?}", mm_matrix.header.symmetry),
    );

    sparse
}

/// Convert enhanced sparse matrix to Matrix Market format
pub fn to_matrix_market<A>(sparse: &SparseMatrix<A>) -> crate::matrix_market::MMSparseMatrix<A>
where
    A: Clone,
{
    let header = crate::matrix_market::MMHeader {
        object: "matrix".to_string(),
        format: crate::matrix_market::MMFormat::Coordinate,
        data_type: crate::matrix_market::MMDataType::Real, // Default, should be determined by A
        symmetry: crate::matrix_market::MMSymmetry::General, // Default
        comments: vec!["Converted from enhanced sparse matrix".to_string()],
    };

    let entries = sparse
        .coo_data
        .row_indices
        .iter()
        .zip(sparse.coo_data.col_indices.iter())
        .zip(sparse.coo_data.values.iter())
        .map(|((&row, &col), value)| crate::matrix_market::SparseEntry {
            row,
            col,
            value: value.clone(),
        })
        .collect();

    crate::matrix_market::MMSparseMatrix {
        header,
        rows: sparse.shape.0,
        cols: sparse.shape.1,
        nnz: sparse.nnz(),
        entries,
    }
}

/// Sparse matrix operations utilities
pub mod sparse_ops {
    use super::*;

    /// Add two sparse matrices (COO format)
    pub fn add_coo<A>(a: &SparseMatrixCOO<A>, b: &SparseMatrixCOO<A>) -> Result<SparseMatrixCOO<A>>
    where
        A: Clone + std::ops::Add<Output = A> + Default + PartialEq,
    {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(IoError::ValidationError(
                "Matrix dimensions must match".to_string(),
            ));
        }

        let mut result = SparseMatrixCOO::new(a.rows, a.cols);
        let mut indices_map: HashMap<(usize, usize), A> = HashMap::new();

        // Add entries from matrix a
        for ((row, col), value) in a
            .row_indices
            .iter()
            .zip(a.col_indices.iter())
            .zip(a.values.iter())
        {
            indices_map.insert((*row, *col), value.clone());
        }

        // Add entries from matrix b
        for ((row, col), value) in b
            .row_indices
            .iter()
            .zip(b.col_indices.iter())
            .zip(b.values.iter())
        {
            let key = (*row, *col);
            if let Some(existing) = indices_map.get(&key) {
                indices_map.insert(key, existing.clone() + value.clone());
            } else {
                indices_map.insert(key, value.clone());
            }
        }

        // Convert back to COO format
        for ((row, col), value) in indices_map {
            if value != A::default() {
                // Only store non-zero values
                result.push(row, col, value);
            }
        }

        Ok(result)
    }

    /// Matrix-vector multiplication for CSR format
    pub fn csr_matvec<A>(matrix: &SparseMatrixCSR<A>, vector: &[A]) -> Result<Vec<A>>
    where
        A: Clone + std::ops::Add<Output = A> + std::ops::Mul<Output = A> + Default,
    {
        if vector.len() != matrix.cols {
            return Err(IoError::ValidationError(
                "Vector dimension must match matrix columns".to_string(),
            ));
        }

        let mut result = vec![A::default(); matrix.rows];

        for (row, result_elem) in result.iter_mut().enumerate() {
            let start = matrix.row_ptrs[row];
            let end = matrix.row_ptrs[row + 1];

            let mut sum = A::default();
            for i in start..end {
                let col = matrix.col_indices[i];
                let val = matrix.values[i].clone();
                sum = sum + (val * vector[col].clone());
            }
            *result_elem = sum;
        }

        Ok(result)
    }

    /// Transpose a COO sparse matrix
    pub fn transpose_coo<A>(matrix: &SparseMatrixCOO<A>) -> SparseMatrixCOO<A>
    where
        A: Clone,
    {
        let mut result = SparseMatrixCOO::new(matrix.cols, matrix.rows);

        for ((row, col), value) in matrix
            .row_indices
            .iter()
            .zip(matrix.col_indices.iter())
            .zip(matrix.values.iter())
        {
            result.push(*col, *row, value.clone());
        }

        result
    }
}
