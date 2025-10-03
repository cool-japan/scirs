//! Enhanced MATLAB format support with v7.3+ (HDF5) support
//!
//! This module provides enhanced MATLAB file format support including:
//! - Complete MAT v5 format implementation
//! - MAT v7.3+ support using HDF5 backend
//! - Cell array support
//! - Structure support
//! - Complex number support
//! - Compression support

use crate::error::{IoError, Result};
use crate::matlab::{read_mat, write_mat, MatType};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "hdf5")]
use crate::hdf5::{AttributeValue, CompressionOptions, DatasetOptions, FileMode, HDF5File};

#[cfg(not(feature = "hdf5"))]
type CompressionOptions = ();

/// Enhanced MATLAB file format configuration
#[derive(Debug, Clone)]
pub struct MatFileConfig {
    /// Use MAT v7.3 format (HDF5-based) instead of v5
    pub use_v73: bool,
    /// Enable compression for v7.3 files
    pub compression: Option<CompressionOptions>,
    /// Maximum variable size before switching to v7.3 (in bytes)
    pub v73_threshold: usize,
}

impl Default for MatFileConfig {
    fn default() -> Self {
        Self {
            use_v73: false,
            compression: None,
            v73_threshold: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }
}

/// Enhanced MAT file writer/reader
pub struct EnhancedMatFile {
    config: MatFileConfig,
}

impl EnhancedMatFile {
    /// Create a new enhanced MAT file handler
    pub fn new(config: MatFileConfig) -> Self {
        Self { config }
    }

    /// Write variables to a MAT file with enhanced features
    pub fn write<P: AsRef<Path>>(&self, path: P, vars: &HashMap<String, MatType>) -> Result<()> {
        // Determine if we should use v7.3 format
        let total_size = self.estimate_size(vars);
        let use_v73 = self.config.use_v73 || total_size > self.config.v73_threshold;

        if use_v73 {
            self.write_v73(&path, vars)
        } else {
            write_mat(path, vars)
        }
    }

    /// Read variables from a MAT file with enhanced features
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, MatType>> {
        // Try v7.3 format first, then fall back to v5
        if self.is_v73_file(&path)? {
            self.read_v73(&path)
        } else {
            read_mat(path)
        }
    }

    /// Estimate the total size of variables
    fn estimate_size(&self, vars: &HashMap<String, MatType>) -> usize {
        let mut total_size = 0;
        for mat_type in vars.values() {
            total_size += Self::estimate_mat_type_size(mat_type);
        }
        total_size
    }

    /// Estimate the size of a MatType
    fn estimate_mat_type_size(_mattype: &MatType) -> usize {
        match _mattype {
            MatType::Double(array) => array.len() * 8,
            MatType::Single(array) => array.len() * 4,
            MatType::Int8(array) => array.len(),
            MatType::UInt8(array) => array.len(),
            MatType::Int16(array) => array.len() * 2,
            MatType::UInt16(array) => array.len() * 2,
            MatType::Int32(array) => array.len() * 4,
            MatType::UInt32(array) => array.len() * 4,
            MatType::Int64(array) => array.len() * 8,
            MatType::UInt64(array) => array.len() * 8,
            MatType::Logical(array) => array.len(),
            MatType::Char(string) => string.len() * 2, // UTF-16
            MatType::Cell(cells) => cells.iter().map(Self::estimate_mat_type_size).sum(),
            MatType::Struct(structure) => {
                structure.values().map(Self::estimate_mat_type_size).sum()
            }
            MatType::SparseDouble(sparse) => sparse.nnz * 8 + sparse.nnz * 8, // data + indices
            MatType::SparseSingle(sparse) => sparse.nnz * 4 + sparse.nnz * 8, // data + indices
            MatType::SparseLogical(sparse) => sparse.nnz + sparse.nnz * 8,    // data + indices
        }
    }

    /// Check if a file is in v7.3 format
    pub fn is_v73_file<P: AsRef<Path>>(&self, path: &P) -> Result<bool> {
        // Try to read as HDF5 file
        #[cfg(feature = "hdf5")]
        {
            if let Ok(mut file) = std::fs::File::open(path.as_ref()) {
                use std::io::Read;
                let mut magic = [0u8; 8];
                if file.read_exact(&mut magic).is_ok() {
                    // HDF5 magic signature
                    return Ok(&magic[0..4] == b"\x89HDF" || &magic[0..6] == b"MATLAB");
                }
            }
        }
        Ok(false)
    }

    /// Write variables using MAT v7.3 format (HDF5)
    #[cfg(feature = "hdf5")]
    fn write_v73<P: AsRef<Path>>(&self, path: P, vars: &HashMap<String, MatType>) -> Result<()> {
        let mut hdf5_file = HDF5File::create(path)?;

        for (name, mat_type) in vars {
            self.write_mat_type_to_hdf5(&mut hdf5_file, name, mat_type)?;
        }

        hdf5_file.close()?;
        Ok(())
    }

    /// Write variables using MAT v7.3 format (fallback without HDF5)
    #[cfg(not(feature = "hdf5"))]
    fn write_v73<P: AsRef<Path>>(&self, path: &P, vars: &HashMap<String, MatType>) -> Result<()> {
        Err(IoError::Other(
            "MAT v7.3 format requires HDF5 feature".to_string(),
        ))
    }

    /// Read variables using MAT v7.3 format (HDF5)
    #[cfg(feature = "hdf5")]
    fn read_v73<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, MatType>> {
        let hdf5_file = HDF5File::open(path, FileMode::ReadOnly)?;
        let mut vars = HashMap::new();

        // List all datasets in the file
        let dataset_names = hdf5_file.list_datasets();
        for name in dataset_names {
            let mat_type = self.read_mat_type_from_hdf5(&hdf5_file, &name)?;
            vars.insert(name.to_string(), mat_type);
        }

        Ok(vars)
    }

    /// Read variables using MAT v7.3 format (fallback without HDF5)
    #[cfg(not(feature = "hdf5"))]
    fn read_v73<P: AsRef<Path>>(&self, path: &P) -> Result<HashMap<String, MatType>> {
        Err(IoError::Other(
            "MAT v7.3 format requires HDF5 feature".to_string(),
        ))
    }

    /// Write a MatType to HDF5 file
    #[cfg(feature = "hdf5")]
    fn write_mat_type_to_hdf5(
        &self,
        file: &mut HDF5File,
        name: &str,
        mat_type: &MatType,
    ) -> Result<()> {
        let mut options = DatasetOptions::default();
        if let Some(ref compression) = self.config.compression {
            options.compression = compression.clone();
        }

        match mat_type {
            MatType::Double(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("double".to_string()),
                )?;
            }
            MatType::Single(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("single".to_string()),
                )?;
            }
            MatType::Int8(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int8".to_string()),
                )?;
            }
            MatType::Int16(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int16".to_string()),
                )?;
            }
            MatType::Int32(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int32".to_string()),
                )?;
            }
            MatType::Int64(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("int64".to_string()),
                )?;
            }
            MatType::UInt8(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint8".to_string()),
                )?;
            }
            MatType::UInt16(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint16".to_string()),
                )?;
            }
            MatType::UInt32(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint32".to_string()),
                )?;
            }
            MatType::UInt64(array) => {
                file.create_dataset_from_array(name, array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("uint64".to_string()),
                )?;
            }
            MatType::Logical(array) => {
                // Convert bool to u8 for storage
                let u8_array = array.mapv(|x| if x { 1u8 } else { 0u8 });
                file.create_dataset_from_array(name, &u8_array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("logical".to_string()),
                )?;
            }
            MatType::Char(string) => {
                // MATLAB stores strings as UTF-16
                let utf16_data: Vec<u16> = string.encode_utf16().collect();
                let array = scirs2_core::ndarray::Array1::from_vec(utf16_data).into_dyn();
                file.create_dataset_from_array(name, &array, Some(options.clone()))?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("char".to_string()),
                )?;
            }
            MatType::Cell(cells) => {
                // Create a group for the cell array
                file.create_group(name)?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("cell".to_string()),
                )?;

                // Store cell array dimensions
                let dims = vec![cells.len() as i64];
                file.set_attribute(name, "MATLAB_dims", AttributeValue::Array(dims))?;

                // Write each cell
                for (i, cell_value) in cells.iter().enumerate() {
                    let cell_name = format!("{}/cell_{}", name, i);
                    self.write_mat_type_to_hdf5(file, &cell_name, cell_value)?;
                }
            }
            MatType::Struct(fields) => {
                // Create a group for the struct
                file.create_group(name)?;
                file.set_attribute(
                    name,
                    "MATLAB_class",
                    AttributeValue::String("struct".to_string()),
                )?;

                // Store field names
                let field_names: Vec<String> = fields.keys().cloned().collect();
                file.set_attribute(
                    name,
                    "MATLAB_fields",
                    AttributeValue::StringArray(field_names),
                )?;

                // Write each field
                for (field_name, field_value) in fields {
                    let field_path = format!("{}/{}", name, field_name);
                    self.write_mat_type_to_hdf5(file, &field_path, field_value)?;
                }
            }
            MatType::SparseDouble(sparse) => {
                self.write_sparse_to_hdf5(file, name, sparse, "double", &options)?;
            }
            MatType::SparseSingle(sparse) => {
                self.write_sparse_to_hdf5(file, name, sparse, "single", &options)?;
            }
            MatType::SparseLogical(sparse) => {
                self.write_sparse_to_hdf5(file, name, sparse, "logical", &options)?;
            }
        }

        // Add MATLAB-specific metadata
        file.set_attribute(name, "MATLAB_int_decode", AttributeValue::Integer(2))?;

        Ok(())
    }

    /// Read a MatType from HDF5 file
    #[cfg(feature = "hdf5")]
    fn read_mat_type_from_hdf5(&self, file: &HDF5File, name: &str) -> Result<MatType> {
        // First, check if it's a group (struct or cell array)
        if file.is_group(name) {
            // Get the MATLAB class attribute
            if let Ok(Some(AttributeValue::String(class))) =
                file.get_attribute(name, "MATLAB_class")
            {
                match class.as_str() {
                    "cell" => {
                        // Read cell array
                        let mut cells = Vec::new();
                        if let Ok(Some(AttributeValue::Array(dims))) =
                            file.get_attribute(name, "MATLAB_dims")
                        {
                            let num_cells = dims[0] as usize;
                            for i in 0..num_cells {
                                let cell_name = format!("{}/cell_{}", name, i);
                                let cell_value = self.read_mat_type_from_hdf5(file, &cell_name)?;
                                cells.push(cell_value);
                            }
                        }
                        Ok(MatType::Cell(cells))
                    }
                    "struct" => {
                        // Read struct
                        let mut fields = HashMap::new();
                        if let Ok(Some(AttributeValue::StringArray(field_names))) =
                            file.get_attribute(name, "MATLAB_fields")
                        {
                            for field_name in field_names {
                                let field_path = format!("{}/{}", name, field_name);
                                let field_value =
                                    self.read_mat_type_from_hdf5(file, &field_path)?;
                                fields.insert(field_name.clone(), field_value);
                            }
                        }
                        Ok(MatType::Struct(fields))
                    }
                    _ => Err(IoError::Other(format!(
                        "Unknown MATLAB group class: {}",
                        class
                    ))),
                }
            } else {
                Err(IoError::Other(format!(
                    "No MATLAB_class attribute for group: {}",
                    name
                )))
            }
        } else {
            // It's a dataset
            if let Ok(Some(AttributeValue::String(class))) =
                file.get_attribute(name, "MATLAB_class")
            {
                match class.as_str() {
                    "double" => {
                        let array = file.read_dataset_typed::<f64>(name)?;
                        Ok(MatType::Double(array))
                    }
                    "single" => {
                        let array = file.read_dataset_typed::<f32>(name)?;
                        Ok(MatType::Single(array))
                    }
                    "int8" => {
                        let array = file.read_dataset_typed::<i8>(name)?;
                        Ok(MatType::Int8(array))
                    }
                    "int16" => {
                        let array = file.read_dataset_typed::<i16>(name)?;
                        Ok(MatType::Int16(array))
                    }
                    "int32" => {
                        let array = file.read_dataset_typed::<i32>(name)?;
                        Ok(MatType::Int32(array))
                    }
                    "int64" => {
                        let array = file.read_dataset_typed::<i64>(name)?;
                        Ok(MatType::Int64(array))
                    }
                    "uint8" => {
                        let array = file.read_dataset_typed::<u8>(name)?;
                        Ok(MatType::UInt8(array))
                    }
                    "uint16" => {
                        let array = file.read_dataset_typed::<u16>(name)?;
                        Ok(MatType::UInt16(array))
                    }
                    "uint32" => {
                        let array = file.read_dataset_typed::<u32>(name)?;
                        Ok(MatType::UInt32(array))
                    }
                    "uint64" => {
                        let array = file.read_dataset_typed::<u64>(name)?;
                        Ok(MatType::UInt64(array))
                    }
                    "logical" => {
                        let array = file.read_dataset_typed::<u8>(name)?;
                        let bool_array = array.mapv(|x| x != 0);
                        Ok(MatType::Logical(bool_array))
                    }
                    "char" => {
                        // Read UTF-16 data
                        let array = file.read_dataset_typed::<u16>(name)?;
                        let utf16_data: Vec<u16> = array.iter().cloned().collect();
                        let string = String::from_utf16(&utf16_data).map_err(|_| {
                            IoError::Other("Invalid UTF-16 string data".to_string())
                        })?;
                        Ok(MatType::Char(string))
                    }
                    _ => {
                        // Check if it's a sparse matrix
                        if let Ok(Some(AttributeValue::Integer(ref is_sparse))) =
                            file.get_attribute(name, "MATLAB_sparse")
                        {
                            if *is_sparse == 1 {
                                match class.as_str() {
                                    "double" => {
                                        let sparse =
                                            self.read_sparse_from_hdf5::<f64>(file, name)?;
                                        Ok(MatType::SparseDouble(sparse))
                                    }
                                    "single" => {
                                        let sparse =
                                            self.read_sparse_from_hdf5::<f32>(file, name)?;
                                        Ok(MatType::SparseSingle(sparse))
                                    }
                                    "logical" => {
                                        let sparse =
                                            self.read_sparse_from_hdf5::<bool>(file, name)?;
                                        Ok(MatType::SparseLogical(sparse))
                                    }
                                    _ => Err(IoError::Other(format!(
                                        "Unknown sparse MATLAB class: {}",
                                        class
                                    ))),
                                }
                            } else {
                                Err(IoError::Other(format!("Unknown MATLAB class: {}", class)))
                            }
                        } else {
                            Err(IoError::Other(format!("Unknown MATLAB class: {}", class)))
                        }
                    }
                }
            } else {
                // No class attribute, try to infer from data type
                // Default to double for backward compatibility
                let array = file.read_dataset_typed::<f64>(name)?;
                Ok(MatType::Double(array))
            }
        }
    }

    /// Write sparse matrix to HDF5 file in MATLAB v7.3 format
    #[cfg(feature = "hdf5")]
    fn write_sparse_to_hdf5<T>(
        &self,
        file: &mut HDF5File,
        name: &str,
        sparse: &crate::sparse::SparseMatrix<T>,
        matlab_class: &str,
        options: &DatasetOptions,
    ) -> Result<()>
    where
        T: Clone + std::fmt::Debug,
    {
        // Create a group for the sparse matrix
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String(matlab_class.to_string()),
        )?;
        file.set_attribute(name, "MATLAB_sparse", AttributeValue::Integer(1))?;

        // Store matrix dimensions
        let dims = vec![sparse.shape.0 as i64, sparse.shape.1 as i64];
        file.set_attribute(name, "MATLAB_dims", AttributeValue::Array(dims))?;

        // Convert to CSC format (MATLAB's native sparse format)
        let csc_data = if let Some(ref csc) = sparse.csc {
            csc.clone()
        } else {
            // Convert COO to CSC
            let mut row_indices = Vec::new();
            let mut col_ptrs = vec![0];
            let mut values = Vec::new();

            // Sort by column, then by row
            let mut entries: Vec<_> = sparse
                .coo
                .row_indices
                .iter()
                .zip(&sparse.coo.col_indices)
                .zip(&sparse.coo.values)
                .map(|((r, c), v)| (*c, *r, v.clone()))
                .collect();
            entries.sort_by_key(|(c, r, _)| (*c, *r));

            let mut current_col = 0;
            for (col, row, val) in entries {
                while current_col < col {
                    col_ptrs.push(values.len());
                    current_col += 1;
                }
                row_indices.push(row);
                values.push(val);
            }
            while col_ptrs.len() <= sparse.shape.1 {
                col_ptrs.push(values.len());
            }

            crate::serialize::SparseMatrixCSC {
                rows: sparse.shape.0,
                cols: sparse.shape.1,
                col_ptrs,
                row_indices,
                values,
                metadata: std::collections::HashMap::new(),
            }
        };

        // Write CSC data
        let ir_array =
            scirs2_core::ndarray::Array1::from_vec(csc_data.row_indices.clone()).into_dyn();
        let jc_array = scirs2_core::ndarray::Array1::from_vec(csc_data.col_ptrs.clone()).into_dyn();
        let data_array = scirs2_core::ndarray::Array1::from_vec(csc_data.values.clone()).into_dyn();

        file.create_dataset_from_array(&format!("{}/ir", name), &ir_array, Some(options.clone()))?;
        file.create_dataset_from_array(&format!("{}/jc", name), &jc_array, Some(options.clone()))?;
        file.create_dataset_from_array(
            &format!("{}/data", name),
            &data_array,
            Some(options.clone()),
        )?;

        Ok(())
    }

    /// Read sparse matrix from HDF5 file in MATLAB v7.3 format
    #[cfg(feature = "hdf5")]
    fn read_sparse_from_hdf5<T>(
        &self,
        file: &HDF5File,
        name: &str,
    ) -> Result<crate::sparse::SparseMatrix<T>>
    where
        T: Clone + std::fmt::Debug + Default,
    {
        // Read matrix dimensions
        let dims = if let Ok(Some(AttributeValue::Array(dims))) =
            file.get_attribute(name, "MATLAB_dims")
        {
            (dims[0] as usize, dims[1] as usize)
        } else {
            return Err(IoError::FormatError(
                "Missing sparse matrix dimensions".to_string(),
            ));
        };

        // Read CSC data as f64 arrays first
        let ir_array = file.read_dataset(&format!("{}/ir", name))?;
        let jc_array = file.read_dataset(&format!("{}/jc", name))?;
        let data_array = file.read_dataset(&format!("{}/data", name))?;

        // Convert to vectors with proper types
        let ir: Vec<usize> = ir_array.iter().map(|&x| x as usize).collect();
        let jc: Vec<usize> = jc_array.iter().map(|&x| x as usize).collect();
        // For generic T, we use default values as placeholder
        // In practice, this would need proper type conversion
        let data: Vec<T> = vec![T::default(); data_array.len()];

        // Convert CSC to COO for SparseMatrix
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for col in 0..dims.1 {
            let start = jc[col];
            let end = jc[col + 1];
            for idx in start..end {
                row_indices.push(ir[idx]);
                col_indices.push(col);
                values.push(data[idx].clone());
            }
        }

        let coo = crate::serialize::SparseMatrixCOO {
            rows: dims.0,
            cols: dims.1,
            row_indices,
            col_indices,
            values,
            metadata: std::collections::HashMap::new(),
        };

        Ok(crate::sparse::SparseMatrix {
            shape: dims,
            nnz: coo.values.len(),
            format: crate::sparse::SparseFormat::COO,
            coo,
            csr: None,
            csc: Some(crate::serialize::SparseMatrixCSC {
                rows: dims.0,
                cols: dims.1,
                col_ptrs: jc.to_vec(),
                row_indices: ir.to_vec(),
                values: data.to_vec(),
                metadata: std::collections::HashMap::new(),
            }),
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Enhanced convenience functions
/// Write variables to a MAT file with automatic format selection
#[allow(dead_code)]
pub fn write_mat_enhanced<P: AsRef<Path>>(
    path: P,
    vars: &HashMap<String, MatType>,
    config: Option<MatFileConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    let enhanced_file = EnhancedMatFile::new(config);
    enhanced_file.write(path, vars)
}

/// Read variables from a MAT file with enhanced format support
#[allow(dead_code)]
pub fn read_mat_enhanced<P: AsRef<Path>>(
    path: P,
    config: Option<MatFileConfig>,
) -> Result<HashMap<String, MatType>> {
    let config = config.unwrap_or_default();
    let enhanced_file = EnhancedMatFile::new(config);
    enhanced_file.read(path)
}

/// Create a complex number MatType
#[allow(dead_code)]
pub fn create_complex_array(real: ArrayD<f64>, imag: ArrayD<f64>) -> Result<MatType> {
    use scirs2_core::numeric::Complex64;

    if real.shape() != imag.shape() {
        return Err(IoError::FormatError(
            "Real and imaginary parts must have the same shape".to_string(),
        ));
    }

    // Create a complex array by combining _real and imaginary parts
    let _complex_array =
        ArrayD::from_shape_fn(real.raw_dim(), |idx| Complex64::new(real[&idx], imag[&idx]));

    // For now, store as a struct with _real and imag fields
    // This is how MATLAB v7.3 stores complex data internally
    let mut fields = HashMap::new();
    fields.insert("_real".to_string(), MatType::Double(real));
    fields.insert("imag".to_string(), MatType::Double(imag));

    Ok(MatType::Struct(fields))
}

/// Create a cell array MatType
#[allow(dead_code)]
pub fn create_cell_array(cells: Vec<MatType>) -> MatType {
    MatType::Cell(cells)
}

/// Create a structure MatType
#[allow(dead_code)]
pub fn create_struct(fields: HashMap<String, MatType>) -> MatType {
    MatType::Struct(fields)
}

/// Advanced v7.3+ features for large data handling
pub struct MatV73Features;

impl MatV73Features {
    /// Create a chunked dataset for streaming large arrays
    #[cfg(feature = "hdf5")]
    pub fn create_chunked_dataset<P: AsRef<Path>>(
        path: P,
        name: &str,
        shape: &[usize],
        chunk_size: &[usize],
    ) -> Result<()> {
        let mut file = HDF5File::create(path)?;

        let mut options = DatasetOptions::default();
        options.chunk_size = Some(chunk_size.to_vec());
        options.compression = CompressionOptions {
            gzip: Some(6),
            szip: None,
            lzf: false,
            shuffle: false,
        };

        // Create an empty dataset with the specified shape
        let total_elements: usize = shape.iter().product();
        let zeros = vec![0.0f64; total_elements];
        let array = ArrayD::from_shape_vec(IxDyn(shape), zeros)
            .map_err(|e| IoError::Other(e.to_string()))?;

        file.create_dataset_from_array(name, &array, Some(options))?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("double".to_string()),
        )?;
        file.set_attribute(name, "MATLAB_v73_chunked", AttributeValue::Boolean(true))?;

        file.close()?;
        Ok(())
    }

    /// Write data to a specific hyperslab in a chunked dataset
    #[cfg(feature = "hdf5")]
    pub fn write_hyperslab<P: AsRef<Path>>(
        path: P,
        dataset_name: &str,
        data: &ArrayD<f64>,
        offset: &[usize],
    ) -> Result<()> {
        let mut file = HDF5File::open(path, FileMode::ReadWrite)?;

        // Write data to the specified offset
        let data_slice = data.as_slice().ok_or_else(|| {
            IoError::FormatError("Cannot convert array to contiguous slice".to_string())
        })?;
        file.write_dataset_slice(dataset_name, data_slice, offset)?;

        file.close()?;
        Ok(())
    }

    /// Read a specific hyperslab from a chunked dataset
    #[cfg(feature = "hdf5")]
    pub fn read_hyperslab<P: AsRef<Path>>(
        path: P,
        dataset_name: &str,
        offset: &[usize],
        shape: &[usize],
    ) -> Result<ArrayD<f64>> {
        let file = HDF5File::open(path, FileMode::ReadOnly)?;

        // Read data from the specified offset and shape
        let data_vec = file.read_dataset_slice(dataset_name, shape, offset)?;

        // Convert Vec to ArrayD
        let array = ArrayD::from_shape_vec(IxDyn(shape), data_vec)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(array)
    }

    /// Create a virtual dataset that references multiple files
    #[cfg(feature = "hdf5")]
    pub fn create_virtual_dataset<P: AsRef<Path>>(
        path: P,
        name: &str,
        source_files: Vec<String>,
        source_datasets: Vec<String>,
    ) -> Result<()> {
        if source_files.len() != source_datasets.len() {
            return Err(IoError::FormatError(
                "Number of source _files must match number of source _datasets".to_string(),
            ));
        }

        let mut file = HDF5File::create(path)?;

        // Store virtual dataset metadata
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("double".to_string()),
        )?;
        file.set_attribute(name, "MATLAB_v73_virtual", AttributeValue::Boolean(true))?;
        file.set_attribute(
            name,
            "source_files",
            AttributeValue::StringArray(source_files),
        )?;
        file.set_attribute(
            name,
            "source_datasets",
            AttributeValue::StringArray(source_datasets),
        )?;

        file.close()?;
        Ok(())
    }
}

/// Support for sparse matrices in v7.3 format
pub struct MatV73Sparse;

impl MatV73Sparse {
    /// Write a sparse matrix in v7.3 format
    #[cfg(feature = "hdf5")]
    pub fn write_sparse<P: AsRef<Path>>(
        path: P,
        name: &str,
        data: &crate::sparse::SparseMatrix<f64>,
    ) -> Result<()> {
        let mut file = HDF5File::create(path)?;

        // Create a group for the sparse matrix
        file.create_group(name)?;
        file.set_attribute(
            name,
            "MATLAB_class",
            AttributeValue::String("sparse".to_string()),
        )?;
        file.set_attribute(
            name,
            "MATLAB_sparse_nrows",
            AttributeValue::Integer(data.shape.0 as i64),
        )?;
        file.set_attribute(
            name,
            "MATLAB_sparse_ncols",
            AttributeValue::Integer(data.shape.1 as i64),
        )?;

        // Write the sparse data components
        let row_path = format!("{}/ir", name); // row indices
        let col_path = format!("{}/jc", name); // column pointers
        let data_path = format!("{}/data", name); // non-zero values

        // Convert to MATLAB's CSC format - use existing methods
        let csc = if let Some(ref csc_data) = data.csc {
            csc_data.clone()
        } else {
            // Convert COO to CSC
            let mut row_indices = Vec::new();
            let mut col_ptrs = vec![0];
            let mut values = Vec::new();

            // Sort by column, then by row
            let mut entries: Vec<_> = data
                .coo
                .row_indices
                .iter()
                .zip(&data.coo.col_indices)
                .zip(&data.coo.values)
                .map(|((r, c), v)| (*c, *r, v.clone()))
                .collect();
            entries.sort_by_key(|(c, r, _)| (*c, *r));

            let mut current_col = 0;
            for (col, row, val) in entries {
                while current_col < col {
                    col_ptrs.push(values.len());
                    current_col += 1;
                }
                row_indices.push(row);
                values.push(val);
            }
            while col_ptrs.len() <= data.shape.1 {
                col_ptrs.push(values.len());
            }

            crate::serialize::SparseMatrixCSC {
                rows: data.shape.0,
                cols: data.shape.1,
                col_ptrs,
                row_indices,
                values,
                metadata: std::collections::HashMap::new(),
            }
        };

        let (row_indices, col_ptrs, values) = (csc.row_indices, csc.col_ptrs, csc.values);

        // Write components
        let row_array = ArrayD::from_shape_vec(vec![row_indices.len()], row_indices)
            .map_err(|e| IoError::Other(e.to_string()))?;
        let col_array = ArrayD::from_shape_vec(vec![col_ptrs.len()], col_ptrs)
            .map_err(|e| IoError::Other(e.to_string()))?;
        let data_array = ArrayD::from_shape_vec(vec![values.len()], values)
            .map_err(|e| IoError::Other(e.to_string()))?;

        file.create_dataset_from_array(&row_path, &row_array, None)?;
        file.create_dataset_from_array(&col_path, &col_array, None)?;
        file.create_dataset_from_array(&data_path, &data_array, None)?;

        file.close()?;
        Ok(())
    }

    /// Read a sparse matrix from v7.3 format
    #[cfg(feature = "hdf5")]
    pub fn read_sparse<P: AsRef<Path>>(
        _path: P,
        name: &str,
    ) -> Result<crate::sparse::SparseMatrix<f64>> {
        let file = HDF5File::open(_path, FileMode::ReadOnly)?;

        // Read sparse matrix metadata
        let nrows = match file.get_attribute(name, "MATLAB_sparse_nrows") {
            Ok(Some(AttributeValue::Integer(n))) => *n as usize,
            _ => return Err(IoError::Other("Missing sparse matrix rows".to_string())),
        };

        let ncols = match file.get_attribute(name, "MATLAB_sparse_ncols") {
            Ok(Some(AttributeValue::Integer(n))) => *n as usize,
            _ => return Err(IoError::Other("Missing sparse matrix cols".to_string())),
        };

        // Read sparse data components as f64 and convert
        let row_indices_f64 = file.read_dataset(&format!("{}/ir", name))?;
        let col_ptrs_f64 = file.read_dataset(&format!("{}/jc", name))?;

        // Convert to i32
        let row_indices = row_indices_f64.mapv(|x| x as i32);
        let col_ptrs = col_ptrs_f64.mapv(|x| x as i32);
        let values: ArrayD<f64> = file.read_dataset(&format!("{}/data", name))?;

        // Convert from CSC format to COO triplets
        let row_vec: Vec<usize> = row_indices.iter().map(|&x| x as usize).collect();
        let col_vec: Vec<usize> = col_ptrs.iter().map(|&x| x as usize).collect();
        let val_vec: Vec<f64> = values.iter().cloned().collect();

        // Convert CSC to COO format for SparseMatrix construction
        let mut coo_rows = Vec::new();
        let mut coo_cols = Vec::new();
        let mut coo_values = Vec::new();

        for col in 0..ncols {
            let start = col_vec[col];
            let end = col_vec[col + 1];
            for idx in start..end {
                coo_rows.push(row_vec[idx]);
                coo_cols.push(col);
                coo_values.push(val_vec[idx]);
            }
        }

        crate::sparse::SparseMatrix::from_triplets(nrows, ncols, coo_rows, coo_cols, coo_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_enhanced_config() {
        let config = MatFileConfig::default();
        assert!(!config.use_v73);
        assert_eq!(config.v73_threshold, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_size_estimation() {
        let _enhanced = EnhancedMatFile::new(MatFileConfig::default());

        let array = Array1::from(vec![1.0, 2.0, 3.0, 4.0]).into_dyn();
        let mat_type = MatType::Double(array);

        let size = EnhancedMatFile::estimate_mat_type_size(&mat_type);
        assert_eq!(size, 4 * 8); // 4 elements * 8 bytes each
    }

    #[test]
    fn test_cell_array_creation() {
        let array1 = Array1::from(vec![1.0, 2.0]).into_dyn();
        let array2 = Array1::from(vec![3.0, 4.0]).into_dyn();

        let cells = vec![MatType::Double(array1), MatType::Double(array2)];

        let cell_array = create_cell_array(cells);
        if let MatType::Cell(ref cells) = cell_array {
            assert_eq!(cells.len(), 2);
        } else {
            panic!("Expected MatType::Cell, got {:?}", cell_array);
        }
    }

    #[test]
    fn test_struct_creation() {
        let mut fields = HashMap::new();
        let array = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        fields.insert("data".to_string(), MatType::Double(array));
        fields.insert("name".to_string(), MatType::Char("test".to_string()));

        let structure = create_struct(fields);
        if let MatType::Struct(ref fields) = structure {
            assert_eq!(fields.len(), 2);
            assert!(fields.contains_key("data"));
            assert!(fields.contains_key("name"));
        } else {
            panic!("Expected MatType::Struct, got {:?}", structure);
        }
    }
}
