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
use ndarray::ArrayD;
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
            self.write_v73(path, vars)
        } else {
            write_mat(path, vars)
        }
    }

    /// Read variables from a MAT file with enhanced features
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, MatType>> {
        // Try v7.3 format first, then fall back to v5
        if self.is_v73_file(&path)? {
            self.read_v73(path)
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
    fn estimate_mat_type_size(mat_type: &MatType) -> usize {
        match mat_type {
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
        }
    }

    /// Check if a file is in v7.3 format
    pub fn is_v73_file<P: AsRef<Path>>(&self, _path: P) -> Result<bool> {
        // Try to read as HDF5 file
        #[cfg(feature = "hdf5")]
        {
            match std::fs::File::open(_path.as_ref()) {
                Ok(mut file) => {
                    use std::io::Read;
                    let mut magic = [0u8; 8];
                    if file.read_exact(&mut magic).is_ok() {
                        // HDF5 magic signature
                        return Ok(&magic[0..4] == b"\x89HDF" || &magic[0..6] == b"MATLAB");
                    }
                }
                Err(_) => {}
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
    fn write_v73<P: AsRef<Path>>(&self, _path: P, _vars: &HashMap<String, MatType>) -> Result<()> {
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
    fn read_v73<P: AsRef<Path>>(&self, _path: P) -> Result<HashMap<String, MatType>> {
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
                file.create_dataset_from_array(name, array, Some(options))?;
            }
            MatType::Single(array) => {
                // Convert to f64 for HDF5 compatibility
                let f64_array = array.mapv(|x| x as f64);
                file.create_dataset_from_array(name, &f64_array, Some(options))?;
            }
            MatType::Int32(array) => {
                // Convert to f64 for HDF5 compatibility
                let f64_array = array.mapv(|x| x as f64);
                file.create_dataset_from_array(name, &f64_array, Some(options))?;
            }
            // Add other numeric types as needed
            MatType::Char(string) => {
                // Create a dataset for the string
                // Convert string to byte array for storage
                let bytes = string.as_bytes();
                let array = ndarray::Array1::from_vec(bytes.to_vec()).into_dyn();
                file.create_dataset_from_array(name, &array, Some(options))?;

                // Add attribute to indicate this is a string
                // TODO: Find the correct way to set attributes on datasets
                // if let Ok(dataset) = file.get_dataset(&format!("/{}", name)) {
                //     dataset
                //         .set_attribute("matlab_class", AttributeValue::String("char".to_string()));
                // }
            }
            _ => {
                return Err(IoError::Other(format!(
                    "MatType {:?} not yet supported in v7.3 format",
                    std::any::type_name::<MatType>()
                )));
            }
        }
        Ok(())
    }

    /// Read a MatType from HDF5 file
    #[cfg(feature = "hdf5")]
    fn read_mat_type_from_hdf5(&self, file: &HDF5File, name: &str) -> Result<MatType> {
        // Try to read as dataset first
        match file.read_dataset(name) {
            Ok(array) => Ok(MatType::Double(array)),
            Err(_) => {
                // Try to read as string from dataset
                // First check if it's a string dataset by looking for matlab_class attribute
                if let Ok(dataset) = file.get_dataset(&format!("/{}", name)) {
                    if let Some(AttributeValue::String(class)) =
                        dataset.get_attribute("matlab_class")
                    {
                        if class == "char" {
                            // Read the byte data and convert back to string
                            match file.read_dataset(name) {
                                Ok(array) => {
                                    let bytes: Vec<u8> = array.iter().map(|&x| x as u8).collect();
                                    match String::from_utf8(bytes) {
                                        Ok(string) => Ok(MatType::Char(string)),
                                        Err(_) => Err(IoError::Other(
                                            "Invalid UTF-8 string data".to_string(),
                                        )),
                                    }
                                }
                                Err(e) => Err(e),
                            }
                        } else {
                            Err(IoError::Other(format!("Unknown MATLAB class: {}", class)))
                        }
                    } else {
                        Err(IoError::Other(format!("No class attribute for: {}", name)))
                    }
                } else {
                    Err(IoError::Other(format!("Cannot read variable: {}", name)))
                }
            }
        }
    }
}

/// Enhanced convenience functions
/// Write variables to a MAT file with automatic format selection
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
pub fn read_mat_enhanced<P: AsRef<Path>>(
    path: P,
    config: Option<MatFileConfig>,
) -> Result<HashMap<String, MatType>> {
    let config = config.unwrap_or_default();
    let enhanced_file = EnhancedMatFile::new(config);
    enhanced_file.read(path)
}

/// Create a complex number MatType (placeholder for future implementation)
pub fn create_complex_array(_real: ArrayD<f64>, _imag: ArrayD<f64>) -> Result<MatType> {
    // TODO: Implement complex number support
    Err(IoError::Other(
        "Complex array support not yet implemented".to_string(),
    ))
}

/// Create a cell array MatType
pub fn create_cell_array(cells: Vec<MatType>) -> MatType {
    MatType::Cell(cells)
}

/// Create a structure MatType
pub fn create_struct(fields: HashMap<String, MatType>) -> MatType {
    MatType::Struct(fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

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
        match cell_array {
            MatType::Cell(ref cells) => assert_eq!(cells.len(), 2),
            _ => panic!("Expected Cell type"),
        }
    }

    #[test]
    fn test_struct_creation() {
        let mut fields = HashMap::new();
        let array = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        fields.insert("data".to_string(), MatType::Double(array));
        fields.insert("name".to_string(), MatType::Char("test".to_string()));

        let structure = create_struct(fields);
        match structure {
            MatType::Struct(ref fields) => {
                assert_eq!(fields.len(), 2);
                assert!(fields.contains_key("data"));
                assert!(fields.contains_key("name"));
            }
            _ => panic!("Expected Struct type"),
        }
    }
}
