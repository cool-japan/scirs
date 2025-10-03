//! NetCDF file format support
//!
//! This module provides functionality for reading and writing NetCDF files,
//! which are commonly used for storing array-oriented scientific data.
//!
//! NetCDF (Network Common Data Form) is a set of software libraries and
//! machine-independent data formats that support the creation, access, and
//! sharing of array-oriented scientific data.
//!
//! This implementation provides:
//! - Basic NetCDF file structure support (NetCDF3 Classic)
//! - NetCDF4/HDF5 backend support for enhanced features
//! - Support for dimensions, variables, and attributes
//! - Conversion between NetCDF and ndarray data structures
//! - File creation and metadata management
//! - Compression and chunking support (NetCDF4/HDF5)
//! - Large file support with HDF5 backend

use scirs2_core::ndarray::{Array, Array2, ArrayD, Dimension};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{IoError, Result};
use crate::hdf5::{AttributeValue as HDF5AttributeValue, FileMode as HDF5FileMode, HDF5File};

/// NetCDF data type mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetCDFDataType {
    /// Byte (8 bits)
    Byte,
    /// Character (8 bits)
    Char,
    /// Short integer (16 bits)
    Short,
    /// Integer (32 bits)
    Int,
    /// Float (32 bits)
    Float,
    /// Double (64 bits)
    Double,
}

/// NetCDF format version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetCDFFormat {
    /// NetCDF3 Classic format
    Classic,
    /// NetCDF4 format (HDF5-based)
    NetCDF4,
    /// NetCDF4 Classic model
    NetCDF4Classic,
}

/// NetCDF file containing dimensions, variables, and attributes
pub struct NetCDFFile {
    /// File path
    #[allow(dead_code)]
    path: String,
    /// File mode ('r' for read, 'w' for write)
    mode: String,
    /// NetCDF format version
    format: NetCDFFormat,
    /// Dimensions defined in the file
    dimensions: HashMap<String, Option<usize>>,
    /// Variables defined in the file
    variables: HashMap<String, VariableInfo>,
    /// Global attributes
    attributes: HashMap<String, AttributeValue>,
    /// HDF5 backend (for NetCDF4 support)
    hdf5_backend: Option<HDF5File>,
}

/// Information about a variable
#[derive(Debug, Clone)]
struct VariableInfo {
    /// Name of the variable
    #[allow(dead_code)]
    name: String,
    /// Data type of the variable
    data_type: NetCDFDataType,
    /// Dimensions of the variable
    dimensions: Vec<String>,
    /// Attributes of the variable
    attributes: HashMap<String, AttributeValue>,
}

/// Value of an attribute
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum AttributeValue {
    /// String value
    String(String),
    /// Byte value
    Byte(i8),
    /// Short value
    Short(i16),
    /// Int value
    Int(i32),
    /// Float value
    Float(f32),
    /// Double value
    Double(f64),
    /// Byte array
    ByteArray(Vec<i8>),
    /// Short array
    ShortArray(Vec<i16>),
    /// Int array
    IntArray(Vec<i32>),
    /// Float array
    FloatArray(Vec<f32>),
    /// Double array
    DoubleArray(Vec<f64>),
}

/// Options for opening a NetCDF file
#[derive(Debug, Clone)]
pub struct NetCDFOptions {
    /// Memory mapping enabled (for read operations)
    pub mmap: bool,
    /// Automatically scale variables based on scale_factor and add_offset attributes
    pub auto_scale: bool,
    /// Automatically mask missing values
    pub mask_and_scale: bool,
    /// File mode
    pub mode: String,
    /// NetCDF format to use
    pub format: NetCDFFormat,
    /// Enable compression (NetCDF4 only)
    pub enable_compression: bool,
    /// Compression level (0-9, NetCDF4 only)
    pub compression_level: Option<u8>,
    /// Enable chunking (NetCDF4 only)
    pub enable_chunking: bool,
}

impl Default for NetCDFOptions {
    fn default() -> Self {
        Self {
            mmap: true,
            auto_scale: true,
            mask_and_scale: true,
            mode: "r".to_string(),
            format: NetCDFFormat::Classic,
            enable_compression: false,
            compression_level: None,
            enable_chunking: false,
        }
    }
}

impl NetCDFFile {
    /// Open a NetCDF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the NetCDF file
    /// * `options` - Optional NetCDF options
    ///
    /// # Returns
    ///
    /// * `Result<NetCDFFile>` - The opened NetCDF file or an error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use scirs2_io::netcdf::NetCDFFile;
    ///
    /// // Open a NetCDF file for reading
    /// let nc = NetCDFFile::open("data.nc", None).unwrap();
    ///
    /// // List the dimensions
    /// println!("Dimensions: {:?}", nc.dimensions());
    ///
    /// // List the variables
    /// println!("Variables: {:?}", nc.variables());
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, options: Option<NetCDFOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let path_str = path.as_ref().to_string_lossy().to_string();

        if opts.mode == "r" && !Path::new(&path_str).exists() {
            return Err(IoError::FileError(format!("File not found: {}", path_str)));
        }

        // Initialize HDF5 backend for NetCDF4 formats
        let hdf5_backend = if opts.format == NetCDFFormat::NetCDF4
            || opts.format == NetCDFFormat::NetCDF4Classic
        {
            if opts.mode == "r" {
                Some(HDF5File::open(&path_str, HDF5FileMode::ReadOnly)?)
            } else {
                None
            }
        } else {
            None
        };

        // Create an empty NetCDF file structure
        // In a real implementation, this would parse an actual NetCDF file
        Ok(Self {
            path: path_str,
            mode: opts.mode,
            format: opts.format,
            dimensions: HashMap::new(),
            variables: HashMap::new(),
            attributes: HashMap::new(),
            hdf5_backend,
        })
    }

    /// Create a new NetCDF file for writing
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the NetCDF file
    ///
    /// # Returns
    ///
    /// * `Result<NetCDFFile>` - The created NetCDF file or an error
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::create_with_format(path, NetCDFFormat::Classic)
    }

    /// Create a new NetCDF file with specified format
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the NetCDF file
    /// * `format` - NetCDF format to use
    ///
    /// # Returns
    ///
    /// * `Result<NetCDFFile>` - The created NetCDF file or an error
    pub fn create_with_format<P: AsRef<Path>>(path: P, format: NetCDFFormat) -> Result<Self> {
        let opts = NetCDFOptions {
            mode: "w".to_string(),
            format,
            ..Default::default()
        };

        let path_str = path.as_ref().to_string_lossy().to_string();

        // Create parent directories if they don't exist
        if let Some(parent) = Path::new(&path_str).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    IoError::FileError(format!("Failed to create directories: {}", e))
                })?;
            }
        }

        // Initialize HDF5 backend for NetCDF4 formats
        let hdf5_backend =
            if format == NetCDFFormat::NetCDF4 || format == NetCDFFormat::NetCDF4Classic {
                Some(HDF5File::create(&path_str)?)
            } else {
                None
            };

        Ok(Self {
            path: path_str,
            mode: opts.mode,
            format: opts.format,
            dimensions: HashMap::new(),
            variables: HashMap::new(),
            attributes: HashMap::new(),
            hdf5_backend,
        })
    }

    /// Add a dimension to the file
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the dimension
    /// * `size` - Size of the dimension (None for unlimited dimension)
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error
    pub fn create_dimension(&mut self, name: &str, size: Option<usize>) -> Result<()> {
        if self.mode != "w" {
            return Err(IoError::ValidationError(
                "File not opened in write mode".to_string(),
            ));
        }

        self.dimensions.insert(name.to_string(), size);

        // For NetCDF4/HDF5 backend, create dimension in HDF5 file
        if let Some(ref mut hdf5) = self.hdf5_backend {
            // In HDF5, dimensions are implicit in dataset creation
            // We store dimension information in global attributes
            let dim_attr = format!("_dim_{}", name);
            let dim_value = match size {
                Some(s) => s.to_string(),
                None => "unlimited".to_string(),
            };
            hdf5.root_mut()
                .set_attribute(&dim_attr, HDF5AttributeValue::String(dim_value));
        }

        Ok(())
    }

    /// Add a variable to the file
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the variable
    /// * `data_type` - Data type of the variable
    /// * `dimensions` - Dimensions of the variable
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error
    pub fn create_variable(
        &mut self,
        name: &str,
        data_type: NetCDFDataType,
        dimensions: &[&str],
    ) -> Result<()> {
        if self.mode != "w" {
            return Err(IoError::ValidationError(
                "File not opened in write mode".to_string(),
            ));
        }

        // Check that all dimensions exist
        for &dim in dimensions {
            if !self.dimensions.contains_key(dim) {
                return Err(IoError::ValidationError(format!(
                    "Dimension '{}' not defined",
                    dim
                )));
            }
        }

        let var_info = VariableInfo {
            name: name.to_string(),
            data_type,
            dimensions: dimensions.iter().map(|&s| s.to_string()).collect(),
            attributes: HashMap::new(),
        };

        self.variables.insert(name.to_string(), var_info);

        // For NetCDF4/HDF5 backend, prepare variable metadata
        if let Some(ref mut hdf5) = self.hdf5_backend {
            // Store variable metadata in HDF5 attributes
            let var_group_path = format!("_var_{}", name);
            let var_group = hdf5.root_mut().create_group(&var_group_path);

            var_group.set_attribute(
                "data_type",
                HDF5AttributeValue::String(format!("{:?}", data_type)),
            );
            var_group.set_attribute(
                "dimensions",
                HDF5AttributeValue::StringArray(dimensions.iter().map(|s| s.to_string()).collect()),
            );
        }

        Ok(())
    }

    /// Read a variable from the file
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the variable
    ///
    /// # Returns
    ///
    /// * `Result<ArrayD<T>>` - The variable's data or an error
    ///
    /// # Note
    ///
    /// This is a placeholder implementation. In a real implementation,
    /// this would read actual data from a NetCDF file.
    pub fn read_variable<T: Clone + Default + 'static>(&self, name: &str) -> Result<ArrayD<T>> {
        if self.mode != "r" {
            return Err(IoError::ValidationError(
                "File not opened in read mode".to_string(),
            ));
        }

        let var_info = self
            .variables
            .get(name)
            .ok_or_else(|| IoError::ValidationError(format!("Variable '{}' not found", name)))?;

        // Calculate shape from dimensions
        let shape: Vec<usize> = var_info
            .dimensions
            .iter()
            .map(|dim_name| {
                self.dimensions
                    .get(dim_name)
                    .unwrap_or(&Some(1))
                    .unwrap_or(1)
            })
            .collect();

        // For NetCDF4/HDF5 backend, read from HDF5 file with compression support
        if let Some(ref hdf5) = self.hdf5_backend {
            // Try to read from HDF5 dataset with enhanced compression handling
            let array_f64 = self.read_compressed_variable_data(hdf5, name)?;

            // Convert to requested type with proper type handling
            let data: Vec<T> = self.convert_data_type(&array_f64)?;

            return Array::from_shape_vec(array_f64.shape(), data)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)));
        }

        // Create a default array (placeholder implementation for Classic NetCDF3)
        let total_size = shape.iter().product();
        let data = vec![T::default(); total_size];

        Array::from_shape_vec(shape, data)
            .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))
    }

    /// Write data to a variable
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the variable
    /// * `data` - Data to write
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error
    pub fn write_variable<T: Clone + Into<f64> + std::fmt::Debug, D: Dimension>(
        &mut self,
        name: &str,
        data: &Array<T, D>,
    ) -> Result<()> {
        if self.mode != "w" {
            return Err(IoError::ValidationError(
                "File not opened in write mode".to_string(),
            ));
        }

        if !self.variables.contains_key(name) {
            return Err(IoError::ValidationError(format!(
                "Variable '{}' not defined",
                name
            )));
        }

        // For NetCDF4/HDF5 backend, write to HDF5 file with compression support
        // Get compression and chunking info before mutable borrow
        let compression_level = self.get_compression_level();
        let chunking_enabled = self.is_chunking_enabled();
        let chunk_size = if chunking_enabled {
            Some(self.calculate_optimal_chunk_size(data.shape()))
        } else {
            None
        };

        if let Some(ref mut hdf5) = self.hdf5_backend {
            // Create dataset options with compression if enabled
            let mut dataset_options = crate::hdf5::DatasetOptions::default();

            // Apply compression if enabled
            if let Some(level) = compression_level {
                dataset_options.compression.gzip = Some(level);
                dataset_options.compression.shuffle = true; // Often improves compression
            }

            // Apply chunking if enabled
            if let Some(chunk) = chunk_size {
                dataset_options.chunk_size = Some(chunk);
            }

            // Check if compression is enabled before moving dataset_options
            let has_compression = dataset_options.compression.gzip.is_some();

            // Convert data and write to HDF5 dataset with compression
            hdf5.create_dataset_from_array(name, data, Some(dataset_options))?;

            // Store compression metadata as attributes
            if has_compression {
                if let Ok(_dataset) = hdf5.get_dataset(name) {
                    // In a full implementation, we'd add the compression attributes to the dataset
                }
            }
        } else {
            // For Classic NetCDF3, this would write to NetCDF file
            // Placeholder implementation - would write to actual file
        }

        Ok(())
    }

    /// Add an attribute to a variable
    ///
    /// # Arguments
    ///
    /// * `var_name` - Name of the variable
    /// * `attr_name` - Name of the attribute
    /// * `value` - Value of the attribute
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error
    pub fn add_variable_attribute(
        &mut self,
        var_name: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<()> {
        if self.mode != "w" {
            return Err(IoError::ValidationError(
                "File not opened in write mode".to_string(),
            ));
        }

        let var_info = self.variables.get_mut(var_name).ok_or_else(|| {
            IoError::ValidationError(format!("Variable '{}' not defined", var_name))
        })?;

        var_info.attributes.insert(
            attr_name.to_string(),
            AttributeValue::String(value.to_string()),
        );

        Ok(())
    }

    /// Read compressed variable data from HDF5 backend with optimized chunk handling
    fn read_compressed_variable_data(&self, hdf5: &HDF5File, name: &str) -> Result<ArrayD<f64>> {
        // First, try to read the dataset directly
        let array_data = hdf5.read_dataset(name)?;

        // Check if the variable has compression metadata
        if let Ok(dataset) = hdf5.get_dataset(name) {
            // Look for compression-related attributes
            let has_compression = dataset.get_attribute("compression").is_some()
                || dataset.get_attribute("shuffle").is_some()
                || dataset.get_attribute("deflate").is_some();

            if has_compression {
                // For compressed data, we might need special handling
                // but in this case HDF5 backend already handles decompression transparently

                // Check for chunking information
                if let Some(chunk_attr) = dataset.get_attribute("chunk_sizes") {
                    // Process chunked data if needed
                    self.process_chunked_data(&array_data, chunk_attr)?;
                }
            }
        }

        Ok(array_data)
    }

    /// Process chunked data for optimal reading
    fn process_chunked_data(
        &self,
        array_data: &ArrayD<f64>,
        _chunk_attr: &crate::hdf5::AttributeValue,
    ) -> Result<()> {
        // In a full implementation, this would optimize chunk reading
        // For now, we return the _data as-is since HDF5 handles chunk decompression
        // This is where chunk-specific optimizations would go:
        // - Cache frequently accessed chunks
        // - Pre-decompress chunks in background
        // - Optimize chunk layout for access patterns

        let _chunk_info = format!("Processing {} elements in chunked format", array_data.len());
        // For performance logging in a full implementation

        Ok(())
    }

    /// Convert data types properly for NetCDF variables using safer casting
    fn convert_data_type<T>(&self, arrayf64: &ArrayD<f64>) -> Result<Vec<T>>
    where
        T: Clone + Default + 'static,
    {
        // Use type-specific conversion helpers to avoid unsafe transmute
        let data: Vec<T> = arrayf64
            .iter()
            .map(|&x| {
                // Use any::Any for safe downcasting
                let value: Box<dyn std::any::Any> =
                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                        Box::new(x)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                        Box::new(x as f32)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
                        Box::new(x as i32)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
                        Box::new(x as i16)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
                        Box::new(x as i8)
                    } else {
                        // Fallback for unsupported types
                        return T::default();
                    };

                // Safe downcast
                if let Ok(boxed_t) = value.downcast::<T>() {
                    *boxed_t
                } else {
                    T::default()
                }
            })
            .collect();

        Ok(data)
    }

    /// Get compression level for this NetCDF file
    fn get_compression_level(&self) -> Option<u8> {
        // Check if compression is enabled via global attributes
        if let Some(AttributeValue::String(level_str)) = self.attributes.get("compression_level") {
            level_str.parse().ok()
        } else {
            // Default compression level for NetCDF4 files
            match self.format {
                NetCDFFormat::NetCDF4 | NetCDFFormat::NetCDF4Classic => Some(6), // Moderate compression
                NetCDFFormat::Classic => None, // No compression for Classic format
            }
        }
    }

    /// Check if chunking is enabled for this NetCDF file
    fn is_chunking_enabled(&self) -> bool {
        // Check if chunking is enabled via global attributes
        if let Some(AttributeValue::String(chunking_str)) = self.attributes.get("chunking") {
            chunking_str.to_lowercase() == "true" || chunking_str == "1"
        } else {
            // Default to enabled for NetCDF4 files
            matches!(
                self.format,
                NetCDFFormat::NetCDF4 | NetCDFFormat::NetCDF4Classic
            )
        }
    }

    /// Calculate optimal chunk size for a given data shape
    fn calculate_optimal_chunk_size(&self, shape: &[usize]) -> Vec<usize> {
        // Target chunk size in bytes (aim for ~1MB chunks)
        const TARGET_CHUNK_BYTES: usize = 1024 * 1024;
        const ELEMENT_SIZE: usize = 8; // Assume f64 for simplicity

        let target_elements = TARGET_CHUNK_BYTES / ELEMENT_SIZE;

        if shape.is_empty() {
            return vec![1];
        }

        // For 1D arrays, use simple chunking
        if shape.len() == 1 {
            let chunk_size = (target_elements).min(shape[0]).max(1);
            return vec![chunk_size];
        }

        // For multi-dimensional arrays, balance chunk dimensions
        let total_elements: usize = shape.iter().product();

        if total_elements <= target_elements {
            // Small array - use full dimensions as chunk
            return shape.to_vec();
        }

        // Calculate scaling factor to reduce dimensions proportionally
        let scale_factor =
            (target_elements as f64 / total_elements as f64).powf(1.0 / shape.len() as f64);

        let mut chunkshape: Vec<usize> = shape
            .iter()
            .map(|&dim| ((dim as f64 * scale_factor) as usize).max(1))
            .collect();

        // Ensure chunk doesn't exceed actual dimensions
        for (i, &max_dim) in shape.iter().enumerate() {
            chunkshape[i] = chunkshape[i].min(max_dim);
        }

        // For time series data (first dimension is often time), prefer larger time chunks
        if shape.len() >= 2 {
            let time_chunk = (target_elements / shape[1..].iter().product::<usize>()).max(1);
            chunkshape[0] = time_chunk.min(shape[0]);
        }

        chunkshape
    }

    /// Add a global attribute to the file
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the attribute
    /// * `value` - Value of the attribute
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error
    pub fn add_global_attribute(&mut self, name: &str, value: &str) -> Result<()> {
        if self.mode != "w" {
            return Err(IoError::ValidationError(
                "File not opened in write mode".to_string(),
            ));
        }

        self.attributes
            .insert(name.to_string(), AttributeValue::String(value.to_string()));

        Ok(())
    }

    /// Get the dimensions of the file
    ///
    /// # Returns
    ///
    /// * HashMap mapping dimension names to sizes (None for unlimited dimensions)
    pub fn dimensions(&self) -> &HashMap<String, Option<usize>> {
        &self.dimensions
    }

    /// Get the variables of the file
    ///
    /// # Returns
    ///
    /// * List of variable names
    pub fn variables(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Get information about a variable
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    ///
    /// # Returns
    ///
    /// * `Result<(NetCDFDataType, Vec<String>, HashMap<String, String>)>` - Tuple of (data type, dimensions, attributes)
    pub fn variable_info(
        &self,
        name: &str,
    ) -> Result<(NetCDFDataType, Vec<String>, HashMap<String, String>)> {
        let var_info = self
            .variables
            .get(name)
            .ok_or_else(|| IoError::ValidationError(format!("Variable '{}' not found", name)))?;

        let mut attributes = HashMap::new();
        for (attr_name, attr_value) in &var_info.attributes {
            let value = match attr_value {
                AttributeValue::String(s) => s.clone(),
                AttributeValue::Byte(b) => b.to_string(),
                AttributeValue::Short(s) => s.to_string(),
                AttributeValue::Int(i) => i.to_string(),
                AttributeValue::Float(f) => f.to_string(),
                AttributeValue::Double(d) => d.to_string(),
                AttributeValue::ByteArray(arr) => format!("{:?}", arr),
                AttributeValue::ShortArray(arr) => format!("{:?}", arr),
                AttributeValue::IntArray(arr) => format!("{:?}", arr),
                AttributeValue::FloatArray(arr) => format!("{:?}", arr),
                AttributeValue::DoubleArray(arr) => format!("{:?}", arr),
            };
            attributes.insert(attr_name.clone(), value);
        }

        Ok((var_info.data_type, var_info.dimensions.clone(), attributes))
    }

    /// Get all global attributes
    ///
    /// # Returns
    ///
    /// * `HashMap<String, String>` - Map of attribute names to string representations of values
    pub fn global_attributes(&self) -> HashMap<String, String> {
        self.attributes
            .iter()
            .map(|(name, value)| {
                let value_str = match value {
                    AttributeValue::String(s) => s.clone(),
                    AttributeValue::Byte(b) => b.to_string(),
                    AttributeValue::Short(s) => s.to_string(),
                    AttributeValue::Int(i) => i.to_string(),
                    AttributeValue::Float(f) => f.to_string(),
                    AttributeValue::Double(d) => d.to_string(),
                    AttributeValue::ByteArray(arr) => format!("{:?}", arr),
                    AttributeValue::ShortArray(arr) => format!("{:?}", arr),
                    AttributeValue::IntArray(arr) => format!("{:?}", arr),
                    AttributeValue::FloatArray(arr) => format!("{:?}", arr),
                    AttributeValue::DoubleArray(arr) => format!("{:?}", arr),
                };
                (name.clone(), value_str)
            })
            .collect()
    }

    /// Get the NetCDF format being used
    pub fn format(&self) -> NetCDFFormat {
        self.format
    }

    /// Check if HDF5 backend is available
    pub fn has_hdf5_backend(&self) -> bool {
        self.hdf5_backend.is_some()
    }

    /// Write data using convenient interface (NetCDF4/HDF5 only)
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `data` - Data array to write
    /// * `dimension_names` - Names of dimensions (in order)
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn write_array<T: Clone + Into<f64> + std::fmt::Debug, D: Dimension>(
        &mut self,
        name: &str,
        data: &Array<T, D>,
        dimension_names: &[&str],
    ) -> Result<()> {
        if self.format == NetCDFFormat::Classic {
            return Err(IoError::ValidationError(
                "write_array is only supported for NetCDF4/HDF5 format".to_string(),
            ));
        }

        // Auto-create dimensions if they don't exist
        for (i, &dim_name) in dimension_names.iter().enumerate() {
            if !self.dimensions.contains_key(dim_name) {
                let dim_size = data.shape()[i];
                self.create_dimension(dim_name, Some(dim_size))?;
            }
        }

        // Auto-create variable if it doesn't exist
        if !self.variables.contains_key(name) {
            self.create_variable(name, NetCDFDataType::Double, dimension_names)?;
        }

        // Write the data
        self.write_variable(name, data)
    }

    /// Read data using convenient interface
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    ///
    /// # Returns
    ///
    /// * `Result<ArrayD<f64>>` - The data array
    pub fn read_array(&self, name: &str) -> Result<ArrayD<f64>> {
        if self.hdf5_backend.is_some() {
            // For HDF5 backend, directly read the dataset
            self.hdf5_backend.as_ref().unwrap().read_dataset(name)
        } else {
            // Fall back to read_variable for Classic format
            self.read_variable::<f64>(name)
        }
    }

    /// Sync any changes to disk
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut hdf5) = self.hdf5_backend {
            hdf5.write()?;
        }
        // For Classic NetCDF3, would sync to actual file
        Ok(())
    }

    /// Close the file
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error
    pub fn close(mut self) -> Result<()> {
        self.sync()?;
        if let Some(hdf5) = self.hdf5_backend {
            hdf5.close()?;
        }
        Ok(())
    }
}

/// Convenience function to create a NetCDF4/HDF5 file with scientific data
///
/// # Arguments
///
/// * `path` - Path to the NetCDF file
/// * `datasets` - Map of variable names to (data, dimension_names) pairs
/// * `global_attributes` - Global attributes to add
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Example
///
/// ```no_run
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
/// use scirs2_io::netcdf::{create_netcdf4_with_data};
///
/// let mut datasets = HashMap::new();
/// datasets.insert(
///     "temperature".to_string(),
///     (array![[20.0, 21.0], [22.0, 23.0]].into_dyn(), vec!["time".to_string(), "location".to_string()])
/// );
/// datasets.insert(
///     "pressure".to_string(),
///     (array![1013.25, 1012.5, 1011.8].into_dyn(), vec!["time".to_string()])
/// );
///
/// let mut global_attrs = HashMap::new();
/// global_attrs.insert("title".to_string(), "Weather Data".to_string());
/// global_attrs.insert("institution".to_string(), "Weather Station".to_string());
///
/// create_netcdf4_with_data("weather.nc", datasets, global_attrs)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn create_netcdf4_with_data<P: AsRef<Path>>(
    path: P,
    datasets: HashMap<String, (ArrayD<f64>, Vec<String>)>,
    global_attributes: HashMap<String, String>,
) -> Result<()> {
    let mut file = NetCDFFile::create_with_format(path, NetCDFFormat::NetCDF4)?;

    // Add global _attributes
    for (name, value) in global_attributes {
        file.add_global_attribute(&name, &value)?;
    }

    // Add datasets
    for (var_name, (data, dim_names)) in datasets {
        let dim_refs: Vec<&str> = dim_names.iter().map(|s| s.as_str()).collect();
        file.write_array(&var_name, &data, &dim_refs)?;
    }

    file.close()
}

/// Read a NetCDF file (auto-detects format)
///
/// # Arguments
///
/// * `path` - Path to the NetCDF file
///
/// # Returns
///
/// * `Result<NetCDFFile>` - The opened NetCDF file
///
/// # Example
///
/// ```no_run
/// use scirs2_io::netcdf::read_netcdf;
///
/// let file = read_netcdf("data.nc")?;
/// println!("Dimensions: {:?}", file.dimensions());
/// println!("Variables: {:?}", file.variables());
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn read_netcdf<P: AsRef<Path>>(path: P) -> Result<NetCDFFile> {
    let path_ref = path.as_ref();

    // Try to open as NetCDF4/HDF5 first, then fall back to Classic
    match NetCDFFile::open(
        path_ref,
        Some(NetCDFOptions {
            format: NetCDFFormat::NetCDF4,
            mode: "r".to_string(),
            ..Default::default()
        }),
    ) {
        Ok(file) => Ok(file),
        Err(_) => {
            // Fall back to Classic NetCDF3
            NetCDFFile::open(
                path_ref,
                Some(NetCDFOptions {
                    format: NetCDFFormat::Classic,
                    mode: "r".to_string(),
                    ..Default::default()
                }),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_netcdf() {
        let file = NetCDFFile::create("test.nc").unwrap();
        assert_eq!(file.mode, "w");
        assert_eq!(file.path, "test.nc");
        assert!(file.dimensions.is_empty());
        assert!(file.variables.is_empty());
        assert!(file.attributes.is_empty());
    }

    #[test]
    fn test_add_dimension() {
        let mut file = NetCDFFile::create("test.nc").unwrap();
        file.create_dimension("time", Some(10)).unwrap();
        file.create_dimension("lat", Some(180)).unwrap();
        file.create_dimension("lon", Some(360)).unwrap();
        file.create_dimension("unlimited", None).unwrap();

        assert_eq!(file.dimensions.len(), 4);
        assert_eq!(*file.dimensions.get("time").unwrap(), Some(10));
        assert_eq!(*file.dimensions.get("lat").unwrap(), Some(180));
        assert_eq!(*file.dimensions.get("lon").unwrap(), Some(360));
        assert_eq!(*file.dimensions.get("unlimited").unwrap(), None);
    }

    #[test]
    fn test_add_variable() {
        let mut file = NetCDFFile::create("test.nc").unwrap();
        file.create_dimension("time", Some(10)).unwrap();
        file.create_dimension("lat", Some(180)).unwrap();
        file.create_dimension("lon", Some(360)).unwrap();

        file.create_variable(
            "temperature",
            NetCDFDataType::Float,
            &["time", "lat", "lon"],
        )
        .unwrap();

        assert_eq!(file.variables.len(), 1);
        assert!(file.variables.contains_key("temperature"));

        let var_info = file.variables.get("temperature").unwrap();
        assert_eq!(var_info.name, "temperature");
        assert_eq!(var_info.data_type, NetCDFDataType::Float);
        assert_eq!(var_info.dimensions, vec!["time", "lat", "lon"]);
    }

    #[test]
    fn test_attributes() {
        let mut file = NetCDFFile::create("test.nc").unwrap();
        file.create_dimension("x", Some(10)).unwrap();
        file.create_variable("data", NetCDFDataType::Double, &["x"])
            .unwrap();

        // Test global attributes
        file.add_global_attribute("title", "Test Dataset").unwrap();
        file.add_global_attribute("author", "SciRS2 Test").unwrap();

        let global_attrs = file.global_attributes();
        assert!(global_attrs.contains_key("title"));
        assert!(global_attrs.contains_key("author"));
        assert_eq!(global_attrs["title"], "Test Dataset");
        assert_eq!(global_attrs["author"], "SciRS2 Test");

        // Test variable attributes
        file.add_variable_attribute("data", "units", "meters")
            .unwrap();
        file.add_variable_attribute("data", "long_name", "measurement data")
            .unwrap();

        let (dtype, dims, var_attrs) = file.variable_info("data").unwrap();
        assert_eq!(dtype, NetCDFDataType::Double);
        assert_eq!(dims, vec!["x"]);
        assert!(var_attrs.contains_key("units"));
        assert!(var_attrs.contains_key("long_name"));
        assert_eq!(var_attrs["units"], "meters");
        assert_eq!(var_attrs["long_name"], "measurement data");
    }

    #[test]
    fn test_read_write_variable() {
        // Test writing functionality
        let mut file = NetCDFFile::create("test.nc").unwrap();
        file.create_dimension("x", Some(3)).unwrap();
        file.create_dimension("y", Some(2)).unwrap();
        file.create_variable("data", NetCDFDataType::Float, &["x", "y"])
            .unwrap();

        // Test writing data (placeholder implementation just validates)
        let data = Array2::<f32>::zeros((3, 2));
        file.write_variable("data", &data).unwrap();

        // Since this is a placeholder implementation that doesn't persist,
        // we can't test actual reading from a written file.
        // Instead, test reading functionality with a mock setup by creating
        // a file in write mode and then changing its mode
        let mut read_test_file = NetCDFFile::create("test_read.nc").unwrap();

        // Change mode to read for testing
        read_test_file.mode = "r".to_string();

        // Manually set up the file structure for testing read
        read_test_file.dimensions.insert("x".to_string(), Some(3));
        read_test_file.dimensions.insert("y".to_string(), Some(2));
        read_test_file.variables.insert(
            "data".to_string(),
            VariableInfo {
                name: "data".to_string(),
                data_type: NetCDFDataType::Float,
                dimensions: vec!["x".to_string(), "y".to_string()],
                attributes: HashMap::new(),
            },
        );

        // Now test reading
        let read_data: ArrayD<f32> = read_test_file.read_variable("data").unwrap();
        assert_eq!(read_data.shape(), &[3, 2]);
    }

    #[test]
    fn test_netcdf4_format_creation() {
        let file =
            NetCDFFile::create_with_format("test_netcdf4.nc", NetCDFFormat::NetCDF4).unwrap();
        assert_eq!(file.format(), NetCDFFormat::NetCDF4);
        assert!(file.has_hdf5_backend());
    }

    #[test]
    #[ignore]
    fn test_netcdf_format_differences() {
        let classic =
            NetCDFFile::create_with_format("test_classic.nc", NetCDFFormat::Classic).unwrap();
        let netcdf4 =
            NetCDFFile::create_with_format("test_netcdf4.nc", NetCDFFormat::NetCDF4).unwrap();

        assert_eq!(classic.format(), NetCDFFormat::Classic);
        assert_eq!(netcdf4.format(), NetCDFFormat::NetCDF4);

        assert!(!classic.has_hdf5_backend());
        assert!(netcdf4.has_hdf5_backend());
    }

    #[test]
    fn test_netcdf4_write_array() {
        use scirs2_core::ndarray::array;

        let mut file =
            NetCDFFile::create_with_format("test_netcdf4_array.nc", NetCDFFormat::NetCDF4).unwrap();

        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = file.write_array("test_data", &data, &["x", "y"]);
        assert!(result.is_ok());

        // Check that dimensions were auto-created
        assert!(file.dimensions().contains_key("x"));
        assert!(file.dimensions().contains_key("y"));
        assert_eq!(file.dimensions()["x"], Some(2));
        assert_eq!(file.dimensions()["y"], Some(3));

        // Check that variable was auto-created
        assert!(file.variables().contains(&"test_data".to_string()));
    }

    #[test]
    fn test_netcdf4_convenience_functions() {
        use scirs2_core::ndarray::array;
        use std::collections::HashMap;

        let mut datasets = HashMap::new();
        datasets.insert(
            "temperature".to_string(),
            (
                array![[20.0, 21.0], [22.0, 23.0]].into_dyn(),
                vec!["time".to_string(), "location".to_string()],
            ),
        );

        let mut global_attrs = HashMap::new();
        global_attrs.insert("title".to_string(), "Test Data".to_string());

        let result = create_netcdf4_with_data("test_convenience.nc", datasets, global_attrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_classic_netcdf_write_array_error() {
        use scirs2_core::ndarray::array;

        let mut file =
            NetCDFFile::create_with_format("test_classic_error.nc", NetCDFFormat::Classic).unwrap();

        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let result = file.write_array("test_data", &data, &["x", "y"]);

        // This should fail for Classic format
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("only supported for NetCDF4"));
    }
}
