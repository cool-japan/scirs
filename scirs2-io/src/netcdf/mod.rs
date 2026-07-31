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

mod classic_backend;

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
    path: String,
    /// File mode ('r' for read, 'w' for write)
    mode: String,
    /// NetCDF format version
    format: NetCDFFormat,
    /// Dimensions defined in the file
    dimensions: HashMap<String, Option<usize>>,
    /// Order in which dimensions were declared. `HashMap` does not preserve
    /// insertion order, but the Classic NetCDF3 on-disk format assigns each
    /// dimension an implicit `dimid` from its declaration order, so this is
    /// needed to serialize/deserialize Classic files deterministically.
    dim_order: Vec<String>,
    /// Variables defined in the file
    variables: HashMap<String, VariableInfo>,
    /// Order in which variables were declared (see `dim_order`).
    var_order: Vec<String>,
    /// Global attributes
    attributes: HashMap<String, AttributeValue>,
    /// Order in which global attributes were declared (see `dim_order`).
    attr_order: Vec<String>,
    /// HDF5 backend (for NetCDF4 support)
    hdf5_backend: Option<HDF5File>,
    /// Buffered/parsed variable data for Classic NetCDF3 format: populated by
    /// `write_variable` (pending a flush to disk) and by `open` when reading
    /// an existing Classic file. Unused for the NetCDF4/HDF5 backend, which
    /// stores and reads data through `hdf5_backend` directly.
    classic_data: HashMap<String, classic_backend::ClassicVarData>,
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
    /// Order in which this variable's attributes were declared (see
    /// `NetCDFFile::dim_order`).
    attr_order: Vec<String>,
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
    /// let nc = NetCDFFile::open("data.nc", None).expect("Operation failed");
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

        // Classic NetCDF3 read mode: actually parse the on-disk file (magic, header,
        // and data sections) rather than starting from an empty in-memory structure.
        if opts.format == NetCDFFormat::Classic && opts.mode == "r" {
            return classic_backend::open_classic(&path_str);
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

        // Create an empty NetCDF file structure (write mode: data is buffered and
        // flushed to disk by `sync`/`close`).
        Ok(Self {
            path: path_str,
            mode: opts.mode,
            format: opts.format,
            dimensions: HashMap::new(),
            dim_order: Vec::new(),
            variables: HashMap::new(),
            var_order: Vec::new(),
            attributes: HashMap::new(),
            attr_order: Vec::new(),
            hdf5_backend,
            classic_data: HashMap::new(),
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
            dim_order: Vec::new(),
            variables: HashMap::new(),
            var_order: Vec::new(),
            attributes: HashMap::new(),
            attr_order: Vec::new(),
            hdf5_backend,
            classic_data: HashMap::new(),
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
        if !self.dim_order.iter().any(|d| d == name) {
            self.dim_order.push(name.to_string());
        }

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
            attr_order: Vec::new(),
        };

        self.variables.insert(name.to_string(), var_info);
        if !self.var_order.iter().any(|v| v == name) {
            self.var_order.push(name.to_string());
        }

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
    /// For a variable that was declared (via `create_variable`) but never
    /// written and never read back from an on-disk file, this returns an
    /// array filled with the variable type's standard NetCDF fill value
    /// (e.g. `NC_FILL_DOUBLE`), matching real NetCDF semantics for unwritten
    /// data -- not zero.
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

        // Classic NetCDF3: return the real, previously-written or parsed-from-disk data.
        if let Some(buf) = self.classic_data.get(name) {
            let array_f64 = Array::from_shape_vec(buf.shape.clone(), buf.values.clone())
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
            let data: Vec<T> = self.convert_data_type(&array_f64)?;
            return Array::from_shape_vec(buf.shape.clone(), data)
                .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)));
        }

        // Variable declared but never written: return the standard NetCDF fill value
        // for its type (matches real NetCDF semantics for unwritten data; NOT zero).
        let fill = classic_backend::fill_value_f64(var_info.data_type);
        let total_size = shape.iter().product();
        let fill_f64 = vec![fill; total_size];
        let array_f64 = Array::from_shape_vec(shape.clone(), fill_f64)
            .map_err(|e| IoError::FormatError(format!("Failed to create array: {}", e)))?;
        let data: Vec<T> = self.convert_data_type(&array_f64)?;
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
        } else if self.format == NetCDFFormat::Classic {
            self.write_variable_classic(name, data)?;
        }

        Ok(())
    }

    /// Buffers `data` for a Classic NetCDF3 variable, validating its shape
    /// against the variable's declared (fixed-size) dimensions. The unlimited
    /// dimension, if the variable uses one, accepts any size: that becomes
    /// this write's contribution to the file's overall record count, computed
    /// when the file is actually flushed to disk (see `sync`/`close`).
    fn write_variable_classic<T: Clone + Into<f64>, D: Dimension>(
        &mut self,
        name: &str,
        data: &Array<T, D>,
    ) -> Result<()> {
        let dim_names = self
            .variables
            .get(name)
            .ok_or_else(|| IoError::ValidationError(format!("Variable '{}' not defined", name)))?
            .dimensions
            .clone();

        let shape = data.shape().to_vec();
        if shape.len() != dim_names.len() {
            return Err(IoError::ValidationError(format!(
                "Variable '{}' expects {} dimension(s), got {}",
                name,
                dim_names.len(),
                shape.len()
            )));
        }
        for (axis, dim_name) in dim_names.iter().enumerate() {
            if let Some(fixed_size) = self.dimensions.get(dim_name).copied().flatten() {
                if shape[axis] != fixed_size {
                    return Err(IoError::ValidationError(format!(
                        "Variable '{}' dimension '{}' expects size {}, got {}",
                        name, dim_name, fixed_size, shape[axis]
                    )));
                }
            }
            // `None` means `dim_name` is the unlimited/record dimension: any size is
            // accepted here (validated to be its own consistent dimension index by
            // `netcdf3::DataSet::add_var` when the file is flushed).
        }

        let values: Vec<f64> = data.iter().cloned().map(Into::into).collect();
        self.classic_data.insert(
            name.to_string(),
            classic_backend::ClassicVarData { values, shape },
        );
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
        if !var_info.attr_order.iter().any(|a| a == attr_name) {
            var_info.attr_order.push(attr_name.to_string());
        }

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
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
                        Box::new(x as i64)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
                        Box::new(x as i32)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
                        Box::new(x as i16)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
                        Box::new(x as i8)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
                        Box::new(x as u64)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
                        Box::new(x as u32)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>() {
                        Box::new(x as u16)
                    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
                        Box::new(x as u8)
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
        if !self.attr_order.iter().any(|a| a == name) {
            self.attr_order.push(name.to_string());
        }

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
        if let Some(backend) = &self.hdf5_backend {
            // For HDF5 backend, directly read the dataset
            backend.read_dataset(name)
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
        } else if self.format == NetCDFFormat::Classic && self.mode == "w" {
            classic_backend::flush_classic(self)?;
        }
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
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_create_netcdf_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let file = NetCDFFile::create(test_path).expect("Operation failed");
        assert_eq!(file.mode, "w");
        assert_eq!(file.path, test_path);
        assert!(file.dimensions.is_empty());
        assert!(file.variables.is_empty());
        assert!(file.attributes.is_empty());

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_add_dimension() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_add_dimension_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("time", Some(10))
            .expect("Operation failed");
        file.create_dimension("lat", Some(180))
            .expect("Operation failed");
        file.create_dimension("lon", Some(360))
            .expect("Operation failed");
        file.create_dimension("unlimited", None)
            .expect("Operation failed");

        assert_eq!(file.dimensions.len(), 4);
        assert_eq!(
            *file.dimensions.get("time").expect("Operation failed"),
            Some(10)
        );
        assert_eq!(
            *file.dimensions.get("lat").expect("Operation failed"),
            Some(180)
        );
        assert_eq!(
            *file.dimensions.get("lon").expect("Operation failed"),
            Some(360)
        );
        assert_eq!(
            *file.dimensions.get("unlimited").expect("Operation failed"),
            None
        );

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_add_variable() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_add_variable_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("time", Some(10))
            .expect("Operation failed");
        file.create_dimension("lat", Some(180))
            .expect("Operation failed");
        file.create_dimension("lon", Some(360))
            .expect("Operation failed");

        file.create_variable(
            "temperature",
            NetCDFDataType::Float,
            &["time", "lat", "lon"],
        )
        .expect("Operation failed");

        assert_eq!(file.variables.len(), 1);
        assert!(file.variables.contains_key("temperature"));

        let var_info = file.variables.get("temperature").expect("Operation failed");
        assert_eq!(var_info.name, "temperature");
        assert_eq!(var_info.data_type, NetCDFDataType::Float);
        assert_eq!(var_info.dimensions, vec!["time", "lat", "lon"]);

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_attributes() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_attributes_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("x", Some(10))
            .expect("Operation failed");
        file.create_variable("data", NetCDFDataType::Double, &["x"])
            .expect("Operation failed");

        // Test global attributes
        file.add_global_attribute("title", "Test Dataset")
            .expect("Operation failed");
        file.add_global_attribute("author", "SciRS2 Test")
            .expect("Operation failed");

        let global_attrs = file.global_attributes();
        assert!(global_attrs.contains_key("title"));
        assert!(global_attrs.contains_key("author"));
        assert_eq!(global_attrs["title"], "Test Dataset");
        assert_eq!(global_attrs["author"], "SciRS2 Test");

        // Test variable attributes
        file.add_variable_attribute("data", "units", "meters")
            .expect("Operation failed");
        file.add_variable_attribute("data", "long_name", "measurement data")
            .expect("Operation failed");

        let (dtype, dims, var_attrs) = file.variable_info("data").expect("Operation failed");
        assert_eq!(dtype, NetCDFDataType::Double);
        assert_eq!(dims, vec!["x"]);
        assert!(var_attrs.contains_key("units"));
        assert!(var_attrs.contains_key("long_name"));
        assert_eq!(var_attrs["units"], "meters");
        assert_eq!(var_attrs["long_name"], "measurement data");

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_read_write_variable() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_read_write_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        // Non-constant data: an all-zero/all-same array could "round-trip correctly"
        // even through a fabricating stub, so use distinct values everywhere.
        let written =
            Array2::<f32>::from_shape_vec((3, 2), vec![1.5, -2.25, 100.0, -0.125, 42.75, -7.0])
                .expect("Operation failed");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("x", Some(3))
            .expect("Operation failed");
        file.create_dimension("y", Some(2))
            .expect("Operation failed");
        file.create_variable("data", NetCDFDataType::Float, &["x", "y"])
            .expect("Operation failed");
        file.write_variable("data", &written)
            .expect("Operation failed");
        file.close().expect("Operation failed");

        // Reopen the file completely fresh (a brand new `NetCDFFile`, parsed from the
        // bytes on disk) and confirm both metadata and data survived the round trip.
        let reopened = NetCDFFile::open(test_path, None).expect("Operation failed");
        assert_eq!(
            *reopened.dimensions().get("x").expect("Operation failed"),
            Some(3)
        );
        assert_eq!(
            *reopened.dimensions().get("y").expect("Operation failed"),
            Some(2)
        );
        assert!(reopened.variables().contains(&"data".to_string()));

        let read_data: ArrayD<f32> = reopened.read_variable("data").expect("Operation failed");
        assert_eq!(read_data.shape(), &[3, 2]);
        assert_eq!(
            read_data.iter().copied().collect::<Vec<_>>(),
            written.iter().copied().collect::<Vec<_>>()
        );

        drop(reopened);
        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_classic_roundtrip_multiple_types_and_attributes() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!(
            "test_multi_type_roundtrip_{}.nc",
            std::process::id()
        ));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let byte_data =
            Array::from_shape_vec(vec![4], vec![-120i8, -5, 3, 119]).expect("Operation failed");
        let short_data = Array::from_shape_vec(vec![4], vec![-30000i16, -1, 12345, 32000])
            .expect("Operation failed");
        let int_data = Array::from_shape_vec(
            vec![4],
            vec![-2_000_000_000i32, -1, 123_456_789, 2_000_000_000],
        )
        .expect("Operation failed");
        let float_data = Array::from_shape_vec(vec![4], vec![-1.5f32, 0.25, 3.75, -100_000.125])
            .expect("Operation failed");
        let double_data = Array::from_shape_vec(
            vec![4],
            vec![-1.234_567_89e10_f64, 5.918_37, -0.000_123, 42.0],
        )
        .expect("Operation failed");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("n", Some(4))
            .expect("Operation failed");
        file.create_variable("b", NetCDFDataType::Byte, &["n"])
            .expect("Operation failed");
        file.create_variable("s", NetCDFDataType::Short, &["n"])
            .expect("Operation failed");
        file.create_variable("i", NetCDFDataType::Int, &["n"])
            .expect("Operation failed");
        file.create_variable("f", NetCDFDataType::Float, &["n"])
            .expect("Operation failed");
        file.create_variable("d", NetCDFDataType::Double, &["n"])
            .expect("Operation failed");

        file.add_global_attribute("title", "multi-type roundtrip")
            .expect("Operation failed");
        file.add_variable_attribute("d", "units", "meters")
            .expect("Operation failed");

        file.write_variable("b", &byte_data)
            .expect("Operation failed");
        file.write_variable("s", &short_data)
            .expect("Operation failed");
        file.write_variable("i", &int_data)
            .expect("Operation failed");
        file.write_variable("f", &float_data)
            .expect("Operation failed");
        file.write_variable("d", &double_data)
            .expect("Operation failed");
        file.close().expect("Operation failed");

        let reopened = NetCDFFile::open(test_path, None).expect("Operation failed");

        let read_b: ArrayD<i8> = reopened.read_variable("b").expect("Operation failed");
        let read_s: ArrayD<i16> = reopened.read_variable("s").expect("Operation failed");
        let read_i: ArrayD<i32> = reopened.read_variable("i").expect("Operation failed");
        let read_f: ArrayD<f32> = reopened.read_variable("f").expect("Operation failed");
        let read_d: ArrayD<f64> = reopened.read_variable("d").expect("Operation failed");

        assert_eq!(
            read_b.iter().copied().collect::<Vec<_>>(),
            byte_data.iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            read_s.iter().copied().collect::<Vec<_>>(),
            short_data.iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            read_i.iter().copied().collect::<Vec<_>>(),
            int_data.iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            read_f.iter().copied().collect::<Vec<_>>(),
            float_data.iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            read_d.iter().copied().collect::<Vec<_>>(),
            double_data.iter().copied().collect::<Vec<_>>()
        );

        let global_attrs = reopened.global_attributes();
        assert_eq!(
            global_attrs.get("title"),
            Some(&"multi-type roundtrip".to_string())
        );

        let (dtype, dims, var_attrs) = reopened.variable_info("d").expect("Operation failed");
        assert_eq!(dtype, NetCDFDataType::Double);
        assert_eq!(dims, vec!["n".to_string()]);
        assert_eq!(var_attrs.get("units"), Some(&"meters".to_string()));

        drop(reopened);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_roundtrip_record_variable() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!(
            "test_record_var_roundtrip_{}.nc",
            std::process::id()
        ));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let num_records = 4usize;
        let per_record = 3usize;
        let values: Vec<f64> = (0..num_records * per_record)
            .map(|k| {
                let i = (k / per_record) as f64;
                let j = (k % per_record) as f64;
                i * 10.5 - j * 0.25 + 1.0
            })
            .collect();
        let written = Array::from_shape_vec(vec![num_records, per_record], values.clone())
            .expect("Operation failed");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("time", None)
            .expect("Operation failed");
        file.create_dimension("loc", Some(per_record))
            .expect("Operation failed");
        file.create_variable("temp", NetCDFDataType::Double, &["time", "loc"])
            .expect("Operation failed");
        file.write_variable("temp", &written)
            .expect("Operation failed");
        file.close().expect("Operation failed");

        let reopened = NetCDFFile::open(test_path, None).expect("Operation failed");
        // The unlimited dimension is reported as `None` (its declared size), matching
        // the public API convention; the concrete record count is only observable
        // through the data shape.
        assert_eq!(
            *reopened.dimensions().get("time").expect("Operation failed"),
            None
        );

        let read_temp: ArrayD<f64> = reopened.read_variable("temp").expect("Operation failed");
        assert_eq!(read_temp.shape(), &[num_records, per_record]);
        assert_eq!(read_temp.iter().copied().collect::<Vec<_>>(), values);

        drop(reopened);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_mixed_fixed_and_record_vars_no_corruption() {
        // Regression test: a fixed-size variable that is declared but never written,
        // in a dataset that ALSO has a fully-written record variable, must not
        // corrupt that record variable's data. `flush_classic` must always write
        // fixed-size variables explicitly rather than leaving them for
        // `netcdf3::FileWriter::close()`'s auto-fill -- verified empirically to
        // otherwise fill them using the dataset's record stride instead of a single
        // copy at their own offset, overwriting whatever data follows (see
        // `classic_backend` module docs).
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_mixed_fixed_record_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let time_values = vec![111.25_f64, -222.5, 333.75];

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("time", None)
            .expect("Operation failed");
        file.create_dimension("loc", Some(2))
            .expect("Operation failed");
        // Declared but intentionally NEVER written:
        file.create_variable("fixedvar", NetCDFDataType::Double, &["loc"])
            .expect("Operation failed");
        file.create_variable("timevar", NetCDFDataType::Double, &["time"])
            .expect("Operation failed");

        let written =
            Array::from_shape_vec(vec![3], time_values.clone()).expect("Operation failed");
        file.write_variable("timevar", &written)
            .expect("Operation failed");
        file.close().expect("Operation failed");

        let reopened = NetCDFFile::open(test_path, None).expect("Operation failed");
        let read_time: ArrayD<f64> = reopened.read_variable("timevar").expect("Operation failed");
        assert_eq!(
            read_time.iter().copied().collect::<Vec<_>>(),
            time_values,
            "fixedvar being left unwritten must not corrupt timevar's already-written data"
        );

        let read_fixed: ArrayD<f64> = reopened
            .read_variable("fixedvar")
            .expect("Operation failed");
        assert_eq!(read_fixed.shape(), &[2]);
        assert!(read_fixed.iter().all(|&x| x == netcdf3::NC_FILL_F64));

        drop(reopened);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_header_golden_bytes() {
        // Golden-byte test: verifies the exact on-disk NetCDF-3 classic byte layout
        // (magic/version, tag constants, name/attribute/variable encoding, padding,
        // and `begin` offset arithmetic) against an independently hand-derived
        // expectation, rather than merely checking that our own reader agrees with
        // our own writer (which a consistently-wrong implementation could still
        // pass).
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_golden_bytes_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("x", Some(2))
            .expect("Operation failed");
        file.create_variable("v", NetCDFDataType::Double, &["x"])
            .expect("Operation failed");
        let data =
            Array::from_shape_vec(vec![2], vec![3.5_f64, -2.25_f64]).expect("Operation failed");
        file.write_variable("v", &data).expect("Operation failed");
        file.close().expect("Operation failed");

        let bytes = std::fs::read(&test_file).expect("Operation failed");

        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            // magic + version (CDF-1 classic)
            0x43, 0x44, 0x46, 0x01,
            // numrecs = 0 (no unlimited dimension in this dataset)
            0x00, 0x00, 0x00, 0x00,
            // dim_list: NC_DIMENSION tag, nelems=1
            0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x01,
            //   dim "x": name (len=1,'x',pad3), dim_length=2
            0x00, 0x00, 0x00, 0x01, 0x78, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x02,
            // gatt_list: ABSENT
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // var_list: NC_VARIABLE tag, nelems=1
            0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x01,
            //   var "v": name (len=1,'v',pad3)
            0x00, 0x00, 0x00, 0x01, 0x76, 0x00, 0x00, 0x00,
            //   ndims=1, dimid=0
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            //   vatt_list: ABSENT
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            //   nc_type=NC_DOUBLE(6), vsize=16, begin=80
            0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x10,
            0x00, 0x00, 0x00, 0x50,
            // data: 3.5f64, -2.25f64 (big-endian)
            0x40, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xc0, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        assert_eq!(bytes, expected);

        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_read_invalid_file_errors() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_invalid_nc3_{}.nc", std::process::id()));
        std::fs::write(&test_file, b"not a netcdf file at all, just garbage bytes")
            .expect("Operation failed");

        let result = NetCDFFile::open(&test_file, None);
        assert!(result.is_err());

        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_scalar_variable_with_no_unlimited_dim() {
        // Regression test for an edge case in the record/fixed-variable
        // classification: a 0-dimensional (scalar) variable in a dataset that
        // has NO unlimited dimension at all must still be written (a naive
        // `record_dim_name == var's_first_dim_name` comparison sees `None ==
        // None` for this case and wrongly treats the scalar as a record
        // variable with zero records, silently dropping its data).
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_scalar_var_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("x", Some(3))
            .expect("Operation failed");
        file.create_variable("scalar_var", NetCDFDataType::Double, &[])
            .expect("Operation failed");
        file.create_variable("arr", NetCDFDataType::Double, &["x"])
            .expect("Operation failed");

        let scalar_value = std::f64::consts::PI;
        let scalar_data =
            Array::from_shape_vec(vec![], vec![scalar_value]).expect("Operation failed");
        file.write_variable("scalar_var", &scalar_data)
            .expect("Operation failed");
        let arr_values = vec![1.5_f64, -2.5, 3.5];
        let arr_data =
            Array::from_shape_vec(vec![3], arr_values.clone()).expect("Operation failed");
        file.write_variable("arr", &arr_data)
            .expect("Operation failed");
        file.close().expect("Operation failed");

        let reopened = NetCDFFile::open(test_path, None).expect("Operation failed");
        let read_scalar: ArrayD<f64> = reopened
            .read_variable("scalar_var")
            .expect("Operation failed");
        assert_eq!(read_scalar.shape(), &[] as &[usize]);
        assert_eq!(
            read_scalar.iter().copied().collect::<Vec<_>>(),
            vec![scalar_value],
            "scalar variable's data must not be silently dropped"
        );

        let read_arr: ArrayD<f64> = reopened.read_variable("arr").expect("Operation failed");
        assert_eq!(read_arr.iter().copied().collect::<Vec<_>>(), arr_values);

        drop(reopened);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_unwritten_variable_reads_fill_value_not_zero() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_fill_value_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create(test_path).expect("Operation failed");
        file.create_dimension("x", Some(2))
            .expect("Operation failed");
        file.create_variable("v", NetCDFDataType::Double, &["x"])
            .expect("Operation failed");
        // Never call write_variable; flip to read mode in-process (without ever
        // flushing to disk) purely to exercise the "declared but unwritten" fallback.
        file.mode = "r".to_string();

        let data: ArrayD<f64> = file.read_variable("v").expect("Operation failed");
        assert_eq!(data.shape(), &[2]);
        assert!(data.iter().all(|&x| x == netcdf3::NC_FILL_F64));
        assert_ne!(netcdf3::NC_FILL_F64, 0.0);

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_netcdf4_format_creation() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_netcdf4_format_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let file = NetCDFFile::create_with_format(test_path, NetCDFFormat::NetCDF4)
            .expect("Operation failed");
        assert_eq!(file.format(), NetCDFFormat::NetCDF4);
        assert!(file.has_hdf5_backend());

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_netcdf_format_differences() {
        let temp_dir = std::env::temp_dir();
        let classic_file = temp_dir.join(format!("test_classic_{}.nc", std::process::id()));
        let netcdf4_file = temp_dir.join(format!("test_netcdf4_{}.nc", std::process::id()));
        let classic_path = classic_file.to_str().expect("path should be valid UTF-8");
        let netcdf4_path = netcdf4_file.to_str().expect("path should be valid UTF-8");

        let classic = NetCDFFile::create_with_format(classic_path, NetCDFFormat::Classic)
            .expect("Operation failed");
        let netcdf4 = NetCDFFile::create_with_format(netcdf4_path, NetCDFFormat::NetCDF4)
            .expect("Operation failed");

        assert_eq!(classic.format(), NetCDFFormat::Classic);
        assert_eq!(netcdf4.format(), NetCDFFormat::NetCDF4);

        assert!(!classic.has_hdf5_backend());
        assert!(netcdf4.has_hdf5_backend());

        drop(classic);
        drop(netcdf4);
        let _ = std::fs::remove_file(classic_file);
        let _ = std::fs::remove_file(netcdf4_file);
    }

    #[test]
    fn test_netcdf4_write_array() {
        use scirs2_core::ndarray::array;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_netcdf4_array_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create_with_format(test_path, NetCDFFormat::NetCDF4)
            .expect("Operation failed");

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

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_netcdf4_convenience_functions() {
        use scirs2_core::ndarray::array;
        use std::collections::HashMap;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_convenience_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

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

        let result = create_netcdf4_with_data(test_path, datasets, global_attrs);
        assert!(result.is_ok());

        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_classic_netcdf_write_array_error() {
        use scirs2_core::ndarray::array;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(format!("test_classic_error_{}.nc", std::process::id()));
        let test_path = test_file.to_str().expect("path should be valid UTF-8");

        let mut file = NetCDFFile::create_with_format(test_path, NetCDFFormat::Classic)
            .expect("Operation failed");

        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let result = file.write_array("test_data", &data, &["x", "y"]);

        // This should fail for Classic format
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("only supported for NetCDF4"));

        drop(file);
        let _ = std::fs::remove_file(test_file);
    }
}
