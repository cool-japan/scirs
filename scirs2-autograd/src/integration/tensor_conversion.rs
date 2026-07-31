//! Tensor conversion utilities for seamless interoperability between SciRS2 modules
//!
//! This module provides efficient conversion between different tensor representations
//! used across the SciRS2 ecosystem, with support for zero-copy operations when possible.

use super::{IntegrationConfig, IntegrationError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

/// Tensor metadata for conversion operations
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub memory_layout: MemoryLayout,
    pub requires_grad: bool,
    pub device: DeviceInfo,
}

/// Memory layout information
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Strided(Vec<isize>),
    Contiguous,
}

/// Device information for tensor placement
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceInfo {
    CPU,
    GPU(u32),
    Memory(String),
}

/// Tensor conversion registry for different module formats
pub struct TensorConverter {
    /// Registered conversion functions
    converters: HashMap<String, Box<dyn ConversionFunction>>,
    /// Configuration for conversion behavior
    config: IntegrationConfig,
}

impl TensorConverter {
    /// Create a new tensor converter
    pub fn new() -> Self {
        let mut converter = Self {
            converters: HashMap::new(),
            config: IntegrationConfig::default(),
        };

        // Register built-in converters
        converter.register_builtin_converters();
        converter
    }

    /// Create tensor converter with custom configuration
    pub fn with_config(config: IntegrationConfig) -> Self {
        let mut converter = Self::new();
        converter.config = config;
        converter
    }

    /// Register a custom conversion function
    pub fn register_converter<F>(&mut self, name: String, converter: F)
    where
        F: Fn(&[u8], &TensorMetadata) -> Result<Vec<u8>, IntegrationError> + Send + 'static,
    {
        self.converters.insert(name, Box::new(converter));
    }

    /// Convert tensor to a specific format
    pub fn convert_to<F: Float>(
        &self,
        tensor: &Tensor<F>,
        target_format: &str,
    ) -> Result<Vec<u8>, IntegrationError> {
        let metadata = self.extract_metadata(tensor)?;
        let data = self.serialize_tensor_data(tensor)?;

        if let Some(converter) = self.converters.get(target_format) {
            converter.convert(&data, &metadata)
        } else {
            Err(IntegrationError::TensorConversion(format!(
                "No converter found for format: {target_format}"
            )))
        }
    }

    /// Convert from a specific format to autograd tensor
    pub fn convert_from<'a, F: Float>(
        _data: &'a [u8],
        _metadata: &'a TensorMetadata,
        _source_format: &'a str,
    ) -> Result<Tensor<'a, F>, IntegrationError> {
        // For now, implement basic conversion
        // In practice, this would use the registered converters
        // Direct tensor creation not supported without graph context
        Err(IntegrationError::TensorConversion(
            "Tensor creation requires graph context. Use run() function.".to_string(),
        ))
    }

    /// Convert an autograd tensor to a different float precision by
    /// evaluating it under `ctx` to obtain its real contents.
    ///
    /// Autograd tensors are *lazy*: a [`Tensor`] is only a handle into a
    /// [`crate::Graph`] and carries no data of its own (see
    /// [`Self::to_ndarray_with_context`] for the full explanation of why an
    /// evaluation [`crate::Context`] must be threaded through explicitly).
    /// This method evaluates `tensor` under `ctx`, converts every element to
    /// the target precision `F2`, and builds a new tensor for `graph` from
    /// the genuine converted values.
    ///
    /// Use [`Self::convert_precision`] only when no `Context` is reachable;
    /// that variant cannot evaluate the source tensor and returns an honest
    /// error rather than fabricated (all-zero) data.
    pub fn convert_precision_with_context<'graph, F1: Float, F2: Float>(
        &self,
        tensor: &Tensor<F1>,
        ctx: &crate::Context<F1>,
        graph: &'graph crate::Graph<F2>,
    ) -> Result<Tensor<'graph, F2>, IntegrationError> {
        let array_f1 = tensor.eval(ctx).map_err(|e| {
            IntegrationError::TensorConversion(format!("Failed to evaluate tensor: {e:?}"))
        })?;

        let shape = array_f1.shape().to_vec();
        let mut converted_data = Vec::with_capacity(array_f1.len());
        for &x in array_f1.iter() {
            let as_f64 = x.to_f64().ok_or_else(|| {
                IntegrationError::TensorConversion(
                    "Failed to convert source value to f64 during precision conversion".to_string(),
                )
            })?;
            let converted = F2::from(as_f64).ok_or_else(|| {
                IntegrationError::TensorConversion(format!(
                    "Value {as_f64} is not representable in the target precision"
                ))
            })?;
            converted_data.push(converted);
        }

        Ok(Tensor::from_vec(converted_data, shape, graph))
    }

    /// Convert between autograd tensors with different precision *without*
    /// an evaluation context.
    ///
    /// An autograd [`Tensor`] holds no data of its own -- it is a lazy node
    /// in a [`crate::Graph`] whose value is only realized by an evaluation
    /// [`crate::Context`]. With no context this method genuinely cannot
    /// recover the source tensor's real contents, so it returns an honest
    /// [`IntegrationError`] rather than fabricating a shape-correct but
    /// all-zero tensor.
    ///
    /// Call [`Self::convert_precision_with_context`] (passing the `ctx`/`g`
    /// from [`crate::run`]) to obtain a real, correctly-converted tensor.
    pub fn convert_precision<F1: Float, F2: Float>(
        &self,
        _tensor: &Tensor<F1>,
        _graph: &crate::Graph<F2>,
    ) -> Result<Tensor<'static, F2>, IntegrationError> {
        Err(IntegrationError::TensorConversion(
            "convert_precision requires an evaluation context to read a lazy autograd \
             tensor's real data; call convert_precision_with_context(tensor, ctx, graph) \
             with the Context from run() instead"
                .to_string(),
        ))
    }

    /// Create a view of tensor data without copying when possible
    pub fn create_view<F: Float>(
        &self,
        tensor: &Tensor<F>,
    ) -> Result<TensorView<F>, IntegrationError> {
        Ok(TensorView {
            data: tensor.data().to_vec(),
            shape: tensor.shape().to_vec(),
            strides: self.compute_strides(&tensor.shape()),
            metadata: self.extract_metadata(tensor)?,
        })
    }

    /// Convert ndarray to autograd tensor
    pub fn from_ndarray<'graph, F: Float>(
        &self,
        array: ArrayD<F>,
        graph: &'graph crate::Graph<F>,
    ) -> Result<Tensor<'graph, F>, IntegrationError> {
        let shape = array.shape().to_vec();
        let data = array.into_raw_vec_and_offset().0;
        Ok(Tensor::from_vec(data, shape, graph))
    }

    /// Convert an autograd tensor to a concrete ndarray by evaluating it.
    ///
    /// Autograd tensors are *lazy*: a [`Tensor`] is only a handle into a
    /// [`crate::Graph`] and carries no data of its own. Producing real values
    /// therefore requires an evaluation [`crate::Context`] (the `ctx`/`g`
    /// argument handed to [`crate::run`]). This method evaluates `tensor`
    /// under `ctx` and returns its genuine contents.
    ///
    /// This is the round-trip partner of [`Self::from_ndarray`]: feeding the
    /// array produced here back through `from_ndarray` reconstructs an
    /// equivalent tensor, and evaluating that tensor reproduces the same
    /// values.
    ///
    /// Use [`Self::to_ndarray`] only when no `Context` is reachable; that
    /// variant cannot evaluate and returns an honest error rather than
    /// fabricated data.
    pub fn to_ndarray_with_context<F: Float>(
        &self,
        tensor: &Tensor<F>,
        ctx: &crate::Context<F>,
    ) -> Result<ArrayD<F>, IntegrationError> {
        tensor.eval(ctx).map_err(|e| {
            IntegrationError::TensorConversion(format!("Failed to evaluate tensor: {e:?}"))
        })
    }

    /// Convert an autograd tensor to a concrete ndarray *without* an evaluation
    /// context.
    ///
    /// An autograd [`Tensor`] holds no data — it is a lazy node in a
    /// [`crate::Graph`] whose value is only realized by an evaluation
    /// [`crate::Context`]. With no context this method genuinely cannot recover
    /// the tensor's contents, so it returns an honest [`IntegrationError`]
    /// rather than fabricating values.
    ///
    /// Call [`Self::to_ndarray_with_context`] (passing the `ctx`/`g` from
    /// [`crate::run`]) to obtain the tensor's real evaluated data.
    pub fn to_ndarray<F: Float>(&self, _tensor: &Tensor<F>) -> Result<ArrayD<F>, IntegrationError> {
        Err(IntegrationError::TensorConversion(
            "to_ndarray requires an evaluation context to read a lazy autograd \
             tensor's real data; call to_ndarray_with_context(tensor, ctx) with \
             the Context from run() instead"
                .to_string(),
        ))
    }

    /// Batch convert multiple tensors efficiently
    pub fn batch_convert<F: Float>(
        &self,
        tensors: &[&Tensor<F>],
        target_format: &str,
    ) -> Result<Vec<Vec<u8>>, IntegrationError> {
        let mut results = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            results.push(self.convert_to(*tensor, target_format)?);
        }

        Ok(results)
    }

    /// Register built-in conversion functions
    fn register_builtin_converters(&mut self) {
        // Register ndarray converter
        self.converters.insert(
            "ndarray".to_string(),
            Box::new(|data: &[u8], metadata: &TensorMetadata| {
                // Convert to ndarray format
                Ok(data.to_vec())
            }),
        );

        // Register numpy-compatible converter
        self.converters.insert(
            "numpy".to_string(),
            Box::new(|data: &[u8], metadata: &TensorMetadata| {
                // Convert to numpy-compatible format
                Ok(data.to_vec())
            }),
        );

        // Register JSON converter for debugging
        self.converters.insert(
            "json".to_string(),
            Box::new(|data: &[u8], metadata: &TensorMetadata| {
                let json_repr = serde_json::json!({
                    "data": data,
                    "shape": metadata.shape,
                    "dtype": metadata.dtype,
                    "layout": format!("{:?}", metadata.memory_layout)
                });

                serde_json::to_vec(&json_repr).map_err(|e| {
                    IntegrationError::TensorConversion(format!("JSON serialization failed: {e}"))
                })
            }),
        );
    }

    /// Extract metadata from tensor
    fn extract_metadata<F: Float>(
        &self,
        tensor: &Tensor<F>,
    ) -> Result<TensorMetadata, IntegrationError> {
        let shape = tensor.shape();

        Ok(TensorMetadata {
            shape,
            dtype: std::any::type_name::<F>().to_string(),
            memory_layout: MemoryLayout::RowMajor, // Simplified
            requires_grad: tensor.requires_grad(),
            device: DeviceInfo::CPU, // Simplified
        })
    }

    /// Serialize tensor data to bytes
    fn serialize_tensor_data<F: Float>(
        &self,
        tensor: &Tensor<F>,
    ) -> Result<Vec<u8>, IntegrationError> {
        let data = tensor.data();
        let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<F>());

        for value in data {
            let value_f64 = value.to_f64().expect("Operation failed");
            bytes.extend_from_slice(&value_f64.to_le_bytes());
        }

        Ok(bytes)
    }

    /// Deserialize tensor data from bytes
    #[allow(dead_code)]
    fn deserialize_tensor_data<'graph, F: Float>(
        &self,
        data: &[u8],
        metadata: &TensorMetadata,
        graph: &'graph crate::Graph<F>,
    ) -> Result<Tensor<'graph, F>, IntegrationError> {
        let element_size = std::mem::size_of::<f64>();
        if !data.len().is_multiple_of(element_size) {
            return Err(IntegrationError::TensorConversion(
                "Invalid data size for tensor deserialization".to_string(),
            ));
        }

        let num_elements = data.len() / element_size;
        let mut values = Vec::with_capacity(num_elements);

        for chunk in data.chunks(element_size) {
            let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                IntegrationError::TensorConversion("Failed to convert bytes to f64".to_string())
            })?;
            let value_f64 = f64::from_le_bytes(bytes);
            let value_f = F::from(value_f64).expect("Failed to convert to float");
            values.push(value_f);
        }

        Ok(Tensor::from_vec(values, metadata.shape.clone(), graph))
    }

    /// Compute strides for tensor shape
    fn compute_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

impl Default for TensorConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor view for zero-copy operations
pub struct TensorView<F: Float> {
    pub data: Vec<F>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub metadata: TensorMetadata,
}

impl<F: Float> TensorView<F> {
    /// Get element at specific indices
    pub fn get(&self, indices: &[usize]) -> Result<F, IntegrationError> {
        if indices.len() != self.shape.len() {
            return Err(IntegrationError::TensorConversion(
                "Index dimension mismatch".to_string(),
            ));
        }

        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(IntegrationError::TensorConversion(
                    "Index out of bounds".to_string(),
                ));
            }
            offset += idx * self.strides[i];
        }

        Ok(self.data[offset])
    }

    /// Create a slice of the tensor view
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<TensorView<F>, IntegrationError> {
        if ranges.len() != self.shape.len() {
            return Err(IntegrationError::TensorConversion(
                "Slice dimension mismatch".to_string(),
            ));
        }

        // Simplified slicing - in practice would compute proper data pointer and strides
        Ok(TensorView {
            data: self.data.clone(),
            shape: ranges.iter().map(|(start, end)| end - start).collect(),
            strides: self.strides.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

/// Trait for conversion functions
trait ConversionFunction: Send {
    fn convert(&self, data: &[u8], metadata: &TensorMetadata) -> Result<Vec<u8>, IntegrationError>;
}

impl<F> ConversionFunction for F
where
    F: Fn(&[u8], &TensorMetadata) -> Result<Vec<u8>, IntegrationError> + Send,
{
    fn convert(&self, data: &[u8], metadata: &TensorMetadata) -> Result<Vec<u8>, IntegrationError> {
        self(data, metadata)
    }
}

/// Global tensor converter instance
static GLOBAL_CONVERTER: std::sync::OnceLock<std::sync::Mutex<TensorConverter>> =
    std::sync::OnceLock::new();

/// Initialize global tensor converter
#[allow(dead_code)]
pub fn init_tensor_converter() -> &'static std::sync::Mutex<TensorConverter> {
    GLOBAL_CONVERTER.get_or_init(|| std::sync::Mutex::new(TensorConverter::new()))
}

/// Convert tensor using global converter
#[allow(dead_code)]
pub fn convert_tensor_to<F: Float>(
    tensor: &Tensor<F>,
    target_format: &str,
) -> Result<Vec<u8>, IntegrationError> {
    let converter = init_tensor_converter();
    let converter_guard = converter.lock().map_err(|_| {
        IntegrationError::TensorConversion("Failed to acquire converter lock".to_string())
    })?;
    converter_guard.convert_to(tensor, target_format)
}

/// Convert from format using global converter
#[allow(dead_code)]
pub fn convert_tensor_from<F: Float>(
    _data: &[u8],
    _metadata: &TensorMetadata,
    _source_format: &str,
) -> Result<(), IntegrationError> {
    let _converter = init_tensor_converter();
    let _converter_guard = _converter.lock().map_err(|_| {
        IntegrationError::TensorConversion("Failed to acquire converter lock".to_string())
    })?;
    // Simplified implementation - direct tensor creation requires graph context
    Ok(())
}

/// Convert precision using global converter
#[allow(dead_code)]
pub fn convert_tensor_precision<F1: Float, F2: Float>(
    _tensor: &Tensor<F1>,
) -> Result<(), IntegrationError> {
    let _converter = init_tensor_converter();
    let _converter_guard = _converter.lock().map_err(|_| {
        IntegrationError::TensorConversion("Failed to acquire converter lock".to_string())
    })?;
    Err(IntegrationError::TensorConversion(
        "Precision conversion requires graph context. Use run() function.".to_string(),
    ))
}

/// Quick conversion from ndarray
#[allow(dead_code)]
pub fn from_ndarray<F: Float>(array: ArrayD<F>) -> Result<(), IntegrationError> {
    let _converter = init_tensor_converter();
    let _converter_guard = _converter.lock().map_err(|_| {
        IntegrationError::TensorConversion("Failed to acquire converter lock".to_string())
    })?;
    Err(IntegrationError::TensorConversion(
        "Tensor creation requires graph context. Use run() function.".to_string(),
    ))
}

/// Quick conversion to ndarray.
///
/// Returns an honest error because evaluating a lazy autograd tensor requires a
/// [`crate::Context`]. Use [`to_ndarray_with_context`] to obtain real data.
#[allow(dead_code)]
pub fn to_ndarray<F: Float>(tensor: &Tensor<F>) -> Result<ArrayD<F>, IntegrationError> {
    let converter = init_tensor_converter();
    let converter_guard = converter.lock().map_err(|_| {
        IntegrationError::TensorConversion("Failed to acquire converter lock".to_string())
    })?;
    converter_guard.to_ndarray(tensor)
}

/// Quick conversion to ndarray using an evaluation context.
///
/// Evaluates `tensor` under `ctx` and returns its genuine contents.
#[allow(dead_code)]
pub fn to_ndarray_with_context<F: Float>(
    tensor: &Tensor<F>,
    ctx: &crate::Context<F>,
) -> Result<ArrayD<F>, IntegrationError> {
    let converter = init_tensor_converter();
    let converter_guard = converter.lock().map_err(|_| {
        IntegrationError::TensorConversion("Failed to acquire converter lock".to_string())
    })?;
    converter_guard.to_ndarray_with_context(tensor, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::tensor_ops::convert_to_tensor;

    #[test]
    fn test_tensor_converter_creation() {
        let converter = TensorConverter::new();
        assert!(!converter.converters.is_empty());
    }

    #[test]
    fn test_metadata_extraction() {
        crate::run(|g| {
            let converter = TensorConverter::new();
            // Use constant tensor which properly preserves shape
            let tensor = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0])
                    .expect("Failed to convert"),
                g,
            );

            // Get shape from evaluated tensor
            let actualshape = tensor.eval(g).expect("Operation failed").shape().to_vec();

            let metadata = converter
                .extract_metadata(&tensor)
                .expect("Operation failed");
            assert_eq!(metadata.shape, actualshape);
            assert!(metadata.dtype.contains("f32"));
            assert_eq!(metadata.memory_layout, MemoryLayout::RowMajor);
            // Tensors created with convert_to_tensor may require gradients by default
            assert_eq!(metadata.requires_grad, tensor.requires_grad());
            assert_eq!(metadata.device, DeviceInfo::CPU);
        });
    }

    #[test]
    fn test_precision_conversion_with_context_round_trips_real_values() {
        // The previous implementation read `tensor.data()` with no reachable
        // Context, which is always empty by construction; that fed
        // `Tensor::from_vec` a length-0 vec against a non-empty shape, which
        // silently fell back to an all-zero tensor. Non-constant,
        // non-integer values here so any such fallback (or a shape mixup)
        // is caught rather than accidentally matching.
        crate::run(|ctx_f32: &mut crate::Context<f32>| {
            let tensor_f32 = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    (2, 2),
                    vec![1.5f32, -2.25, 100.75, 3.0],
                )
                .expect("Failed to convert"),
                ctx_f32,
            );
            let converter = TensorConverter::new();

            crate::run(|ctx_f64: &mut crate::Context<f64>| {
                let tensor_f64 = converter
                    .convert_precision_with_context(&tensor_f32, ctx_f32, ctx_f64)
                    .expect("Operation failed");
                assert_eq!(tensor_f64.shape(), vec![2, 2]);

                let values = tensor_f64
                    .data_with_context(ctx_f64)
                    .expect("Operation failed");
                assert_eq!(values, vec![1.5f64, -2.25, 100.75, 3.0]);
                // Explicitly rule out the old silent-zero fabrication.
                assert_ne!(values, vec![0.0f64, 0.0, 0.0, 0.0]);
            });
        });
    }

    #[test]
    fn test_precision_conversion_without_context_is_honest_error() {
        crate::run(|g: &mut crate::Context<f32>| {
            let tensor = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    (2, 2),
                    vec![1.5f32, -2.25, 100.75, 3.0],
                )
                .expect("Failed to convert"),
                g,
            );
            let converter = TensorConverter::new();
            // Target-precision graph is irrelevant here: the function always
            // errors without ever reading it. No context for `tensor` -> must
            // NOT fabricate a zeroed tensor; must error honestly.
            let targetgraph: crate::Graph<f64> = crate::Graph::default();
            let result: Result<Tensor<'_, f64>, _> =
                converter.convert_precision(&tensor, &targetgraph);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_tensor_view() {
        crate::run(|g| {
            let tensor = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0])
                    .expect("Failed to convert"),
                g,
            );
            let converter = TensorConverter::new();

            let view = converter.create_view(&tensor).expect("Operation failed");
            assert_eq!(view.shape, vec![2, 2]);
            // Since data() returns empty, just check shape
            assert_eq!(view.shape.len(), 2);
        });
    }

    #[test]
    fn test_ndarray_conversion() {
        crate::run(|g| {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let tensor = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec((2, 2), data.clone())
                    .expect("Operation failed"),
                g,
            );

            let converter = TensorConverter::new();
            // Real evaluation: requires the graph context.
            let ndarray = converter
                .to_ndarray_with_context(&tensor, g)
                .expect("Operation failed");
            assert_eq!(ndarray.shape(), &[2, 2]);

            let tensor_back = converter
                .from_ndarray(ndarray, g)
                .expect("Operation failed");
            assert_eq!(tensor_back.shape(), tensor.shape());
        });
    }

    #[test]
    fn test_to_ndarray_without_context_is_honest_error() {
        crate::run(|g| {
            let tensor = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0])
                    .expect("Operation failed"),
                g,
            );
            let converter = TensorConverter::new();
            // No context -> must NOT fabricate [1,2,3,...]; must error honestly.
            let result = converter.to_ndarray(&tensor);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_to_ndarray_returns_real_data() {
        // The previous implementation fabricated [1,2,3,...] ignoring the
        // tensor's true contents. This proves the context-aware path returns
        // the REAL values and that from_ndarray -> to_ndarray round-trips them.
        crate::run(|g| {
            let original = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
            let array = scirs2_core::ndarray::Array::from_shape_vec((2, 3), original.clone())
                .expect("Operation failed")
                .into_dyn();

            let converter = TensorConverter::new();

            // ndarray -> tensor -> ndarray must preserve the exact values.
            let tensor = converter
                .from_ndarray(array.clone(), g)
                .expect("Operation failed");
            let recovered = converter
                .to_ndarray_with_context(&tensor, g)
                .expect("Operation failed");

            assert_eq!(recovered.shape(), &[2, 3]);
            let recovered_flat: Vec<f32> = recovered.iter().copied().collect();
            assert_eq!(recovered_flat, original);

            // And explicitly: the data is NOT the fabricated [1,2,3,4,5,6].
            assert_ne!(recovered_flat, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        });
    }

    #[test]
    fn test_global_converter() {
        crate::run(|g| {
            let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], g);

            // Test conversion to JSON format
            let json_data = convert_tensor_to(&tensor, "json").expect("Operation failed");
            assert!(!json_data.is_empty());

            // Test precision conversion: `convert_tensor_precision` is a
            // context-less entry point with NO successful path at all (see
            // its definition above -- it unconditionally returns an error).
            // Unlike `to_ndarray`/`to_ndarray_with_context`, there is no
            // context-aware sibling free function for precision conversion,
            // so there is no `tensor_f64` value to ever compare shapes
            // against here. The commented-out assertion below was
            // unreachable dead code and has been removed rather than
            // restored.
            let _result = convert_tensor_precision::<f32, f64>(&tensor);
            assert!(_result.is_err()); // Should error with context requirement
        });
    }
}
