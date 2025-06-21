//! Source file generation and C++ wrapper generation
//!
//! This module handles the generation of C source files and C++ wrapper files
//! that implement the API defined in the header files.

use crate::error::{NeuralError, Result};
use std::fs;
use std::path::PathBuf;

use super::config::{BindingConfig, BindingLanguage};

/// Source file generator
pub struct SourceGenerator<'a> {
    config: &'a BindingConfig,
    output_dir: &'a PathBuf,
}

impl<'a> SourceGenerator<'a> {
    /// Create a new source generator
    pub fn new(config: &'a BindingConfig, output_dir: &'a PathBuf) -> Self {
        Self { config, output_dir }
    }

    /// Generate all source files
    pub fn generate(&self) -> Result<Vec<PathBuf>> {
        let mut sources = Vec::new();

        // Generate main source file
        let main_source = self.generate_source()?;
        sources.push(main_source);

        // Generate C++ wrapper if needed
        if self.config.language == BindingLanguage::CWithCppWrapper {
            let (header, source) = self.generate_cpp_wrapper()?;
            sources.push(header);
            sources.push(source);
        }

        Ok(sources)
    }

    /// Generate main source file
    fn generate_source(&self) -> Result<PathBuf> {
        let source_path = match self.config.language {
            BindingLanguage::C | BindingLanguage::CWithCppWrapper => self
                .output_dir
                .join("src")
                .join(format!("{}.c", self.config.library_name)),
            BindingLanguage::Cpp => self
                .output_dir
                .join("src")
                .join(format!("{}.cpp", self.config.library_name)),
        };

        let mut source_content = String::new();

        // Includes
        source_content.push_str(&format!(
            r#"#include "{}.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

"#,
            self.config.library_name
        ));

        // Generate implementation
        source_content.push_str(&self.generate_implementation()?);

        fs::write(&source_path, source_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(source_path)
    }

    /// Generate implementation code
    fn generate_implementation(&self) -> Result<String> {
        let mut impl_content = String::new();

        // Global state for error handling
        impl_content.push_str(
            r#"// Global error state
static scirs2_error_t g_last_error = SCIRS2_SUCCESS;
static char g_error_message[1024] = {0};

// Error handling implementation
const char* scirs2_get_error_string(scirs2_error_t error) {
    switch (error) {
        case SCIRS2_SUCCESS: return "Success";
        case SCIRS2_ERROR_INVALID_MODEL: return "Invalid model";
        case SCIRS2_ERROR_INVALID_INPUT: return "Invalid input";
        case SCIRS2_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case SCIRS2_ERROR_COMPUTATION: return "Computation error";
        case SCIRS2_ERROR_IO: return "I/O error";
        case SCIRS2_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        default: return "Unknown error";
    }
}

scirs2_error_t scirs2_get_last_error(void) {
    return g_last_error;
}

void scirs2_clear_error(void) {
    g_last_error = SCIRS2_SUCCESS;
    memset(g_error_message, 0, sizeof(g_error_message));
}

// Memory management implementation
void* scirs2_malloc(size_t size) {
    return malloc(size);
}

void scirs2_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

scirs2_error_t scirs2_get_memory_info(size_t* allocated, size_t* peak) {
    // Placeholder implementation
    if (allocated) *allocated = 0;
    if (peak) *peak = 0;
    return SCIRS2_SUCCESS;
}

// Model implementation
struct scirs2_model {
    void* model_data;
    bool is_loaded;
    char model_path[512];
};

scirs2_error_t scirs2_model_load(const char* model_path, scirs2_model_t* model) {
    if (!model_path || !model) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    *model = (scirs2_model_t)malloc(sizeof(struct scirs2_model));
    if (!*model) {
        g_last_error = SCIRS2_ERROR_MEMORY_ALLOCATION;
        return g_last_error;
    }

    // Initialize model structure
    (*model)->model_data = NULL;
    (*model)->is_loaded = false;
    strncpy((*model)->model_path, model_path, sizeof((*model)->model_path) - 1);
    (*model)->model_path[sizeof((*model)->model_path) - 1] = '\0';

    // Placeholder: actual model loading would happen here
    (*model)->is_loaded = true;

    return SCIRS2_SUCCESS;
}

scirs2_error_t scirs2_model_save(scirs2_model_t model, const char* model_path) {
    if (!model || !model_path) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    if (!model->is_loaded) {
        g_last_error = SCIRS2_ERROR_INVALID_MODEL;
        return g_last_error;
    }

    // Placeholder: actual model saving would happen here
    strncpy(model->model_path, model_path, sizeof(model->model_path) - 1);
    model->model_path[sizeof(model->model_path) - 1] = '\0';

    return SCIRS2_SUCCESS;
}

void scirs2_model_free(scirs2_model_t model) {
    if (model) {
        // Free any model-specific data
        if (model->model_data) {
            free(model->model_data);
        }
        free(model);
    }
}

// Tensor implementation
scirs2_error_t scirs2_tensor_create(size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor) {
    if (!shape || !tensor || ndim == 0) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    // Allocate tensor structure
    tensor->data = NULL;
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        g_last_error = SCIRS2_ERROR_MEMORY_ALLOCATION;
        return g_last_error;
    }

    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->strides) {
        free(tensor->shape);
        g_last_error = SCIRS2_ERROR_MEMORY_ALLOCATION;
        return g_last_error;
    }

    // Copy shape and calculate strides
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->owns_data = false;

    // Calculate strides (C-order)
    tensor->strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        tensor->strides[i] = tensor->strides[i + 1] * shape[i + 1];
    }

    return SCIRS2_SUCCESS;
}

scirs2_error_t scirs2_tensor_from_data(void* data, size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor) {
    scirs2_error_t result = scirs2_tensor_create(shape, ndim, dtype, tensor);
    if (result != SCIRS2_SUCCESS) {
        return result;
    }

    tensor->data = data;
    tensor->owns_data = false;

    return SCIRS2_SUCCESS;
}

void scirs2_tensor_free(scirs2_tensor_t* tensor) {
    if (tensor) {
        if (tensor->owns_data && tensor->data) {
            free(tensor->data);
        }
        if (tensor->shape) {
            free(tensor->shape);
        }
        if (tensor->strides) {
            free(tensor->strides);
        }
        memset(tensor, 0, sizeof(scirs2_tensor_t));
    }
}

scirs2_error_t scirs2_tensor_get_data(scirs2_tensor_t* tensor, void** data) {
    if (!tensor || !data) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    *data = tensor->data;
    return SCIRS2_SUCCESS;
}

scirs2_error_t scirs2_tensor_get_shape(scirs2_tensor_t* tensor, size_t** shape, size_t* ndim) {
    if (!tensor || !shape || !ndim) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    *shape = tensor->shape;
    *ndim = tensor->ndim;
    return SCIRS2_SUCCESS;
}

// Inference implementation
scirs2_error_t scirs2_model_predict(scirs2_model_t model, scirs2_tensor_t* input, scirs2_tensor_t* output) {
    if (!model || !input || !output) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    if (!model->is_loaded) {
        g_last_error = SCIRS2_ERROR_INVALID_MODEL;
        return g_last_error;
    }

    // Placeholder: actual inference would happen here
    // For now, just copy input to output
    if (input->data && output->data) {
        size_t input_size = 1;
        size_t output_size = 1;
        
        for (size_t i = 0; i < input->ndim; i++) {
            input_size *= input->shape[i];
        }
        for (size_t i = 0; i < output->ndim; i++) {
            output_size *= output->shape[i];
        }
        
        size_t copy_size = (input_size < output_size) ? input_size : output_size;
        memcpy(output->data, input->data, copy_size * sizeof(float));
    }

    return SCIRS2_SUCCESS;
}

scirs2_error_t scirs2_model_predict_batch(scirs2_model_t model, scirs2_tensor_t* inputs, size_t batch_size, scirs2_tensor_t* outputs) {
    if (!model || !inputs || !outputs || batch_size == 0) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }

    // Process each input in the batch
    for (size_t i = 0; i < batch_size; i++) {
        scirs2_error_t result = scirs2_model_predict(model, &inputs[i], &outputs[i]);
        if (result != SCIRS2_SUCCESS) {
            return result;
        }
    }

    return SCIRS2_SUCCESS;
}

"#,
        );

        // Threading functions if enabled
        if !self.config.threading.sync_primitives.is_empty() {
            impl_content.push_str(
                r#"
// Threading implementation
static int g_num_threads = 1;

scirs2_error_t scirs2_set_num_threads(int num_threads) {
    if (num_threads <= 0) {
        g_last_error = SCIRS2_ERROR_INVALID_INPUT;
        return g_last_error;
    }
    g_num_threads = num_threads;
    return SCIRS2_SUCCESS;
}

int scirs2_get_num_threads(void) {
    return g_num_threads;
}

"#,
            );
        }

        Ok(impl_content)
    }

    /// Generate C++ wrapper files
    fn generate_cpp_wrapper(&self) -> Result<(PathBuf, PathBuf)> {
        let header_path = self
            .output_dir
            .join("include")
            .join(format!("{}_cpp.hpp", self.config.library_name));
        let source_path = self
            .output_dir
            .join("src")
            .join(format!("{}_cpp.cpp", self.config.library_name));

        // Generate C++ header
        let header_content = format!(
            r#"#ifndef {}_CPP_HPP
#define {}_CPP_HPP

#include "{}.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace scirs2 {{

class Exception : public std::runtime_error {{
public:
    explicit Exception(scirs2_error_t error_code);
    explicit Exception(const std::string& message);
    scirs2_error_t error_code() const {{ return error_code_; }}
    
private:
    scirs2_error_t error_code_;
}};

class Tensor {{
public:
    Tensor();
    Tensor(const std::vector<size_t>& shape, int dtype);
    Tensor(void* data, const std::vector<size_t>& shape, int dtype);
    ~Tensor();
    
    // Copy and move semantics
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Accessors
    void* data() const {{ return tensor_.data; }}
    const std::vector<size_t>& shape() const {{ return shape_; }}
    size_t ndim() const {{ return tensor_.ndim; }}
    int dtype() const {{ return tensor_.dtype; }}
    
    // Operations
    void fill(double value);
    
private:
    scirs2_tensor_t tensor_;
    std::vector<size_t> shape_;
    bool owns_data_;
}};

class Model {{
public:
    Model();
    explicit Model(const std::string& model_path);
    ~Model();
    
    // Non-copyable, movable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    
    // Operations
    void load(const std::string& model_path);
    void save(const std::string& model_path);
    Tensor predict(const Tensor& input);
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs);
    
    bool is_loaded() const {{ return loaded_; }}
    
private:
    scirs2_model_t model_;
    bool loaded_;
}};

// Utility functions
void set_num_threads(int num_threads);
int get_num_threads();
std::pair<size_t, size_t> get_memory_info();

}} // namespace scirs2

#endif // {}_CPP_HPP
"#,
            self.config.library_name.to_uppercase(),
            self.config.library_name.to_uppercase(),
            self.config.library_name,
            self.config.library_name.to_uppercase()
        );

        // Generate C++ source
        let source_content = format!(
            r#"#include "{}_cpp.hpp"
#include <cstring>
#include <algorithm>

namespace scirs2 {{

// Exception implementation
Exception::Exception(scirs2_error_t error_code)
    : std::runtime_error(scirs2_get_error_string(error_code)), error_code_(error_code) {{}}

Exception::Exception(const std::string& message)
    : std::runtime_error(message), error_code_(SCIRS2_ERROR_COMPUTATION) {{}}

// Tensor implementation
Tensor::Tensor() : owns_data_(false) {{
    std::memset(&tensor_, 0, sizeof(tensor_));
}}

Tensor::Tensor(const std::vector<size_t>& shape, int dtype) : shape_(shape), owns_data_(true) {{
    scirs2_error_t result = scirs2_tensor_create(const_cast<size_t*>(shape.data()), shape.size(), dtype, &tensor_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
}}

Tensor::~Tensor() {{
    if (owns_data_) {{
        scirs2_tensor_free(&tensor_);
    }}
}}

// Model implementation
Model::Model() : model_(nullptr), loaded_(false) {{}}

Model::Model(const std::string& model_path) : model_(nullptr), loaded_(false) {{
    load(model_path);
}}

Model::~Model() {{
    if (model_) {{
        scirs2_model_free(model_);
    }}
}}

void Model::load(const std::string& model_path) {{
    scirs2_error_t result = scirs2_model_load(model_path.c_str(), &model_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
    loaded_ = true;
}}

Tensor Model::predict(const Tensor& input) {{
    if (!loaded_) {{
        throw Exception("Model not loaded");
    }}
    
    // Create output tensor with same shape as input for now
    Tensor output(input.shape(), input.dtype());
    
    scirs2_error_t result = scirs2_model_predict(model_, 
        const_cast<scirs2_tensor_t*>(&input.tensor_), &output.tensor_);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
    
    return output;
}}

// Utility functions
void set_num_threads(int num_threads) {{
    scirs2_error_t result = scirs2_set_num_threads(num_threads);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
}}

int get_num_threads() {{
    return scirs2_get_num_threads();
}}

std::pair<size_t, size_t> get_memory_info() {{
    size_t allocated, peak;
    scirs2_error_t result = scirs2_get_memory_info(&allocated, &peak);
    if (result != SCIRS2_SUCCESS) {{
        throw Exception(result);
    }}
    return {{allocated, peak}};
}}

}} // namespace scirs2
"#,
            self.config.library_name
        );

        fs::write(&header_path, header_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        fs::write(&source_path, source_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok((header_path, source_path))
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_source_generator_creation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = SourceGenerator::new(&config, &output_dir);
        assert_eq!(generator.config.library_name, "scirs2_model");
    }

    #[test]
    fn test_implementation_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = SourceGenerator::new(&config, &output_dir);
        let impl_code = generator.generate_implementation().unwrap();

        assert!(impl_code.contains("scirs2_model_load"));
        assert!(impl_code.contains("scirs2_tensor_create"));
        assert!(impl_code.contains("scirs2_get_error_string"));
    }
}
