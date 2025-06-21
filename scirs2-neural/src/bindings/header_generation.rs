//! Header file generation for C/C++ bindings
//!
//! This module handles the generation of C and C++ header files including
//! type definitions, API declarations, and different API styles.

use crate::error::{NeuralError, Result};
use std::fs;
use std::path::PathBuf;

use super::config::{ApiStyle, ArrayMapping, BindingConfig, BindingLanguage, StringMapping};

/// Header file generator
pub struct HeaderGenerator<'a> {
    config: &'a BindingConfig,
    output_dir: &'a PathBuf,
}

impl<'a> HeaderGenerator<'a> {
    /// Create a new header generator
    pub fn new(config: &'a BindingConfig, output_dir: &'a PathBuf) -> Self {
        Self { config, output_dir }
    }

    /// Generate all header files
    pub fn generate(&self) -> Result<Vec<PathBuf>> {
        let mut headers = Vec::new();

        // Generate main header
        let main_header = self.generate_header()?;
        headers.push(main_header);

        Ok(headers)
    }

    /// Generate main header file
    fn generate_header(&self) -> Result<PathBuf> {
        let header_path = self
            .output_dir
            .join("include")
            .join(format!("{}.h", self.config.library_name));

        let header_guard = format!("{}_H", self.config.library_name.to_uppercase());
        let mut header_content = String::new();

        // Header guard and includes
        header_content.push_str(&format!(
            r#"#ifndef {}
#define {}

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

"#,
            header_guard, header_guard
        ));

        // Add custom includes
        for custom_type in &self.config.type_mappings.custom_types {
            for include in &custom_type.includes {
                header_content.push_str(&format!("#include <{}>\n", include));
            }
        }

        // C++ compatibility
        header_content.push_str(
            r#"
#ifdef __cplusplus
extern "C" {
#endif

"#,
        );

        // Generate type definitions
        header_content.push_str(&self.generate_type_definitions()?);

        // Generate API declarations
        header_content.push_str(&self.generate_api_declarations()?);

        // Close C++ compatibility
        header_content.push_str(
            r#"
#ifdef __cplusplus
}
#endif

"#,
        );

        // Close header guard
        header_content.push_str(&format!("#endif // {}\n", header_guard));

        fs::write(&header_path, header_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(header_path)
    }

    /// Generate type definitions
    fn generate_type_definitions(&self) -> Result<String> {
        let mut defs = String::new();

        // Generate tensor type based on array mapping
        match self.config.type_mappings.arrays {
            ArrayMapping::PlainArrays => {
                defs.push_str(
                    r#"
// Plain array interface
typedef struct {
    void* data;
    size_t element_size;
    size_t* shape;
    size_t ndim;
    size_t total_elements;
} scirs2_tensor_t;

"#,
                );
            }
            ArrayMapping::StructuredArrays => {
                defs.push_str(
                    r#"
// Structured array interface
typedef struct {
    void* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    int dtype;
    bool owns_data;
} scirs2_tensor_t;

"#,
                );
            }
            ArrayMapping::CustomArrayType(ref custom_type) => {
                defs.push_str(&format!(
                    "// Custom array type\ntypedef {} scirs2_tensor_t;\n\n",
                    custom_type
                ));
            }
        }

        // Generate string type based on string mapping
        match self.config.type_mappings.strings {
            StringMapping::CString => {
                defs.push_str("typedef char* scirs2_string_t;\n\n");
            }
            StringMapping::LengthPrefixed => {
                defs.push_str(
                    r#"typedef struct {
    char* data;
    size_t length;
} scirs2_string_t;

"#,
                );
            }
            StringMapping::CustomString(ref custom_type) => {
                defs.push_str(&format!("typedef {} scirs2_string_t;\n\n", custom_type));
            }
        }

        // Model handle type
        defs.push_str("typedef struct scirs2_model* scirs2_model_t;\n\n");

        // Error code enumeration
        defs.push_str(
            r#"typedef enum {
    SCIRS2_SUCCESS = 0,
    SCIRS2_ERROR_INVALID_MODEL = -1,
    SCIRS2_ERROR_INVALID_INPUT = -2,
    SCIRS2_ERROR_MEMORY_ALLOCATION = -3,
    SCIRS2_ERROR_COMPUTATION = -4,
    SCIRS2_ERROR_IO = -5,
    SCIRS2_ERROR_NOT_IMPLEMENTED = -6
} scirs2_error_t;

"#,
        );

        // Add custom type definitions
        for custom_type in &self.config.type_mappings.custom_types {
            defs.push_str(&format!("{}\n\n", custom_type.definition));
        }

        Ok(defs)
    }

    /// Generate API declarations
    fn generate_api_declarations(&self) -> Result<String> {
        match self.config.api_style {
            ApiStyle::Procedural => self.generate_procedural_api(),
            ApiStyle::ObjectOriented => self.generate_oo_api(),
            ApiStyle::Hybrid => self.generate_hybrid_api(),
        }
    }

    /// Generate procedural API declarations
    fn generate_procedural_api(&self) -> Result<String> {
        let mut api = String::new();

        api.push_str("// Model management functions\n");
        api.push_str(
            "scirs2_error_t scirs2_model_load(const char* model_path, scirs2_model_t* model);\n",
        );
        api.push_str(
            "scirs2_error_t scirs2_model_save(scirs2_model_t model, const char* model_path);\n",
        );
        api.push_str("void scirs2_model_free(scirs2_model_t model);\n\n");

        api.push_str("// Tensor management functions\n");
        api.push_str("scirs2_error_t scirs2_tensor_create(size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor);\n");
        api.push_str("scirs2_error_t scirs2_tensor_from_data(void* data, size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor);\n");
        api.push_str("void scirs2_tensor_free(scirs2_tensor_t* tensor);\n");
        api.push_str(
            "scirs2_error_t scirs2_tensor_get_data(scirs2_tensor_t* tensor, void** data);\n",
        );
        api.push_str("scirs2_error_t scirs2_tensor_get_shape(scirs2_tensor_t* tensor, size_t** shape, size_t* ndim);\n\n");

        api.push_str("// Inference functions\n");
        api.push_str("scirs2_error_t scirs2_model_predict(scirs2_model_t model, scirs2_tensor_t* input, scirs2_tensor_t* output);\n");
        api.push_str("scirs2_error_t scirs2_model_predict_batch(scirs2_model_t model, scirs2_tensor_t* inputs, size_t batch_size, scirs2_tensor_t* outputs);\n\n");

        // Threading functions if enabled
        if !self.config.threading.sync_primitives.is_empty() {
            api.push_str("// Threading functions\n");
            api.push_str("scirs2_error_t scirs2_set_num_threads(int num_threads);\n");
            api.push_str("int scirs2_get_num_threads(void);\n\n");
        }

        // Memory management functions
        api.push_str("// Memory management functions\n");
        api.push_str("void* scirs2_malloc(size_t size);\n");
        api.push_str("void scirs2_free(void* ptr);\n");
        api.push_str("scirs2_error_t scirs2_get_memory_info(size_t* allocated, size_t* peak);\n\n");

        // Error handling functions
        api.push_str("// Error handling functions\n");
        api.push_str("const char* scirs2_get_error_string(scirs2_error_t error);\n");
        api.push_str("scirs2_error_t scirs2_get_last_error(void);\n");
        api.push_str("void scirs2_clear_error(void);\n\n");

        Ok(api)
    }

    /// Generate object-oriented API declarations (C++ only)
    fn generate_oo_api(&self) -> Result<String> {
        let mut api = String::new();

        if self.config.language == BindingLanguage::Cpp {
            api.push_str(
                r#"
#ifdef __cplusplus

namespace scirs2 {

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<size_t>& shape, int dtype);
    Tensor(void* data, const std::vector<size_t>& shape, int dtype);
    ~Tensor();
    
    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Data access
    void* data() const;
    const std::vector<size_t>& shape() const;
    size_t ndim() const;
    int dtype() const;
    size_t size() const;
    
    // Tensor operations
    Tensor copy() const;
    void fill(double value);
    
private:
    scirs2_tensor_t tensor_;
    bool owns_data_;
};

class Model {
public:
    Model();
    explicit Model(const std::string& model_path);
    ~Model();
    
    // Non-copyable but movable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    
    // Model operations
    bool load(const std::string& model_path);
    bool save(const std::string& model_path) const;
    Tensor predict(const Tensor& input) const;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) const;
    
    // Model information
    bool is_loaded() const;
    std::string get_info() const;
    
private:
    scirs2_model_t model_;
    bool loaded_;
};

// Exception class for SCIRS2 errors
class ScIRS2Error : public std::runtime_error {
public:
    explicit ScIRS2Error(scirs2_error_t error_code);
    explicit ScIRS2Error(const std::string& message);
    scirs2_error_t error_code() const;
    
private:
    scirs2_error_t error_code_;
};

// Utility functions
void set_num_threads(int num_threads);
int get_num_threads();
std::pair<size_t, size_t> get_memory_info();

} // namespace scirs2

#endif // __cplusplus

"#,
            );
        }

        Ok(api)
    }

    /// Generate hybrid API declarations
    fn generate_hybrid_api(&self) -> Result<String> {
        let mut api = String::new();

        // Combine procedural and object-oriented approaches
        api.push_str(&self.generate_procedural_api()?);

        if self.config.language == BindingLanguage::Cpp {
            api.push_str(&self.generate_oo_api()?);
        }

        Ok(api)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_header_generator_creation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = HeaderGenerator::new(&config, &output_dir);
        assert_eq!(generator.config.library_name, "scirs2_model");
    }

    #[test]
    fn test_type_definitions_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = HeaderGenerator::new(&config, &output_dir);
        let type_defs = generator.generate_type_definitions().unwrap();

        assert!(type_defs.contains("scirs2_tensor_t"));
        assert!(type_defs.contains("scirs2_model_t"));
        assert!(type_defs.contains("scirs2_error_t"));
    }

    #[test]
    fn test_procedural_api_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = HeaderGenerator::new(&config, &output_dir);
        let api = generator.generate_procedural_api().unwrap();

        assert!(api.contains("scirs2_model_load"));
        assert!(api.contains("scirs2_model_predict"));
        assert!(api.contains("scirs2_tensor_create"));
    }

    #[test]
    fn test_cpp_api_generation() {
        let mut config = BindingConfig::default();
        config.language = BindingLanguage::Cpp;
        config.api_style = ApiStyle::ObjectOriented;

        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = HeaderGenerator::new(&config, &output_dir);
        let api = generator.generate_oo_api().unwrap();

        assert!(api.contains("class Tensor"));
        assert!(api.contains("class Model"));
        assert!(api.contains("namespace scirs2"));
    }
}
