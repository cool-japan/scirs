//! Examples and documentation generation for C/C++ bindings
//!
//! This module handles the generation of example code and documentation
//! files to help users understand how to use the generated bindings.

use crate::error::{NeuralError, Result};
use std::fs;
use std::path::PathBuf;

use super::config::{BindingConfig, BindingLanguage};

/// Examples and documentation generator
pub struct ExamplesDocsGenerator<'a> {
    config: &'a BindingConfig,
    output_dir: &'a PathBuf,
}

impl<'a> ExamplesDocsGenerator<'a> {
    /// Create a new examples and docs generator
    pub fn new(config: &'a BindingConfig, output_dir: &'a PathBuf) -> Self {
        Self { config, output_dir }
    }

    /// Generate examples and documentation
    pub fn generate(&self) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
        let examples = self.generate_examples()?;
        let docs = self.generate_documentation()?;
        Ok((examples, docs))
    }

    /// Generate example files
    fn generate_examples(&self) -> Result<Vec<PathBuf>> {
        let mut examples = Vec::new();

        // Generate C example
        let c_example_path = self.output_dir.join("examples").join("basic_usage.c");
        let c_example_content = format!(
            r#"#include <stdio.h>
#include <stdlib.h>
#include "{}.h"

void error_callback(int error_code, const char* message, void* user_data) {{
    fprintf(stderr, "Error %d: %s\n", error_code, message);
}}

int main() {{
    // Load model
    scirs2_model_t model;
    scirs2_error_t result = scirs2_model_load("model.scirs2", &model);
    if (result != SCIRS2_SUCCESS) {{
        printf("Failed to load model: %s\n", scirs2_get_error_string(result));
        return 1;
    }}
    
    printf("Model loaded successfully\n");
    
    // Create input tensor
    size_t shape[] = {{1, 3, 224, 224}};
    scirs2_tensor_t input;
    result = scirs2_tensor_create(shape, 4, 0, &input); // dtype 0 = float32
    if (result != SCIRS2_SUCCESS) {{
        printf("Failed to create input tensor: %s\n", scirs2_get_error_string(result));
        scirs2_model_free(model);
        return 1;
    }}
    
    // Create output tensor
    size_t output_shape[] = {{1, 1000}};
    scirs2_tensor_t output;
    result = scirs2_tensor_create(output_shape, 2, 0, &output);
    if (result != SCIRS2_SUCCESS) {{
        printf("Failed to create output tensor: %s\n", scirs2_get_error_string(result));
        scirs2_tensor_free(&input);
        scirs2_model_free(model);
        return 1;
    }}
    
    // Run inference
    result = scirs2_model_predict(model, &input, &output);
    if (result != SCIRS2_SUCCESS) {{
        printf("Prediction failed: %s\n", scirs2_get_error_string(result));
    }} else {{
        printf("Prediction completed successfully\n");
    }}
    
    // Cleanup
    scirs2_tensor_free(&input);
    scirs2_tensor_free(&output);
    scirs2_model_free(model);
    
    return 0;
}}
"#,
            self.config.library_name
        );

        fs::write(&c_example_path, c_example_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        examples.push(c_example_path);

        // Generate C++ example if applicable
        if self.config.language == BindingLanguage::Cpp
            || self.config.language == BindingLanguage::CWithCppWrapper
        {
            let cpp_example_path = self.output_dir.join("examples").join("basic_usage.cpp");
            let cpp_example_content = format!(
                r#"#include <iostream>
#include <vector>
#include "{}_cpp.hpp"

int main() {{
    try {{
        // Load model
        scirs2::Model model("model.scirs2");
        std::cout << "Model loaded successfully" << std::endl;
        
        // Create input tensor
        std::vector<size_t> input_shape = {{1, 3, 224, 224}};
        scirs2::Tensor input(input_shape, 0); // dtype 0 = float32
        
        // Fill input with sample data
        input.fill(0.5);
        
        // Run inference
        scirs2::Tensor output = model.predict(input);
        std::cout << "Prediction completed successfully" << std::endl;
        
        // Print output shape
        const auto& output_shape = output.shape();
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {{
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }}
        std::cout << "]" << std::endl;
        
    }} catch (const scirs2::Exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
    
    return 0;
}}
"#,
                self.config.library_name
            );

            fs::write(&cpp_example_path, cpp_example_content)
                .map_err(|e| NeuralError::IOError(e.to_string()))?;
            examples.push(cpp_example_path);
        }

        // Generate example Makefile
        let example_makefile_path = self.output_dir.join("examples").join("Makefile");
        let example_makefile_content = format!(
            r#"CC = gcc
CXX = g++
CFLAGS = -std=c99 -Wall -Wextra
CXXFLAGS = -std=c++17 -Wall -Wextra
INCLUDES = -I../include
LIBS = -L../build -l{}

TARGETS = basic_usage_c basic_usage_cpp

.PHONY: all clean

all: $(TARGETS)

basic_usage_c: basic_usage.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

basic_usage_cpp: basic_usage.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)

help:
	@echo "Available targets:"
	@echo "  all    - Build all examples"
	@echo "  clean  - Remove example binaries"
	@echo "  help   - Show this help"
"#,
            self.config.library_name
        );

        fs::write(&example_makefile_path, example_makefile_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        examples.push(example_makefile_path);

        Ok(examples)
    }

    /// Generate documentation files
    fn generate_documentation(&self) -> Result<Vec<PathBuf>> {
        let mut docs = Vec::new();

        // Generate README
        let readme_path = self.output_dir.join("README.md");
        let readme_content = format!(
            r#"# {} - SciRS2 Neural Network C/C++ Bindings

This library provides C/C++ bindings for SciRS2 neural network models.

## Features

- Load and run SciRS2 neural network models
- C and C++ APIs available
- Cross-platform support
- Memory-safe tensor operations
- Error handling with detailed messages

## Building

### Using CMake

```bash
mkdir build
cd build
cmake ..
make
```

### Using Make

```bash
make
```

## Usage

### C API

```c
#include "{}.h"

int main() {{
    scirs2_model_t model;
    scirs2_model_load("model.scirs2", &model);
    
    // Create tensors and run inference
    // ...
    
    scirs2_model_free(&model);
    return 0;
}}
```

### C++ API

```cpp
#include "{}_cpp.hpp"

int main() {{
    try {{
        scirs2::Model model("model.scirs2");
        
        std::vector<size_t> shape = {{1, 3, 224, 224}};
        scirs2::Tensor input(shape, 0);
        
        scirs2::Tensor output = model.predict(input);
        
    }} catch (const scirs2::Exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
    }}
    
    return 0;
}}
```

## API Reference

### Error Handling

All C functions return `scirs2_error_t` to indicate success or failure:

- `SCIRS2_SUCCESS`: Operation completed successfully
- `SCIRS2_ERROR_INVALID_MODEL`: Invalid model handle or model not loaded
- `SCIRS2_ERROR_INVALID_INPUT`: Invalid input parameters
- `SCIRS2_ERROR_MEMORY_ALLOCATION`: Memory allocation failed
- `SCIRS2_ERROR_COMPUTATION`: Computation error during inference
- `SCIRS2_ERROR_IO`: File I/O error
- `SCIRS2_ERROR_NOT_IMPLEMENTED`: Feature not implemented

### Model Management

#### C API

```c
// Load a model from file
scirs2_error_t scirs2_model_load(const char* model_path, scirs2_model_t* model);

// Save a model to file
scirs2_error_t scirs2_model_save(scirs2_model_t model, const char* model_path);

// Free model resources
void scirs2_model_free(scirs2_model_t model);
```

#### C++ API

```cpp
class Model {{
public:
    Model();
    explicit Model(const std::string& model_path);
    
    void load(const std::string& model_path);
    void save(const std::string& model_path);
    
    Tensor predict(const Tensor& input);
    bool is_loaded() const;
}};
```

### Tensor Operations

#### C API

```c
// Create a tensor with specified shape and data type
scirs2_error_t scirs2_tensor_create(size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor);

// Create a tensor from existing data
scirs2_error_t scirs2_tensor_from_data(void* data, size_t* shape, size_t ndim, int dtype, scirs2_tensor_t* tensor);

// Free tensor resources
void scirs2_tensor_free(scirs2_tensor_t* tensor);

// Get tensor data pointer
scirs2_error_t scirs2_tensor_get_data(scirs2_tensor_t* tensor, void** data);

// Get tensor shape information
scirs2_error_t scirs2_tensor_get_shape(scirs2_tensor_t* tensor, size_t** shape, size_t* ndim);
```

#### C++ API

```cpp
class Tensor {{
public:
    Tensor();
    Tensor(const std::vector<size_t>& shape, int dtype);
    Tensor(void* data, const std::vector<size_t>& shape, int dtype);
    
    void* data() const;
    const std::vector<size_t>& shape() const;
    size_t ndim() const;
    int dtype() const;
    
    void fill(double value);
}};
```

### Inference

#### C API

```c
// Run inference on a single input
scirs2_error_t scirs2_model_predict(scirs2_model_t model, scirs2_tensor_t* input, scirs2_tensor_t* output);

// Run inference on a batch of inputs
scirs2_error_t scirs2_model_predict_batch(scirs2_model_t model, scirs2_tensor_t* inputs, size_t batch_size, scirs2_tensor_t* outputs);
```

#### C++ API

```cpp
Tensor predict(const Tensor& input);
std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs);
```

## Installation

### System-wide Installation

```bash
make install
```

This will install the library to `/usr/local` by default. You can customize the installation prefix:

```bash
make install PREFIX=/usr
```

### Using pkg-config

After installation, you can use pkg-config to get compiler and linker flags:

```bash
pkg-config --cflags {}
pkg-config --libs {}
```

## Threading

The library supports multi-threaded inference. You can control the number of threads:

### C API

```c
scirs2_error_t scirs2_set_num_threads(int num_threads);
int scirs2_get_num_threads(void);
```

### C++ API

```cpp
void scirs2::set_num_threads(int num_threads);
int scirs2::get_num_threads();
```

## Memory Management

The library provides utilities for memory management and monitoring:

### C API

```c
void* scirs2_malloc(size_t size);
void scirs2_free(void* ptr);
scirs2_error_t scirs2_get_memory_info(size_t* allocated, size_t* peak);
```

### C++ API

```cpp
std::pair<size_t, size_t> scirs2::get_memory_info();
```

## License

This library is released under the same license as the SciRS2 project.

## Contributing

Please see the main SciRS2 project for contribution guidelines.
"#,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name,
            self.config.library_name
        );

        fs::write(&readme_path, readme_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        docs.push(readme_path);

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_examples_docs_generator() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        // Create examples directory
        fs::create_dir_all(output_dir.join("examples")).unwrap();

        let generator = ExamplesDocsGenerator::new(&config, &output_dir);
        let (examples, docs) = generator.generate().unwrap();

        assert!(!examples.is_empty());
        assert!(!docs.is_empty());
        assert!(examples[0]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains(".c"));
        assert!(docs[0]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("README"));
    }

    #[test]
    fn test_example_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        // Create examples directory
        fs::create_dir_all(output_dir.join("examples")).unwrap();

        let generator = ExamplesDocsGenerator::new(&config, &output_dir);
        let examples = generator.generate_examples().unwrap();

        assert!(!examples.is_empty());

        // Check that C example was created
        let c_example = examples
            .iter()
            .find(|p| p.file_name().unwrap().to_str().unwrap().ends_with(".c"));
        assert!(c_example.is_some());

        let content = std::fs::read_to_string(c_example.unwrap()).unwrap();
        assert!(content.contains("scirs2_model_load"));
        assert!(content.contains("scirs2_tensor_create"));
    }

    #[test]
    fn test_documentation_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = ExamplesDocsGenerator::new(&config, &output_dir);
        let docs = generator.generate_documentation().unwrap();

        assert!(!docs.is_empty());

        let readme = docs
            .iter()
            .find(|p| p.file_name().unwrap().to_str().unwrap() == "README.md");
        assert!(readme.is_some());

        let content = std::fs::read_to_string(readme.unwrap()).unwrap();
        assert!(content.contains("SciRS2 Neural Network"));
        assert!(content.contains("## Features"));
        assert!(content.contains("## Building"));
    }
}
