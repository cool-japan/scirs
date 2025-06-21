//! Build system file generation for C/C++ bindings
//!
//! This module handles the generation of build system files including
//! CMake, Makefile, and pkg-config files.

use crate::error::{NeuralError, Result};
use std::fs;
use std::path::PathBuf;

use super::config::{BindingConfig, BuildSystem};

/// Build system file generator
pub struct BuildSystemGenerator<'a> {
    config: &'a BindingConfig,
    output_dir: &'a PathBuf,
}

impl<'a> BuildSystemGenerator<'a> {
    /// Create a new build system generator
    pub fn new(config: &'a BindingConfig, output_dir: &'a PathBuf) -> Self {
        Self { config, output_dir }
    }

    /// Generate all build system files
    pub fn generate(&self) -> Result<Vec<PathBuf>> {
        let mut build_files = Vec::new();

        match self.config.build_system.system {
            BuildSystem::CMake => {
                let cmake_path = self.generate_cmake()?;
                build_files.push(cmake_path);
            }
            BuildSystem::Make => {
                let make_path = self.generate_makefile()?;
                build_files.push(make_path);
            }
            _ => {
                // Generate CMake as default
                let cmake_path = self.generate_cmake()?;
                build_files.push(cmake_path);
            }
        }

        // Generate pkg-config file if requested
        if self.config.build_system.install_config.generate_pkgconfig {
            let pc_path = self.generate_pkgconfig()?;
            build_files.push(pc_path);
        }

        Ok(build_files)
    }

    /// Generate CMakeLists.txt
    fn generate_cmake(&self) -> Result<PathBuf> {
        let cmake_path = self.output_dir.join("CMakeLists.txt");

        let cmake_content = format!(
            r#"cmake_minimum_required(VERSION 3.12)
project({} VERSION 1.0.0 LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} {}")
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} {}")

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.c" "src/*.cpp")

# Create library
add_library({} SHARED ${{SOURCES}})

# Link libraries
target_link_libraries({} m)

# Install targets
install(TARGETS {}
    LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
    ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
    RUNTIME DESTINATION ${{CMAKE_INSTALL_BINDIR}})

install(DIRECTORY include/
    DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}})

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS AND EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/tests")
    enable_testing()
    add_subdirectory(tests)
endif()
"#,
            self.config.library_name,
            self.config.build_system.compiler_flags.join(" "),
            self.config.build_system.compiler_flags.join(" "),
            self.config.library_name,
            self.config.library_name,
            self.config.library_name
        );

        fs::write(&cmake_path, cmake_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(cmake_path)
    }

    /// Generate Makefile
    fn generate_makefile(&self) -> Result<PathBuf> {
        let makefile_path = self.output_dir.join("Makefile");

        let makefile_content = format!(
            r#"# Makefile for {}

CC = gcc
CXX = g++
CFLAGS = -std=c99 -fPIC {}
CXXFLAGS = -std=c++17 -fPIC {}
LDFLAGS = {}
INCLUDES = -Iinclude

# Source and object files
SRCDIR = src
SOURCES = $(wildcard $(SRCDIR)/*.c $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:.c=.o)
OBJECTS := $(OBJECTS:.cpp=.o)

# Library name
LIBRARY = lib{}.so
STATIC_LIB = lib{}.a

# Installation directories
PREFIX = {}
LIBDIR = $(PREFIX)/{}
INCDIR = $(PREFIX)/{}
BINDIR = $(PREFIX)/{}

.PHONY: all clean install uninstall examples

all: $(LIBRARY) $(STATIC_LIB)

$(LIBRARY): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(LIBRARY) $(STATIC_LIB)
	$(MAKE) -C examples clean

install: $(LIBRARY) $(STATIC_LIB)
	mkdir -p $(LIBDIR) $(INCDIR) $(BINDIR)
	cp $(LIBRARY) $(STATIC_LIB) $(LIBDIR)/
	cp -r include/* $(INCDIR)/

uninstall:
	rm -f $(LIBDIR)/$(LIBRARY) $(LIBDIR)/$(STATIC_LIB)
	rm -rf $(INCDIR)/{}.h

examples:
	$(MAKE) -C examples

help:
	@echo "Available targets:"
	@echo "  all      - Build library (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install library and headers"
	@echo "  uninstall - Remove installed files"
	@echo "  examples - Build example programs"
	@echo "  help     - Show this help"
"#,
            self.config.library_name,
            self.config.build_system.compiler_flags.join(" "),
            self.config.build_system.compiler_flags.join(" "),
            self.config.build_system.linker_flags.join(" "),
            self.config.library_name,
            self.config.library_name,
            self.config.build_system.install_config.prefix,
            self.config.build_system.install_config.lib_dir,
            self.config.build_system.install_config.include_dir,
            self.config.build_system.install_config.bin_dir,
            self.config.library_name
        );

        fs::write(&makefile_path, makefile_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(makefile_path)
    }

    /// Generate pkg-config file
    fn generate_pkgconfig(&self) -> Result<PathBuf> {
        let pc_path = self
            .output_dir
            .join(format!("{}.pc", self.config.library_name));

        let pc_content = format!(
            r#"prefix={}
libdir=${{prefix}}/{}
includedir=${{prefix}}/{}

Name: {}
Description: SciRS2 neural network C/C++ bindings
Version: 1.0.0
Libs: -L${{libdir}} -l{}
Cflags: -I${{includedir}}
"#,
            self.config.build_system.install_config.prefix,
            self.config.build_system.install_config.lib_dir,
            self.config.build_system.install_config.include_dir,
            self.config.library_name,
            self.config.library_name
        );

        fs::write(&pc_path, pc_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(pc_path)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::*;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_build_system_generator() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = BuildSystemGenerator::new(&config, &output_dir);
        let build_files = generator.generate().unwrap();

        assert!(!build_files.is_empty());
        assert!(build_files[0]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("CMakeLists.txt"));
    }

    #[test]
    fn test_cmake_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = BuildSystemGenerator::new(&config, &output_dir);
        let cmake_path = generator.generate_cmake().unwrap();

        assert!(cmake_path.exists());
        let content = std::fs::read_to_string(&cmake_path).unwrap();
        assert!(content.contains("cmake_minimum_required"));
        assert!(content.contains("project(scirs2_model"));
    }

    #[test]
    fn test_pkgconfig_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().to_path_buf();

        let generator = BuildSystemGenerator::new(&config, &output_dir);
        let pc_path = generator.generate_pkgconfig().unwrap();

        assert!(pc_path.exists());
        let content = std::fs::read_to_string(&pc_path).unwrap();
        assert!(content.contains("Name: scirs2_model"));
        assert!(content.contains("Libs: -L"));
    }
}
