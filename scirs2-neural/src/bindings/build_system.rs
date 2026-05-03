//! Build system file generation for C/C++ bindings
//!
//! This module handles the generation of build system files including
//! CMake, Makefile, and pkg-config files.

use crate::error::{NeuralError, Result};
use std::fs;
use std::path::{Path, PathBuf};

use super::config::{BindingConfig, BuildSystem};

/// Build system file generator
pub struct BuildSystemGenerator<'a> {
    config: &'a BindingConfig,
    output_dir: &'a Path,
}

impl<'a> BuildSystemGenerator<'a> {
    /// Create a new build system generator
    pub fn new(config: &'a BindingConfig, output_dir: &'a Path) -> Self {
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
    pub fn generate_cmake(&self) -> Result<PathBuf> {
        let cmake_path = self.output_dir.join("CMakeLists.txt");
        let lib_name = &self.config.library_name;
        let cflags = self.config.build_system.compiler_flags.join(" ");
        let cmake_content = format!(
            "cmake_minimum_required(VERSION 3.12)\n\
             project({lib_name} VERSION 1.0.0 LANGUAGES C CXX)\n\
             set(CMAKE_C_STANDARD 99)\n\
             set(CMAKE_CXX_STANDARD 17)\n\
             if(NOT CMAKE_BUILD_TYPE)\n  set(CMAKE_BUILD_TYPE Release)\nendif()\n\
             set(CMAKE_C_FLAGS \"${{CMAKE_C_FLAGS}} {cflags}\")\n\
             set(CMAKE_CXX_FLAGS \"${{CMAKE_CXX_FLAGS}} {cflags}\")\n\
             include_directories(include)\n\
             file(GLOB_RECURSE SOURCES \"src/*.c\" \"src/*.cpp\")\n\
             add_library({lib_name} SHARED ${{SOURCES}})\n\
             target_link_libraries({lib_name} m)\n\
             install(TARGETS {lib_name}\n\
               LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}}\n\
               ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}}\n\
               RUNTIME DESTINATION ${{CMAKE_INSTALL_BINDIR}})\n\
             install(DIRECTORY include/ DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}})\n"
        );
        fs::write(&cmake_path, cmake_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(cmake_path)
    }

    /// Generate Makefile
    fn generate_makefile(&self) -> Result<PathBuf> {
        let makefile_path = self.output_dir.join("Makefile");
        let lib_name = &self.config.library_name;
        let cflags = self.config.build_system.compiler_flags.join(" ");
        let ldflags = self.config.build_system.linker_flags.join(" ");
        let prefix = &self.config.build_system.install_config.prefix;
        let libdir = &self.config.build_system.install_config.lib_dir;
        let incdir = &self.config.build_system.install_config.include_dir;
        let bindir = &self.config.build_system.install_config.bin_dir;
        let makefile_content = format!(
            "# Makefile for {lib_name}\n\
             CC = gcc\nCXX = g++\n\
             CFLAGS = -std=c99 -fPIC {cflags}\n\
             CXXFLAGS = -std=c++17 -fPIC {cflags}\n\
             LDFLAGS = {ldflags}\n\
             INCLUDES = -Iinclude\n\
             SRCDIR = src\n\
             SOURCES = $(wildcard $(SRCDIR)/*.c $(SRCDIR)/*.cpp)\n\
             OBJECTS = $(SOURCES:.c=.o)\n\
             OBJECTS := $(OBJECTS:.cpp=.o)\n\
             LIBRARY = lib{lib_name}.so\n\
             STATIC_LIB = lib{lib_name}.a\n\
             PREFIX = {prefix}\n\
             LIBDIR = $(PREFIX)/{libdir}\n\
             INCDIR = $(PREFIX)/{incdir}\n\
             BINDIR = $(PREFIX)/{bindir}\n\
             .PHONY: all clean install\n\
             all: $(LIBRARY) $(STATIC_LIB)\n\
             $(LIBRARY): $(OBJECTS)\n\
             \t$(CC) -shared -o $@ $^ $(LDFLAGS)\n\
             $(STATIC_LIB): $(OBJECTS)\n\
             \tar rcs $@ $^\n\
             %.o: %.c\n\
             \t$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@\n\
             %.o: %.cpp\n\
             \t$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@\n\
             clean:\n\
             \trm -f $(OBJECTS) $(LIBRARY) $(STATIC_LIB)\n\
             install: $(LIBRARY) $(STATIC_LIB)\n\
             \tmkdir -p $(LIBDIR) $(INCDIR) $(BINDIR)\n\
             \tcp $(LIBRARY) $(STATIC_LIB) $(LIBDIR)/\n\
             \tcp -r include/* $(INCDIR)/\n"
        );
        fs::write(&makefile_path, makefile_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(makefile_path)
    }

    /// Generate pkg-config file
    pub fn generate_pkgconfig(&self) -> Result<PathBuf> {
        let lib_name = &self.config.library_name;
        let pc_path = self.output_dir.join(format!("{lib_name}.pc"));
        let prefix = &self.config.build_system.install_config.prefix;
        let libdir = &self.config.build_system.install_config.lib_dir;
        let incdir = &self.config.build_system.install_config.include_dir;
        let pc_content = format!(
            "prefix={prefix}\n\
             libdir=${{prefix}}/{libdir}\n\
             includedir=${{prefix}}/{incdir}\n\
             Name: {lib_name}\n\
             Description: SciRS2 neural network C/C++ bindings\n\
             Version: 1.0.0\n\
             Libs: -L${{libdir}} -l{lib_name}\n\
             Cflags: -I${{includedir}}\n"
        );
        fs::write(&pc_path, pc_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
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
        let temp_dir = TempDir::new().expect("TempDir::new failed");
        let output_dir = temp_dir.path();
        let generator = BuildSystemGenerator::new(&config, output_dir);
        let build_files = generator.generate().expect("generate failed");
        assert!(!build_files.is_empty());
    }

    #[test]
    fn test_cmake_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().expect("TempDir::new failed");
        let output_dir = temp_dir.path();
        let generator = BuildSystemGenerator::new(&config, output_dir);
        let cmake_path = generator.generate_cmake().expect("generate_cmake failed");
        assert!(cmake_path.exists());
        let content = std::fs::read_to_string(&cmake_path).expect("read_to_string failed");
        assert!(content.contains("cmake_minimum_required"));
    }

    #[test]
    fn test_pkgconfig_generation() {
        let config = BindingConfig::default();
        let temp_dir = TempDir::new().expect("TempDir::new failed");
        let output_dir = temp_dir.path();
        let generator = BuildSystemGenerator::new(&config, output_dir);
        let pc_path = generator.generate_pkgconfig().expect("generate_pkgconfig failed");
        assert!(pc_path.exists());
        let content = std::fs::read_to_string(&pc_path).expect("read_to_string failed");
        assert!(content.contains("Libs:"));
    }
}
