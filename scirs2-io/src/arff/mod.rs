//! ARFF (Attribute-Relation File Format) handling module
//!
//! This module provides functionality for reading and writing ARFF files,
//! commonly used in machine learning applications like WEKA.

use scirs2_core::ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// ARFF attribute types
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeType {
    /// Numeric attribute (real or integer)
    Numeric,
    /// String attribute
    String,
    /// Date attribute
    Date(String), // with optional date format
    /// Nominal attribute with possible values
    Nominal(Vec<String>),
}

/// ARFF dataset representation
#[derive(Debug, Clone)]
pub struct ArffData {
    /// Name of the relation
    pub relation: String,
    /// Attributes with their names and types
    pub attributes: Vec<(String, AttributeType)>,
    /// Data as a 2D array where rows are instances and columns are attributes
    pub data: Array2<ArffValue>,
}

/// ARFF data value
#[derive(Debug, Clone, PartialEq)]
pub enum ArffValue {
    /// Numeric value
    Numeric(f64),
    /// String value
    String(String),
    /// Date value as string
    Date(String),
    /// Nominal value
    Nominal(String),
    /// Missing value
    Missing,
}

impl ArffValue {
    /// Try to convert the value to a float if possible
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            ArffValue::Numeric(val) => Some(*val),
            _ => None,
        }
    }

    /// Try to convert the value to a string
    pub fn as_string(&self) -> String {
        match self {
            ArffValue::Numeric(val) => val.to_string(),
            ArffValue::String(val) => val.clone(),
            ArffValue::Date(val) => val.clone(),
            ArffValue::Nominal(val) => val.clone(),
            ArffValue::Missing => "?".to_string(),
        }
    }
}

/// Parse attribute definition from ARFF file
#[allow(dead_code)]
fn parse_attribute(line: &str) -> Result<(String, AttributeType)> {
    // Expected format: @attribute name type
    let _line = line.trim();
    if !_line.starts_with("@attribute") {
        return Err(IoError::FormatError("Invalid attribute format".to_string()));
    }

    // Split into parts: @attribute, name, type
    let parts: Vec<&str> = line.splitn(3, ' ').collect();
    if parts.len() < 3 {
        return Err(IoError::FormatError("Invalid attribute format".to_string()));
    }

    let name = parts[1].trim().to_string();
    let type_str = parts[2].trim();

    // Parse attribute type
    let attr_type = if type_str.eq_ignore_ascii_case("numeric")
        || type_str.eq_ignore_ascii_case("real")
        || type_str.eq_ignore_ascii_case("integer")
    {
        AttributeType::Numeric
    } else if type_str.eq_ignore_ascii_case("string") {
        AttributeType::String
    } else if type_str.starts_with("date") {
        let format = if type_str.len() > 4 && type_str.contains(' ') {
            let format_str = type_str.split_once(' ').map(|x| x.1).unwrap_or("").trim();
            // Remove quotes if present
            if format_str.starts_with('"') && format_str.ends_with('"') {
                format_str[1..format_str.len() - 1].to_string()
            } else {
                format_str.to_string()
            }
        } else {
            // Default format
            "yyyy-MM-dd'T'HH:mm:ss".to_string()
        };
        AttributeType::Date(format)
    } else if type_str.starts_with('{') && type_str.ends_with('}') {
        // Nominal attribute with values
        let values_str = &type_str[1..type_str.len() - 1];
        let values: Vec<String> = values_str
            .split(',')
            .map(|s| {
                let s = s.trim();
                // Remove quotes if present
                if s.starts_with('"') && s.ends_with('"') {
                    s[1..s.len() - 1].to_string()
                } else {
                    s.to_string()
                }
            })
            .collect();
        AttributeType::Nominal(values)
    } else {
        return Err(IoError::FormatError(format!(
            "Unknown attribute type: {type_str}"
        )));
    };

    Ok((name, attr_type))
}

/// Parse an ARFF data line into ArffValue instances
#[allow(dead_code)]
fn parse_data_line(line: &str, attributes: &[(String, AttributeType)]) -> Result<Vec<ArffValue>> {
    let _line = line.trim();
    if line.is_empty() {
        return Err(IoError::FormatError("Empty data _line".to_string()));
    }

    let mut values = Vec::new();
    let parts: Vec<&str> = line.split(',').collect();

    if parts.len() != attributes.len() {
        return Err(IoError::FormatError(format!(
            "Data _line has {} values but expected {}",
            parts.len(),
            attributes.len()
        )));
    }

    for (i, part) in parts.iter().enumerate() {
        let part = part.trim();

        // Handle missing values
        if part == "?" {
            values.push(ArffValue::Missing);
            continue;
        }

        let attr_type = &attributes[i].1;
        let value = match attr_type {
            AttributeType::Numeric => {
                let num = part
                    .parse::<f64>()
                    .map_err(|_| IoError::FormatError(format!("Invalid numeric value: {part}")))?;
                ArffValue::Numeric(num)
            }
            AttributeType::String => {
                // Remove quotes if present
                let s = if part.starts_with('"') && part.ends_with('"') {
                    part[1..part.len() - 1].to_string()
                } else {
                    part.to_string()
                };
                ArffValue::String(s)
            }
            AttributeType::Date(_) => {
                // Remove quotes if present
                let s = if part.starts_with('"') && part.ends_with('"') {
                    part[1..part.len() - 1].to_string()
                } else {
                    part.to_string()
                };
                ArffValue::Date(s)
            }
            AttributeType::Nominal(allowed_values) => {
                // Remove quotes if present
                let s = if part.starts_with('"') && part.ends_with('"') {
                    part[1..part.len() - 1].to_string()
                } else {
                    part.to_string()
                };

                // Check if value is in allowed values
                if !allowed_values.contains(&s) {
                    return Err(IoError::FormatError(format!(
                        "Invalid nominal value: {s}, expected one of {allowed_values:?}"
                    )));
                }

                ArffValue::Nominal(s)
            }
        };

        values.push(value);
    }

    Ok(values)
}

/// Reads an ARFF file
///
/// # Arguments
///
/// * `path` - Path to the ARFF file
///
/// # Returns
///
/// * ARFF data structure containing the relation, attributes, and data
///
/// # Example
///
/// ```no_run
/// use scirs2_io::arff::read_arff;
/// use std::path::Path;
///
/// let arff_data = read_arff(Path::new("dataset.arff")).unwrap();
/// println!("Relation: {}", arff_data.relation);
/// println!("Number of attributes: {}", arff_data.attributes.len());
/// println!("Number of instances: {}", arff_data.data.shape()[0]);
/// ```
#[allow(dead_code)]
pub fn read_arff<P: AsRef<Path>>(path: P) -> Result<ArffData> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut relation = String::new();
    let mut attributes = Vec::new();
    let mut data_lines = Vec::new();
    let mut in_data_section = false;

    // Parse ARFF file
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .map_err(|e| IoError::FileError(format!("Error reading line {}: {e}", line_num + 1)))?;

        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            // Skip empty lines and comments
            continue;
        }

        if in_data_section {
            // We're in the data section, collect data lines
            data_lines.push(line.to_string());
        } else {
            // We're in the header section
            if line.starts_with("@relation") {
                // Parse relation name
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    return Err(IoError::FormatError("Invalid relation format".to_string()));
                }

                let mut rel_name = parts[1].trim().to_string();
                // Remove quotes if present
                if rel_name.starts_with('"') && rel_name.ends_with('"') {
                    rel_name = rel_name[1..rel_name.len() - 1].to_string();
                }
                relation = rel_name;
            } else if line.starts_with("@attribute") {
                // Parse attribute
                let (name, attr_type) = parse_attribute(line)?;
                attributes.push((name, attr_type));
            } else if line.starts_with("@data") {
                // Start of data section
                in_data_section = true;
            } else {
                return Err(IoError::FormatError(format!(
                    "Unexpected line in header section: {line}"
                )));
            }
        }
    }

    if !in_data_section {
        return Err(IoError::FormatError("No @data section found".to_string()));
    }

    if attributes.is_empty() {
        return Err(IoError::FormatError("No attributes defined".to_string()));
    }

    // Parse data lines
    let mut data_values = Vec::new();
    for (i, line) in data_lines.iter().enumerate() {
        let values = parse_data_line(line, &attributes)
            .map_err(|e| IoError::FormatError(format!("Error parsing data line {}: {e}", i + 1)))?;

        data_values.push(values);
    }

    // Create data array
    let num_instances = data_values.len();
    let num_attributes = attributes.len();

    // Initialize data array with missing values
    let mut data = Array2::from_elem((num_instances, num_attributes), ArffValue::Missing);

    // Fill in data array
    for (i, row) in data_values.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            data[[i, j]] = value.clone();
        }
    }

    Ok(ArffData {
        relation,
        attributes,
        data,
    })
}

/// Extracts a numeric matrix from ARFF data
///
/// # Arguments
///
/// * `arff_data` - ARFF data structure
/// * `numeric_attributes` - List of attribute names to include in the output (must be numeric)
///
/// # Returns
///
/// * A 2D array containing the numeric data, with rows as instances and columns as the specified attributes
/// * Missing values are replaced with NaN
///
/// # Example
///
/// ```no_run
/// use scirs2_io::arff::{read_arff, get_numeric_matrix};
/// use std::path::Path;
///
/// let arff_data = read_arff(Path::new("iris.arff")).unwrap();
/// // Get only the first 4 numeric attributes from the Iris dataset
/// let numeric_attrs = arff_data.attributes.iter()
///     .take(4)
///     .map(|(name, _)| name.clone())
///     .collect::<Vec<_>>();
///
/// let matrix = get_numeric_matrix(&arff_data, &numeric_attrs).unwrap();
/// println!("Matrix shape: {:?}", matrix.shape());
/// ```
#[allow(dead_code)]
pub fn get_numeric_matrix(
    arff_data: &ArffData,
    numeric_attributes: &[String],
) -> Result<Array2<f64>> {
    // Find indices of requested _attributes
    let mut indices = Vec::new();
    let mut attr_names = Vec::new();

    for attr_name in numeric_attributes {
        let mut found = false;
        for (i, (name, attr_type)) in arff_data.attributes.iter().enumerate() {
            if name == attr_name {
                match attr_type {
                    AttributeType::Numeric => {
                        indices.push(i);
                        attr_names.push(name.clone());
                        found = true;
                        break;
                    }
                    _ => {
                        return Err(IoError::FormatError(format!(
                            "Attribute '{name}' is not numeric"
                        )));
                    }
                }
            }
        }

        if !found {
            return Err(IoError::FormatError(format!(
                "Attribute '{attr_name}' not found"
            )));
        }
    }

    // Create output matrix
    let num_instances = arff_data.data.shape()[0];
    let num_selected = indices.len();
    let mut output = Array2::from_elem((num_instances, num_selected), f64::NAN);

    // Fill output matrix
    for (out_col, &in_col) in indices.iter().enumerate() {
        for row in 0..num_instances {
            match &arff_data.data[[row, in_col]] {
                ArffValue::Numeric(val) => {
                    output[[row, out_col]] = *val;
                }
                ArffValue::Missing => {
                    // Leave as NaN
                }
                _ => {
                    // This shouldn't happen if the attribute types were checked correctly
                    return Err(IoError::FormatError(format!(
                        "Non-numeric value found in numeric attribute '{}' at row {}",
                        attr_names[out_col], row
                    )));
                }
            }
        }
    }

    Ok(output)
}

/// Writes data to an ARFF file
///
/// # Arguments
///
/// * `path` - Path where the ARFF file should be written
/// * `arff_data` - ARFF data structure to write
///
/// # Example
///
/// ```no_run
/// use scirs2_io::arff::{ArffData, ArffValue, AttributeType, write_arff};
/// use scirs2_core::ndarray::Array2;
/// use std::path::Path;
///
/// // Create a simple ARFF dataset
/// let mut arff_data = ArffData {
///     relation: "weather".to_string(),
///     attributes: vec![
///         ("outlook".to_string(), AttributeType::Nominal(vec![
///             "sunny".to_string(), "overcast".to_string(), "rainy".to_string()
///         ])),
///         ("temperature".to_string(), AttributeType::Numeric),
///     ],
///     data: Array2::from_shape_vec((2, 2), vec![
///         ArffValue::Nominal("sunny".to_string()), ArffValue::Numeric(85.0),
///         ArffValue::Nominal("overcast".to_string()), ArffValue::Numeric(72.0),
///     ]).unwrap(),
/// };
///
/// write_arff(Path::new("weather.arff"), &arff_data).unwrap();
/// ```
#[allow(dead_code)]
pub fn write_arff<P: AsRef<Path>>(path: P, arff_data: &ArffData) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write relation
    writeln!(
        writer,
        "@relation {}",
        format_arff_string(&arff_data.relation)
    )
    .map_err(|e| IoError::FileError(format!("Failed to write relation: {e}")))?;

    // Add an empty line
    writeln!(writer).map_err(|e| IoError::FileError(format!("Failed to write newline: {e}")))?;

    // Write attributes
    for (name, attr_type) in &arff_data.attributes {
        let type_str = match attr_type {
            AttributeType::Numeric => "numeric".to_string(),
            AttributeType::String => "string".to_string(),
            AttributeType::Date(format) => {
                if format.is_empty() {
                    "date".to_string()
                } else {
                    format!("date {}", format_arff_string(format))
                }
            }
            AttributeType::Nominal(values) => {
                let values_str: Vec<String> =
                    values.iter().map(|v| format_arff_string(v)).collect();
                format!("{{{}}}", values_str.join(", "))
            }
        };

        writeln!(
            writer,
            "@attribute {} {}",
            format_arff_string(name),
            type_str
        )
        .map_err(|e| IoError::FileError(format!("Failed to write attribute: {e}")))?;
    }

    // Write data section header
    writeln!(writer, "\n@data")
        .map_err(|e| IoError::FileError(format!("Failed to write data header: {e}")))?;

    // Write data lines
    let shape = arff_data.data.shape();
    let num_instances = shape[0];
    let num_attributes = shape[1];

    for i in 0..num_instances {
        let mut line = String::new();

        for j in 0..num_attributes {
            let value = &arff_data.data[[i, j]];
            let attr_type = &arff_data.attributes[j].1;

            let value_str = match (value, attr_type) {
                (ArffValue::Missing, _) => "?".to_string(),
                (ArffValue::Numeric(val), _) => val.to_string(),
                (ArffValue::String(val), _) => format_arff_string(val),
                (ArffValue::Date(val), _) => format_arff_string(val),
                (ArffValue::Nominal(val), _) => format_arff_string(val),
            };

            if j > 0 {
                line.push(',');
            }
            line.push_str(&value_str);
        }

        writeln!(writer, "{line}")
            .map_err(|e| IoError::FileError(format!("Failed to write data line: {e}")))?;
    }

    Ok(())
}

/// Creates an ARFF data structure from a numeric matrix
///
/// # Arguments
///
/// * `relation` - Name of the relation
/// * `attribute_names` - Names of the attributes (columns) in the matrix
/// * `data` - Numeric data matrix, where rows are instances and columns are attributes
///
/// # Returns
///
/// * ARFF data structure containing the relation, attributes, and data
///
/// # Example
///
/// ```no_run
/// use scirs2_io::arff::numeric_matrix_to_arff;
/// use scirs2_core::ndarray::Array2;
/// use std::path::Path;
///
/// // Create a simple matrix
/// let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let attribute_names = vec!["attr1".to_string(), "attr2".to_string()];
///
/// let arff_data = numeric_matrix_to_arff("simple_data", &attribute_names, &matrix);
///
/// // Now you can write it to a file
/// use scirs2_io::arff::write_arff;
/// write_arff(Path::new("simple_data.arff"), &arff_data).unwrap();
/// ```
#[allow(dead_code)]
pub fn numeric_matrix_to_arff(
    relation: &str,
    attribute_names: &[String],
    data: &Array2<f64>,
) -> ArffData {
    let shape = data.shape();
    let num_instances = shape[0];
    let num_attributes = shape[1];

    // Create attributes
    let mut attributes = Vec::with_capacity(num_attributes);
    for name in attribute_names {
        attributes.push((name.clone(), AttributeType::Numeric));
    }

    // Create data array
    let mut arff_data = Array2::from_elem((num_instances, num_attributes), ArffValue::Missing);

    // Fill data array
    for i in 0..num_instances {
        for j in 0..num_attributes {
            let val = data[[i, j]];
            arff_data[[i, j]] = if val.is_nan() {
                ArffValue::Missing
            } else {
                ArffValue::Numeric(val)
            };
        }
    }

    ArffData {
        relation: relation.to_string(),
        attributes,
        data: arff_data,
    }
}

/// Format a string for ARFF output, adding quotes if needed
#[allow(dead_code)]
fn format_arff_string(s: &str) -> String {
    if s.contains(' ')
        || s.contains(',')
        || s.contains('\'')
        || s.contains('"')
        || s.contains('{')
        || s.contains('}')
    {
        format!("\"{}\"", s.replace("\"", "\\\""))
    } else {
        s.to_string()
    }
}
