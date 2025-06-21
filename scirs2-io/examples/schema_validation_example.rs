//! Schema-based validation example
//!
//! This example demonstrates the new schema validation capabilities
//! for validating structured data against defined schemas.

use scirs2_io::validation::{schema_helpers, SchemaConstraint, SchemaValidator};
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Schema-based Validation Example");
    println!("==================================");

    // Demonstrate basic schema validation
    demonstrate_basic_schema_validation()?;

    // Demonstrate complex nested schemas
    demonstrate_nested_schema_validation()?;

    // Demonstrate custom validators
    demonstrate_custom_validators()?;

    // Demonstrate format validators
    demonstrate_format_validators()?;

    // Demonstrate file validation
    demonstrate_file_validation()?;

    // Demonstrate JSON Schema compatibility
    demonstrate_json_schema_compatibility()?;

    println!("\n✅ All schema validation demonstrations completed successfully!");
    println!("💡 Schema validation ensures data integrity and structure compliance");

    Ok(())
}

fn demonstrate_basic_schema_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Demonstrating Basic Schema Validation...");

    let validator = SchemaValidator::new();

    // Create a simple user schema
    let user_schema = {
        let mut properties = HashMap::new();

        properties.insert(
            "name".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::MinLength(1))
                .with_constraint(SchemaConstraint::MaxLength(100))
                .required(),
        );

        properties.insert(
            "age".to_string(),
            schema_helpers::integer()
                .with_constraint(SchemaConstraint::MinValue(0.0))
                .with_constraint(SchemaConstraint::MaxValue(150.0))
                .required(),
        );

        properties.insert("email".to_string(), schema_helpers::email().required());

        properties.insert(
            "score".to_string(),
            schema_helpers::number()
                .with_constraint(SchemaConstraint::MinValue(0.0))
                .with_constraint(SchemaConstraint::MaxValue(100.0))
                .optional(),
        );

        schema_helpers::object(properties)
    };

    // Test valid data
    println!("  ✅ Testing valid user data...");
    let valid_user = json!({
        "name": "Alice Johnson",
        "age": 30,
        "email": "alice@example.com",
        "score": 95.5
    });

    let result = validator.validate(&valid_user, &user_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Fields validated: {}", result.fields_validated);
    println!("     Validation time: {:.2}ms", result.validation_time_ms);

    // Test invalid data
    println!("  ❌ Testing invalid user data...");
    let invalid_user = json!({
        "name": "", // Too short
        "age": -5,  // Negative age
        "email": "not-an-email", // Invalid email format
        "score": 150.0 // Score too high
    });

    let result = validator.validate(&invalid_user, &user_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Number of errors: {}", result.errors.len());

    for (i, error) in result.errors.iter().enumerate() {
        println!("     Error {}: {} - {}", i + 1, error.path, error.message);
    }

    // Test missing required fields
    println!("  ⚠️  Testing data with missing required fields...");
    let incomplete_user = json!({
        "name": "Bob Smith"
        // Missing age and email
    });

    let result = validator.validate(&incomplete_user, &user_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Number of errors: {}", result.errors.len());

    Ok(())
}

fn demonstrate_nested_schema_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏗️  Demonstrating Nested Schema Validation...");

    let validator = SchemaValidator::new();

    // Create a complex nested schema for a scientific dataset
    let dataset_schema = {
        let mut properties = HashMap::new();

        // Metadata object
        let mut metadata_props = HashMap::new();
        metadata_props.insert("title".to_string(), schema_helpers::string().required());
        metadata_props.insert(
            "description".to_string(),
            schema_helpers::string().optional(),
        );
        metadata_props.insert(
            "created_date".to_string(),
            schema_helpers::date().required(),
        );
        metadata_props.insert(
            "version".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::Pattern(r"^\d+\.\d+\.\d+$".to_string()))
                .required(),
        );

        properties.insert(
            "metadata".to_string(),
            schema_helpers::object(metadata_props).required(),
        );

        // Array of measurements
        let measurement_schema = {
            let mut measurement_props = HashMap::new();
            measurement_props.insert(
                "timestamp".to_string(),
                schema_helpers::string()
                    .with_constraint(SchemaConstraint::Format("date-time".to_string()))
                    .required(),
            );
            measurement_props.insert("value".to_string(), schema_helpers::number().required());
            measurement_props.insert(
                "unit".to_string(),
                schema_helpers::string()
                    .with_constraint(SchemaConstraint::Enum(vec![
                        json!("celsius"),
                        json!("fahrenheit"),
                        json!("kelvin"),
                    ]))
                    .required(),
            );
            measurement_props.insert(
                "quality".to_string(),
                schema_helpers::string()
                    .with_constraint(SchemaConstraint::Enum(vec![
                        json!("good"),
                        json!("acceptable"),
                        json!("poor"),
                    ]))
                    .optional(),
            );

            schema_helpers::object(measurement_props)
        };

        properties.insert(
            "measurements".to_string(),
            schema_helpers::array(measurement_schema)
                .with_constraint(SchemaConstraint::MinLength(1))
                .with_constraint(SchemaConstraint::MaxLength(10000))
                .required(),
        );

        // Statistics object
        let mut stats_props = HashMap::new();
        stats_props.insert(
            "count".to_string(),
            schema_helpers::positive_integer().required(),
        );
        stats_props.insert("mean".to_string(), schema_helpers::number().required());
        stats_props.insert(
            "std_dev".to_string(),
            schema_helpers::non_negative_number().required(),
        );

        properties.insert(
            "statistics".to_string(),
            schema_helpers::object(stats_props).optional(),
        );

        schema_helpers::object(properties)
    };

    // Test valid nested data
    println!("  ✅ Testing valid nested dataset...");
    let valid_dataset = json!({
        "metadata": {
            "title": "Temperature Measurements",
            "description": "Daily temperature readings from weather station",
            "created_date": "2024-01-15",
            "version": "1.2.3"
        },
        "measurements": [
            {
                "timestamp": "2024-01-15T08:00:00Z",
                "value": 20.5,
                "unit": "celsius",
                "quality": "good"
            },
            {
                "timestamp": "2024-01-15T12:00:00Z",
                "value": 25.2,
                "unit": "celsius",
                "quality": "good"
            },
            {
                "timestamp": "2024-01-15T16:00:00Z",
                "value": 22.8,
                "unit": "celsius"
            }
        ],
        "statistics": {
            "count": 3,
            "mean": 22.83,
            "std_dev": 2.35
        }
    });

    let result = validator.validate(&valid_dataset, &dataset_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Fields validated: {}", result.fields_validated);
    println!("     Validation time: {:.2}ms", result.validation_time_ms);

    // Test invalid nested data
    println!("  ❌ Testing invalid nested dataset...");
    let invalid_dataset = json!({
        "metadata": {
            "title": "Temperature Measurements",
            "created_date": "not-a-date", // Invalid date format
            "version": "invalid-version"  // Invalid version pattern
            // Missing required description field (actually description is optional)
        },
        "measurements": [
            {
                "timestamp": "not-a-timestamp", // Invalid timestamp format
                "value": "not-a-number",        // String instead of number
                "unit": "invalid-unit",         // Not in enum
                "quality": "excellent"          // Not in enum
            }
        ],
        "statistics": {
            "count": -1,    // Negative count
            "mean": "NaN",  // String instead of number
            "std_dev": -5.0 // Negative standard deviation
        }
    });

    let result = validator.validate(&invalid_dataset, &dataset_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Number of errors: {}", result.errors.len());

    // Show first few errors
    for (i, error) in result.errors.iter().take(5).enumerate() {
        println!("     Error {}: {} - {}", i + 1, error.path, error.message);
    }

    if result.errors.len() > 5 {
        println!("     ... and {} more errors", result.errors.len() - 5);
    }

    Ok(())
}

fn demonstrate_custom_validators() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Demonstrating Custom Validators...");

    let mut validator = SchemaValidator::new();

    // Add custom validators
    validator.add_custom_validator("is_even", |value| {
        if let Some(num) = value.as_i64() {
            num % 2 == 0
        } else {
            false
        }
    });

    validator.add_custom_validator("is_positive_and_prime", |value| {
        if let Some(num) = value.as_i64() {
            if num <= 1 {
                return false;
            }
            for i in 2..=((num as f64).sqrt() as i64) {
                if num % i == 0 {
                    return false;
                }
            }
            true
        } else {
            false
        }
    });

    // Create schema with custom validators
    let custom_schema = {
        let mut properties = HashMap::new();

        properties.insert(
            "even_number".to_string(),
            schema_helpers::integer()
                .with_constraint(SchemaConstraint::Custom("is_even".to_string()))
                .required(),
        );

        properties.insert(
            "prime_number".to_string(),
            schema_helpers::integer()
                .with_constraint(SchemaConstraint::Custom(
                    "is_positive_and_prime".to_string(),
                ))
                .required(),
        );

        schema_helpers::object(properties)
    };

    // Test valid data with custom validators
    println!("  ✅ Testing valid data with custom validators...");
    let valid_data = json!({
        "even_number": 42,
        "prime_number": 17
    });

    let result = validator.validate(&valid_data, &custom_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );

    // Test invalid data with custom validators
    println!("  ❌ Testing invalid data with custom validators...");
    let invalid_data = json!({
        "even_number": 43,  // Odd number
        "prime_number": 15  // Not prime (3 * 5)
    });

    let result = validator.validate(&invalid_data, &custom_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    for error in &result.errors {
        println!("     Error: {} - {}", error.path, error.message);
    }

    Ok(())
}

fn demonstrate_format_validators() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📧 Demonstrating Format Validators...");

    let mut validator = SchemaValidator::new();

    // Add a custom phone number format validator
    validator.add_format_validator("phone", |s| {
        let phone_regex = regex::Regex::new(r"^\+?[\d\s\-\(\)]{10,15}$").unwrap();
        phone_regex.is_match(s)
    });

    // Create schema with various format validators
    let contact_schema = {
        let mut properties = HashMap::new();

        properties.insert("email".to_string(), schema_helpers::email().required());

        properties.insert(
            "website".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::Format("uri".to_string()))
                .optional(),
        );

        properties.insert(
            "registration_date".to_string(),
            schema_helpers::date().required(),
        );

        properties.insert(
            "last_login".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::Format("date-time".to_string()))
                .optional(),
        );

        properties.insert("user_id".to_string(), schema_helpers::uuid().required());

        properties.insert(
            "ip_address".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::Format("ipv4".to_string()))
                .optional(),
        );

        properties.insert(
            "phone".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::Format("phone".to_string()))
                .optional(),
        );

        schema_helpers::object(properties)
    };

    // Test valid format data
    println!("  ✅ Testing valid format data...");
    let valid_contact = json!({
        "email": "user@example.com",
        "website": "https://example.com",
        "registration_date": "2024-01-15",
        "last_login": "2024-01-15T14:30:00Z",
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "ip_address": "192.168.1.1",
        "phone": "+1 (555) 123-4567"
    });

    let result = validator.validate(&valid_contact, &contact_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );

    // Test invalid format data
    println!("  ❌ Testing invalid format data...");
    let invalid_contact = json!({
        "email": "not-an-email",
        "website": "not-a-url",
        "registration_date": "2024-13-45", // Invalid date
        "last_login": "not-a-datetime",
        "user_id": "not-a-uuid",
        "ip_address": "999.999.999.999", // Invalid IP
        "phone": "abc"  // Invalid phone
    });

    let result = validator.validate(&invalid_contact, &contact_schema);
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Number of format errors: {}", result.errors.len());

    for error in &result.errors {
        println!("     Format error: {} - {}", error.path, error.message);
    }

    Ok(())
}

fn demonstrate_file_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📁 Demonstrating File Validation...");

    let temp_dir = tempdir()?;
    let validator = SchemaValidator::new();

    // Create a schema for a configuration file
    let config_schema = {
        let mut properties = HashMap::new();

        properties.insert(
            "app_name".to_string(),
            schema_helpers::string()
                .with_constraint(SchemaConstraint::MinLength(1))
                .required(),
        );

        properties.insert(
            "port".to_string(),
            schema_helpers::integer()
                .with_constraint(SchemaConstraint::MinValue(1.0))
                .with_constraint(SchemaConstraint::MaxValue(65535.0))
                .required(),
        );

        properties.insert("debug".to_string(), schema_helpers::boolean().required());

        let mut db_props = HashMap::new();
        db_props.insert("host".to_string(), schema_helpers::string().required());
        db_props.insert(
            "port".to_string(),
            schema_helpers::positive_integer().required(),
        );
        db_props.insert("database".to_string(), schema_helpers::string().required());

        properties.insert(
            "database".to_string(),
            schema_helpers::object(db_props).required(),
        );

        schema_helpers::object(properties)
    };

    // Write valid config file
    let valid_config_path = temp_dir.path().join("valid_config.json");
    let valid_config_content = json!({
        "app_name": "MyApp",
        "port": 8080,
        "debug": true,
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "myapp_db"
        }
    });

    std::fs::write(
        &valid_config_path,
        serde_json::to_string_pretty(&valid_config_content)?,
    )?;

    println!("  ✅ Validating valid configuration file...");
    let result = validator.validate_file(&valid_config_path, &config_schema)?;
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Fields validated: {}", result.fields_validated);

    // Write invalid config file
    let invalid_config_path = temp_dir.path().join("invalid_config.json");
    let invalid_config_content = json!({
        "app_name": "",  // Empty string
        "port": 70000,   // Port too high
        "debug": "yes",  // String instead of boolean
        "database": {
            "host": "localhost",
            "port": -1   // Negative port
            // Missing database field
        }
    });

    std::fs::write(
        &invalid_config_path,
        serde_json::to_string_pretty(&invalid_config_content)?,
    )?;

    println!("  ❌ Validating invalid configuration file...");
    let result = validator.validate_file(&invalid_config_path, &config_schema)?;
    println!(
        "     Validation result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Number of errors: {}", result.errors.len());

    for error in &result.errors {
        println!("     Error: {} - {}", error.path, error.message);
    }

    Ok(())
}

fn demonstrate_json_schema_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating JSON Schema Compatibility...");

    let validator = SchemaValidator::new();

    // Define a JSON Schema
    let json_schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 50
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 120
            },
            "email": {
                "type": "string",
                "format": "email"
            },
            "scores": {
                "type": "array",
                "items": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100
                }
            }
        }
    });

    // Convert JSON Schema to our internal format
    let schema = scirs2_io::validation::schema_from_json_schema(&json_schema)?;

    println!("  🔄 Testing JSON Schema compatibility...");

    // Test valid data
    let valid_data = json!({
        "name": "Jane Doe",
        "age": 28,
        "email": "jane@example.com",
        "scores": [85.5, 92.0, 78.5, 95.0]
    });

    let result = validator.validate(&valid_data, &schema);
    println!(
        "     Valid data result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );

    // Test invalid data
    let invalid_data = json!({
        "name": "",  // Too short
        "age": 150,  // Too old
        "email": "not-an-email",
        "scores": [85.5, 105.0, -10.0]  // Scores out of range
    });

    let result = validator.validate(&invalid_data, &schema);
    println!(
        "     Invalid data result: {}",
        if result.valid { "PASSED" } else { "FAILED" }
    );
    println!("     Errors found: {}", result.errors.len());

    println!("  ✅ JSON Schema conversion and validation successful!");

    Ok(())
}
