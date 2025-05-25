//! Enhanced NetCDF file format example
//!
//! This example demonstrates the enhanced NetCDF capabilities:
//! - Creating NetCDF files with dimensions and variables
//! - Adding metadata and attributes
//! - Working with different data types
//! - Managing NetCDF file structure

use scirs2_io::error::Result;
use scirs2_io::netcdf::{NetCDFDataType, NetCDFFile};

fn main() -> Result<()> {
    println!("=== Enhanced NetCDF Example ===");

    // Example 1: Create a simple climate dataset structure
    create_climate_dataset_structure()?;

    // Example 2: Create a multi-dimensional scientific dataset structure
    create_scientific_dataset_structure()?;

    // Example 3: Demonstrate attributes and metadata
    demonstrate_metadata_features()?;

    println!("Enhanced NetCDF example completed successfully!");
    println!("Note: This is a structural demonstration - actual NetCDF file I/O");
    println!("would require a complete NetCDF library implementation.");
    Ok(())
}

fn create_climate_dataset_structure() -> Result<()> {
    println!("\n1. Creating climate dataset structure...");

    // Create a new NetCDF file for climate data
    let mut nc_file = NetCDFFile::create("climate_data.nc")?;

    // Define dimensions
    nc_file.create_dimension("time", Some(365))?; // 365 days
    nc_file.create_dimension("lat", Some(180))?; // 180 latitude points
    nc_file.create_dimension("lon", Some(360))?; // 360 longitude points

    // Create coordinate variables
    nc_file.create_variable("time", NetCDFDataType::Double, &["time"])?;
    nc_file.create_variable("lat", NetCDFDataType::Float, &["lat"])?;
    nc_file.create_variable("lon", NetCDFDataType::Float, &["lon"])?;

    // Create data variables
    nc_file.create_variable(
        "temperature",
        NetCDFDataType::Float,
        &["time", "lat", "lon"],
    )?;
    nc_file.create_variable(
        "precipitation",
        NetCDFDataType::Float,
        &["time", "lat", "lon"],
    )?;

    // Add metadata attributes
    nc_file.add_global_attribute("title", "Sample Climate Dataset")?;
    nc_file.add_global_attribute("institution", "SciRS2 Example")?;
    nc_file.add_global_attribute("source", "Synthetic climate model")?;
    nc_file.add_global_attribute("history", "Created with scirs2-io NetCDF example")?;

    // Add variable attributes
    nc_file.add_variable_attribute("time", "units", "days since 2023-01-01")?;
    nc_file.add_variable_attribute("time", "long_name", "time")?;

    nc_file.add_variable_attribute("lat", "units", "degrees_north")?;
    nc_file.add_variable_attribute("lat", "long_name", "latitude")?;

    nc_file.add_variable_attribute("lon", "units", "degrees_east")?;
    nc_file.add_variable_attribute("lon", "long_name", "longitude")?;

    nc_file.add_variable_attribute("temperature", "units", "degrees_C")?;
    nc_file.add_variable_attribute("temperature", "long_name", "2m temperature")?;
    nc_file.add_variable_attribute("temperature", "standard_name", "air_temperature")?;

    nc_file.add_variable_attribute("precipitation", "units", "mm/day")?;
    nc_file.add_variable_attribute("precipitation", "long_name", "precipitation rate")?;
    nc_file.add_variable_attribute("precipitation", "standard_name", "precipitation_flux")?;

    // Display the structure
    println!("  Climate dataset structure:");
    println!("    Dimensions: {:?}", nc_file.dimensions());
    println!("    Variables: {:?}", nc_file.variables());
    println!("    Global attributes: {:?}", nc_file.global_attributes());

    // Show variable information
    for var_name in nc_file.variables() {
        let (dtype, dims, attrs) = nc_file.variable_info(&var_name)?;
        println!(
            "    Variable '{}': {:?}, dimensions: {:?}",
            var_name, dtype, dims
        );
        if !attrs.is_empty() {
            println!("      Attributes: {:?}", attrs);
        }
    }

    nc_file.close()?;
    println!("  Climate dataset structure created successfully");

    Ok(())
}

fn create_scientific_dataset_structure() -> Result<()> {
    println!("\n2. Creating scientific measurement dataset structure...");

    // Create a new NetCDF file for scientific measurements
    let mut nc_file = NetCDFFile::create("scientific_measurements.nc")?;

    // Define dimensions for a time series of 2D measurements
    nc_file.create_dimension("time", Some(100))?; // 100 time steps
    nc_file.create_dimension("x", Some(50))?; // 50 x points
    nc_file.create_dimension("y", Some(30))?; // 30 y points
    nc_file.create_dimension("sensors", Some(5))?; // 5 different sensors
    nc_file.create_dimension("unlimited", None)?; // Unlimited dimension

    // Create coordinate variables
    nc_file.create_variable("time", NetCDFDataType::Double, &["time"])?;
    nc_file.create_variable("x", NetCDFDataType::Float, &["x"])?;
    nc_file.create_variable("y", NetCDFDataType::Float, &["y"])?;
    nc_file.create_variable("sensor_id", NetCDFDataType::Int, &["sensors"])?;

    // Create data variables with different data types
    nc_file.create_variable(
        "field_intensity",
        NetCDFDataType::Double,
        &["time", "x", "y"],
    )?;
    nc_file.create_variable(
        "sensor_readings",
        NetCDFDataType::Float,
        &["time", "sensors"],
    )?;
    nc_file.create_variable("quality_flags", NetCDFDataType::Byte, &["time", "x", "y"])?;
    nc_file.create_variable("experiment_notes", NetCDFDataType::Char, &["unlimited"])?;

    // Add comprehensive metadata
    nc_file.add_global_attribute("title", "Scientific Field Measurements")?;
    nc_file.add_global_attribute("institution", "SciRS2 Research Lab")?;
    nc_file.add_global_attribute("experiment", "Wave interference study")?;
    nc_file.add_global_attribute("created_by", "scirs2-io enhanced NetCDF example")?;
    nc_file.add_global_attribute("version", "1.0")?;

    // Coordinate attributes
    nc_file.add_variable_attribute("time", "units", "seconds since experiment start")?;
    nc_file.add_variable_attribute("time", "long_name", "measurement time")?;

    nc_file.add_variable_attribute("x", "units", "meters")?;
    nc_file.add_variable_attribute("x", "long_name", "x coordinate")?;
    nc_file.add_variable_attribute("x", "axis", "X")?;

    nc_file.add_variable_attribute("y", "units", "meters")?;
    nc_file.add_variable_attribute("y", "long_name", "y coordinate")?;
    nc_file.add_variable_attribute("y", "axis", "Y")?;

    nc_file.add_variable_attribute("sensor_id", "long_name", "sensor identifier")?;

    // Data variable attributes
    nc_file.add_variable_attribute("field_intensity", "units", "arbitrary_units")?;
    nc_file.add_variable_attribute(
        "field_intensity",
        "long_name",
        "electromagnetic field intensity",
    )?;
    nc_file.add_variable_attribute(
        "field_intensity",
        "description",
        "Measured field intensity showing wave interference patterns",
    )?;
    nc_file.add_variable_attribute("field_intensity", "valid_range", "-1000.0 1000.0")?;

    nc_file.add_variable_attribute("sensor_readings", "units", "volts")?;
    nc_file.add_variable_attribute("sensor_readings", "long_name", "sensor voltage readings")?;
    nc_file.add_variable_attribute("sensor_readings", "calibration", "Factory calibrated")?;
    nc_file.add_variable_attribute("sensor_readings", "accuracy", "±0.1V")?;

    nc_file.add_variable_attribute("quality_flags", "long_name", "data quality flags")?;
    nc_file.add_variable_attribute("quality_flags", "flag_values", "0, 1")?;
    nc_file.add_variable_attribute("quality_flags", "flag_meanings", "bad_data good_data")?;

    nc_file.add_variable_attribute(
        "experiment_notes",
        "long_name",
        "experimental notes and comments",
    )?;
    nc_file.add_variable_attribute(
        "experiment_notes",
        "description",
        "Free-form text describing experimental conditions",
    )?;

    // Display the structure
    println!("  Scientific dataset structure:");
    println!("    Dimensions: {:?}", nc_file.dimensions());
    println!("    Variables: {:?}", nc_file.variables());
    println!("    Global attributes: {:?}", nc_file.global_attributes());

    // Show variable information with enhanced details
    for var_name in nc_file.variables() {
        let (dtype, dims, attrs) = nc_file.variable_info(&var_name)?;
        println!("    Variable '{}': {:?}", var_name, dtype);
        println!("      Dimensions: {:?}", dims);
        if !attrs.is_empty() {
            println!("      Attributes:");
            for (attr_name, attr_value) in &attrs {
                println!("        {}: {}", attr_name, attr_value);
            }
        }
    }

    nc_file.close()?;
    println!("  Scientific dataset structure created successfully");

    Ok(())
}

fn demonstrate_metadata_features() -> Result<()> {
    println!("\n3. Demonstrating metadata and attribute features...");

    let mut nc_file = NetCDFFile::create("metadata_demo.nc")?;

    // Create a simple structure for metadata demonstration
    nc_file.create_dimension("time", Some(24))?; // 24 hours
    nc_file.create_dimension("station", Some(10))?; // 10 weather stations

    // Create variables
    nc_file.create_variable("time", NetCDFDataType::Double, &["time"])?;
    nc_file.create_variable("station_id", NetCDFDataType::Int, &["station"])?;
    nc_file.create_variable("temperature", NetCDFDataType::Float, &["time", "station"])?;
    nc_file.create_variable("humidity", NetCDFDataType::Float, &["time", "station"])?;
    nc_file.create_variable("pressure", NetCDFDataType::Double, &["time", "station"])?;

    // Comprehensive global metadata
    nc_file.add_global_attribute("title", "Weather Station Hourly Data")?;
    nc_file.add_global_attribute(
        "summary",
        "Hourly meteorological measurements from 10 weather stations",
    )?;
    nc_file.add_global_attribute("institution", "National Weather Service")?;
    nc_file.add_global_attribute("source", "Automated weather stations")?;
    nc_file.add_global_attribute("history", "Created 2024-01-01: Initial dataset")?;
    nc_file.add_global_attribute(
        "references",
        "Station network described in Smith et al. (2023)",
    )?;
    nc_file.add_global_attribute("comment", "Quality controlled and validated data")?;
    nc_file.add_global_attribute("Conventions", "CF-1.8")?;
    nc_file.add_global_attribute("creator_name", "Dr. Jane Doe")?;
    nc_file.add_global_attribute("creator_email", "jane.doe@weather.gov")?;
    nc_file.add_global_attribute("date_created", "2024-01-01T00:00:00Z")?;
    nc_file.add_global_attribute("geospatial_lat_min", "35.0")?;
    nc_file.add_global_attribute("geospatial_lat_max", "45.0")?;
    nc_file.add_global_attribute("geospatial_lon_min", "-125.0")?;
    nc_file.add_global_attribute("geospatial_lon_max", "-115.0")?;

    // Time variable metadata
    nc_file.add_variable_attribute("time", "units", "hours since 2024-01-01 00:00:00")?;
    nc_file.add_variable_attribute("time", "long_name", "time of measurement")?;
    nc_file.add_variable_attribute("time", "standard_name", "time")?;
    nc_file.add_variable_attribute("time", "axis", "T")?;
    nc_file.add_variable_attribute("time", "calendar", "gregorian")?;

    // Station variable metadata
    nc_file.add_variable_attribute("station_id", "long_name", "weather station identifier")?;
    nc_file.add_variable_attribute("station_id", "cf_role", "timeseries_id")?;

    // Temperature variable metadata
    nc_file.add_variable_attribute("temperature", "units", "degrees_C")?;
    nc_file.add_variable_attribute("temperature", "long_name", "air temperature")?;
    nc_file.add_variable_attribute("temperature", "standard_name", "air_temperature")?;
    nc_file.add_variable_attribute("temperature", "valid_min", "-50.0")?;
    nc_file.add_variable_attribute("temperature", "valid_max", "60.0")?;
    nc_file.add_variable_attribute("temperature", "missing_value", "-999.0")?;
    nc_file.add_variable_attribute(
        "temperature",
        "instrument",
        "Platinum resistance thermometer",
    )?;
    nc_file.add_variable_attribute("temperature", "accuracy", "±0.1°C")?;
    nc_file.add_variable_attribute("temperature", "resolution", "0.01°C")?;

    // Humidity variable metadata
    nc_file.add_variable_attribute("humidity", "units", "percent")?;
    nc_file.add_variable_attribute("humidity", "long_name", "relative humidity")?;
    nc_file.add_variable_attribute("humidity", "standard_name", "relative_humidity")?;
    nc_file.add_variable_attribute("humidity", "valid_min", "0.0")?;
    nc_file.add_variable_attribute("humidity", "valid_max", "100.0")?;
    nc_file.add_variable_attribute("humidity", "missing_value", "-999.0")?;
    nc_file.add_variable_attribute("humidity", "instrument", "Capacitive humidity sensor")?;

    // Pressure variable metadata
    nc_file.add_variable_attribute("pressure", "units", "hPa")?;
    nc_file.add_variable_attribute("pressure", "long_name", "atmospheric pressure")?;
    nc_file.add_variable_attribute("pressure", "standard_name", "air_pressure")?;
    nc_file.add_variable_attribute("pressure", "valid_min", "800.0")?;
    nc_file.add_variable_attribute("pressure", "valid_max", "1200.0")?;
    nc_file.add_variable_attribute("pressure", "missing_value", "-999.0")?;
    nc_file.add_variable_attribute("pressure", "instrument", "Aneroid barometer")?;
    nc_file.add_variable_attribute("pressure", "calibration_date", "2023-12-01")?;

    // Display comprehensive metadata
    println!("  Metadata demonstration:");
    println!("    File structure:");
    println!("      Dimensions: {:?}", nc_file.dimensions());
    println!("      Variables: {:?}", nc_file.variables());

    println!("    Global attributes:");
    for (attr_name, attr_value) in nc_file.global_attributes() {
        println!("      {}: {}", attr_name, attr_value);
    }

    println!("    Variable metadata:");
    for var_name in nc_file.variables() {
        let (dtype, dims, attrs) = nc_file.variable_info(&var_name)?;
        println!("      Variable '{}' ({:?}):", var_name, dtype);
        println!("        Dimensions: {:?}", dims);
        if !attrs.is_empty() {
            println!("        Attributes:");
            for (attr_name, attr_value) in &attrs {
                println!("          {}: {}", attr_name, attr_value);
            }
        }
        println!();
    }

    nc_file.close()?;
    println!("  Metadata demonstration completed successfully");

    Ok(())
}
