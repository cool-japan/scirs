//! NetCDF4/HDF5 integration example
//!
//! This example demonstrates the enhanced NetCDF4 format support with HDF5 backend,
//! showcasing improved performance, compression, and larger dataset handling.

use ndarray::{Array2, Array3};
use scirs2_io::netcdf::{
    create_netcdf4_with_data, read_netcdf, NetCDFFile, NetCDFFormat, NetCDFOptions,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 NetCDF4/HDF5 Integration Example");
    println!("===================================");

    // Demonstrate NetCDF4 vs Classic format
    demonstrate_format_comparison()?;

    // Demonstrate convenient NetCDF4 data creation
    demonstrate_netcdf4_convenience()?;

    // Demonstrate large dataset handling with NetCDF4
    demonstrate_large_dataset_support()?;

    // Demonstrate reading NetCDF4 files
    demonstrate_netcdf4_reading()?;

    // Demonstrate NetCDF4 advanced features
    demonstrate_netcdf4_advanced_features()?;

    println!("\n✅ All NetCDF4/HDF5 integration demonstrations completed successfully!");
    println!("💡 NetCDF4 format provides enhanced features through HDF5 backend:");
    println!("   - Larger dataset support");
    println!("   - Better compression");
    println!("   - Hierarchical data organization");
    println!("   - Improved performance for large files");

    Ok(())
}

fn demonstrate_format_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demonstrating NetCDF Format Comparison...");

    // Create NetCDF3 Classic format file
    println!("  📄 Creating NetCDF3 Classic format file...");
    let mut classic_file =
        NetCDFFile::create_with_format("classic_example.nc", NetCDFFormat::Classic)?;

    // Add dimensions and variables using traditional methods
    classic_file.create_dimension("time", Some(10))?;
    classic_file.create_dimension("lat", Some(180))?;
    classic_file.create_dimension("lon", Some(360))?;

    classic_file.create_variable(
        "temperature",
        scirs2_io::netcdf::NetCDFDataType::Float,
        &["time", "lat", "lon"],
    )?;
    classic_file.add_global_attribute("title", "Classic NetCDF3 Example")?;
    classic_file.add_global_attribute("format", "NetCDF3 Classic")?;

    println!("     Format: {:?}", classic_file.format());
    println!("     HDF5 Backend: {}", classic_file.has_hdf5_backend());
    classic_file.close()?;

    // Create NetCDF4 format file
    println!("  🔬 Creating NetCDF4 format file...");
    let mut netcdf4_file =
        NetCDFFile::create_with_format("netcdf4_example.nc", NetCDFFormat::NetCDF4)?;

    // Use convenient array-based interface (NetCDF4 only)
    let temperature_data = Array3::<f64>::zeros((10, 180, 360));
    netcdf4_file.write_array("temperature", &temperature_data, &["time", "lat", "lon"])?;

    netcdf4_file.add_global_attribute("title", "NetCDF4/HDF5 Example")?;
    netcdf4_file.add_global_attribute("format", "NetCDF4 with HDF5 backend")?;

    println!("     Format: {:?}", netcdf4_file.format());
    println!("     HDF5 Backend: {}", netcdf4_file.has_hdf5_backend());
    netcdf4_file.close()?;

    println!("  ✅ Format comparison completed!");

    Ok(())
}

fn demonstrate_netcdf4_convenience() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Demonstrating NetCDF4 Convenience Functions...");

    // Create multiple datasets with metadata
    let mut datasets = HashMap::new();

    // Temperature data (time x location)
    let temperature = Array2::from_shape_fn((24, 10), |(t, l)| {
        20.0 + 5.0 * (t as f64 * 0.26).sin() + (l as f64 * 0.1).cos()
    });
    datasets.insert(
        "temperature".to_string(),
        (
            temperature.into_dyn(),
            vec!["time".to_string(), "location".to_string()],
        ),
    );

    // Pressure data (time only)
    let pressure = Array2::from_shape_fn((24, 1), |(t, _)| 1013.25 - 2.0 * (t as f64 * 0.26).cos());
    datasets.insert(
        "pressure".to_string(),
        (
            pressure.into_dyn(),
            vec!["time".to_string(), "pressure_level".to_string()],
        ),
    );

    // Wind speed data (time x location x altitude)
    let wind_speed = Array3::from_shape_fn((24, 10, 5), |(t, l, a)| {
        10.0 + 3.0 * (t as f64 * 0.26).sin() + (l as f64 * 0.2).cos() + (a as f64 * 0.5)
    });
    datasets.insert(
        "wind_speed".to_string(),
        (
            wind_speed.into_dyn(),
            vec![
                "time".to_string(),
                "location".to_string(),
                "altitude".to_string(),
            ],
        ),
    );

    // Global attributes
    let mut global_attrs = HashMap::new();
    global_attrs.insert("title".to_string(), "Weather Station Data".to_string());
    global_attrs.insert(
        "institution".to_string(),
        "SciRS2 Weather Network".to_string(),
    );
    global_attrs.insert(
        "source".to_string(),
        "Automated weather stations".to_string(),
    );
    global_attrs.insert("conventions".to_string(), "CF-1.8".to_string());
    global_attrs.insert(
        "history".to_string(),
        format!(
            "Created {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ),
    );

    println!("  📊 Creating comprehensive weather dataset...");
    println!("     Variables: {}", datasets.len());
    println!("     Global attributes: {}", global_attrs.len());

    create_netcdf4_with_data("weather_comprehensive.nc", datasets, global_attrs)?;

    println!("  ✅ NetCDF4 convenience function demonstration completed!");

    Ok(())
}

fn demonstrate_large_dataset_support() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demonstrating Large Dataset Support...");

    let mut large_file = NetCDFFile::create_with_format("large_dataset.nc", NetCDFFormat::NetCDF4)?;

    // Create a large 3D dataset (simulating climate model output)
    println!("  🌍 Creating large climate dataset (100 x 200 x 365)...");
    let climate_data = Array3::from_shape_fn((100, 200, 365), |(lat, lon, day)| {
        // Simulate temperature variation
        let lat_factor = (lat as f64 - 50.0) / 50.0; // Latitude effect
        let lon_factor = (lon as f64 * 0.018).sin(); // Longitude effect
        let seasonal = (day as f64 * 2.0 * std::f64::consts::PI / 365.0).sin(); // Seasonal variation

        15.0 - 20.0 * lat_factor + 5.0 * seasonal + 2.0 * lon_factor
    });

    // Write the large dataset
    large_file.write_array(
        "global_temperature",
        &climate_data,
        &["latitude", "longitude", "day_of_year"],
    )?;

    // Add metadata
    large_file.add_global_attribute("title", "Global Climate Model Output")?;
    large_file.add_global_attribute("model", "SciRS2-Climate-v1.0")?;
    large_file.add_global_attribute("resolution", "1.8° x 1.8° global grid")?;

    println!("     Dataset size: {} elements", climate_data.len());
    println!(
        "     Memory usage: ~{:.1} MB",
        climate_data.len() * 8 / 1_000_000
    );

    large_file.close()?;

    println!("  ✅ Large dataset demonstration completed!");

    Ok(())
}

fn demonstrate_netcdf4_reading() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📖 Demonstrating NetCDF4 Reading...");

    // Read the weather dataset we created earlier
    println!("  📊 Reading weather dataset...");
    let weather_file = read_netcdf("weather_comprehensive.nc")?;

    println!("     File format: {:?}", weather_file.format());
    println!("     Has HDF5 backend: {}", weather_file.has_hdf5_backend());

    // List dimensions
    println!("     Dimensions:");
    for (name, size) in weather_file.dimensions() {
        match size {
            Some(s) => println!("       {}: {}", name, s),
            None => println!("       {}: unlimited", name),
        }
    }

    // List variables
    println!("     Variables:");
    for var_name in weather_file.variables() {
        println!("       {}", var_name);

        // Read a sample of the data
        if var_name == "temperature" {
            let temp_data = weather_file.read_array(&var_name)?;
            println!("         Shape: {:?}", temp_data.shape());
            println!(
                "         Sample values: {:?}",
                &temp_data.as_slice().unwrap()[..5.min(temp_data.len())]
            );
        }
    }

    // List global attributes
    println!("     Global attributes:");
    for (name, value) in weather_file.global_attributes() {
        println!("       {}: {}", name, value);
    }

    println!("  ✅ NetCDF4 reading demonstration completed!");

    Ok(())
}

fn demonstrate_netcdf4_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Demonstrating NetCDF4 Advanced Features...");

    // Create a file with advanced NetCDF4 options
    let advanced_options = NetCDFOptions {
        format: NetCDFFormat::NetCDF4,
        mode: "w".to_string(),
        enable_compression: true,
        compression_level: Some(6),
        enable_chunking: true,
        ..Default::default()
    };

    println!("  ⚙️  Creating file with advanced options...");
    println!("     Compression: enabled (level 6)");
    println!("     Chunking: enabled");

    let mut advanced_file = NetCDFFile::open("advanced_features.nc", Some(advanced_options))?;

    // Create a dataset with complex structure
    let time_series = Array2::from_shape_fn((1000, 50), |(t, sensor)| {
        // Simulate sensor data with noise
        let signal = (t as f64 * 0.01).sin() * (1.0 + sensor as f64 * 0.1);
        let noise = (t as f64 * sensor as f64 * 0.001).sin() * 0.1;
        signal + noise
    });

    advanced_file.write_array("sensor_timeseries", &time_series, &["time", "sensor_id"])?;

    // Add detailed metadata
    advanced_file.add_global_attribute("title", "High-Resolution Sensor Network")?;
    advanced_file.add_global_attribute("sampling_rate", "100 Hz")?;
    advanced_file.add_global_attribute("sensor_count", "50")?;
    advanced_file.add_global_attribute("duration", "10 seconds")?;
    advanced_file.add_global_attribute("compression", "enabled")?;

    // Add variable-specific attributes
    advanced_file.add_variable_attribute("sensor_timeseries", "units", "volts")?;
    advanced_file.add_variable_attribute(
        "sensor_timeseries",
        "long_name",
        "Sensor voltage readings",
    )?;
    advanced_file.add_variable_attribute("sensor_timeseries", "sampling_rate", "100")?;

    advanced_file.close()?;

    // Read back and verify
    println!("  🔍 Verifying advanced features...");
    let verification_file = read_netcdf("advanced_features.nc")?;

    let read_data = verification_file.read_array("sensor_timeseries")?;
    println!("     Data shape: {:?}", read_data.shape());
    println!("     Data size: {} elements", read_data.len());

    // Get variable information
    let (dtype, dims, attrs) = verification_file.variable_info("sensor_timeseries")?;
    println!("     Variable data type: {:?}", dtype);
    println!("     Variable dimensions: {:?}", dims);
    println!("     Variable attributes: {}", attrs.len());

    println!("  ✅ Advanced features demonstration completed!");

    Ok(())
}
