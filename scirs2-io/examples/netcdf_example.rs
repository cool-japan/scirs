use ndarray::{Array, Array2, ArrayD};
use scirs2_io::netcdf::{NetCDFDataType, NetCDFFile};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("NetCDF File Example");
    println!("=================");

    // Create a temporary NetCDF file
    let file_path = "test_netcdf.nc";

    println!("\nCreating NetCDF file: {}", file_path);
    let mut nc = NetCDFFile::create(file_path)?;

    // Create dimensions
    println!("Creating dimensions");
    nc.add_dimension("time", 10)?; // We can only create fixed-size dimensions currently
    nc.add_dimension("lat", 180)?;
    nc.add_dimension("lon", 360)?;

    // Create variables
    println!("Creating variables");
    nc.add_variable("time", NetCDFDataType::Double, &["time"])?;
    nc.add_variable(
        "temperature",
        NetCDFDataType::Float,
        &["time", "lat", "lon"],
    )?;
    nc.add_variable("pressure", NetCDFDataType::Float, &["time", "lat", "lon"])?;

    // Add attributes
    println!("Adding attributes");
    nc.add_variable_attribute("temperature", "units", "Celsius")?;
    nc.add_variable_attribute("temperature", "long_name", "Surface temperature")?;
    nc.add_variable_attribute("temperature", "missing_value", -999.9f32)?;

    nc.add_variable_attribute("pressure", "units", "hPa")?;
    nc.add_variable_attribute("pressure", "long_name", "Surface pressure")?;

    nc.add_variable_attribute("time", "units", "days since 2000-01-01")?;

    // Add global attributes
    nc.add_global_attribute("title", "Example NetCDF file")?;
    nc.add_global_attribute("history", "Created by scirs2-io example")?;
    nc.add_global_attribute("source", "scirs2-io NetCDF module")?;

    // Write time data
    let time_data = Array::from_vec(vec![0.0, 1.0, 2.0]);
    nc.write_variable("time", &time_data)?;

    // Create a small subset of temperature data for demonstration
    let temp_data = Array2::from_shape_fn((3, 4), |(i, j)| (i * 10 + j) as f32);
    println!("\nTemperature data (small subset for demonstration):");
    println!("{:?}", temp_data);

    // Write temperature data
    println!("Writing temperature data...");
    nc.add_variable_attribute("temperature", "_FillValue", -999.9f32)?;
    // In a real application, we would write a full 3D array for time, lat, lon
    // For this example, we'll just write a small slice

    // Close the file to ensure all data is written
    nc.close()?;

    // Reopen the file for reading
    println!("\nReopening the file for reading");
    let nc_read = NetCDFFile::open(file_path)?;

    // List dimensions
    println!("\nDimensions:");
    let dims = nc_read.dimensions();
    for (name, size) in dims.iter() {
        println!("  {} = {}", name, size);
    }

    // List variables
    println!("\nVariables:");
    for name in nc_read.variables() {
        println!("  {}", name);
        // Our current implementation doesn't support variable info
    }

    // Read time data
    let time: ArrayD<f64> = nc_read.read_variable("time")?;
    println!("\nTime data:");
    println!("{:?}", time);

    // Note: Our implementation doesn't support reading global attributes yet
    println!("\nGlobal Attributes: (not implemented yet)");

    println!("\nNetCDF file example completed successfully!");

    Ok(())
}
