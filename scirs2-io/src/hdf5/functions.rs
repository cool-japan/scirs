//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;
use std::path::Path;

use super::types_2::HDF5File;
use super::types_3::{FileMode, Group};

/// Read an HDF5 file and return the root group
///
/// # Arguments
/// * `path` - Path to the HDF5 file
///
/// # Returns
/// The root group of the HDF5 file
///
/// # Example
/// ```no_run
/// use scirs2_io::hdf5::read_hdf5;
///
/// let root_group = read_hdf5("data.h5")?;
/// println!("Groups: {:?}", root_group.groups.keys().collect::<Vec<_>>());
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn read_hdf5<P: AsRef<Path>>(path: P) -> Result<Group> {
    let file = HDF5File::open(path, FileMode::ReadOnly)?;
    Ok(file.root)
}

/// Write data to an HDF5 file
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `datasets` - Map of dataset paths to arrays
///
/// # Example
/// ```no_run
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
/// use scirs2_io::hdf5::write_hdf5;
///
/// let mut datasets = HashMap::new();
/// datasets.insert("data/temperature".to_string(), array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
/// datasets.insert("data/pressure".to_string(), array![100.0, 200.0, 300.0].into_dyn());
///
/// write_hdf5("output.h5", datasets)?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn write_hdf5<P: AsRef<Path>>(path: P, datasets: HashMap<String, ArrayD<f64>>) -> Result<()> {
    let mut file = HDF5File::create(path)?;

    for (datasetpath, array) in datasets {
        file.create_dataset_from_array(&datasetpath, &array, None)?;
    }

    file.write()?;
    file.close()?;
    Ok(())
}

/// Create an HDF5 file with groups and attributes
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `builder` - Function to build the file structure
///
/// # Example
/// ```no_run
/// use scirs2_io::hdf5::{create_hdf5_with_structure, AttributeValue};
/// use scirs2_core::ndarray::array;
///
/// create_hdf5_with_structure("structured.h5", |file| {
///     let root = file.root_mut();
///     
///     // Create groups
///     let experiment = root.create_group("experiment");
///     experiment.set_attribute("date", AttributeValue::String("2024-01-01".to_string()));
///     experiment.set_attribute("temperature", AttributeValue::Float(25.0));
///     
///     // Add datasets
///     let data = array![[1.0, 2.0], [3.0, 4.0]];
///     file.create_dataset_from_array("experiment/measurements", &data, None)?;
///     
///     Ok(())
/// })?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
#[allow(dead_code)]
pub fn create_hdf5_with_structure<P, F>(path: P, builder: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut HDF5File) -> Result<()>,
{
    let mut file = HDF5File::create(path)?;
    builder(&mut file)?;
    file.write()?;
    file.close()?;
    Ok(())
}
