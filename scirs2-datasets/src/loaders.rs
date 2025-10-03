//! Data loading utilities

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use csv::ReaderBuilder;
use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Load a dataset from a CSV file (legacy API)
#[allow(dead_code)]
pub fn load_csv_legacy<P: AsRef<Path>>(
    path: P,
    has_header: bool,
    target_column: Option<usize>,
) -> Result<Dataset> {
    let config = CsvConfig::new()
        .with_header(has_header)
        .with_target_column(target_column);
    load_csv(path, config)
}

/// Load a dataset from a JSON file
#[allow(dead_code)]
pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    let file = File::open(path).map_err(DatasetsError::IoError)?;
    let reader = BufReader::new(file);

    let dataset: Dataset = serde_json::from_reader(reader)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to parse JSON: {e}")))?;

    Ok(dataset)
}

/// Save a dataset to a JSON file
#[allow(dead_code)]
pub fn save_json<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<()> {
    let file = File::create(path).map_err(DatasetsError::IoError)?;

    serde_json::to_writer_pretty(file, dataset)
        .map_err(|e| DatasetsError::SerdeError(format!("Failed to write JSON: {e}")))?;

    Ok(())
}

/// Load raw data from a file
#[allow(dead_code)]
pub fn load_raw<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(DatasetsError::IoError)?;
    let mut buffer = Vec::new();

    file.read_to_end(&mut buffer)
        .map_err(DatasetsError::IoError)?;

    Ok(buffer)
}

/// Configuration for CSV loading operations
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Whether the CSV has a header row
    pub has_header: bool,
    /// Index of the target column (if any)
    pub target_column: Option<usize>,
    /// Delimiter character
    pub delimiter: u8,
    /// Quote character
    pub quote: u8,
    /// Whether to use double quotes
    pub double_quote: bool,
    /// Escape character
    pub escape: Option<u8>,
    /// Flexible parsing (ignore inconsistent columns)
    pub flexible: bool,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            has_header: true,
            target_column: None,
            delimiter: b',',
            quote: b'"',
            double_quote: true,
            escape: None,
            flexible: false,
        }
    }
}

impl CsvConfig {
    /// Create a new CSV configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether the CSV has headers
    pub fn with_header(mut self, hasheader: bool) -> Self {
        self.has_header = hasheader;
        self
    }

    /// Set the target column index
    pub fn with_target_column(mut self, targetcolumn: Option<usize>) -> Self {
        self.target_column = targetcolumn;
        self
    }

    /// Set the delimiter character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set flexible parsing mode
    pub fn with_flexible(mut self, flexible: bool) -> Self {
        self.flexible = flexible;
        self
    }
}

/// Configuration for streaming dataset loading
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each chunk (number of rows)
    pub chunk_size: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of parallel threads (0 = auto-detect)
    pub num_threads: usize,
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory: usize,
    /// Whether to use memory mapping for large files
    pub use_mmap: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            parallel: true,
            num_threads: 0, // Auto-detect
            max_memory: 0,  // Unlimited
            use_mmap: false,
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, chunksize: usize) -> Self {
        self.chunk_size = chunksize;
        self
    }

    /// Enable or disable parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the number of threads
    pub fn with_num_threads(mut self, numthreads: usize) -> Self {
        self.num_threads = numthreads;
        self
    }

    /// Set maximum memory usage
    pub fn with_max_memory(mut self, maxmemory: usize) -> Self {
        self.max_memory = maxmemory;
        self
    }

    /// Enable or disable memory mapping
    pub fn with_mmap(mut self, usemmap: bool) -> Self {
        self.use_mmap = usemmap;
        self
    }
}

/// Iterator for streaming dataset chunks
pub struct DatasetChunkIterator {
    reader: csv::Reader<File>,
    chunk_size: usize,
    target_column: Option<usize>,
    featurenames: Option<Vec<String>>,
    n_features: usize,
    buffer: Vec<Vec<f64>>,
    finished: bool,
}

impl DatasetChunkIterator {
    /// Create a new chunk iterator
    pub fn new<P: AsRef<Path>>(path: P, csv_config: CsvConfig, chunksize: usize) -> Result<Self> {
        let file = File::open(path).map_err(DatasetsError::IoError)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(csv_config.has_header)
            .delimiter(csv_config.delimiter)
            .quote(csv_config.quote)
            .double_quote(csv_config.double_quote)
            .flexible(csv_config.flexible)
            .from_reader(file);

        // Read header if present
        let featurenames = if csv_config.has_header {
            let headers = reader.headers().map_err(|e| {
                DatasetsError::InvalidFormat(format!("Failed to read CSV headers: {e}"))
            })?;
            Some(
                headers
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            )
        } else {
            None
        };

        // Determine number of features
        let n_features = if let Some(ref names) = featurenames {
            if csv_config.target_column.is_some() {
                names.len() - 1
            } else {
                names.len()
            }
        } else {
            // We'll determine this from the first row
            0
        };

        Ok(Self {
            reader,
            chunk_size: chunksize,
            target_column: csv_config.target_column,
            featurenames,
            n_features,
            buffer: Vec::new(),
            finished: false,
        })
    }

    /// Get feature names
    pub fn featurenames(&self) -> Option<&Vec<String>> {
        self.featurenames.as_ref()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Iterator for DatasetChunkIterator {
    type Item = Result<Dataset>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        self.buffer.clear();

        // Read chunk_size rows
        for _ in 0..self.chunk_size {
            match self.reader.records().next() {
                Some(Ok(record)) => {
                    let values: Vec<f64> = match record
                        .iter()
                        .map(|s| s.parse::<f64>())
                        .collect::<std::result::Result<Vec<f64>, _>>()
                    {
                        Ok(vals) => vals,
                        Err(e) => {
                            return Some(Err(DatasetsError::InvalidFormat(format!(
                                "Failed to parse value: {e}"
                            ))))
                        }
                    };

                    if !values.is_empty() {
                        // Update n_features if not set
                        if self.n_features == 0 {
                            self.n_features = if self.target_column.is_some() {
                                values.len() - 1
                            } else {
                                values.len()
                            };
                        }
                        self.buffer.push(values);
                    }
                }
                Some(Err(e)) => {
                    return Some(Err(DatasetsError::InvalidFormat(format!(
                        "Failed to read CSV record: {e}"
                    ))))
                }
                None => {
                    self.finished = true;
                    break;
                }
            }
        }

        if self.buffer.is_empty() {
            return None;
        }

        // Create dataset from buffer
        let n_rows = self.buffer.len();
        let n_cols = self.buffer[0].len();

        let (data, target) = if let Some(idx) = self.target_column {
            if idx >= n_cols {
                return Some(Err(DatasetsError::InvalidFormat(format!(
                    "Target column index {idx} is out of bounds (max: {})",
                    n_cols - 1
                ))));
            }

            let mut data_array = Array2::zeros((n_rows, n_cols - 1));
            let mut target_array = Array1::zeros(n_rows);

            for (i, row) in self.buffer.iter().enumerate() {
                let mut data_col = 0;
                for (j, &val) in row.iter().enumerate() {
                    if j == idx {
                        target_array[i] = val;
                    } else {
                        data_array[[i, data_col]] = val;
                        data_col += 1;
                    }
                }
            }

            (data_array, Some(target_array))
        } else {
            let mut data_array = Array2::zeros((n_rows, n_cols));

            for (i, row) in self.buffer.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    data_array[[i, j]] = val;
                }
            }

            (data_array, None)
        };

        let mut dataset = Dataset::new(data, target);

        // Set feature names (excluding target column)
        if let Some(ref names) = self.featurenames {
            let featurenames = if let Some(target_idx) = self.target_column {
                names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| {
                        if i != target_idx {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                names.clone()
            };
            dataset = dataset.with_featurenames(featurenames);
        }

        Some(Ok(dataset))
    }
}

/// Load a CSV file using streaming with configurable chunking
#[allow(dead_code)]
pub fn load_csv_streaming<P: AsRef<Path>>(
    path: P,
    csv_config: CsvConfig,
    streaming_config: StreamingConfig,
) -> Result<DatasetChunkIterator> {
    DatasetChunkIterator::new(path, csv_config, streaming_config.chunk_size)
}

/// Load a large CSV file efficiently by processing in parallel chunks
#[allow(dead_code)]
pub fn load_csv_parallel<P: AsRef<Path>>(
    path: P,
    csv_config: CsvConfig,
    streaming_config: StreamingConfig,
) -> Result<Dataset> {
    // First pass: determine dataset dimensions
    let file = File::open(&path).map_err(DatasetsError::IoError)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(csv_config.has_header)
        .delimiter(csv_config.delimiter)
        .from_reader(file);

    let featurenames = if csv_config.has_header {
        let headers = reader.headers().map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to read CSV headers: {e}"))
        })?;
        Some(
            headers
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
        )
    } else {
        None
    };

    // Count rows and determine column count
    let mut row_count = 0;
    let mut col_count = 0;

    for result in reader.records() {
        let record = result
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to read CSV record: {e}")))?;

        if col_count == 0 {
            col_count = record.len();
        }
        row_count += 1;
    }

    if row_count == 0 {
        return Err(DatasetsError::InvalidFormat(
            "CSV file is empty".to_string(),
        ));
    }

    // Determine final dimensions
    let data_cols = if csv_config.target_column.is_some() {
        col_count - 1
    } else {
        col_count
    };

    // Create output arrays
    let data = Arc::new(Mutex::new(Array2::zeros((row_count, data_cols))));
    let target = if csv_config.target_column.is_some() {
        Some(Arc::new(Mutex::new(Array1::zeros(row_count))))
    } else {
        None
    };

    // Second pass: parallel processing in chunks
    if streaming_config.parallel && row_count > streaming_config.chunk_size {
        load_csv_parallel_chunks(
            &path,
            csv_config.clone(),
            streaming_config,
            data.clone(),
            target.clone(),
            row_count,
        )?;
    } else {
        load_csv_sequential(&path, csv_config.clone(), data.clone(), target.clone())?;
    }

    // Extract final arrays
    let final_data = Arc::try_unwrap(data)
        .map_err(|_| DatasetsError::Other("Failed to unwrap data array".to_string()))?
        .into_inner()
        .map_err(|_| DatasetsError::Other("Failed to acquire data lock".to_string()))?;

    let final_target = if let Some(target_arc) = target {
        Some(
            Arc::try_unwrap(target_arc)
                .map_err(|_| DatasetsError::Other("Failed to unwrap target array".to_string()))?
                .into_inner()
                .map_err(|_| DatasetsError::Other("Failed to acquire target lock".to_string()))?,
        )
    } else {
        None
    };

    let mut dataset = Dataset::new(final_data, final_target);

    // Set feature names
    if let Some(names) = featurenames {
        let featurenames = if let Some(target_idx) = csv_config.target_column {
            names
                .iter()
                .enumerate()
                .filter_map(|(i, name)| {
                    if i != target_idx {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            names
        };
        dataset = dataset.with_featurenames(featurenames);
    }

    Ok(dataset)
}

/// Load CSV using parallel chunks
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn load_csv_parallel_chunks<P: AsRef<Path>>(
    path: P,
    csv_config: CsvConfig,
    streaming_config: StreamingConfig,
    data: Arc<Mutex<Array2<f64>>>,
    target: Option<Arc<Mutex<Array1<f64>>>>,
    total_rows: usize,
) -> Result<()> {
    let chunk_size = streaming_config.chunk_size;
    let num_chunks = total_rows.div_ceil(chunk_size);

    // Process chunks sequentially (parallel processing disabled for now)
    for chunk_idx in 0..num_chunks {
        let start_row = chunk_idx * chunk_size;
        let end_row = std::cmp::min(start_row + chunk_size, total_rows);

        if let Err(e) = process_csv_chunk(
            &path,
            &csv_config,
            start_row,
            end_row,
            data.clone(),
            target.clone(),
        ) {
            eprintln!("Error processing chunk {chunk_idx}: {e}");
        }
    }

    Ok(())
}

/// Process a single CSV chunk
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn process_csv_chunk<P: AsRef<Path>>(
    path: P,
    csv_config: &CsvConfig,
    start_row: usize,
    end_row: usize,
    data: Arc<Mutex<Array2<f64>>>,
    target: Option<Arc<Mutex<Array1<f64>>>>,
) -> Result<()> {
    let file = File::open(path).map_err(DatasetsError::IoError)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(csv_config.has_header)
        .delimiter(csv_config.delimiter)
        .from_reader(file);

    // Skip to start _row
    if csv_config.has_header {
        reader
            .headers()
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to read headers: {e}")))?;
    }

    for (current_row, result) in reader.records().enumerate() {
        if current_row >= end_row {
            break;
        }

        if current_row >= start_row {
            let record = result.map_err(|e| {
                DatasetsError::InvalidFormat(format!("Failed to read CSV record: {e}"))
            })?;

            let values: Vec<f64> = record
                .iter()
                .map(|s| s.parse::<f64>())
                .collect::<std::result::Result<Vec<f64>, _>>()
                .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to parse value: {e}")))?;

            // Write to shared arrays
            {
                let mut data_lock = data.lock().unwrap();
                if let Some(target_idx) = csv_config.target_column {
                    let mut data_col = 0;
                    for (j, &val) in values.iter().enumerate() {
                        if j == target_idx {
                            if let Some(ref target_arc) = target {
                                let mut target_lock = target_arc.lock().unwrap();
                                target_lock[current_row] = val;
                            }
                        } else {
                            data_lock[[current_row, data_col]] = val;
                            data_col += 1;
                        }
                    }
                } else {
                    for (j, &val) in values.iter().enumerate() {
                        data_lock[[current_row, j]] = val;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Load CSV sequentially (fallback)
#[allow(dead_code)]
fn load_csv_sequential<P: AsRef<Path>>(
    path: P,
    csv_config: CsvConfig,
    data: Arc<Mutex<Array2<f64>>>,
    target: Option<Arc<Mutex<Array1<f64>>>>,
) -> Result<()> {
    let file = File::open(path).map_err(DatasetsError::IoError)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(csv_config.has_header)
        .delimiter(csv_config.delimiter)
        .from_reader(file);

    if csv_config.has_header {
        reader
            .headers()
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to read headers: {e}")))?;
    }

    for (row_idx, result) in reader.records().enumerate() {
        let record = result
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to read CSV record: {e}")))?;

        let values: Vec<f64> = record
            .iter()
            .map(|s| s.parse::<f64>())
            .collect::<std::result::Result<Vec<f64>, _>>()
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to parse value: {e}")))?;

        {
            let mut data_lock = data.lock().unwrap();
            if let Some(target_idx) = csv_config.target_column {
                let mut data_col = 0;
                for (j, &val) in values.iter().enumerate() {
                    if j == target_idx {
                        if let Some(ref target_arc) = target {
                            let mut target_lock = target_arc.lock().unwrap();
                            target_lock[row_idx] = val;
                        }
                    } else {
                        data_lock[[row_idx, data_col]] = val;
                        data_col += 1;
                    }
                }
            } else {
                for (j, &val) in values.iter().enumerate() {
                    data_lock[[row_idx, j]] = val;
                }
            }
        }
    }

    Ok(())
}

/// Enhanced CSV loader with improved configuration
#[allow(dead_code)]
pub fn load_csv<P: AsRef<Path>>(path: P, config: CsvConfig) -> Result<Dataset> {
    let file = File::open(path).map_err(DatasetsError::IoError)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(config.has_header)
        .delimiter(config.delimiter)
        .quote(config.quote)
        .double_quote(config.double_quote)
        .flexible(config.flexible)
        .from_reader(file);

    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut header: Option<Vec<String>> = None;

    // Read header if needed
    if config.has_header {
        let headers = reader.headers().map_err(|e| {
            DatasetsError::InvalidFormat(format!("Failed to read CSV headers: {e}"))
        })?;
        header = Some(headers.iter().map(|s| s.to_string()).collect());
    }

    // Read rows
    for result in reader.records() {
        let record = result
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to read CSV record: {e}")))?;

        let values: Vec<f64> = record
            .iter()
            .map(|s| {
                s.parse::<f64>().map_err(|_| {
                    DatasetsError::InvalidFormat(format!("Failed to parse value: {s}"))
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        if !values.is_empty() {
            records.push(values);
        }
    }

    if records.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "CSV file is empty".to_string(),
        ));
    }

    // Create data array and target array if needed
    let n_rows = records.len();
    let n_cols = records[0].len();

    let (data, target, featurenames, _targetname) = if let Some(idx) = config.target_column {
        if idx >= n_cols {
            return Err(DatasetsError::InvalidFormat(format!(
                "Target column index {idx} is out of bounds (max: {})",
                n_cols - 1
            )));
        }

        let mut data_array = Array2::zeros((n_rows, n_cols - 1));
        let mut target_array = Array1::zeros(n_rows);

        for (i, row) in records.iter().enumerate() {
            let mut data_col = 0;
            for (j, &val) in row.iter().enumerate() {
                if j == idx {
                    target_array[i] = val;
                } else {
                    data_array[[i, data_col]] = val;
                    data_col += 1;
                }
            }
        }

        let featurenames = header.as_ref().map(|h| {
            let mut names = Vec::new();
            for (j, name) in h.iter().enumerate() {
                if j != idx {
                    names.push(name.clone());
                }
            }
            names
        });

        (
            data_array,
            Some(target_array),
            featurenames,
            header.as_ref().map(|h| h[idx].clone()),
        )
    } else {
        let mut data_array = Array2::zeros((n_rows, n_cols));

        for (i, row) in records.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data_array[[i, j]] = val;
            }
        }

        (data_array, None, header, None)
    };

    let mut dataset = Dataset::new(data, target);

    if let Some(names) = featurenames {
        dataset = dataset.with_featurenames(names);
    }

    Ok(dataset)
}
