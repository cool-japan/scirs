//! Memory-efficient statistical operations
//!
//! This module provides memory-optimized implementations of statistical functions
//! that minimize allocations and use streaming/chunked processing for large datasets.

use crate::error::{StatsError, StatsResult};
use crate::error_standardization::ErrorMessages;
#[cfg(feature = "memmap")]
use memmap2::Mmap;
use scirs2_core::ndarray::{s, ArrayBase, ArrayViewMut1, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Chunk size for streaming operations (tuned for cache efficiency)
const CHUNK_SIZE: usize = 8192;

/// Streaming mean calculation that processes data in chunks
///
/// This function computes the mean without loading the entire dataset into memory
/// at once, making it suitable for very large datasets.
///
/// # Arguments
///
/// * `data_iter` - Iterator over data chunks
/// * `total_count` - Total number of elements across all chunks
///
/// # Returns
///
/// The arithmetic mean
#[allow(dead_code)]
pub fn streaming_mean<F, I>(mut data_iter: I, totalcount: usize) -> StatsResult<F>
where
    F: Float + NumCast,
    I: Iterator<Item = F> + std::fmt::Display,
{
    if totalcount == 0 {
        return Err(ErrorMessages::empty_array("dataset"));
    }

    let mut sum = F::zero();
    let mut _count = 0;

    // Process in chunks to maintain precision
    while _count < totalcount {
        let chunk_sum = data_iter
            .by_ref()
            .take(CHUNK_SIZE)
            .fold(F::zero(), |acc, val| acc + val);

        sum = sum + chunk_sum;
        _count += CHUNK_SIZE.min(totalcount - _count);
    }

    Ok(sum / F::from(totalcount).unwrap())
}

/// Welford's online algorithm for variance computation
///
/// This algorithm computes variance in a single pass with minimal memory usage
/// and improved numerical stability.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// * Tuple of (mean, variance)
#[allow(dead_code)]
pub fn welford_variance<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<(F, F)>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n = x.len();
    if n <= ddof {
        return Err(StatsError::InvalidArgument(
            "Not enough data points for the given degrees of freedom".to_string(),
        ));
    }

    let mut mean = F::zero();
    let mut m2 = F::zero();
    let mut count = 0;

    for &value in x.iter() {
        count += 1;
        let delta = value - mean;
        mean = mean + delta / F::from(count).unwrap();
        let delta2 = value - mean;
        m2 = m2 + delta * delta2;
    }

    let variance = m2 / F::from(n - ddof).unwrap();
    Ok((mean, variance))
}

/// In-place normalization (standardization) of data
///
/// This function normalizes data in-place to have zero mean and unit variance,
/// avoiding the need for additional memory allocation.
///
/// # Arguments
///
/// * `data` - Mutable array to normalize
/// * `ddof` - Delta degrees of freedom for variance calculation
#[allow(dead_code)]
pub fn normalize_inplace<F>(data: &mut ArrayViewMut1<F>, ddof: usize) -> StatsResult<()>
where
    F: Float + NumCast + std::fmt::Display,
{
    let (mean, variance) = welford_variance(&data.to_owned(), ddof)?;

    if variance <= F::epsilon() {
        return Err(StatsError::InvalidArgument(
            "Cannot normalize data with zero variance".to_string(),
        ));
    }

    let std_dev = variance.sqrt();

    // Normalize in-place
    for val in data.iter_mut() {
        *val = (*val - mean) / std_dev;
    }

    Ok(())
}

/// Memory-efficient quantile computation using quickselect
///
/// This function computes quantiles without fully sorting the array,
/// which saves memory and time for large datasets.
///
/// # Arguments
///
/// * `data` - Input data array (will be partially reordered)
/// * `q` - Quantile to compute (0 to 1)
///
/// # Returns
///
/// The computed quantile value
#[allow(dead_code)]
pub fn quantile_quickselect<F>(data: &mut [F], q: F) -> StatsResult<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute quantile of empty array".to_string(),
        ));
    }

    if q < F::zero() || q > F::one() {
        return Err(StatsError::domain("Quantile must be between 0 and 1"));
    }

    let n = data.len();
    let pos = q * F::from(n - 1).unwrap();
    let k = NumCast::from(pos.floor()).unwrap();

    // Use quickselect to find k-th element
    quickselect(data, k);

    let lower = data[k];

    // Handle interpolation if needed
    let frac = pos - pos.floor();
    if frac > F::zero() && k + 1 < n {
        quickselect(&mut data[(k + 1)..], 0);
        let upper = data[k + 1];
        Ok(lower + frac * (upper - lower))
    } else {
        Ok(lower)
    }
}

/// Quickselect algorithm for finding k-th smallest element
#[allow(dead_code)]
fn quickselect<F: Float>(data: &mut [F], k: usize) {
    let len = data.len();
    if len <= 1 {
        return;
    }

    let mut left = 0;
    let mut right = len - 1;

    while left < right {
        let pivot_idx = partition(data, left, right);

        match k.cmp(&pivot_idx) {
            Ordering::Less => right = pivot_idx - 1,
            Ordering::Greater => left = pivot_idx + 1,
            Ordering::Equal => return,
        }
    }
}

/// Partition function for quickselect
#[allow(dead_code)]
fn partition<F: Float>(data: &mut [F], left: usize, right: usize) -> usize {
    let pivot_idx = left + (right - left) / 2;
    let pivot = data[pivot_idx];

    data.swap(pivot_idx, right);

    let mut store_idx = left;
    for i in left..right {
        if data[i] < pivot {
            data.swap(i, store_idx);
            store_idx += 1;
        }
    }

    data.swap(store_idx, right);
    store_idx
}

/// Memory-efficient covariance matrix computation
///
/// Computes covariance matrix using a streaming algorithm that processes
/// data in chunks to minimize memory usage.
///
/// # Arguments
///
/// * `data` - 2D array where columns are variables
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Covariance matrix
#[allow(dead_code)]
pub fn covariance_chunked<F, D>(
    data: &ArrayBase<D, Ix2>,
    ddof: usize,
) -> StatsResult<scirs2_core::ndarray::Array2<F>>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n_obs = data.nrows();
    let n_vars = data.ncols();

    if n_obs <= ddof {
        return Err(StatsError::InvalidArgument(
            "Not enough observations for the given degrees of freedom".to_string(),
        ));
    }

    // Compute means for each variable
    let mut means = scirs2_core::ndarray::Array1::zeros(n_vars);
    for j in 0..n_vars {
        let col = data.slice(s![.., j]);
        means[j] = col.iter().fold(F::zero(), |acc, &val| acc + val) / F::from(n_obs).unwrap();
    }

    // Initialize covariance matrix
    let mut cov_matrix = scirs2_core::ndarray::Array2::zeros((n_vars, n_vars));

    // Process data in chunks to compute covariance
    let chunksize = CHUNK_SIZE / n_vars;
    for chunk_start in (0..n_obs).step_by(chunksize) {
        let chunk_end = (chunk_start + chunksize).min(n_obs);
        let chunk = data.slice(s![chunk_start..chunk_end, ..]);

        // Update covariance for this chunk
        for i in 0..n_vars {
            for j in i..n_vars {
                let mut sum = F::zero();
                for k in 0..chunk.nrows() {
                    let xi = chunk[(k, i)] - means[i];
                    let xj = chunk[(k, j)] - means[j];
                    sum = sum + xi * xj;
                }
                cov_matrix[(i, j)] = cov_matrix[(i, j)] + sum;
            }
        }
    }

    // Normalize and fill symmetric entries
    let factor = F::from(n_obs - ddof).unwrap();
    for i in 0..n_vars {
        for j in i..n_vars {
            cov_matrix[(i, j)] = cov_matrix[(i, j)] / factor;
            if i != j {
                cov_matrix[(j, i)] = cov_matrix[(i, j)];
            }
        }
    }

    Ok(cov_matrix)
}

/// Streaming correlation computation for large datasets
///
/// Computes correlation between two variables without loading all data at once.
#[allow(dead_code)]
pub struct StreamingCorrelation<F: Float> {
    n: usize,
    sum_x: F,
    sum_y: F,
    sum_xx: F,
    sum_yy: F,
    sum_xy: F,
}

#[allow(dead_code)]
impl<F: Float + NumCast + std::fmt::Display> StreamingCorrelation<F> {
    /// Create a new streaming correlation calculator
    pub fn new() -> Self {
        Self {
            n: 0,
            sum_x: F::zero(),
            sum_y: F::zero(),
            sum_xx: F::zero(),
            sum_yy: F::zero(),
            sum_xy: F::zero(),
        }
    }

    /// Update with a new pair of values
    pub fn update(&mut self, x: F, y: F) {
        self.n += 1;
        self.sum_x = self.sum_x + x;
        self.sum_y = self.sum_y + y;
        self.sum_xx = self.sum_xx + x * x;
        self.sum_yy = self.sum_yy + y * y;
        self.sum_xy = self.sum_xy + x * y;
    }

    /// Update with arrays of values
    pub fn update_batch<D>(&mut self, x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>)
    where
        D: Data<Elem = F>,
    {
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            self.update(xi, yi);
        }
    }

    /// Compute the correlation coefficient
    pub fn correlation(&self) -> StatsResult<F> {
        if self.n < 2 {
            return Err(StatsError::InvalidArgument(
                "Need at least 2 observations to compute correlation".to_string(),
            ));
        }

        let n = F::from(self.n).unwrap();
        let mean_x = self.sum_x / n;
        let mean_y = self.sum_y / n;

        let cov_xy = (self.sum_xy - n * mean_x * mean_y) / (n - F::one());
        let var_x = (self.sum_xx - n * mean_x * mean_x) / (n - F::one());
        let var_y = (self.sum_yy - n * mean_y * mean_y) / (n - F::one());

        if var_x <= F::epsilon() || var_y <= F::epsilon() {
            return Err(StatsError::InvalidArgument(
                "Cannot compute correlation when one or both variables have zero variance"
                    .to_string(),
            ));
        }

        Ok(cov_xy / (var_x * var_y).sqrt())
    }

    /// Merge with another streaming correlation
    pub fn merge(&mut self, other: &Self) {
        self.n += other.n;
        self.sum_x = self.sum_x + other.sum_x;
        self.sum_y = self.sum_y + other.sum_y;
        self.sum_xx = self.sum_xx + other.sum_xx;
        self.sum_yy = self.sum_yy + other.sum_yy;
        self.sum_xy = self.sum_xy + other.sum_xy;
    }
}

/// Incremental covariance matrix computation
///
/// Updates covariance matrix incrementally as new observations arrive.
#[allow(dead_code)]
pub struct IncrementalCovariance<F: Float> {
    n: usize,
    means: scirs2_core::ndarray::Array1<F>,
    cov_matrix: scirs2_core::ndarray::Array2<F>,
    n_vars: usize,
}

#[allow(dead_code)]
impl<F: Float + NumCast + scirs2_core::ndarray::ScalarOperand + std::fmt::Display>
    IncrementalCovariance<F>
{
    /// Create a new incremental covariance calculator
    pub fn new(_nvars: usize) -> Self {
        Self {
            n: 0,
            means: scirs2_core::ndarray::Array1::zeros(_nvars),
            cov_matrix: scirs2_core::ndarray::Array2::zeros((_nvars, _nvars)),
            n_vars: _nvars,
        }
    }

    /// Update with a new observation
    pub fn update(&mut self, observation: &scirs2_core::ndarray::ArrayView1<F>) -> StatsResult<()> {
        if observation.len() != self.n_vars {
            return Err(StatsError::DimensionMismatch(
                "Observation dimension doesn't match".to_string(),
            ));
        }

        self.n += 1;
        let n = F::from(self.n).unwrap();

        // Update means and covariance using Welford's algorithm
        let mut delta = scirs2_core::ndarray::Array1::zeros(self.n_vars);

        for i in 0..self.n_vars {
            delta[i] = observation[i] - self.means[i];
            self.means[i] = self.means[i] + delta[i] / n;
        }

        if self.n > 1 {
            for i in 0..self.n_vars {
                for j in i..self.n_vars {
                    let delta_new = observation[j] - self.means[j];
                    let cov_update = delta[i] * delta_new * (n - F::one()) / n;
                    self.cov_matrix[(i, j)] = self.cov_matrix[(i, j)] + cov_update;
                    if i != j {
                        self.cov_matrix[(j, i)] = self.cov_matrix[(i, j)];
                    }
                }
            }
        }

        Ok(())
    }

    /// Get current covariance matrix
    pub fn covariance(&self, ddof: usize) -> StatsResult<scirs2_core::ndarray::Array2<F>> {
        if self.n <= ddof {
            return Err(StatsError::InvalidArgument(
                "Not enough observations for the given degrees of freedom".to_string(),
            ));
        }

        let factor = F::from(self.n - ddof).unwrap();
        Ok(&self.cov_matrix / factor)
    }

    /// Get current means
    pub fn means(&self) -> &scirs2_core::ndarray::Array1<F> {
        &self.means
    }
}

/// Memory-efficient rolling window statistics
///
/// Computes statistics over a sliding window without storing all data.
#[allow(dead_code)]
pub struct RollingStats<F: Float> {
    windowsize: usize,
    buffer: Vec<F>,
    position: usize,
    is_full: bool,
    sum: F,
    sum_squares: F,
}

#[allow(dead_code)]
impl<F: Float + NumCast + std::fmt::Display> RollingStats<F> {
    /// Create a new rolling statistics calculator
    pub fn new(_windowsize: usize) -> StatsResult<Self> {
        if _windowsize == 0 {
            return Err(StatsError::InvalidArgument(
                "Window size must be positive".to_string(),
            ));
        }

        Ok(Self {
            windowsize: _windowsize,
            buffer: vec![F::zero(); _windowsize],
            position: 0,
            is_full: false,
            sum: F::zero(),
            sum_squares: F::zero(),
        })
    }

    /// Add a new value to the rolling window
    pub fn push(&mut self, value: F) {
        let old_value = self.buffer[self.position];

        // Update running sums
        self.sum = self.sum - old_value + value;
        self.sum_squares = self.sum_squares - old_value * old_value + value * value;

        // Store new value
        self.buffer[self.position] = value;
        self.position = (self.position + 1) % self.windowsize;

        if !self.is_full && self.position == 0 {
            self.is_full = true;
        }
    }

    /// Get current window size
    pub fn len(&self) -> usize {
        if self.is_full {
            self.windowsize
        } else {
            self.position
        }
    }

    /// Compute mean of current window
    pub fn mean(&self) -> F {
        let n = self.len();
        if n == 0 {
            F::zero()
        } else {
            self.sum / F::from(n).unwrap()
        }
    }

    /// Compute variance of current window
    pub fn variance(&self, ddof: usize) -> StatsResult<F> {
        let n = self.len();
        if n <= ddof {
            return Err(StatsError::InvalidArgument(
                "Not enough data for the given degrees of freedom".to_string(),
            ));
        }

        let mean = self.mean();
        let n_f = F::from(n).unwrap();
        let variance = (self.sum_squares / n_f) - mean * mean;
        Ok(variance * n_f / F::from(n - ddof).unwrap())
    }

    /// Get current buffer as array
    pub fn as_array(&self) -> scirs2_core::ndarray::Array1<F> {
        if self.is_full {
            scirs2_core::ndarray::Array1::from_vec(self.buffer.clone())
        } else {
            scirs2_core::ndarray::Array1::from_vec(self.buffer[..self.position].to_vec())
        }
    }
}

/// Memory-efficient histogram computation
///
/// Computes histogram without storing all data, using a streaming approach.
pub struct StreamingHistogram<F: Float> {
    bins: Vec<F>,
    counts: Vec<usize>,
    min_val: F,
    max_val: F,
    total_count: usize,
}

impl<F: Float + NumCast + std::fmt::Display> StreamingHistogram<F> {
    /// Create a new streaming histogram
    pub fn new(_n_bins: usize, min_val: F, maxval: F) -> Self {
        let bin_width = (maxval - min_val) / F::from(_n_bins).unwrap();
        let bins: Vec<F> = (0..=_n_bins)
            .map(|i| min_val + F::from(i).unwrap() * bin_width)
            .collect();

        Self {
            bins,
            counts: vec![0; _n_bins],
            min_val,
            max_val: maxval,
            total_count: 0,
        }
    }

    /// Add a value to the histogram
    pub fn add_value(&mut self, value: F) {
        if value >= self.min_val && value <= self.max_val {
            let n_bins = self.counts.len();
            let bin_width = (self.max_val - self.min_val) / F::from(n_bins).unwrap();
            let bin_idx = ((value - self.min_val) / bin_width).floor();
            let bin_idx: usize = NumCast::from(bin_idx).unwrap_or(n_bins - 1).min(n_bins - 1);
            self.counts[bin_idx] += 1;
            self.total_count += 1;
        }
    }

    /// Add multiple values
    pub fn add_values<D>(&mut self, values: &ArrayBase<D, Ix1>)
    where
        D: Data<Elem = F>,
    {
        for &value in values.iter() {
            self.add_value(value);
        }
    }

    /// Get the histogram results
    pub fn get_histogram(&self) -> (Vec<F>, Vec<usize>) {
        (self.bins.clone(), self.counts.clone())
    }

    /// Get normalized histogram (density)
    pub fn get_density(&self) -> (Vec<F>, Vec<F>) {
        let n_bins = self.counts.len();
        let bin_width = (self.max_val - self.min_val) / F::from(n_bins).unwrap();
        let total = F::from(self.total_count).unwrap() * bin_width;

        let density: Vec<F> = self
            .counts
            .iter()
            .map(|&count| F::from(count).unwrap() / total)
            .collect();

        (self.bins.clone(), density)
    }
}

/// Out-of-core statistics for datasets larger than memory
///
/// Processes data from files in chunks without loading entire dataset.
#[allow(dead_code)]
pub struct OutOfCoreStats<F: Float> {
    chunksize: usize,
    _phantom: std::marker::PhantomData<F>,
}

#[allow(dead_code)]
impl<F: Float + NumCast + std::str::FromStr + std::fmt::Display> OutOfCoreStats<F> {
    /// Create a new out-of-core statistics processor
    pub fn new(_chunksize: usize) -> Self {
        Self {
            chunksize: _chunksize,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute mean from a CSV file
    pub fn mean_from_csv<P: AsRef<Path>>(
        &self,
        path: P,
        column: usize,
        has_header: bool,
    ) -> StatsResult<F> {
        let file = File::open(path)
            .map_err(|e| StatsError::InvalidArgument(format!("Failed to open file: {e}")))?;
        let mut reader = BufReader::new(file);

        let mut sum = F::zero();
        let mut count = 0;
        let mut line = String::new();
        let mut line_num = 0;

        // Skip _header if present
        if has_header {
            reader
                .read_line(&mut line)
                .map_err(|e| StatsError::InvalidArgument(format!("Failed to read header: {e}")))?;
            line.clear();
        }

        // Process file in lines
        loop {
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    line_num += 1;
                    let fields: Vec<&str> = line.trim().split(',').collect();

                    if fields.len() > column {
                        if let Ok(value) = fields[column].parse::<F>() {
                            sum = sum + value;
                            count += 1;
                        }
                    }

                    line.clear();
                }
                Err(e) => {
                    return Err(StatsError::ComputationError(format!(
                        "Error reading line {line_num}: {e}"
                    )))
                }
            }
        }

        if count == 0 {
            return Err(StatsError::InvalidArgument(
                "No valid data found".to_string(),
            ));
        }

        Ok(sum / F::from(count).unwrap())
    }

    /// Compute variance from a CSV file using two-pass algorithm
    pub fn variance_from_csv<P: AsRef<Path>>(
        &self,
        path: P,
        column: usize,
        has_header: bool,
        ddof: usize,
    ) -> StatsResult<(F, F)> {
        // First pass: compute mean
        let mean = self.mean_from_csv(&path, column, has_header)?;

        // Second pass: compute variance
        let file = File::open(path)
            .map_err(|e| StatsError::InvalidArgument(format!("Failed to open file: {e}")))?;
        let mut reader = BufReader::new(file);

        let mut sum_sq = F::zero();
        let mut count = 0;
        let mut line = String::new();

        // Skip _header if present
        if has_header {
            reader
                .read_line(&mut line)
                .map_err(|e| StatsError::InvalidArgument(format!("Failed to read header: {e}")))?;
            line.clear();
        }

        // Process file
        loop {
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let fields: Vec<&str> = line.trim().split(',').collect();

                    if fields.len() > column {
                        if let Ok(value) = fields[column].parse::<F>() {
                            let diff = value - mean;
                            sum_sq = sum_sq + diff * diff;
                            count += 1;
                        }
                    }

                    line.clear();
                }
                Err(e) => {
                    return Err(StatsError::ComputationError(format!(
                        "Error reading file: {}",
                        e
                    )))
                }
            }
        }

        if count <= ddof {
            return Err(StatsError::InvalidArgument(
                "Not enough data for the given degrees of freedom".to_string(),
            ));
        }

        let variance = sum_sq
            / F::from(count - ddof).ok_or_else(|| {
                StatsError::ComputationError(
                    "Failed to convert count - ddof to target type".to_string(),
                )
            })?;
        Ok((mean, variance))
    }

    /// Process large binary file of floats
    pub fn process_binary_file<P: AsRef<Path>, G>(
        &self,
        path: P,
        mut processor: G,
    ) -> StatsResult<()>
    where
        G: FnMut(&[F]) -> StatsResult<()>,
    {
        use std::mem;

        let file = File::open(path)
            .map_err(|e| StatsError::InvalidArgument(format!("Failed to open file: {e}")))?;
        let mut reader = BufReader::new(file);

        let elementsize = mem::size_of::<F>();
        let buffersize = self.chunksize * elementsize;
        let mut buffer = vec![0u8; buffersize];

        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(bytes_read) => {
                    let n_elements = bytes_read / elementsize;

                    // Convert bytes to floats (unsafe but efficient)
                    let floats = unsafe {
                        std::slice::from_raw_parts(buffer.as_ptr() as *const F, n_elements)
                    };

                    processor(floats)?;
                }
                Err(e) => {
                    return Err(StatsError::ComputationError(format!(
                        "Error reading file: {}",
                        e
                    )))
                }
            }
        }

        Ok(())
    }
}

/// Memory-mapped statistics for very large files
///
/// Uses memory-mapped I/O for efficient access to large datasets.
#[cfg(feature = "memmap")]
pub struct MemoryMappedStats<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

#[cfg(feature = "memmap")]
impl<F: Float + NumCast + std::fmt::Display> MemoryMappedStats<F> {
    /// Create a new memory-mapped statistics processor
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process a memory-mapped file
    pub fn process_mmap_file<P: AsRef<Path>, G>(&self, path: P, processor: G) -> StatsResult<()>
    where
        G: FnOnce(&[F]) -> StatsResult<()>,
    {
        let file = File::open(path)
            .map_err(|e| StatsError::InvalidArgument(format!("Failed to open file: {e}")))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| StatsError::ComputationError(format!("Failed to mmap file: {}", e)))?
        };

        // Interpret memory-mapped data as array of floats
        let data = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr() as *const F,
                mmap.len() / std::mem::size_of::<F>(),
            )
        };

        processor(data)
    }
}
