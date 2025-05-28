//! Waterfall plot module for time-frequency analysis visualization
//!
//! This module provides functionality for creating waterfall plot data, which
//! represents how the frequency spectrum of a signal changes over time.
//! Waterfall plots are particularly useful for visualizing dynamic spectral
//! characteristics of signals.
//!
//! Unlike spectrograms (which typically show time on the x-axis, frequency on
//! the y-axis, and intensity as color), waterfall plots can be thought of as
//! a 3D representation with time, frequency, and amplitude/intensity axes.
//!
//! This module provides functions to generate the data needed for waterfall
//! plots in different formats and coordinate systems.

use crate::error::{FFTError, FFTResult};
use crate::spectrogram::{spectrogram, spectrogram_normalized};
use crate::window::Window;
use ndarray::{Array1, Array2, Array3};
use num_traits::NumCast;

/// Generate data for a 3D waterfall plot from a time-domain signal.
///
/// This function computes a spectrogram and organizes the data for a 3D
/// waterfall plot visualization, with time, frequency, and amplitude as axes.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `fs` - Sampling frequency of the signal (default: 1.0)
/// * `nperseg` - Length of each segment (default: 256)
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `log_scale` - Whether to use logarithmic (dB) scale for amplitudes (default: true)
/// * `db_range` - Dynamic range in dB for normalization when using log scale (default: 60.0)
///
/// # Returns
///
/// * A tuple containing:
///   - Time values array (t) - Times at the center of each segment
///   - Frequency values array (f) - Frequency bins
///   - Waterfall data array (Z) - 3D array with shape [time, frequency, 3] where the last dimension
///     contains [t, f, amplitude] coordinates for each point
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::waterfall_3d;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = t.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute waterfall plot data
/// let (times, freqs, data) = waterfall_3d(
///     &chirp,
///     Some(fs),
///     Some(128),
///     Some(64),
///     Some(true),
///     Some(80.0),
/// ).unwrap();
///
/// // Verify dimensions
/// assert_eq!(data.shape()[0], times.len());    // Time dimension
/// assert_eq!(data.shape()[1], freqs.len());    // Frequency dimension
/// assert_eq!(data.shape()[2], 3);              // X, Y, Z coordinates
/// ```
pub fn waterfall_3d<T>(
    x: &[T],
    fs: Option<f64>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    log_scale: Option<bool>,
    db_range: Option<f64>,
) -> FFTResult<(Array1<f64>, Array1<f64>, Array3<f64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    let nperseg = nperseg.unwrap_or(256);
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let log_scale = log_scale.unwrap_or(true);
    let db_range = db_range.unwrap_or(60.0);

    // Compute spectrogram
    let (freqs, times, spec_data) = if log_scale {
        // Use normalized spectrogram (log scale)
        spectrogram_normalized(x, Some(fs), Some(nperseg), Some(noverlap), Some(db_range))?
    } else {
        // Use linear amplitude spectrogram
        spectrogram(
            x,
            Some(fs),
            Some(Window::Hann),
            Some(nperseg),
            Some(noverlap),
            None,
            None,
            Some("density"),
            Some("magnitude"),
        )?
    };

    // Get dimensions
    let n_times = times.len();
    let n_freqs = freqs.len();

    // Create arrays for coordinates and copy data
    let mut waterfall_data = Array3::zeros((n_times, n_freqs, 3));

    // Fill the data array with [t, f, amplitude] coordinates
    for t_idx in 0..n_times {
        for f_idx in 0..n_freqs {
            let amplitude = spec_data[[f_idx, t_idx]];
            waterfall_data[[t_idx, f_idx, 0]] = times[t_idx]; // Time coordinate (x)
            waterfall_data[[t_idx, f_idx, 1]] = freqs[f_idx]; // Frequency coordinate (y)
            waterfall_data[[t_idx, f_idx, 2]] = amplitude; // Amplitude (z)
        }
    }

    // Convert to array1d for return
    let times_array = Array1::from_vec(times);
    let freqs_array = Array1::from_vec(freqs);

    Ok((times_array, freqs_array, waterfall_data))
}

/// Generate data for a mesh grid waterfall plot from a time-domain signal.
///
/// This function computes a spectrogram and organizes the data as separate
/// time, frequency, and amplitude arrays suitable for mesh or surface plotting.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `fs` - Sampling frequency of the signal (default: 1.0)
/// * `nperseg` - Length of each segment (default: 256)
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `log_scale` - Whether to use logarithmic (dB) scale for amplitudes (default: true)
/// * `db_range` - Dynamic range in dB for normalization when using log scale (default: 60.0)
///
/// # Returns
///
/// * A tuple containing:
///   - Time mesh grid (T) - 2D array of time values
///   - Frequency mesh grid (F) - 2D array of frequency values
///   - Amplitude data (Z) - 2D array of amplitude values
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::waterfall_mesh;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = t.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute waterfall mesh data
/// let (time_mesh, freq_mesh, amplitudes) = waterfall_mesh(
///     &chirp,
///     Some(fs),
///     Some(128),
///     Some(64),
///     Some(true),
///     Some(80.0),
/// ).unwrap();
///
/// // Verify dimensions are consistent
/// assert_eq!(time_mesh.shape(), freq_mesh.shape());
/// assert_eq!(time_mesh.shape(), amplitudes.shape());
/// ```
pub fn waterfall_mesh<T>(
    x: &[T],
    fs: Option<f64>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    log_scale: Option<bool>,
    db_range: Option<f64>,
) -> FFTResult<(Array2<f64>, Array2<f64>, Array2<f64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    let nperseg = nperseg.unwrap_or(256);
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let log_scale = log_scale.unwrap_or(true);
    let db_range = db_range.unwrap_or(60.0);

    // Compute spectrogram
    let (freqs, times, spec_data) = if log_scale {
        // Use normalized spectrogram (log scale)
        spectrogram_normalized(x, Some(fs), Some(nperseg), Some(noverlap), Some(db_range))?
    } else {
        // Use linear amplitude spectrogram
        spectrogram(
            x,
            Some(fs),
            Some(Window::Hann),
            Some(nperseg),
            Some(noverlap),
            None,
            None,
            Some("density"),
            Some("magnitude"),
        )?
    };

    // Create mesh grids for time and frequency
    // These are 2D arrays where each row/column has the same time/frequency value
    let n_times = times.len();
    let n_freqs = freqs.len();

    let mut time_mesh = Array2::zeros((n_freqs, n_times));
    let mut freq_mesh = Array2::zeros((n_freqs, n_times));

    // Fill the mesh grids
    for t_idx in 0..n_times {
        for f_idx in 0..n_freqs {
            time_mesh[[f_idx, t_idx]] = times[t_idx];
            freq_mesh[[f_idx, t_idx]] = freqs[f_idx];
        }
    }

    // Return the time mesh, frequency mesh, and amplitude data
    // Note: we transpose the amplitude data to match the mesh orientation
    Ok((time_mesh, freq_mesh, spec_data))
}

/// Generate data for a stacked line waterfall plot from a time-domain signal.
///
/// This function computes a spectrogram and organizes the data as a series of
/// line plots (spectra) at different time points, with optional offset between
/// lines to create a 3D stacking effect.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `fs` - Sampling frequency of the signal (default: 1.0)
/// * `nperseg` - Length of each segment (default: 256)
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `n_lines` - Number of spectral lines to include (default: 20, evenly spaced in time)
/// * `offset` - Vertical offset between consecutive lines (default: 0.1)
/// * `log_scale` - Whether to use logarithmic (dB) scale for amplitudes (default: true)
/// * `db_range` - Dynamic range in dB for normalization when using log scale (default: 60.0)
///
/// # Returns
///
/// * A tuple containing:
///   - Times array (t) - Times for each spectral line
///   - Frequencies array (f) - Frequency bins
///   - Stacked spectra data (Z) - 3D array where each [i, :, :] slice contains
///     the [frequency, amplitude] coordinates for the i-th time point
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::waterfall_lines;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = t.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute stacked line data
/// let (times, freqs, lines) = waterfall_lines(
///     &chirp,
///     Some(fs),
///     Some(128),
///     Some(64),
///     Some(10),    // 10 lines
///     Some(0.2),   // Offset between lines
///     Some(true),
///     Some(80.0),
/// ).unwrap();
///
/// // Each line contains frequency and amplitude data
/// assert_eq!(lines.shape()[0], times.len());      // Number of time points
/// assert_eq!(lines.shape()[1], freqs.len());      // Number of frequency points
/// assert_eq!(lines.shape()[2], 2);                // [frequency, amplitude] pairs
/// ```
#[allow(clippy::too_many_arguments)]
pub fn waterfall_lines<T>(
    x: &[T],
    fs: Option<f64>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    n_lines: Option<usize>,
    offset: Option<f64>,
    log_scale: Option<bool>,
    db_range: Option<f64>,
) -> FFTResult<(Array1<f64>, Array1<f64>, Array3<f64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    let nperseg = nperseg.unwrap_or(256);
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let log_scale = log_scale.unwrap_or(true);
    let db_range = db_range.unwrap_or(60.0);

    // Compute spectrogram
    let (freqs, times, spec_data) = if log_scale {
        // Use normalized spectrogram (log scale)
        spectrogram_normalized(x, Some(fs), Some(nperseg), Some(noverlap), Some(db_range))?
    } else {
        // Use linear amplitude spectrogram
        spectrogram(
            x,
            Some(fs),
            Some(Window::Hann),
            Some(nperseg),
            Some(noverlap),
            None,
            None,
            Some("density"),
            Some("magnitude"),
        )?
    };

    // Determine the time points to use
    let n_times = times.len();
    let n_lines = n_lines.unwrap_or(20).min(n_times);
    let offset = offset.unwrap_or(0.1);

    // Select evenly spaced time indices
    let time_indices: Vec<usize> = if n_lines >= n_times {
        // Use all time points if n_lines >= number of available times
        (0..n_times).collect()
    } else {
        // Select evenly spaced indices
        (0..n_lines)
            .map(|i| i * (n_times - 1) / (n_lines - 1))
            .collect()
    };

    // Extract the times for these indices
    let selected_times: Vec<f64> = time_indices.iter().map(|&idx| times[idx]).collect();

    // Create 3D array for line data
    let n_freqs = freqs.len();
    let mut line_data = Array3::zeros((n_lines, n_freqs, 2));

    // Fill the line data with [frequency, amplitude] pairs
    // Add increasing offsets to amplitude for 3D effect
    for (line_idx, &time_idx) in time_indices.iter().enumerate() {
        let line_offset = line_idx as f64 * offset;

        for f_idx in 0..n_freqs {
            // Original amplitude from spectrogram
            let amplitude = spec_data[[f_idx, time_idx]];

            // Store frequency and offset amplitude
            line_data[[line_idx, f_idx, 0]] = freqs[f_idx]; // Frequency
            line_data[[line_idx, f_idx, 1]] = amplitude + line_offset; // Amplitude with offset
        }
    }

    // Convert vectors to Array1 for return
    let times_array = Array1::from_vec(selected_times);
    let freqs_array = Array1::from_vec(freqs);

    Ok((times_array, freqs_array, line_data))
}

/// Generate coordinates for a color mesh waterfall plot from a time-domain signal.
///
/// This function computes a spectrogram and organizes the data as arrays of
/// vertices, faces, and colors suitable for triangulated surface plots.
///
/// # Arguments
///
/// * `x` - Input signal array
/// * `fs` - Sampling frequency of the signal (default: 1.0)
/// * `nperseg` - Length of each segment (default: 256)
/// * `noverlap` - Number of points to overlap between segments (default: `nperseg // 2`)
/// * `log_scale` - Whether to use logarithmic (dB) scale for amplitudes (default: true)
/// * `db_range` - Dynamic range in dB for normalization when using log scale (default: 60.0)
///
/// # Returns
///
/// * A tuple containing:
///   - Vertices array - Nx3 array of \[x,y,z\] coordinates for each vertex
///   - Faces array - Mx3 array of vertex indices defining triangular faces
///   - Colors array - Nx3 array of \[r,g,b\] colors for each vertex (normalized to \[0,1\])
///
/// # Errors
///
/// Returns an error if the computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::waterfall_mesh_colored;
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let fs = 1000.0; // 1 kHz sampling rate
/// let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
/// let chirp = t.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();
///
/// // Compute colored mesh data
/// let (vertices, faces, colors) = waterfall_mesh_colored(
///     &chirp,
///     Some(fs),
///     Some(128),
///     Some(64),
///     Some(true),
///     Some(80.0),
/// ).unwrap();
///
/// // Verify dimensions
/// assert_eq!(vertices.shape()[1], 3);  // [x,y,z] coordinates
/// assert_eq!(faces.shape()[1], 3);     // Triangle vertex indices
/// assert_eq!(colors.shape()[1], 3);    // [r,g,b] values
/// assert_eq!(vertices.shape()[0], colors.shape()[0]);  // Same number of vertices and colors
/// ```
pub fn waterfall_mesh_colored<T>(
    x: &[T],
    fs: Option<f64>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    log_scale: Option<bool>,
    db_range: Option<f64>,
) -> FFTResult<(Array2<f64>, Array2<usize>, Array2<f64>)>
where
    T: NumCast + Copy + std::fmt::Debug,
{
    // Set default parameters
    let fs = fs.unwrap_or(1.0);
    let nperseg = nperseg.unwrap_or(256);
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let log_scale = log_scale.unwrap_or(true);
    let db_range = db_range.unwrap_or(60.0);

    // Compute spectrogram
    let (freqs, times, spec_data) = if log_scale {
        // Use normalized spectrogram (log scale)
        spectrogram_normalized(x, Some(fs), Some(nperseg), Some(noverlap), Some(db_range))?
    } else {
        // Use linear amplitude spectrogram
        spectrogram(
            x,
            Some(fs),
            Some(Window::Hann),
            Some(nperseg),
            Some(noverlap),
            None,
            None,
            Some("density"),
            Some("magnitude"),
        )?
    };

    let n_times = times.len();
    let n_freqs = freqs.len();

    // Create vertices (time, frequency, amplitude coordinates)
    let n_vertices = n_times * n_freqs;
    let mut vertices = Array2::zeros((n_vertices, 3));
    let mut colors = Array2::zeros((n_vertices, 3));

    // Fill vertices and calculate colors
    let mut vertex_idx = 0;
    for (t_idx, &t) in times.iter().enumerate() {
        for (f_idx, &f) in freqs.iter().enumerate() {
            let amplitude = spec_data[[f_idx, t_idx]];

            // Set vertex coordinates
            vertices[[vertex_idx, 0]] = t; // Time (x coordinate)
            vertices[[vertex_idx, 1]] = f; // Frequency (y coordinate)
            vertices[[vertex_idx, 2]] = amplitude; // Amplitude (z coordinate)

            // Set vertex color using a simple blue -> red color map based on amplitude
            colors[[vertex_idx, 0]] = amplitude; // Red component
            colors[[vertex_idx, 1]] = amplitude * 0.5; // Green component
            colors[[vertex_idx, 2]] = 1.0 - amplitude; // Blue component

            vertex_idx += 1;
        }
    }

    // Create triangular faces
    // Each rectangular cell in the grid becomes two triangular faces
    let n_cells_t = n_times - 1;
    let n_cells_f = n_freqs - 1;
    let n_faces = 2 * n_cells_t * n_cells_f;
    let mut faces = Array2::zeros((n_faces, 3));

    let mut face_idx = 0;
    for t in 0..n_cells_t {
        for f in 0..n_cells_f {
            // Calculate the vertex indices for this cell
            let v00 = t * n_freqs + f; // Top-left
            let v01 = t * n_freqs + (f + 1); // Top-right
            let v10 = (t + 1) * n_freqs + f; // Bottom-left
            let v11 = (t + 1) * n_freqs + (f + 1); // Bottom-right

            // First triangle (top-left, bottom-left, bottom-right)
            faces[[face_idx, 0]] = v00;
            faces[[face_idx, 1]] = v10;
            faces[[face_idx, 2]] = v11;
            face_idx += 1;

            // Second triangle (top-left, bottom-right, top-right)
            faces[[face_idx, 0]] = v00;
            faces[[face_idx, 1]] = v11;
            faces[[face_idx, 2]] = v01;
            face_idx += 1;
        }
    }

    Ok((vertices, faces, colors))
}

/// Apply a colormap to amplitude values for visualization.
///
/// This function maps amplitude values to RGB colors using various
/// colormap schemes commonly used in scientific visualization.
///
/// # Arguments
///
/// * `amplitudes` - Array of amplitude values in range [0, 1]
/// * `colormap` - Name of the colormap to use (default: "jet")
///
/// # Returns
///
/// * RGB array where each row corresponds to an input amplitude
///
/// # Errors
///
/// Returns an error if the colormap name is not recognized.
///
/// # Examples
///
/// ```
/// use scirs2_fft::waterfall::apply_colormap;
/// use ndarray::Array1;
///
/// // Create some test amplitudes
/// let amplitudes = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
///
/// // Apply jet colormap
/// let colors = apply_colormap(&amplitudes, "jet").unwrap();
///
/// // Check output dimensions
/// assert_eq!(colors.shape(), &[5, 3]);  // 5 colors, each with RGB components
///
/// // Check that values are in valid range [0, 1]
/// for color in colors.outer_iter() {
///     for &component in color {
///         assert!(0.0 <= component && component <= 1.0);
///     }
/// }
/// ```
pub fn apply_colormap(amplitudes: &Array1<f64>, colormap: &str) -> FFTResult<Array2<f64>> {
    // Verify that amplitudes are in range [0, 1]
    for &amp in amplitudes.iter() {
        if !(0.0..=1.0).contains(&amp) {
            return Err(FFTError::ValueError(
                "Amplitude values must be in range [0, 1]".to_string(),
            ));
        }
    }

    // Create output array
    let n_values = amplitudes.len();
    let mut colors = Array2::zeros((n_values, 3));

    match colormap {
        "jet" => {
            // Jet colormap: blue -> cyan -> yellow -> red
            for (i, &amp) in amplitudes.iter().enumerate() {
                // Red component
                colors[[i, 0]] = if amp < 0.35 {
                    0.0
                } else if amp < 0.66 {
                    (amp - 0.35) / 0.31
                } else {
                    1.0
                };

                // Green component
                colors[[i, 1]] = if amp < 0.125 {
                    0.0
                } else if amp < 0.375 {
                    (amp - 0.125) / 0.25
                } else if amp < 0.875 {
                    1.0
                } else {
                    (1.0 - amp) / 0.125
                };

                // Blue component
                colors[[i, 2]] = if amp < 0.125 {
                    (0.5 + amp * 4.0).min(1.0)
                } else if amp < 0.375 {
                    1.0
                } else if amp < 0.625 {
                    (0.625 - amp) / 0.25
                } else {
                    0.0
                };
            }
        }
        "viridis" => {
            // Viridis colormap: dark blue -> green -> yellow
            for (i, &amp) in amplitudes.iter().enumerate() {
                // Simplified approximation of viridis
                colors[[i, 0]] = amp.powf(1.5) * 0.9; // Red increases nonlinearly
                colors[[i, 1]] = amp.powf(0.8) * 0.9; // Green increases faster
                colors[[i, 2]] = if amp < 0.5 {
                    0.4 + (0.5 - amp) * 0.6 // Blue decreases from start
                } else {
                    0.4 * (1.0 - amp) // Blue continues decreasing
                };
            }
        }
        "plasma" => {
            // Plasma colormap: dark purple -> red -> yellow
            for (i, &amp) in amplitudes.iter().enumerate() {
                colors[[i, 0]] = 0.05 + amp.powf(0.7) * 0.95; // Red increases quickly
                colors[[i, 1]] = amp.powf(2.0) * 0.9; // Green increases slowly then faster
                colors[[i, 2]] = if amp < 0.7 {
                    0.5 - amp * 0.5 // Blue decreases linearly at first
                } else {
                    0.15 * (1.0 - amp) / 0.3 // Blue continues decreasing
                };
            }
        }
        "grayscale" => {
            // Grayscale: black -> white
            for (i, &amp) in amplitudes.iter().enumerate() {
                colors[[i, 0]] = amp; // Red
                colors[[i, 1]] = amp; // Green
                colors[[i, 2]] = amp; // Blue
            }
        }
        "hot" => {
            // Hot colormap: black -> red -> yellow -> white
            for (i, &amp) in amplitudes.iter().enumerate() {
                colors[[i, 0]] = (amp * 3.0).min(1.0); // Red rises fastest
                colors[[i, 1]] = ((amp - 0.33) * 3.0).clamp(0.0, 1.0); // Green rises next
                colors[[i, 2]] = ((amp - 0.66) * 3.0).clamp(0.0, 1.0); // Blue rises last
            }
        }
        _ => {
            return Err(FFTError::ValueError(format!(
                "Unknown colormap: {}. Use 'jet', 'viridis', 'plasma', 'grayscale', or 'hot'.",
                colormap
            )));
        }
    }

    Ok(colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Generate a test signal (chirp: frequency increases with time)
    fn generate_chirp(fs: f64, duration: f64) -> Vec<f64> {
        let n_samples = (fs * duration) as usize;
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();

        t.iter()
            .map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin())
            .collect()
    }

    #[test]
    fn test_waterfall_3d_dimensions() {
        // Generate a test signal
        let fs = 1000.0;
        let signal = generate_chirp(fs, 1.0);

        // Compute waterfall plot data
        let (times, freqs, data) = waterfall_3d(
            &signal,
            Some(fs),
            Some(128),
            Some(64),
            Some(true),
            Some(60.0),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(data.shape()[0], times.len()); // Time dimension
        assert_eq!(data.shape()[1], freqs.len()); // Frequency dimension
        assert_eq!(data.shape()[2], 3); // X, Y, Z coordinates

        // Check ranges
        for t in times.iter() {
            assert!(0.0 <= *t && *t <= 1.0); // Time should be in [0, 1] range
        }

        for f in freqs.iter() {
            assert!(0.0 <= *f && *f <= fs / 2.0); // Frequency should be in [0, fs/2] range
        }

        // Check amplitudes are normalized
        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[1] {
                assert!(0.0 <= data[[i, j, 2]] && data[[i, j, 2]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_waterfall_mesh_dimensions() {
        // Generate a test signal
        let fs = 1000.0;
        let signal = generate_chirp(fs, 1.0);

        // Compute waterfall mesh data
        let (time_mesh, freq_mesh, amplitudes) = waterfall_mesh(
            &signal,
            Some(fs),
            Some(128),
            Some(64),
            Some(true),
            Some(60.0),
        )
        .unwrap();

        // Check that dimensions are consistent
        assert_eq!(time_mesh.shape(), freq_mesh.shape());
        assert_eq!(time_mesh.shape(), amplitudes.shape());

        // Check that values are in expected ranges
        assert!(0.0 <= time_mesh.iter().fold(f64::MAX, |a, &b| a.min(b)));
        assert!(time_mesh.iter().fold(f64::MIN, |a, &b| a.max(b)) <= 1.0);

        assert!(0.0 <= freq_mesh.iter().fold(f64::MAX, |a, &b| a.min(b)));
        assert!(freq_mesh.iter().fold(f64::MIN, |a, &b| a.max(b)) <= fs / 2.0);

        // Amplitudes should be normalized
        assert!(0.0 <= amplitudes.iter().fold(f64::MAX, |a, &b| a.min(b)));
        assert!(amplitudes.iter().fold(f64::MIN, |a, &b| a.max(b)) <= 1.0);
    }

    #[test]
    fn test_waterfall_lines() {
        // Generate a test signal
        let fs = 1000.0;
        let signal = generate_chirp(fs, 1.0);

        // Number of lines to use
        let n_lines = 10;

        // Compute stacked lines data
        let (times, freqs, lines) = waterfall_lines(
            &signal,
            Some(fs),
            Some(128),
            Some(64),
            Some(n_lines),
            Some(0.2),
            Some(true),
            Some(60.0),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(times.len(), n_lines);
        assert_eq!(lines.shape()[0], n_lines);
        assert_eq!(lines.shape()[1], freqs.len());
        assert_eq!(lines.shape()[2], 2);

        // Check that times are monotonically increasing
        for i in 1..times.len() {
            assert!(times[i] > times[i - 1]);
        }

        // Check that frequencies match
        for i in 0..n_lines {
            for j in 0..freqs.len() {
                assert_eq!(lines[[i, j, 0]], freqs[j]);
            }
        }

        // Check that successive lines have the correct offset
        let offset = 0.2;
        for i in 1..n_lines {
            let prev_line_max = (0..freqs.len())
                .map(|j| lines[[i - 1, j, 1]])
                .fold(f64::MIN, f64::max);

            let curr_line_min = (0..freqs.len())
                .map(|j| lines[[i, j, 1]])
                .fold(f64::MAX, f64::min);

            // The minimum of the current line should be at least the offset higher
            // than the base level (not necessarily the max) of the previous line
            assert!(curr_line_min > prev_line_max - 1.0 + offset);
        }
    }

    #[test]
    fn test_waterfall_mesh_colored() {
        // Generate a test signal
        let fs = 1000.0;
        let signal = generate_chirp(fs, 1.0);

        // Compute colored mesh data
        let (vertices, faces, colors) = waterfall_mesh_colored(
            &signal,
            Some(fs),
            Some(128),
            Some(64),
            Some(true),
            Some(60.0),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(vertices.shape()[1], 3); // [x,y,z] coordinates
        assert_eq!(faces.shape()[1], 3); // Triangle vertex indices
        assert_eq!(colors.shape()[1], 3); // [r,g,b] values
        assert_eq!(vertices.shape()[0], colors.shape()[0]); // Same number of vertices and colors

        // Check that vertex indices in faces are within range
        let max_vertex_idx = vertices.shape()[0] - 1;
        for face in faces.outer_iter() {
            for &idx in face {
                assert!(idx <= max_vertex_idx);
            }
        }

        // Check valid color range
        for color in colors.outer_iter() {
            for &component in color {
                assert!((0.0..=1.0).contains(&component));
            }
        }
    }

    #[test]
    fn test_apply_colormap() {
        // Create test amplitudes
        let amplitudes = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        // Test each colormap
        let colormaps = ["jet", "viridis", "plasma", "grayscale", "hot"];

        for &cmap in &colormaps {
            let colors = apply_colormap(&amplitudes, cmap).unwrap();

            // Check dimensions
            assert_eq!(colors.shape(), &[5, 3]);

            // Check valid range for colors (with small epsilon for floating-point precision)
            let epsilon = 1e-10;
            for row in colors.outer_iter() {
                for &component in row {
                    assert!(
                        -epsilon <= component && component <= 1.0 + epsilon,
                        "Color component out of range: {}",
                        component
                    );
                }
            }

            // For grayscale colormap, all RGB components should be equal
            if cmap == "grayscale" {
                for i in 0..amplitudes.len() {
                    assert_eq!(colors[[i, 0]], colors[[i, 1]]);
                    assert_eq!(colors[[i, 1]], colors[[i, 2]]);
                    assert_eq!(colors[[i, 0]], amplitudes[i]);
                }
            }
        }

        // Test invalid colormap name
        let result = apply_colormap(&amplitudes, "invalid");
        assert!(result.is_err());
    }
}
