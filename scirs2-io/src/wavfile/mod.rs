//! WAV file format handling module
//!
//! This module provides functionality for reading and writing WAV audio files.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use scirs2_core::ndarray::ArrayD;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// WAV audio format codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavFormat {
    /// PCM format (1)
    Pcm = 1,
    /// IEEE float format (3)
    Float = 3,
    /// A-law format (6)
    Alaw = 6,
    /// μ-law format (7)
    Mulaw = 7,
}

impl TryFrom<u16> for WavFormat {
    type Error = IoError;

    fn try_from(value: u16) -> std::result::Result<Self, Self::Error> {
        match value {
            1 => Ok(WavFormat::Pcm),
            3 => Ok(WavFormat::Float),
            6 => Ok(WavFormat::Alaw),
            7 => Ok(WavFormat::Mulaw),
            _ => Err(IoError::FormatError(format!(
                "Unknown WAV format code: {}",
                value
            ))),
        }
    }
}

/// WAV file header information
#[derive(Debug, Clone)]
pub struct WavHeader {
    /// WAV format type
    pub format: WavFormat,
    /// Number of channels
    pub channels: u16,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Total number of samples per channel
    pub samples_per_channel: usize,
}

/// RIFF chunk type
#[derive(Debug, Clone, PartialEq, Eq)]
struct RiffChunk {
    /// Chunk ID (4 bytes)
    id: [u8; 4],
    /// Chunk size
    size: u32,
}

impl RiffChunk {
    /// Read a RIFF chunk from a reader
    fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut id = [0u8; 4];
        reader.read_exact(&mut id)?;
        let size = reader.read_u32::<LittleEndian>()?;
        Ok(RiffChunk { id, size })
    }

    /// Write a RIFF chunk to a writer
    fn _write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.id)?;
        writer.write_u32::<LittleEndian>(self.size)?;
        Ok(())
    }

    /// Check if the chunk ID matches a given string
    fn is_id(&self, id: &str) -> bool {
        id.as_bytes() == self.id
    }
}

/// Reads a WAV file
///
/// # Arguments
///
/// * `path` - Path to the WAV file
///
/// # Returns
///
/// * A tuple containing the WAV header and the audio data as a 2D array
///   where the first dimension represents channels and the second dimension represents samples
///
/// # Example
///
/// ```no_run
/// use scirs2_io::wavfile::read_wav;
/// use std::path::Path;
///
/// let (header, data) = read_wav(Path::new("audio.wav")).unwrap();
/// println!("Sample rate: {}", header.sample_rate);
/// println!("Channels: {}", header.channels);
/// println!("Samples per channel: {}", header.samples_per_channel);
/// ```
#[allow(dead_code)]
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<(WavHeader, ArrayD<f32>)> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    // Read RIFF chunk
    let riff_chunk = RiffChunk::read(&mut reader)
        .map_err(|e| IoError::FormatError(format!("Failed to read RIFF chunk: {}", e)))?;

    if !riff_chunk.is_id("RIFF") {
        return Err(IoError::FormatError("Not a RIFF file".to_string()));
    }

    // Read format (should be "WAVE")
    let mut format = [0u8; 4];
    reader
        .read_exact(&mut format)
        .map_err(|e| IoError::FormatError(format!("Failed to read WAVE format: {}", e)))?;

    if format != *b"WAVE" {
        return Err(IoError::FormatError("Not a WAVE file".to_string()));
    }

    // Read fmt chunk
    let fmt_chunk = RiffChunk::read(&mut reader)
        .map_err(|e| IoError::FormatError(format!("Failed to read fmt chunk: {}", e)))?;

    if !fmt_chunk.is_id("fmt ") {
        return Err(IoError::FormatError("Expected fmt chunk".to_string()));
    }

    // Read format data
    let audio_format = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read audio format: {}", e)))?;
    let format = WavFormat::try_from(audio_format)?;

    let channels = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read channel count: {}", e)))?;

    let sample_rate = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read sample rate: {}", e)))?;

    let _byte_rate = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read byte rate: {}", e)))?;

    let _block_align = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read block align: {}", e)))?;

    let bits_per_sample = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read bits per sample: {}", e)))?;

    // Skip any extra fmt data
    if fmt_chunk.size > 16 {
        let extra_bytes = fmt_chunk.size as usize - 16;
        let mut extra_data = vec![0u8; extra_bytes];
        reader
            .read_exact(&mut extra_data)
            .map_err(|e| IoError::FormatError(format!("Failed to read extra fmt data: {}", e)))?;
    }

    // Find data chunk
    let data_size;
    loop {
        let chunk = RiffChunk::read(&mut reader)
            .map_err(|e| IoError::FormatError(format!("Failed to read chunk: {}", e)))?;

        if chunk.is_id("data") {
            data_size = chunk.size;
            break;
        }

        // Skip this chunk
        reader
            .seek(SeekFrom::Current(chunk.size as i64))
            .map_err(|e| IoError::FormatError(format!("Failed to skip chunk: {}", e)))?;
    }

    // Calculate number of samples
    let bytes_per_sample = bits_per_sample / 8;
    let samples_per_channel = (data_size / (channels as u32 * bytes_per_sample as u32)) as usize;

    // We'll create the header at the end to return it

    // Read audio data
    let shape = scirs2_core::ndarray::IxDyn(&[channels as usize, samples_per_channel]);
    let mut data: scirs2_core::ndarray::ArrayD<f32> = scirs2_core::ndarray::Array::zeros(shape);

    // Read samples based on bits per sample
    match bits_per_sample {
        8 => {
            // 8-bit samples are unsigned
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let byte = reader.read_u8().map_err(|e| {
                        IoError::FormatError(format!("Failed to read 8-bit sample: {}", e))
                    })?;
                    // Convert from 0-255 to -1.0 to 1.0
                    data[[ch, sample_idx]] = (byte as f32 - 128.0) / 127.0;
                }
            }
        }
        16 => {
            // 16-bit samples are signed
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = reader.read_i16::<LittleEndian>().map_err(|e| {
                        IoError::FormatError(format!("Failed to read 16-bit sample: {}", e))
                    })?;
                    // Convert from -32768 to 32767 to -1.0 to 1.0
                    data[[ch, sample_idx]] = sample as f32 / 32767.0;
                }
            }
        }
        24 => {
            // 24-bit samples require special handling
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let mut bytes = [0u8; 3];
                    reader.read_exact(&mut bytes).map_err(|e| {
                        IoError::FormatError(format!("Failed to read 24-bit sample: {}", e))
                    })?;
                    // Convert 24-bit signed to i32
                    let sample = if bytes[2] & 0x80 != 0 {
                        // Sign extend for negative values
                        ((bytes[2] as i32) << 16)
                            | ((bytes[1] as i32) << 8)
                            | (bytes[0] as i32)
                            | 0xFF000000u32 as i32
                    } else {
                        ((bytes[2] as i32) << 16) | ((bytes[1] as i32) << 8) | (bytes[0] as i32)
                    };
                    // Convert to -1.0 to 1.0
                    data[[ch, sample_idx]] = sample as f32 / 8388607.0;
                }
            }
        }
        32 => {
            // 32-bit samples can be int or float
            if format == WavFormat::Float {
                // IEEE float format
                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels as usize {
                        let sample = reader.read_f32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!(
                                "Failed to read 32-bit float sample: {}",
                                e
                            ))
                        })?;
                        data[[ch, sample_idx]] = sample;
                    }
                }
            } else {
                // 32-bit integer
                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels as usize {
                        let sample = reader.read_i32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read 32-bit sample: {}", e))
                        })?;
                        // Convert to -1.0 to 1.0
                        data[[ch, sample_idx]] = sample as f32 / 2147483647.0;
                    }
                }
            }
        }
        _ => {
            return Err(IoError::FormatError(format!(
                "Unsupported bits per sample: {}",
                bits_per_sample
            )));
        }
    }

    let header = WavHeader {
        format,
        channels,
        sample_rate,
        bits_per_sample,
        samples_per_channel,
    };

    Ok((header, data))
}

/// Writes audio data to a WAV file
///
/// # Arguments
///
/// * `path` - Path where the WAV file should be written
/// * `sample_rate` - Sample rate in Hz
/// * `data` - Audio data where the first dimension represents channels and the second dimension represents samples
///
/// # Example
///
/// ```no_run
/// use scirs2_core::ndarray::Array2;
/// use scirs2_io::wavfile::write_wav;
/// use std::path::Path;
///
/// // Create a simple sine wave
/// let sample_rate = 44100;
/// let duration = 1.0; // seconds
/// let frequency = 440.0; // Hz (A4 note)
/// let num_samples = (sample_rate as f64 * duration) as usize;
///
/// let mut samples = Array2::zeros((1, num_samples)); // mono audio
/// for i in 0..num_samples {
///     let t = i as f64 / sample_rate as f64;
///     samples[[0, i]] = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32;
/// }
///
/// write_wav(Path::new("sine_wave.wav"), sample_rate, &samples.into_dyn()).unwrap();
/// ```
#[allow(dead_code)]
pub fn write_wav<P: AsRef<Path>>(path: P, samplerate: u32, data: &ArrayD<f32>) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Check that data is at least 2D (channels, samples)
    if data.ndim() < 2 {
        return Err(IoError::FormatError(
            "Audio data must be at least 2D (channels, samples)".to_string(),
        ));
    }

    let shape = data.shape();
    let channels = shape[0] as u16;
    let samples_per_channel = shape[1];

    // Set up WAV parameters
    let bits_per_sample: u16 = 32; // 32-bit float
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels * bytes_per_sample;
    let byte_rate = samplerate * block_align as u32;

    // Calculate total data size
    let data_size = samples_per_channel * channels as usize * bytes_per_sample as usize;

    // Write RIFF header
    writer
        .write_all(b"RIFF")
        .map_err(|e| IoError::FileError(format!("Failed to write RIFF header: {}", e)))?;

    // File size = 36 (header) + data size
    writer
        .write_u32::<LittleEndian>(36 + data_size as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write file size: {}", e)))?;

    writer
        .write_all(b"WAVE")
        .map_err(|e| IoError::FileError(format!("Failed to write WAVE header: {}", e)))?;

    // Write fmt chunk
    writer
        .write_all(b"fmt ")
        .map_err(|e| IoError::FileError(format!("Failed to write fmt chunk: {}", e)))?;

    writer.write_u32::<LittleEndian>(16) // fmt chunk size
        .map_err(|e| IoError::FileError(format!("Failed to write fmt chunk size: {}", e)))?;

    writer.write_u16::<LittleEndian>(3) // format = 3 (IEEE float)
        .map_err(|e| IoError::FileError(format!("Failed to write audio format: {}", e)))?;

    writer
        .write_u16::<LittleEndian>(channels)
        .map_err(|e| IoError::FileError(format!("Failed to write channel count: {}", e)))?;

    writer
        .write_u32::<LittleEndian>(samplerate)
        .map_err(|e| IoError::FileError(format!("Failed to write sample rate: {}", e)))?;

    writer
        .write_u32::<LittleEndian>(byte_rate)
        .map_err(|e| IoError::FileError(format!("Failed to write byte rate: {}", e)))?;

    writer
        .write_u16::<LittleEndian>(block_align)
        .map_err(|e| IoError::FileError(format!("Failed to write block align: {}", e)))?;

    writer
        .write_u16::<LittleEndian>(bits_per_sample)
        .map_err(|e| IoError::FileError(format!("Failed to write bits per sample: {}", e)))?;

    // Write data chunk
    writer
        .write_all(b"data")
        .map_err(|e| IoError::FileError(format!("Failed to write data chunk: {}", e)))?;

    writer
        .write_u32::<LittleEndian>(data_size as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write data size: {}", e)))?;

    // Write actual audio samples
    // The data is expected to be in format [channels, samples]
    for sample_idx in 0..samples_per_channel {
        for ch in 0..channels as usize {
            let sample = data[[ch, sample_idx]];
            writer
                .write_f32::<LittleEndian>(sample)
                .map_err(|e| IoError::FileError(format!("Failed to write sample: {}", e)))?;
        }
    }

    // Flush the writer to ensure all data is written
    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Failed to flush writer: {}", e)))?;

    Ok(())
}
