//! Positional encoding implementations for transformer sequences

use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};
use crate::error::Result;

/// Positional encoding types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (original Transformer)
    Sinusoidal,
    /// Learned positional encoding
    Learned,
    /// Rotary Position Embedding (RoPE)
    Rotary,
    /// No positional encoding
    None,
}

/// Positional encoding implementation
pub struct PositionalEncoding<T: Float> {
    /// Type of positional encoding
    encoding_type: PositionalEncodingType,

    /// Maximum sequence length
    max_sequence_length: usize,

    /// Model dimension
    model_dimension: usize,

    /// Precomputed encoding matrix
    encoding_matrix: Array2<T>,

    /// Learned parameters (for learned encoding)
    learned_embeddings: Option<Array2<T>>,

    /// RoPE frequency base
    rope_base: T,
}

impl<T: Float> PositionalEncoding<T> {
    /// Create new positional encoding
    pub fn new(
        max_sequence_length: usize,
        model_dimension: usize,
        encoding_type: PositionalEncodingType,
    ) -> Result<Self> {
        let rope_base = T::from(10000.0).unwrap();
        let mut encoding = Self {
            encoding_type,
            max_sequence_length,
            model_dimension,
            encoding_matrix: Array2::zeros((max_sequence_length, model_dimension)),
            learned_embeddings: None,
            rope_base,
        };

        encoding.initialize_encoding()?;
        Ok(encoding)
    }

    /// Initialize the encoding based on type
    fn initialize_encoding(&mut self) -> Result<()> {
        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => self.initialize_sinusoidal(),
            PositionalEncodingType::Learned => self.initialize_learned(),
            PositionalEncodingType::Rotary => self.initialize_rotary(),
            PositionalEncodingType::None => Ok(()),
        }
    }

    /// Initialize sinusoidal positional encoding
    fn initialize_sinusoidal(&mut self) -> Result<()> {
        for pos in 0..self.max_sequence_length {
            for i in 0..self.model_dimension {
                let position = T::from(pos).unwrap();
                let dimension = T::from(i).unwrap();
                let model_dim = T::from(self.model_dimension).unwrap();

                let angle = position / T::from(10000.0).unwrap().powf(
                    T::from(2.0).unwrap() * dimension / model_dim
                );

                if i % 2 == 0 {
                    // Even dimensions: sin
                    self.encoding_matrix[[pos, i]] = angle.sin();
                } else {
                    // Odd dimensions: cos
                    self.encoding_matrix[[pos, i]] = angle.cos();
                }
            }
        }
        Ok(())
    }

    /// Initialize learned positional encoding
    fn initialize_learned(&mut self) -> Result<()> {
        let learned_embeddings = Array2::zeros((self.max_sequence_length, self.model_dimension));
        self.learned_embeddings = Some(learned_embeddings);
        Ok(())
    }

    /// Initialize rotary positional encoding
    fn initialize_rotary(&mut self) -> Result<()> {
        // RoPE doesn't use a precomputed matrix, it's applied during attention
        // We'll store frequency inverse for each dimension pair
        for i in (0..self.model_dimension).step_by(2) {
            let dim_pair = T::from(i).unwrap() / T::from(self.model_dimension).unwrap();
            let freq = T::one() / self.rope_base.powf(dim_pair);

            if i < self.model_dimension {
                self.encoding_matrix[[0, i]] = freq;
            }
            if i + 1 < self.model_dimension {
                self.encoding_matrix[[0, i + 1]] = freq;
            }
        }
        Ok(())
    }

    /// Apply positional encoding to input
    pub fn encode(&self, input: &Array2<T>) -> Result<Array2<T>> {
        match self.encoding_type {
            PositionalEncodingType::None => Ok(input.clone()),
            PositionalEncodingType::Sinusoidal => self.apply_sinusoidal(input),
            PositionalEncodingType::Learned => self.apply_learned(input),
            PositionalEncodingType::Rotary => self.apply_rotary(input),
        }
    }

    /// Apply sinusoidal encoding
    fn apply_sinusoidal(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let batch_size = input.shape()[0];
        let sequence_length = input.shape()[1];

        if sequence_length > self.max_sequence_length {
            return Err(crate::error::OptimError::Other(
                "Sequence length exceeds maximum".to_string()
            ));
        }

        let mut output = input.clone();

        for batch in 0..batch_size {
            for pos in 0..sequence_length {
                for dim in 0..self.model_dimension {
                    output[[batch, pos]] = output[[batch, pos]] + self.encoding_matrix[[pos, dim]];
                }
            }
        }

        Ok(output)
    }

    /// Apply learned encoding
    fn apply_learned(&self, input: &Array2<T>) -> Result<Array2<T>> {
        if let Some(ref learned) = self.learned_embeddings {
            let batch_size = input.shape()[0];
            let sequence_length = input.shape()[1];

            if sequence_length > self.max_sequence_length {
                return Err(crate::error::OptimError::Other(
                    "Sequence length exceeds maximum".to_string()
                ));
            }

            let mut output = input.clone();

            for batch in 0..batch_size {
                for pos in 0..sequence_length {
                    for dim in 0..self.model_dimension {
                        output[[batch, pos]] = output[[batch, pos]] + learned[[pos, dim]];
                    }
                }
            }

            Ok(output)
        } else {
            Err(crate::error::OptimError::Other(
                "Learned embeddings not initialized".to_string()
            ))
        }
    }

    /// Apply rotary encoding (simplified implementation)
    fn apply_rotary(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let batch_size = input.shape()[0];
        let sequence_length = input.shape()[1];
        let mut output = input.clone();

        for batch in 0..batch_size {
            for pos in 0..sequence_length {
                let position = T::from(pos).unwrap();

                for i in (0..self.model_dimension).step_by(2) {
                    if i + 1 < self.model_dimension {
                        let freq = self.encoding_matrix[[0, i]];
                        let angle = position * freq;

                        let cos_val = angle.cos();
                        let sin_val = angle.sin();

                        let x = input[[batch, pos]]; // Simplified: treating as single value
                        let y = if i + 1 < input.shape()[1] {
                            input[[batch, pos]]
                        } else {
                            T::zero()
                        };

                        output[[batch, pos]] = x * cos_val - y * sin_val;
                        // For complete RoPE, we'd need to handle the paired dimension
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get encoding for specific position
    pub fn get_position_encoding(&self, position: usize) -> Result<Array1<T>> {
        if position >= self.max_sequence_length {
            return Err(crate::error::OptimError::Other(
                "Position exceeds maximum sequence length".to_string()
            ));
        }

        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => {
                Ok(self.encoding_matrix.row(position).to_owned())
            }
            PositionalEncodingType::Learned => {
                if let Some(ref learned) = self.learned_embeddings {
                    Ok(learned.row(position).to_owned())
                } else {
                    Err(crate::error::OptimError::Other(
                        "Learned embeddings not available".to_string()
                    ))
                }
            }
            PositionalEncodingType::Rotary => {
                // Return frequency information for RoPE
                Ok(self.encoding_matrix.row(0).to_owned())
            }
            PositionalEncodingType::None => {
                Ok(Array1::zeros(self.model_dimension))
            }
        }
    }

    /// Update learned embeddings (for training)
    pub fn update_learned_embeddings(&mut self, gradients: &Array2<T>) -> Result<()> {
        if let Some(ref mut learned) = self.learned_embeddings {
            *learned = &*learned - gradients;
            Ok(())
        } else {
            Err(crate::error::OptimError::Other(
                "No learned embeddings to update".to_string()
            ))
        }
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        match self.encoding_type {
            PositionalEncodingType::Learned => self.max_sequence_length * self.model_dimension,
            _ => 0, // Sinusoidal, RoPE, and None have no learnable parameters
        }
    }

    /// Reset parameters
    pub fn reset(&mut self) -> Result<()> {
        self.initialize_encoding()
    }

    /// Get encoding type
    pub fn get_encoding_type(&self) -> PositionalEncodingType {
        self.encoding_type
    }

    /// Get maximum sequence length
    pub fn get_max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    /// Get model dimension
    pub fn get_model_dimension(&self) -> usize {
        self.model_dimension
    }

    /// Create sinusoidal encoding with custom base
    pub fn sinusoidal_with_base(
        max_sequence_length: usize,
        model_dimension: usize,
        base: T,
    ) -> Result<Self> {
        let mut encoding = Self::new(
            max_sequence_length,
            model_dimension,
            PositionalEncodingType::Sinusoidal,
        )?;

        // Reinitialize with custom base
        for pos in 0..max_sequence_length {
            for i in 0..model_dimension {
                let position = T::from(pos).unwrap();
                let dimension = T::from(i).unwrap();
                let model_dim = T::from(model_dimension).unwrap();

                let angle = position / base.powf(
                    T::from(2.0).unwrap() * dimension / model_dim
                );

                if i % 2 == 0 {
                    encoding.encoding_matrix[[pos, i]] = angle.sin();
                } else {
                    encoding.encoding_matrix[[pos, i]] = angle.cos();
                }
            }
        }

        Ok(encoding)
    }

    /// Create rotary encoding with custom base
    pub fn rotary_with_base(
        max_sequence_length: usize,
        model_dimension: usize,
        base: T,
    ) -> Result<Self> {
        let mut encoding = Self::new(
            max_sequence_length,
            model_dimension,
            PositionalEncodingType::Rotary,
        )?;

        encoding.rope_base = base;
        encoding.initialize_rotary()?;

        Ok(encoding)
    }
}

/// Relative positional encoding for local attention patterns
pub struct RelativePositionalEncoding<T: Float> {
    /// Maximum relative distance
    max_relative_distance: usize,

    /// Model dimension
    model_dimension: usize,

    /// Relative encoding table
    relative_encoding: Array2<T>,
}

impl<T: Float> RelativePositionalEncoding<T> {
    /// Create new relative positional encoding
    pub fn new(max_relative_distance: usize, model_dimension: usize) -> Result<Self> {
        let table_size = 2 * max_relative_distance + 1;
        let relative_encoding = Array2::zeros((table_size, model_dimension));

        Ok(Self {
            max_relative_distance,
            model_dimension,
            relative_encoding,
        })
    }

    /// Get relative encoding between two positions
    pub fn get_relative_encoding(&self, from_pos: usize, to_pos: usize) -> Array1<T> {
        let relative_distance = (to_pos as i32 - from_pos as i32)
            .max(-(self.max_relative_distance as i32))
            .min(self.max_relative_distance as i32);

        let index = (relative_distance + self.max_relative_distance as i32) as usize;
        self.relative_encoding.row(index).to_owned()
    }

    /// Initialize with sinusoidal patterns
    pub fn initialize_sinusoidal(&mut self) -> Result<()> {
        let table_size = 2 * self.max_relative_distance + 1;

        for i in 0..table_size {
            let relative_pos = i as i32 - self.max_relative_distance as i32;
            let position = T::from(relative_pos).unwrap();

            for j in 0..self.model_dimension {
                let dimension = T::from(j).unwrap();
                let model_dim = T::from(self.model_dimension).unwrap();

                let angle = position / T::from(10000.0).unwrap().powf(
                    T::from(2.0).unwrap() * dimension / model_dim
                );

                if j % 2 == 0 {
                    self.relative_encoding[[i, j]] = angle.sin();
                } else {
                    self.relative_encoding[[i, j]] = angle.cos();
                }
            }
        }

        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        (2 * self.max_relative_distance + 1) * self.model_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_encoding() {
        let encoding = PositionalEncoding::<f32>::new(100, 64, PositionalEncodingType::Sinusoidal);
        assert!(encoding.is_ok());

        let pe = encoding.unwrap();
        assert_eq!(pe.parameter_count(), 0);

        let input = Array2::<f32>::zeros((2, 50));
        let result = pe.encode(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_learned_encoding() {
        let encoding = PositionalEncoding::<f32>::new(100, 64, PositionalEncodingType::Learned);
        assert!(encoding.is_ok());

        let pe = encoding.unwrap();
        assert_eq!(pe.parameter_count(), 100 * 64);
    }

    #[test]
    fn test_rotary_encoding() {
        let encoding = PositionalEncoding::<f32>::new(100, 64, PositionalEncodingType::Rotary);
        assert!(encoding.is_ok());

        let pe = encoding.unwrap();
        let input = Array2::<f32>::ones((2, 50));
        let result = pe.encode(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_position_encoding_retrieval() {
        let pe = PositionalEncoding::<f32>::new(100, 64, PositionalEncodingType::Sinusoidal).unwrap();

        let pos_encoding = pe.get_position_encoding(10);
        assert!(pos_encoding.is_ok());

        let encoding = pos_encoding.unwrap();
        assert_eq!(encoding.len(), 64);
    }

    #[test]
    fn test_relative_positional_encoding() {
        let mut rel_pe = RelativePositionalEncoding::<f32>::new(10, 64);
        assert!(rel_pe.is_ok());

        let mut rpe = rel_pe.unwrap();
        assert!(rpe.initialize_sinusoidal().is_ok());

        let encoding = rpe.get_relative_encoding(5, 8);
        assert_eq!(encoding.len(), 64);
    }

    #[test]
    fn test_encoding_types() {
        let types = [
            PositionalEncodingType::Sinusoidal,
            PositionalEncodingType::Learned,
            PositionalEncodingType::Rotary,
            PositionalEncodingType::None,
        ];

        for encoding_type in types.iter() {
            let pe = PositionalEncoding::<f32>::new(50, 32, *encoding_type);
            assert!(pe.is_ok());
        }
    }
}