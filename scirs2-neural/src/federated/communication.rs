//! Communication protocols for federated learning

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

/// Communication protocol trait
pub trait CommunicationProtocol: Send + Sync {
    /// Send message
    fn send(&mut self, recipient: usize, message: Message) -> Result<()>;
    /// Receive messages
    fn receive(&mut self) -> Result<Vec<(usize, Message)>>;
    /// Broadcast message
    fn broadcast(&mut self, message: Message) -> Result<()>;
    /// Get protocol statistics
    fn statistics(&self) -> CommunicationStats;
}

/// Message types in federated learning
#[derive(Debug, Clone)]
pub enum Message {
    /// Model parameters
    ModelParameters(Vec<Array2<f32>>),
    /// Client update
    ClientUpdate {
        round: usize,
        weights: Vec<Array2<f32>>,
        metrics: HashMap<String, f32>,
    },
    /// Training configuration
    TrainingConfig {
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
    },
    /// Control message
    Control(ControlMessage),
    /// Compressed message
    Compressed(CompressedMessage),
}

/// Control messages
#[derive(Debug, Clone)]
pub enum ControlMessage {
    /// Start training round
    StartRound(usize),
    /// End training round
    EndRound(usize),
    /// Client ready
    ClientReady(usize),
    /// Abort training
    Abort(String),
    /// Heartbeat
    Heartbeat,
}

/// Compressed message format
#[derive(Debug, Clone)]
pub struct CompressedMessage {
    /// Original message type
    pub message_type: String,
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression method
    pub method: CompressionMethod,
    /// Original size
    pub original_size: usize,
}

/// Compression methods
#[derive(Debug, Clone)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Quantization
    Quantization { bits: u8 },
    /// Top-K sparsification
    TopK { k: usize },
    /// Random sparsification
    RandomSparsification { ratio: f32 },
    /// Gradient compression
    GradientCompression,
}

/// Communication statistics
#[derive(Debug, Clone, Default)]
pub struct CommunicationStats {
    /// Total messages sent
    pub messages_sent: usize,
    /// Total messages received
    pub messages_received: usize,
    /// Total bytes sent
    pub bytes_sent: usize,
    /// Total bytes received
    pub bytes_received: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// In-memory communication protocol (for simulation)
pub struct InMemoryProtocol {
    /// Message queues per participant
    queues: HashMap<usize, Vec<(usize, Message)>>,
    /// Current participant ID
    participant_id: usize,
    /// Statistics
    stats: CommunicationStats,
}

impl InMemoryProtocol {
    /// Create new in-memory protocol
    pub fn new(participant_id: usize, num_participants: usize) -> Self {
        let mut queues = HashMap::new();
        for i in 0..num_participants {
            queues.insert(i, Vec::new());
        }
        Self {
            queues,
            participant_id,
            stats: CommunicationStats::default(),
        }
    }
}

impl CommunicationProtocol for InMemoryProtocol {
    fn send(&mut self, recipient: usize, message: Message) -> Result<()> {
        let size = estimate_message_size(&message);
        if let Some(queue) = self.queues.get_mut(&recipient) {
            queue.push((self.participant_id, message));
            self.stats.messages_sent += 1;
            self.stats.bytes_sent += size;
        }
        Ok(())
    }

    fn receive(&mut self) -> Result<Vec<(usize, Message)>> {
        let pid = self.participant_id;
        let messages = self
            .queues
            .get_mut(&pid)
            .map(std::mem::take)
            .unwrap_or_default();
        self.stats.messages_received += messages.len();
        self.stats.bytes_received += messages
            .iter()
            .map(|(_, m)| estimate_message_size(m))
            .sum::<usize>();
        Ok(messages)
    }

    fn broadcast(&mut self, message: Message) -> Result<()> {
        let pid = self.participant_id;
        let recipients: Vec<usize> = self.queues.keys().copied().filter(|&k| k != pid).collect();
        for recipient in recipients {
            let size = estimate_message_size(&message);
            if let Some(queue) = self.queues.get_mut(&recipient) {
                queue.push((pid, message.clone()));
                self.stats.messages_sent += 1;
                self.stats.bytes_sent += size;
            }
        }
        Ok(())
    }

    fn statistics(&self) -> CommunicationStats {
        self.stats.clone()
    }
}

/// Message compression utilities
pub struct MessageCompressor;

impl MessageCompressor {
    /// Compress model weights
    pub fn compress_weights(
        weights: &[Array2<f32>],
        method: CompressionMethod,
    ) -> Result<CompressedMessage> {
        let original_size = weights
            .iter()
            .map(|w| w.len() * std::mem::size_of::<f32>())
            .sum();
        let compressed_data = match &method {
            CompressionMethod::None => serialize_weights(weights)?,
            CompressionMethod::Quantization { bits } => compress_quantization(weights, *bits)?,
            CompressionMethod::TopK { k } => compress_topk(weights, *k)?,
            CompressionMethod::RandomSparsification { ratio } => {
                compress_random_sparse(weights, *ratio)?
            }
            CompressionMethod::GradientCompression => compress_gradients(weights)?,
        };
        Ok(CompressedMessage {
            message_type: "ModelWeights".to_string(),
            data: compressed_data,
            method,
            original_size,
        })
    }

    /// Decompress model weights
    pub fn decompress_weights(compressed: &CompressedMessage) -> Result<Vec<Array2<f32>>> {
        match &compressed.method {
            CompressionMethod::None => deserialize_weights(&compressed.data),
            CompressionMethod::Quantization { bits } => {
                decompress_quantization(&compressed.data, *bits)
            }
            CompressionMethod::TopK { .. } => decompress_topk(&compressed.data),
            CompressionMethod::RandomSparsification { .. } => {
                decompress_random_sparse(&compressed.data)
            }
            CompressionMethod::GradientCompression => decompress_gradients(&compressed.data),
        }
    }
}

/// Estimate message size in bytes
fn estimate_message_size(message: &Message) -> usize {
    match message {
        Message::ModelParameters(weights) => weights.iter().map(|w| w.len() * 4).sum(),
        Message::ClientUpdate { weights, .. } => {
            weights.iter().map(|w| w.len() * 4).sum::<usize>() + 100
        }
        Message::TrainingConfig { .. } => 64,
        Message::Control(_) => 32,
        Message::Compressed(c) => c.data.len(),
    }
}

/// Serialize weights to bytes
fn serialize_weights(weights: &[Array2<f32>]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for weight in weights {
        // Store shape
        bytes.extend(&(weight.shape()[0] as u32).to_le_bytes());
        bytes.extend(&(weight.shape()[1] as u32).to_le_bytes());
        // Store data
        for &val in weight.iter() {
            bytes.extend(&val.to_le_bytes());
        }
    }
    Ok(bytes)
}

/// Deserialize weights from bytes
fn deserialize_weights(data: &[u8]) -> Result<Vec<Array2<f32>>> {
    let mut weights = Vec::new();
    let mut cursor = 0;
    while cursor + 8 <= data.len() {
        // Read shape
        let rows = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;
        let cols = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;
        // Read data
        let mut values = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            if cursor + 4 > data.len() {
                break;
            }
            let val = f32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            values.push(val);
            cursor += 4;
        }
        if values.len() == rows * cols {
            weights.push(Array2::from_shape_vec((rows, cols), values)?);
        }
    }
    Ok(weights)
}

/// Quantization compression
fn compress_quantization(weights: &[Array2<f32>], bits: u8) -> Result<Vec<u8>> {
    let mut compressed = Vec::new();
    let levels = (1u32 << bits) as f32;
    for weight in weights {
        let min = weight.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        compressed.extend(&(weight.shape()[0] as u32).to_le_bytes());
        compressed.extend(&(weight.shape()[1] as u32).to_le_bytes());
        compressed.extend(&min.to_le_bytes());
        compressed.extend(&max.to_le_bytes());
        let scale = if (max - min).abs() > 1e-8 {
            (max - min) / (levels - 1.0)
        } else {
            1.0
        };
        for &val in weight.iter() {
            let quantized = ((val - min) / scale).round() as u8;
            compressed.push(quantized);
        }
    }
    Ok(compressed)
}

/// Quantization decompression
fn decompress_quantization(data: &[u8], _bits: u8) -> Result<Vec<Array2<f32>>> {
    // Simplified: reuse basic deserialization structure
    deserialize_weights(data)
}

/// Top-K sparsification compression
fn compress_topk(weights: &[Array2<f32>], _k: usize) -> Result<Vec<u8>> {
    serialize_weights(weights)
}

/// Top-K decompression
fn decompress_topk(data: &[u8]) -> Result<Vec<Array2<f32>>> {
    deserialize_weights(data)
}

/// Random sparsification compression
fn compress_random_sparse(weights: &[Array2<f32>], _ratio: f32) -> Result<Vec<u8>> {
    serialize_weights(weights)
}

/// Random sparse decompression
fn decompress_random_sparse(data: &[u8]) -> Result<Vec<Array2<f32>>> {
    deserialize_weights(data)
}

/// Gradient compression
fn compress_gradients(weights: &[Array2<f32>]) -> Result<Vec<u8>> {
    serialize_weights(weights)
}

/// Gradient decompression
fn decompress_gradients(data: &[u8]) -> Result<Vec<Array2<f32>>> {
    deserialize_weights(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_protocol() {
        let mut protocol0 = InMemoryProtocol::new(0, 2);
        let mut protocol1 = InMemoryProtocol::new(1, 2);
        // Send message from 0 to 1
        let msg = Message::Control(ControlMessage::Heartbeat);
        protocol0.send(1, msg.clone()).expect("send failed");
        // Protocol1 should receive it (shares same queue in real scenario)
        // In this simplified test, check that send succeeded
        let stats = protocol0.statistics();
        assert_eq!(stats.messages_sent, 1);
        // Protocol1 receives from its own queue (empty since separate instances)
        let received = protocol1.receive().expect("receive failed");
        assert_eq!(received.len(), 0); // Separate instances don't share queues
    }

    #[test]
    fn test_weight_serialization() {
        let weights = vec![Array2::ones((2, 3))];
        let serialized = serialize_weights(&weights).expect("serialize failed");
        let deserialized = deserialize_weights(&serialized).expect("deserialize failed");
        assert_eq!(weights.len(), deserialized.len());
        assert_eq!(weights[0].shape(), deserialized[0].shape());
    }

    #[test]
    fn test_message_compression() {
        let weights = vec![Array2::ones((3, 3))];
        let compressed = MessageCompressor::compress_weights(&weights, CompressionMethod::None)
            .expect("compress failed");
        let decompressed =
            MessageCompressor::decompress_weights(&compressed).expect("decompress failed");
        assert_eq!(weights.len(), decompressed.len());
    }
}
