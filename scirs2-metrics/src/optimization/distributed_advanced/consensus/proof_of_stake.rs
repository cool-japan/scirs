//! Proof of Stake Consensus Implementation
//!
//! Implementation of Proof of Stake consensus for distributed optimization.

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Proof of Stake consensus implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfStakeConsensus {
    node_id: String,
    stake: u64,
    validators: HashMap<String, u64>,
    current_epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validator {
    id: String,
    stake: u64,
    is_active: bool,
}

/// Proof of Stake algorithm state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoSState {
    pub current_epoch: u64,
    pub validators: HashMap<String, Validator>,
    pub finalized_blocks: Vec<String>,
    pub pending_transactions: Vec<String>,
}

impl ProofOfStakeConsensus {
    pub fn new(node_id: String, initial_stake: u64) -> Self {
        let mut validators = HashMap::new();
        validators.insert(node_id.clone(), initial_stake);

        Self {
            node_id,
            stake: initial_stake,
            validators,
            current_epoch: 0,
        }
    }

    pub fn add_validator(&mut self, validator_id: String, stake: u64) -> Result<()> {
        self.validators.insert(validator_id, stake);
        Ok(())
    }

    pub fn select_validator(&self) -> Result<String> {
        let total_stake: u64 = self.validators.values().sum();
        if total_stake == 0 {
            return Err(MetricsError::InvalidOperation(
                "No validators with stake".into(),
            ));
        }

        // Simple deterministic selection based on stake
        let mut max_stake = 0;
        let mut selected = self.node_id.clone();

        for (id, &stake) in &self.validators {
            if stake > max_stake {
                max_stake = stake;
                selected = id.clone();
            }
        }

        Ok(selected)
    }

    pub fn validate_block(&self, validator_id: &str) -> Result<bool> {
        match self.validators.get(validator_id) {
            Some(&stake) => Ok(stake > 0),
            None => Ok(false),
        }
    }

    pub fn next_epoch(&mut self) {
        self.current_epoch += 1;
    }
}
