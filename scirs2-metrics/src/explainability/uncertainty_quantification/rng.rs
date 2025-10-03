//! Random number generators for uncertainty quantification
//!
//! This module provides various random number generators used in
//! Monte Carlo sampling and uncertainty estimation.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use scirs2_core::numeric::Float;
use std::f64::consts::PI;

/// Trait for random number generators
pub trait RandomNumberGeneratorTrait {
    fn uniform_01<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F;
    fn normal<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F;
    fn seed(&mut self, seed: u64);
}

/// Linear Congruential Generator implementation
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomNumberGeneratorTrait for LcgRng {
    fn uniform_01<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        F::from((self.state >> 16) as f64 / (1u64 << 32) as f64).unwrap()
    }

    fn normal<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        // Box-Muller transform
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// Xorshift random number generator
pub struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
}

impl RandomNumberGeneratorTrait for XorshiftRng {
    fn uniform_01<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        F::from(self.state as f64 / u64::MAX as f64).unwrap()
    }

    fn normal<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        // Box-Muller transform
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state = if seed == 0 { 1 } else { seed };
    }
}

/// Permuted Congruential Generator
pub struct PcgRng {
    state: u64,
    inc: u64,
}

impl PcgRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            inc: 1,
        }
    }
}

impl RandomNumberGeneratorTrait for PcgRng {
    fn uniform_01<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc | 1);
        let xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        let rot = oldstate >> 59;
        let output = (xorshifted >> rot) | (xorshifted << ((!rot + 1) & 31));
        F::from(output as f64 / u32::MAX as f64).unwrap()
    }

    fn normal<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        // Box-Muller transform
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// ChaCha random number generator (simplified implementation)
pub struct ChaChaRng {
    state: [u32; 16],
    counter: u64,
}

impl ChaChaRng {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 16];
        state[0] = seed as u32;
        state[1] = (seed >> 32) as u32;
        Self { state, counter: 0 }
    }
}

impl RandomNumberGeneratorTrait for ChaChaRng {
    fn uniform_01<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        // Simplified ChaCha implementation
        self.counter = self.counter.wrapping_add(1);
        let output = self.state[0].wrapping_add(self.counter as u32);
        F::from(output as f64 / u32::MAX as f64).unwrap()
    }

    fn normal<F: Float + scirs2_core::numeric::FromPrimitive>(&mut self) -> F {
        // Box-Muller transform
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state[0] = seed as u32;
        self.state[1] = (seed >> 32) as u32;
        self.counter = 0;
    }
}
