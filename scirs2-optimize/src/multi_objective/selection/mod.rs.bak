//! Selection operators for multi-objective optimization
//!
//! Methods for selecting individuals from populations.

use crate::multi_objective::solutions::Solution;
use rand::seq::SliceRandom;
use rand::Rng;

/// Trait for selection operators
pub trait SelectionOperator {
    /// Select individuals from population
    fn select(&self, population: &[Solution], n_select: usize) -> Vec<Solution>;
}

/// Tournament selection
#[derive(Debug, Clone)]
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }

    fn binary_tournament(&self, population: &[Solution]) -> Solution {
        let mut rng = rand::rng();
        let idx1 = rng.gen_range(0..population.len());
        let idx2 = rng.gen_range(0..population.len());

        let sol1 = &population[idx1];
        let sol2 = &population[idx2];

        // Prefer lower rank (better solutions)
        if sol1.rank < sol2.rank {
            sol1.clone()
        } else if sol2.rank < sol1.rank {
            sol2.clone()
        } else {
            // Same rank, prefer higher crowding distance
            if sol1.crowding_distance > sol2.crowding_distance {
                sol1.clone()
            } else {
                sol2.clone()
            }
        }
    }
}

impl SelectionOperator for TournamentSelection {
    fn select(&self, population: &[Solution], n_select: usize) -> Vec<Solution> {
        let mut selected = Vec::with_capacity(n_select);

        for _ in 0..n_select {
            selected.push(self.binary_tournament(population));
        }

        selected
    }
}

/// Random selection
#[derive(Debug, Clone)]
pub struct RandomSelection;

impl RandomSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionOperator for RandomSelection {
    fn select(&self, population: &[Solution], n_select: usize) -> Vec<Solution> {
        let mut rng = rand::rng();
        let mut selected = Vec::with_capacity(n_select);
        for _ in 0..n_select.min(population.len()) {
            let idx = rng.gen_range(0..population.len());
            selected.push(population[idx].clone());
        }
        selected
    }
}

/// Roulette wheel selection based on fitness
#[derive(Debug, Clone)]
pub struct RouletteWheelSelection;

impl RouletteWheelSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionOperator for RouletteWheelSelection {
    fn select(&self, population: &[Solution], n_select: usize) -> Vec<Solution> {
        let mut rng = rand::rng();
        let mut selected = Vec::with_capacity(n_select);

        // Calculate fitness scores (inverse of rank for maximization)
        let max_rank = population.iter().map(|s| s.rank).max().unwrap_or(1);
        let fitness_scores: Vec<f64> = population
            .iter()
            .map(|s| (max_rank + 1 - s.rank) as f64)
            .collect();

        let total_fitness: f64 = fitness_scores.iter().sum();

        for _ in 0..n_select {
            let mut accumulator = 0.0;
            let random_value = rng.gen::<f64>() * total_fitness;

            for (i, &fitness) in fitness_scores.iter().enumerate() {
                accumulator += fitness;
                if accumulator >= random_value {
                    selected.push(population[i].clone());
                    break;
                }
            }
        }

        selected
    }
}

/// Truncation selection - select best individuals
#[derive(Debug, Clone)]
pub struct TruncationSelection;

impl TruncationSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionOperator for TruncationSelection {
    fn select(&self, population: &[Solution], n_select: usize) -> Vec<Solution> {
        let mut sorted = population.to_vec();
        sorted.sort_by(|a, b| {
            a.rank.cmp(&b.rank).then(
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap(),
            )
        });

        sorted.into_iter().take(n_select).collect()
    }
}
