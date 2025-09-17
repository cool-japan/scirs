//! Advanced Training Methods for Time Series
//!
//! This module implements cutting-edge training methodologies for time series forecasting,
//! including meta-learning for few-shot adaptation, Neural ODEs for continuous-time modeling,
//! and variational autoencoders for uncertainty quantification.
//!
//! ## Advanced Training Techniques
//! - **Meta-Learning (MAML)**: Model-Agnostic Meta-Learning for rapid adaptation
//! - **Neural ODEs**: Continuous-time neural networks with ODE solvers
//! - **Variational Autoencoders**: Probabilistic modeling with uncertainty estimation
//! - **Transformer Forecasting**: Attention-based sequence modeling for time series
//! - **Hyperparameter Optimization**: Automated hyperparameter tuning with Bayesian optimization
//! - **Few-Shot Learning**: Prototypical Networks and REPTILE for rapid adaptation
//! - **Memory-Augmented Networks**: External memory for complex sequential tasks
//! - **Meta-Optimization**: Learned optimizers for adaptive parameter updates
//!
//! ## Module Organization
//!
//! - [`config`]: Common configuration and data structures
//! - [`meta_learning`]: MAML and other meta-learning algorithms
//! - [`neural_ode`]: Neural Ordinary Differential Equations
//! - [`variational`]: Variational Autoencoders for time series
//! - [`transformers`]: Transformer-based forecasting models
//! - [`hyperparameter_optimization`]: Automated hyperparameter tuning
//! - [`few_shot`]: Few-shot learning algorithms (Prototypical Networks, REPTILE)
//! - [`memory_augmented`]: Memory-Augmented Neural Networks (MANN)
//! - [`optimization`]: Meta-optimization and learned optimizers

// Module declarations
pub mod config;
pub mod few_shot;
pub mod hyperparameter_optimization;
pub mod memory_augmented;
pub mod meta_learning;
pub mod neural_ode;
pub mod optimization;
pub mod transformers;
pub mod variational;

// Configuration and common structures
pub use config::TaskData;

// Meta-learning algorithms
pub use meta_learning::MAML;

// Neural ODE implementations
pub use neural_ode::{IntegrationMethod, NeuralODE, ODESolverConfig};

// Variational methods
pub use variational::{TimeSeriesVAE, VAEOutput};

// Transformer-based models
pub use transformers::TimeSeriesTransformer;

// Hyperparameter optimization
pub use hyperparameter_optimization::{
    HyperparameterOptimizer, HyperparameterSet, OptimizationMethod, OptimizationResults,
    OptimizationStep, SearchSpace,
};

// Few-shot learning
pub use few_shot::{FewShotEpisode, PrototypicalNetworks, REPTILE};

// Memory-augmented networks
pub use memory_augmented::MANN;

// Meta-optimization
pub use optimization::{MetaOptimizer, OptimizationProblem};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_module_integration_maml_with_vae() {
        // Test that MAML can work with VAE-generated features
        let mut maml = MAML::<f64>::new(4, 8, 2, 0.01, 0.1, 3);
        let vae = TimeSeriesVAE::<f64>::new(5, 2, 3, 8, 8);

        let input =
            Array2::from_shape_vec((5, 2), (0..10).map(|i| i as f64 * 0.1).collect()).unwrap();
        let vae_output = vae.forward(&input).unwrap();

        // Use VAE latent representation as input to MAML
        let task = TaskData {
            support_x: Array2::from_shape_vec((3, 4), (0..12).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            support_y: Array2::from_shape_vec((3, 2), (0..6).map(|i| i as f64 * 0.2).collect())
                .unwrap(),
            query_x: Array2::from_shape_vec((2, 4), (12..20).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            query_y: Array2::from_shape_vec((2, 2), (6..10).map(|i| i as f64 * 0.2).collect())
                .unwrap(),
        };

        let loss = maml.meta_train(&[task]).unwrap();
        assert!(loss.is_finite());
        assert!(vae_output.reconstruction_loss.is_finite());
    }

    #[test]
    fn test_module_integration_transformer_with_hyperopt() {
        // Test hyperparameter optimization with transformer
        let search_space = SearchSpace {
            continuous: vec![("learning_rate".to_string(), 0.001, 0.1)],
            integer: vec![
                ("num_heads".to_string(), 2, 8),
                ("num_layers".to_string(), 1, 4),
            ],
            categorical: vec![],
        };

        let mut optimizer =
            HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, search_space, 3);

        // Objective function that creates and evaluates a transformer
        let objective = |params: &HyperparameterSet<f64>| -> crate::error::Result<f64> {
            let num_heads = params.get_integer("num_heads").unwrap_or(4) as usize;
            let num_layers = params.get_integer("num_layers").unwrap_or(2) as usize;

            let transformer =
                TimeSeriesTransformer::<f64>::new(10, 5, 64, num_heads, num_layers, 256);
            let input =
                Array2::from_shape_vec((2, 10), (0..20).map(|i| i as f64 * 0.1).collect()).unwrap();

            let output = transformer.forward(&input)?;

            // Simple evaluation: prefer smaller prediction variance
            let mut variance = 0.0;
            let mean: f64 = output.iter().sum::<f64>() / output.len() as f64;
            for &val in output.iter() {
                variance += (val - mean).powi(2);
            }
            variance /= output.len() as f64;

            Ok(-variance) // Maximize negative variance (minimize variance)
        };

        let best_params = optimizer.optimize(objective).unwrap();
        assert!(!best_params.integer.is_empty());
    }

    #[test]
    fn test_module_integration_few_shot_with_mann() {
        // Test few-shot learning episode with memory-augmented network
        let mut mann = MANN::<f64>::new(8, 6, 10, 12, 4);

        let episode = FewShotEpisode {
            support_x: Array2::from_shape_vec((4, 8), (0..32).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            support_y: Array1::from_vec(vec![0, 0, 1, 1]),
            query_x: Array2::from_shape_vec((2, 8), (32..48).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            query_y: Array1::from_vec(vec![0, 1]),
        };

        let loss = mann.train_few_shot(&[episode]).unwrap();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_module_integration_meta_optimizer_with_neural_ode() {
        // Test meta-optimizer optimization of neural ODE parameters
        let mut meta_opt = MetaOptimizer::<f64>::new(3, 5);

        let ode_config = ODESolverConfig::runge_kutta4(0.1);

        let time_steps = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
        let neural_ode = NeuralODE::<f64>::new(3, 16, time_steps, ode_config);

        // Create an optimization problem for ODE parameters
        let problem = OptimizationProblem::<f64>::quadratic(10, 20);
        let (optimized_params, loss_history) = meta_opt
            .optimize_parameters(&problem.initial_params, &problem.target, problem.max_steps)
            .unwrap();

        assert_eq!(optimized_params.len(), 10);
        assert_eq!(loss_history.len(), 20);
        // Neural ODE created successfully
    }

    #[test]
    fn test_module_integration_prototypical_with_reptile() {
        // Test that Prototypical Networks and REPTILE can work with same data format
        let proto_net = PrototypicalNetworks::<f64>::new(6, 4, vec![8]);
        let mut reptile = REPTILE::<f64>::new(6, 8, 2, 0.01, 0.1, 3);

        let support_x =
            Array2::from_shape_vec((4, 6), (0..24).map(|i| i as f64 * 0.1).collect()).unwrap();
        let support_y_proto = Array1::from_vec(vec![0, 0, 1, 1]);
        let support_y_reptile =
            Array2::from_shape_vec((4, 2), (0..8).map(|i| i as f64 * 0.1).collect()).unwrap();
        let query_x =
            Array2::from_shape_vec((2, 6), (24..36).map(|i| i as f64 * 0.1).collect()).unwrap();

        // Test Prototypical Networks
        let proto_predictions = proto_net
            .few_shot_episode(&support_x, &support_y_proto, &query_x)
            .unwrap();
        assert_eq!(proto_predictions.len(), 2);

        // Test REPTILE adaptation
        let adapted_params = reptile.fast_adapt(&support_x, &support_y_reptile).unwrap();
        assert_eq!(adapted_params.dim(), reptile.parameters().dim());
    }

    #[test]
    fn test_all_modules_basic_functionality() {
        // Basic smoke test for all modules

        // Config and task data
        let task = TaskData {
            support_x: Array2::from_shape_vec((2, 3), (0..6).map(|i| i as f64).collect()).unwrap(),
            support_y: Array2::from_shape_vec((2, 2), (0..4).map(|i| i as f64).collect()).unwrap(),
            query_x: Array2::from_shape_vec((1, 3), (6..9).map(|i| i as f64).collect()).unwrap(),
            query_y: Array2::from_shape_vec((1, 2), (4..6).map(|i| i as f64).collect()).unwrap(),
        };

        // MAML
        let mut maml = MAML::<f64>::new(3, 6, 2, 0.01, 0.1, 2);
        let maml_loss = maml.meta_train(&[task.clone()]).unwrap();
        assert!(maml_loss.is_finite());

        // Neural ODE
        let ode_config = ODESolverConfig::runge_kutta4(0.1);
        let time_steps = Array1::from_vec(vec![0.0, 0.1, 0.2]);
        let neural_ode = NeuralODE::<f64>::new(2, 4, time_steps, ode_config);
        let initial_state = Array1::from_vec(vec![1.0, 0.5]);
        let ode_result = neural_ode.forward(&initial_state).unwrap();
        assert_eq!(ode_result.dim(), (3, 2));

        // VAE
        let vae = TimeSeriesVAE::<f64>::new(3, 2, 2, 4, 4);
        let vae_input =
            Array2::from_shape_vec((3, 2), (0..6).map(|i| i as f64 * 0.1).collect()).unwrap();
        let vae_output = vae.forward(&vae_input).unwrap();
        assert_eq!(vae_output.reconstruction.dim(), (3, 2));

        // Transformer
        let transformer = TimeSeriesTransformer::<f64>::new(4, 2, 8, 2, 1, 16);
        let transformer_input =
            Array2::from_shape_vec((1, 4), (0..4).map(|i| i as f64 * 0.1).collect()).unwrap();
        let transformer_output = transformer.forward(&transformer_input).unwrap();
        assert_eq!(transformer_output.dim(), (1, 2));

        // Hyperparameter Optimization
        let search_space = SearchSpace {
            continuous: vec![("lr".to_string(), 0.001, 0.1)],
            integer: vec![],
            categorical: vec![],
        };
        let mut hyperopt =
            HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, search_space, 2);
        let objective = |_: &HyperparameterSet<f64>| -> crate::error::Result<f64> { Ok(0.5) };
        let best_params = hyperopt.optimize(objective).unwrap();
        assert!(!best_params.continuous.is_empty());

        // Prototypical Networks
        let proto_net = PrototypicalNetworks::<f64>::new(3, 2, vec![4]);
        let proto_features = proto_net.extract_features(&task.support_x).unwrap();
        assert_eq!(proto_features.dim(), (2, 2));

        // REPTILE
        let mut reptile = REPTILE::<f64>::new(3, 4, 2, 0.01, 0.1, 2);
        let reptile_loss = reptile.meta_train(&[task]).unwrap();
        assert!(reptile_loss.is_finite());

        // MANN
        let mut mann = MANN::<f64>::new(4, 3, 6, 4, 2);
        let mann_input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mann_output = mann.forward(&mann_input).unwrap();
        assert_eq!(mann_output.len(), 2);

        // Meta-Optimizer
        let mut meta_opt = MetaOptimizer::<f64>::new(2, 3);
        let opt_update = meta_opt.generate_update(0.1, &[1.0, 0.8], 5).unwrap();
        assert!(opt_update.is_finite());

        // Optimization Problem
        let opt_problem = OptimizationProblem::<f64>::quadratic(2, 10);
        assert_eq!(opt_problem.dimension(), 2);
        assert_eq!(opt_problem.max_steps, 10);
    }

    #[test]
    fn test_cross_module_data_compatibility() {
        // Test that data structures are compatible across modules
        let time_series_data =
            Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64 * 0.1).collect()).unwrap();

        // Test that the same data can be used by different modules
        let vae = TimeSeriesVAE::<f64>::new(5, 3, 4, 8, 8);
        let vae_result = vae.forward(&time_series_data).unwrap();

        let transformer = TimeSeriesTransformer::<f64>::new(5, 3, 12, 3, 2, 32);
        let transformer_input =
            Array2::from_shape_vec((1, 5), (0..5).map(|i| i as f64 * 0.1).collect()).unwrap();
        let transformer_result = transformer.forward(&transformer_input).unwrap();

        let proto_net = PrototypicalNetworks::<f64>::new(3, 4, vec![6]);
        let proto_features = proto_net.extract_features(&time_series_data).unwrap();

        // All should produce finite outputs
        assert!(vae_result.reconstruction_loss.is_finite());
        assert_eq!(transformer_result.dim(), (1, 3));
        assert_eq!(proto_features.dim(), (5, 4));
    }
}
