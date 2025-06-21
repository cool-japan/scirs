//! Advanced neural network features demonstration
//!
//! This example showcases the most advanced features of the neural network module:
//! - Data augmentation (image, mix-based)
//! - Enhanced model evaluation with statistical analysis
//! - Model compression techniques
//! - Knowledge distillation
//! - Transfer learning utilities
//! - Model interpretation and explanation

use ndarray::{Array, Array2, Array4};
use scirs2_neural::{
    augmentation::AugmentationPipelineBuilder,
    compression::{
        CalibrationMethod, ModelPruner, PostTrainingQuantizer, PruningMethod, QuantizationBits,
        QuantizationScheme,
    },
    distillation::{DistillationMethod, DistillationTrainer},
    error::Result,
    interpretation::{AttributionMethod, BaselineMethod, ModelInterpreter},
    model_evaluation::{CrossValidationStrategy, EvaluationBuilder},
    transfer_learning::{TransferLearningManager, TransferStrategy},
};

fn main() -> Result<()> {
    println!("=== Advanced Neural Network Features Demonstration ===\n");

    // 1. Advanced Data Augmentation
    demonstrate_advanced_augmentation()?;

    // 2. Enhanced Model Evaluation
    demonstrate_enhanced_evaluation()?;

    // 3. Model Compression
    demonstrate_model_compression()?;

    // 4. Knowledge Distillation
    demonstrate_knowledge_distillation()?;

    // 5. Transfer Learning
    demonstrate_transfer_learning()?;

    // 6. Model Interpretation
    demonstrate_model_interpretation()?;

    println!("=== All advanced demonstrations completed successfully! ===");
    Ok(())
}

fn demonstrate_advanced_augmentation() -> Result<()> {
    println!("🎨 Advanced Data Augmentation Demonstration");
    println!("==========================================\n");

    // Create comprehensive augmentation pipeline
    let mut augmentation_manager = AugmentationPipelineBuilder::<f64>::new()
        .with_seed(42)
        .with_strong_image_augmentations()
        .with_mixup(1.0)
        .with_cutmix(1.0, (0.1, 0.5))
        .build();

    // Create sample image batch (NCHW format: batch=4, channels=3, height=32, width=32)
    let images = Array4::<f64>::from_shape_fn((4, 3, 32, 32), |(b, c, h, w)| {
        (b + c + h + w) as f64 / 100.0
    })
    .into_dyn();

    let labels =
        Array2::<f64>::from_shape_fn((4, 10), |(b, c)| if c == b % 10 { 1.0 } else { 0.0 })
            .into_dyn();

    println!("Original images shape: {:?}", images.shape());
    println!("Original labels shape: {:?}", labels.shape());

    // Apply standard augmentations
    println!("\n1. Applying image augmentations...");
    let augmented_images = augmentation_manager.augment_images(&images)?;
    println!("   Augmented images shape: {:?}", augmented_images.shape());

    // Apply MixUp
    println!("\n2. Applying MixUp augmentation...");
    let (mixup_images, mixup_labels) = augmentation_manager.apply_mixup(&images, &labels, 1.0)?;
    println!("   MixUp images shape: {:?}", mixup_images.shape());
    println!("   MixUp labels shape: {:?}", mixup_labels.shape());

    // Apply CutMix
    println!("\n3. Applying CutMix augmentation...");
    let (cutmix_images, cutmix_labels) =
        augmentation_manager.apply_cutmix(&images, &labels, 1.0, (0.1, 0.5))?;
    println!("   CutMix images shape: {:?}", cutmix_images.shape());
    println!("   CutMix labels shape: {:?}", cutmix_labels.shape());

    // Display statistics
    let stats = augmentation_manager.get_statistics();
    println!("\n4. Augmentation Statistics:");
    println!("   Samples processed: {}", stats.samples_processed);
    println!("   Processing time: {:.2}ms", stats.processing_time_ms);
    println!("   Transform counts: {:?}", stats.transform_counts);

    println!("✅ Advanced augmentation demonstration completed!\n");
    Ok(())
}

fn demonstrate_enhanced_evaluation() -> Result<()> {
    println!("📊 Enhanced Model Evaluation Demonstration");
    println!("=========================================\n");

    // Create comprehensive evaluation pipeline
    let mut evaluator = EvaluationBuilder::<f64>::new()
        .with_classification_metrics()
        .with_regression_metrics()
        .with_cross_validation(CrossValidationStrategy::KFold {
            k: 5,
            shuffle: true,
        })
        .with_bootstrap(1000)
        .build();

    // Generate sample predictions and ground truth
    let y_true_class =
        Array::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).into_dyn();
    let y_pred_class =
        Array::from_vec(vec![0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2]).into_dyn();

    let y_true_reg =
        Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into_dyn();
    let y_pred_reg =
        Array::from_vec(vec![1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.1, 7.9, 9.2, 9.8]).into_dyn();

    println!("1. Evaluating classification model...");
    let class_results = evaluator.evaluate(
        &y_true_class,
        &y_pred_class,
        Some("classifier_model".to_string()),
    )?;
    println!(
        "   Classification metrics computed: {}",
        class_results.scores.len()
    );

    for (metric, score) in &class_results.scores {
        println!("   {}: {:.4}", metric, score.value);
    }

    println!("\n2. Evaluating regression model...");
    let reg_results = evaluator.evaluate(
        &y_true_reg,
        &y_pred_reg,
        Some("regression_model".to_string()),
    )?;
    println!(
        "   Regression metrics computed: {}",
        reg_results.scores.len()
    );

    for (metric, score) in &reg_results.scores {
        println!("   {}: {:.4}", metric, score.value);
    }

    // Generate comprehensive report
    println!("\n3. Comprehensive Evaluation Report:");
    println!("{}", evaluator.generate_report(&class_results));

    // Compare models
    println!("4. Statistical Model Comparison:");
    let comparison = evaluator.compare_models("classifier_model", "regression_model")?;
    if let Some(t_test) = &comparison.t_test {
        println!(
            "   T-test: t={:.3}, p={:.3}, significant={}",
            t_test.t_statistic, t_test.p_value, t_test.significant
        );
    }

    println!("✅ Enhanced evaluation demonstration completed!\n");
    Ok(())
}

fn demonstrate_model_compression() -> Result<()> {
    println!("🗜️  Model Compression Demonstration");
    println!("==================================\n");

    // 1. Post-training quantization
    println!("1. Post-training Quantization:");
    let mut quantizer = PostTrainingQuantizer::<f64>::new(
        QuantizationBits::Int8,
        QuantizationScheme::Symmetric,
        CalibrationMethod::MinMax,
    );

    // Simulate layer activations
    let activations =
        Array::from_shape_fn((100, 256), |(i, j)| ((i + j) as f64 / 10.0).sin() * 2.0).into_dyn();

    quantizer.calibrate("conv1".to_string(), &activations)?;

    let quantized = quantizer.quantize_tensor("conv1", &activations)?;
    let _dequantized = quantizer.dequantize_tensor("conv1", &quantized)?;

    println!("   Original shape: {:?}", activations.shape());
    println!("   Quantized shape: {:?}", quantized.shape());
    println!(
        "   Compression ratio: {:.1}x",
        quantizer.get_compression_ratio()
    );

    // 2. Model pruning
    println!("\n2. Model Pruning:");
    let mut pruner = ModelPruner::<f64>::new(PruningMethod::MagnitudeBased { threshold: 0.1 });

    // Simulate model weights
    let weights =
        Array::from_shape_fn((128, 256), |(i, j)| ((i * j) as f64 / 1000.0).tanh()).into_dyn();

    let mask = pruner.generate_pruning_mask("fc1".to_string(), &weights)?;
    println!("   Weights shape: {:?}", weights.shape());
    println!("   Pruning mask shape: {:?}", mask.shape());
    println!(
        "   Model sparsity: {:.1}%",
        pruner.get_model_sparsity() * 100.0
    );

    let sparsity_stats = pruner.get_sparsity_statistics();
    for (layer_name, stats) in sparsity_stats {
        println!(
            "   {}: {:.1}% sparse ({}/{} params)",
            layer_name,
            stats.sparsity_ratio * 100.0,
            stats.pruned_params,
            stats.total_params
        );
    }

    println!("✅ Model compression demonstration completed!\n");
    Ok(())
}

fn demonstrate_knowledge_distillation() -> Result<()> {
    println!("🎓 Knowledge Distillation Demonstration");
    println!("======================================\n");

    // Create distillation trainer
    let mut trainer = DistillationTrainer::<f64>::new(DistillationMethod::ResponseBased {
        temperature: 3.0,
        alpha: 0.7,
        beta: 0.3,
    });

    // Simulate teacher and student outputs
    let mut teacher_outputs = std::collections::HashMap::new();
    let mut student_outputs = std::collections::HashMap::new();

    teacher_outputs.insert(
        "output".to_string(),
        Array2::from_shape_fn(
            (4, 10),
            |(b, c)| {
                if c == b % 3 {
                    2.0
                } else {
                    c as f64 * 0.1
                }
            },
        )
        .into_dyn(),
    );

    student_outputs.insert(
        "output".to_string(),
        Array2::from_shape_fn(
            (4, 10),
            |(b, c)| {
                if c == b % 3 {
                    1.8
                } else {
                    c as f64 * 0.12
                }
            },
        )
        .into_dyn(),
    );

    println!("1. Computing distillation loss...");
    let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None)?;
    println!("   Distillation loss: {:.4}", loss);

    // Simulate training steps
    println!("\n2. Simulating distillation training:");
    for step in 1..=5 {
        let step_loss =
            trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None)?;
        println!("   Step {}: loss = {:.4}", step, step_loss);
    }

    let stats = trainer.get_statistics();
    println!("\n3. Training Statistics:");
    println!("   Distillation steps: {}", stats.current_step);
    println!(
        "   Loss history length: {}",
        stats.distillation_loss_history.len()
    );

    println!("✅ Knowledge distillation demonstration completed!\n");
    Ok(())
}

fn demonstrate_transfer_learning() -> Result<()> {
    println!("🔄 Transfer Learning Demonstration");
    println!("=================================\n");

    // Create transfer learning manager
    let mut transfer_manager = TransferLearningManager::<f64>::new(
        TransferStrategy::FeatureExtraction { unfrozen_layers: 2 },
        0.001,
    )?;

    let layer_names = vec![
        "backbone.conv1".to_string(),
        "backbone.conv2".to_string(),
        "backbone.conv3".to_string(),
        "head.fc1".to_string(),
        "head.fc2".to_string(),
    ];

    println!("1. Initializing transfer learning strategy...");
    transfer_manager.initialize_layer_states(&layer_names)?;

    let summary = transfer_manager.get_summary();
    println!("   Strategy: {:?}", summary.strategy);
    println!("   Total layers: {}", summary.total_layers);
    println!("   Frozen layers: {}", summary.frozen_layers);
    println!("   Trainable layers: {}", summary.trainable_layers);

    println!("\n2. Layer-wise learning rates:");
    for layer_name in &layer_names {
        let lr = transfer_manager.get_layer_learning_rate(layer_name);
        let frozen = transfer_manager.is_layer_frozen(layer_name);
        println!("   {}: lr={:.6}, frozen={}", layer_name, lr, frozen);
    }

    // Simulate progressive unfreezing
    println!("\n3. Progressive unfreezing simulation:");
    for epoch in [5, 10, 15] {
        transfer_manager.update_epoch(epoch)?;
        println!(
            "   Epoch {}: {} frozen layers",
            epoch,
            transfer_manager.get_summary().frozen_layers
        );
    }

    println!("✅ Transfer learning demonstration completed!\n");
    Ok(())
}

fn demonstrate_model_interpretation() -> Result<()> {
    println!("🔍 Model Interpretation Demonstration");
    println!("====================================\n");

    // Create model interpreter
    let mut interpreter = ModelInterpreter::<f64>::new();

    // Add attribution methods
    interpreter.add_attribution_method(AttributionMethod::Saliency);
    interpreter.add_attribution_method(AttributionMethod::IntegratedGradients {
        baseline: BaselineMethod::Zero,
        num_steps: 50,
    });

    // Simulate input data and gradients
    let input = Array2::from_shape_fn((2, 10), |(i, j)| (i as f64 + j as f64) / 10.0).into_dyn();

    let gradients =
        Array2::from_shape_fn((2, 10), |(i, j)| ((i + j) as f64 / 20.0).sin()).into_dyn();

    // Cache gradients for attribution computation
    interpreter.cache_gradients("input_gradient".to_string(), gradients.clone());

    println!("1. Computing feature attributions...");
    println!("   Input shape: {:?}", input.shape());

    // Compute saliency attribution
    let saliency =
        interpreter.compute_attribution(&AttributionMethod::Saliency, &input, Some(1))?;
    println!("   Saliency attribution shape: {:?}", saliency.shape());

    // Compute integrated gradients
    let integrated_grad = interpreter.compute_attribution(
        &AttributionMethod::IntegratedGradients {
            baseline: BaselineMethod::Zero,
            num_steps: 50,
        },
        &input,
        Some(1),
    )?;
    println!(
        "   Integrated gradients shape: {:?}",
        integrated_grad.shape()
    );

    // Analyze layer activations
    println!("\n2. Analyzing layer activations...");
    let _layer_activations = Array2::from_shape_fn((20, 64), |(i, j)| {
        if (i + j) % 7 == 0 {
            0.0
        } else {
            ((i * j) as f64 / 100.0).tanh()
        }
    })
    .into_dyn();

    interpreter.analyze_layer_activations("conv_layer")?;

    if let Some(stats) = interpreter.layer_statistics().get("conv_layer") {
        println!("   Layer statistics:");
        println!("     Mean activation: {:.4}", stats.mean_activation);
        println!("     Std activation: {:.4}", stats.std_activation);
        println!("     Sparsity: {:.1}%", stats.sparsity);
        println!("     Dead neurons: {:.1}%", stats.dead_neuron_percentage);
    }

    // Generate comprehensive interpretation report
    println!("\n3. Generating interpretation report...");
    let report = interpreter.generate_report(&input)?;
    println!("{}", report);

    println!("✅ Model interpretation demonstration completed!\n");
    Ok(())
}
