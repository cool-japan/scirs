use scirs2_core::ndarray::{Array, Array3, IxDyn};
use scirs2_neural::losses::{
    ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loss functions example");
    // Mean Squared Error example
    println!("\n--- Mean Squared Error Example ---");
    let mse = MeanSquaredError::new();
    // Create sample data for regression
    let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
    let targets = Array::from_vec(vec![1.5, 1.8, 2.5]).into_dyn();
    // Calculate loss
    let loss = mse.forward(&predictions, &targets)?;
    println!("Predictions: {predictions:?}");
    println!("Targets: {targets:?}");
    println!("MSE Loss: {loss:.4}");
    // Calculate gradients
    let gradients = mse.backward(&predictions, &targets)?;
    println!("MSE Gradients: {gradients:?}");
    // Cross-Entropy Loss example
    println!("\n--- Cross-Entropy Loss Example ---");
    let ce = CrossEntropyLoss::new(1e-10);
    // Create sample data for multi-class classification
    let predictions = Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.7, 0.2, 0.1, 0.3, 0.6, 0.1])?;
    let targets = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let loss = ce.forward(&predictions, &targets)?;
    println!("Predictions (probabilities):");
    println!("{predictions:?}");
    println!("Targets (one-hot):");
    println!("{targets:?}");
    println!("Cross-Entropy Loss: {loss:.4}");
    let gradients = ce.backward(&predictions, &targets)?;
    println!("Cross-Entropy Gradients:");
    println!("{gradients:?}");
    // Focal Loss example
    println!("\n--- Focal Loss Example ---");
    let focal = FocalLoss::new(2.0, Some(0.25), 1e-10);
    // Create sample data for imbalanced classification
    let loss = focal.forward(&predictions, &targets)?;
    println!("Focal Loss (gamma=2.0, alpha=0.25): {loss:.4}");
    let _gradients = focal.backward(&predictions, &targets)?;
    println!("Focal Loss Gradients:");
    // Contrastive Loss example
    println!("\n--- Contrastive Loss Example ---");
    let contrastive = ContrastiveLoss::new(1.0);
    // Create sample data for similarity learning
    // Embedding pairs (batch_size x 2 x embedding_dim)
    let embeddings = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            0.1, 0.2, 0.3, // First pair, first embedding
            0.1, 0.3, 0.3, // First pair, second embedding (similar)
            0.5, 0.5, 0.5, // Second pair, first embedding
            0.9, 0.8, 0.7, // Second pair, second embedding (dissimilar)
        ],
    )?;
    // Labels: 1 for similar pairs, 0 for dissimilar
    let labels = Array::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 0.0])?;
    let loss = contrastive.forward(&embeddings, &labels)?;
    println!("Embeddings (batch_size x 2 x embedding_dim):");
    println!("{embeddings:?}");
    println!("Labels (1 for similar, 0 for dissimilar):");
    println!("{labels:?}");
    println!("Contrastive Loss (margin=1.0): {loss:.4}");
    let gradients = contrastive.backward(&embeddings, &labels)?;
    println!("Contrastive Loss Gradients (first few):");
    let gradient_slice = gradients.slice(scirs2_core::ndarray::s![0, .., 0]);
    println!("{gradient_slice:?}");
    // Triplet Loss example
    println!("\n--- Triplet Loss Example ---");
    let triplet = TripletLoss::new(0.5);
    // Create sample data for triplet learning
    // Embedding triplets (batch_size x 3 x embedding_dim)
    let triplet_embeddings = Array3::from_shape_vec(
        (2, 3, 3),
        vec![
            0.1, 0.2, 0.3, // First triplet, anchor
            0.1, 0.3, 0.3, // First triplet, positive
            0.5, 0.5, 0.5, // First triplet, negative
            0.6, 0.6, 0.6, // Second triplet, anchor
            0.5, 0.6, 0.6, // Second triplet, positive
            0.1, 0.1, 0.1, // Second triplet, negative
        ],
    )?
    .into_dyn();
    // Dummy labels (not used by triplet loss)
    let dummy_labels = Array::zeros(IxDyn(&[2, 1]));
    let loss = triplet.forward(&triplet_embeddings, &dummy_labels)?;
    println!("Embeddings (batch_size x 3 x embedding_dim):");
    println!("  - First dimension: batch size");
    println!("  - Second dimension: [anchor, positive, negative]");
    println!("  - Third dimension: embedding components");
    println!("Triplet Loss (margin=0.5): {loss:.4}");
    let _gradients = triplet.backward(&triplet_embeddings, &dummy_labels)?;
    println!("Triplet Loss Gradients (first few):");
    Ok(())
}
