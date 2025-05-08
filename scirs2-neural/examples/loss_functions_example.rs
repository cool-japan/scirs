use ndarray::{Array, IxDyn};
use scirs2_neural::losses::{
    ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
};

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
    println!("Predictions: {:?}", predictions);
    println!("Targets: {:?}", targets);
    println!("MSE Loss: {:.4}", loss);

    // Calculate gradients
    let gradients = mse.backward(&predictions, &targets)?;
    println!("MSE Gradients: {:?}", gradients);

    // Cross-Entropy Loss example
    println!("\n--- Cross-Entropy Loss Example ---");
    let ce = CrossEntropyLoss::new(1e-10);

    // Create sample data for multi-class classification
    let predictions = Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.7, 0.2, 0.1, 0.3, 0.6, 0.1])?;
    let targets = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;

    // Calculate loss
    let loss = ce.forward(&predictions, &targets)?;
    println!("Predictions (probabilities):");
    println!("{:?}", predictions);
    println!("Targets (one-hot):");
    println!("{:?}", targets);
    println!("Cross-Entropy Loss: {:.4}", loss);

    // Calculate gradients
    let gradients = ce.backward(&predictions, &targets)?;
    println!("Cross-Entropy Gradients:");
    println!("{:?}", gradients);

    // Focal Loss example
    println!("\n--- Focal Loss Example ---");
    let focal = FocalLoss::new(2.0, Some(0.25), 1e-10);

    // Create sample data for imbalanced classification
    let predictions = Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.7, 0.2, 0.1, 0.3, 0.6, 0.1])?;
    let targets = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;

    // Calculate loss
    let loss = focal.forward(&predictions, &targets)?;
    println!("Predictions (probabilities):");
    println!("{:?}", predictions);
    println!("Targets (one-hot):");
    println!("{:?}", targets);
    println!("Focal Loss (gamma=2.0, alpha=0.25): {:.4}", loss);

    // Calculate gradients
    let gradients = focal.backward(&predictions, &targets)?;
    println!("Focal Loss Gradients:");
    println!("{:?}", gradients);

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

    // Calculate loss
    let loss = contrastive.forward(&embeddings, &labels)?;
    println!("Embeddings (batch_size x 2 x embedding_dim):");
    println!("{:?}", embeddings);
    println!("Labels (1 for similar, 0 for dissimilar):");
    println!("{:?}", labels);
    println!("Contrastive Loss (margin=1.0): {:.4}", loss);

    // Calculate gradients
    let gradients = contrastive.backward(&embeddings, &labels)?;
    println!("Contrastive Loss Gradients (first few):");
    println!("{:?}", gradients.slice(ndarray::s![0, .., 0]));

    // Triplet Loss example
    println!("\n--- Triplet Loss Example ---");
    let triplet = TripletLoss::new(0.5);

    // Create sample data for triplet learning
    // Embedding triplets (batch_size x 3 x embedding_dim)
    let embeddings = Array::from_shape_vec(
        IxDyn(&[2, 3, 3]),
        vec![
            0.1, 0.2, 0.3, // First triplet, anchor
            0.1, 0.3, 0.3, // First triplet, positive
            0.5, 0.5, 0.5, // First triplet, negative
            0.6, 0.6, 0.6, // Second triplet, anchor
            0.5, 0.6, 0.6, // Second triplet, positive
            0.1, 0.1, 0.1, // Second triplet, negative
        ],
    )?;

    // Dummy labels (not used by triplet loss)
    let dummy_labels = Array::zeros(IxDyn(&[2, 1]));

    // Calculate loss
    let loss = triplet.forward(&embeddings, &dummy_labels)?;
    println!("Embeddings (batch_size x 3 x embedding_dim):");
    println!("  - First dimension: batch size");
    println!("  - Second dimension: [anchor, positive, negative]");
    println!("  - Third dimension: embedding components");
    println!("{:?}", embeddings);
    println!("Triplet Loss (margin=0.5): {:.4}", loss);

    // Calculate gradients
    let gradients = triplet.backward(&embeddings, &dummy_labels)?;
    println!("Triplet Loss Gradients (first few):");
    println!("{:?}", gradients.slice(ndarray::s![0, .., 0]));

    Ok(())
}
