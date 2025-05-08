use ndarray::Array1;
use scirs2_text::embeddings::{cosine_similarity, Word2Vec, Word2VecAlgorithm};
use std::time::Instant;

fn main() {
    println!("Word2Vec Example");
    println!("================\n");

    // Sample corpus for demonstration
    let corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown fox jumps over a lazy dog",
        "the fox is quick and brown",
        "the dog is lazy and sleepy",
        "quick brown foxes jump over lazy dogs",
        "the quick fox jumped over the lazy sleeping dog",
        "a brown dog chased the quick fox",
        "foxes and dogs are natural enemies",
        "the quick brown cat jumps over the lazy fox",
        "a quick brown cat jumps over a lazy fox",
    ];

    println!("Training Word2Vec model on a small corpus...");
    let start = Instant::now();

    // Create a Word2Vec model with Skip-gram algorithm
    let mut skipgram_model = Word2Vec::new()
        .with_vector_size(50)
        .with_window_size(3)
        .with_min_count(1)
        .with_epochs(100)
        .with_algorithm(Word2VecAlgorithm::SkipGram)
        .with_negative_samples(5);

    // Train the model
    skipgram_model
        .train(&corpus)
        .expect("Failed to train Skip-gram model");
    let elapsed = start.elapsed();

    println!(
        "Training completed in {:.2} seconds\n",
        elapsed.as_secs_f32()
    );

    // Find similar words
    println!("Finding words similar to 'fox':");
    let similar_to_fox = skipgram_model
        .most_similar("fox", 5)
        .expect("Failed to find similar words");

    for (word, similarity) in similar_to_fox {
        println!("{}: {:.4}", word, similarity);
    }

    println!("\nFinding words similar to 'dog':");
    let similar_to_dog = skipgram_model
        .most_similar("dog", 5)
        .expect("Failed to find similar words");

    for (word, similarity) in similar_to_dog {
        println!("{}: {:.4}", word, similarity);
    }

    // Compute analogies (e.g., fox is to dog as quick is to ?)
    println!("\nAnalogy: fox is to dog as quick is to ?");
    let analogy_result = skipgram_model
        .analogy("fox", "dog", "quick", 3)
        .expect("Failed to compute analogy");

    for (word, similarity) in analogy_result {
        println!("{}: {:.4}", word, similarity);
    }

    // Get word vectors and calculate cosine similarity manually
    println!("\nComparing word vectors:");
    let fox_vector = skipgram_model
        .get_word_vector("fox")
        .expect("Failed to get vector for 'fox'");
    let dog_vector = skipgram_model
        .get_word_vector("dog")
        .expect("Failed to get vector for 'dog'");
    let quick_vector = skipgram_model
        .get_word_vector("quick")
        .expect("Failed to get vector for 'quick'");

    println!(
        "Cosine similarity between 'fox' and 'dog': {:.4}",
        cosine_similarity(&fox_vector, &dog_vector)
    );
    println!(
        "Cosine similarity between 'fox' and 'quick': {:.4}",
        cosine_similarity(&fox_vector, &quick_vector)
    );
    println!(
        "Cosine similarity between 'quick' and 'dog': {:.4}",
        cosine_similarity(&quick_vector, &dog_vector)
    );

    // Train a CBOW model on the same corpus
    println!("\nTraining CBOW model on the same corpus...");
    let start = Instant::now();

    let mut cbow_model = Word2Vec::new()
        .with_vector_size(50)
        .with_window_size(3)
        .with_min_count(1)
        .with_epochs(100)
        .with_algorithm(Word2VecAlgorithm::CBOW)
        .with_negative_samples(5);

    cbow_model
        .train(&corpus)
        .expect("Failed to train CBOW model");
    let elapsed = start.elapsed();

    println!(
        "Training completed in {:.2} seconds\n",
        elapsed.as_secs_f32()
    );

    // Compare results from CBOW model
    println!("CBOW model - Words similar to 'fox':");
    let similar_to_fox_cbow = cbow_model
        .most_similar("fox", 5)
        .expect("Failed to find similar words");

    for (word, similarity) in similar_to_fox_cbow {
        println!("{}: {:.4}", word, similarity);
    }

    // Vector arithmetic: fox - dog + cat = ?
    println!("\nVector arithmetic: fox - dog + cat = ?");

    // Manual vector arithmetic
    let fox_vec = skipgram_model.get_word_vector("fox").unwrap();
    let dog_vec = skipgram_model.get_word_vector("dog").unwrap();
    let cat_vec = skipgram_model.get_word_vector("cat").unwrap();

    // Compute the result vector
    let mut result_vec = Array1::zeros(fox_vec.dim());
    result_vec.assign(&fox_vec);
    result_vec -= &dog_vec;
    result_vec += &cat_vec;

    // Normalize the vector
    let norm = (result_vec.iter().fold(0.0, |sum, &val| sum + val * val)).sqrt();
    result_vec.mapv_inplace(|val| val / norm);

    // Find words similar to the result vector
    let similar_to_result = skipgram_model
        .most_similar_by_vector(&result_vec, 5, &["fox", "dog", "cat"])
        .expect("Failed to find similar words");

    for (word, similarity) in similar_to_result {
        println!("{}: {:.4}", word, similarity);
    }

    // Save and load the model
    println!("\nSaving and loading the model...");
    skipgram_model
        .save("word2vec_model.txt")
        .expect("Failed to save model");
    println!("Model saved to 'word2vec_model.txt'");

    let loaded_model = Word2Vec::load("word2vec_model.txt").expect("Failed to load model");
    println!("Model loaded successfully");

    // Verify the loaded model works
    let similar_words_loaded = loaded_model
        .most_similar("fox", 3)
        .expect("Failed to find similar words with loaded model");

    println!("\nWords similar to 'fox' using loaded model:");
    for (word, similarity) in similar_words_loaded {
        println!("{}: {:.4}", word, similarity);
    }
}
