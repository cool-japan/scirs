//! Topic modeling example using LDA

use scirs2_text::{
    CountVectorizer, LatentDirichletAllocation, LdaBuilder, LdaLearningMethod, Vectorizer,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Topic Modeling with LDA Demo");
    println!("===========================\n");

    // Sample documents about different topics
    let documents = vec![
        // Technology documents
        "Artificial intelligence and machine learning are transforming the tech industry",
        "Deep learning neural networks require powerful GPUs for training",
        "Computer vision algorithms can now recognize objects in real time",
        "Natural language processing helps computers understand human language",
        // Sports documents
        "The basketball team won the championship after a thrilling final game",
        "Football players need excellent physical conditioning and teamwork",
        "Tennis requires both physical fitness and mental concentration",
        "Swimming is an excellent full-body workout and competitive sport",
        // Science documents
        "Climate change is affecting global weather patterns and ecosystems",
        "Quantum physics explores the behavior of matter at atomic scales",
        "Genetic research is unlocking the secrets of human DNA",
        "Space exploration continues to reveal mysteries of the universe",
    ];

    // Convert documents to document-term matrix
    let mut vectorizer = CountVectorizer::default();
    let doc_term_matrix = vectorizer.fit_transform(&documents)?;

    println!("Document-Term Matrix:");
    println!(
        "  Shape: ({}, {})",
        doc_term_matrix.nrows(),
        doc_term_matrix.ncols()
    );
    println!("  Vocabulary size: {}\n", vectorizer.vocabulary_size());

    // Create vocabulary mapping
    let vocabulary = vectorizer.vocabulary();
    let mut word_index_map = HashMap::new();
    for (word, &idx) in vocabulary.token_to_index().iter() {
        word_index_map.insert(idx, word.clone());
    }

    // Train LDA model
    let mut lda = LdaBuilder::new()
        .n_topics(3)
        .max_iter(100)
        .random_seed(42)
        .doc_topic_prior(0.1)
        .topic_word_prior(0.01)
        .learning_method(LdaLearningMethod::Batch)
        .build();

    println!("Training LDA model with 3 topics...");
    let doc_topics = lda.fit_transform(&doc_term_matrix)?;
    println!("Training completed!\n");

    // Display document-topic assignments
    println!("Document-Topic Assignments:");
    for (doc_idx, topic_dist) in doc_topics.outer_iter().enumerate() {
        let max_topic = topic_dist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        println!(
            "Document {}: Topic {} (probabilities: {:.3}, {:.3}, {:.3})",
            doc_idx + 1,
            max_topic,
            topic_dist[0],
            topic_dist[1],
            topic_dist[2]
        );
    }
    println!();

    // Get topics with top words
    let topics = lda.get_topics(10, &word_index_map)?;

    println!("Discovered Topics:");
    for topic in &topics {
        println!("\nTopic {}:", topic.id);
        println!("Top words:");
        for (word, weight) in &topic.top_words {
            println!("  {} ({:.4})", word, weight);
        }
    }

    // Analyze a new document
    println!("\n\nAnalyzing a new document:");
    let new_doc = "Machine learning algorithms are revolutionizing artificial intelligence";
    let new_doc_vec = vectorizer.transform(new_doc)?;
    let new_doc_topics = lda.transform(&new_doc_vec.insert_axis(ndarray::Axis(0)))?;

    println!("Document: \"{}\"", new_doc);
    println!("Topic distribution:");
    for (topic_idx, &prob) in new_doc_topics.row(0).iter().enumerate() {
        println!("  Topic {}: {:.3}", topic_idx, prob);
    }

    // Create another LDA model with different configuration
    println!("\n\nTrying different LDA configuration:");
    let mut lda2 = LatentDirichletAllocation::with_n_topics(4);
    lda2.fit(&doc_term_matrix)?;

    let topics2 = lda2.get_topics(5, &word_index_map)?;
    println!("Discovered {} topics with top 5 words each:", topics2.len());
    for topic in &topics2 {
        let words: Vec<String> = topic
            .top_words
            .iter()
            .map(|(word, _)| word.clone())
            .collect();
        println!("Topic {}: {}", topic.id, words.join(", "));
    }

    Ok(())
}
