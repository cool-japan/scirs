//! Topic coherence evaluation demonstration

use scirs2_text::{
    LatentDirichletAllocation, Tokenizer, Topic, TopicCoherence, TopicDiversity,
    WhitespaceTokenizer,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Topic Coherence Evaluation Demo");
    println!("==============================\n");

    // Sample documents for topic modeling
    let documents = vec![
        "Machine learning algorithms are used in artificial intelligence",
        "Deep learning neural networks process complex data patterns",
        "Natural language processing enables text understanding",
        "Computer vision algorithms detect objects in images",
        "Reinforcement learning agents learn through trial and error",
        "Supervised learning requires labeled training data",
        "Unsupervised learning discovers hidden patterns",
        "Transfer learning reuses pretrained models",
        "Statistical models analyze numerical data distributions",
        "Regression analysis predicts continuous outcomes",
        "Classification algorithms categorize data points",
        "Time series analysis forecasts temporal patterns",
        "Clustering groups similar data together",
        "Feature engineering improves model performance",
        "Model validation prevents overfitting",
    ];

    // Tokenize documents
    let tokenizer = WhitespaceTokenizer::new();
    let tokenized_docs: Vec<Vec<String>> = documents
        .iter()
        .map(|doc| tokenizer.tokenize(doc).unwrap())
        .collect();

    // Create a simple vocabulary for demonstration
    let mut vocabulary = HashMap::new();
    let mut word_id = 0;

    for doc in &tokenized_docs {
        for word in doc {
            if !vocabulary.contains_key(word) {
                vocabulary.insert(word.clone(), word_id);
                word_id += 1;
            }
        }
    }

    // Create document-term matrix
    let n_docs = tokenized_docs.len();
    let n_words = vocabulary.len();
    let mut doc_term_matrix = ndarray::Array2::zeros((n_docs, n_words));

    for (doc_idx, doc) in tokenized_docs.iter().enumerate() {
        for word in doc {
            if let Some(&word_id) = vocabulary.get(word) {
                doc_term_matrix[[doc_idx, word_id]] += 1.0;
            }
        }
    }

    // Train LDA model
    println!("1. Training LDA Model");
    println!("--------------------");

    let mut lda = LatentDirichletAllocation::with_n_topics(3);
    lda.fit(&doc_term_matrix)?;

    // Create reverse vocabulary mapping
    let id_to_word: HashMap<usize, String> = vocabulary
        .iter()
        .map(|(word, &id)| (id, word.clone()))
        .collect();

    // Get topics
    let topics = lda.get_topics(5, &id_to_word)?;

    println!("Discovered topics:");
    for (i, topic) in topics.iter().enumerate() {
        println!("\nTopic {}: ", i + 1);
        for (word, prob) in &topic.top_words {
            println!("  {} ({:.4})", word, prob);
        }
    }

    // Calculate coherence metrics
    println!("\n2. Topic Coherence Metrics");
    println!("-------------------------");

    let coherence_calc = TopicCoherence::new().with_window_size(5);

    // C_v coherence
    let cv_coherence = coherence_calc.cv_coherence(&topics, &tokenized_docs)?;
    println!("C_v coherence: {:.4}", cv_coherence);

    // UMass coherence
    let umass_coherence = coherence_calc.umass_coherence(&topics, &tokenized_docs)?;
    println!("UMass coherence: {:.4}", umass_coherence);

    // UCI coherence
    let uci_coherence = coherence_calc.uci_coherence(&topics, &tokenized_docs)?;
    println!("UCI coherence: {:.4}", uci_coherence);

    // Topic diversity
    println!("\n3. Topic Diversity");
    println!("-----------------");

    let diversity = TopicDiversity::calculate(&topics);
    println!("Topic diversity: {:.4}", diversity);

    // Pairwise distances
    let distances = TopicDiversity::pairwise_distances(&topics);
    println!("\nPairwise Jaccard distances between topics:");
    for i in 0..distances.nrows() {
        for j in 0..distances.ncols() {
            print!("{:.3} ", distances[[i, j]]);
        }
        println!();
    }

    // Compare different numbers of topics
    println!("\n4. Optimal Topic Number Analysis");
    println!("-------------------------------");

    let topic_counts = vec![2, 3, 4, 5];
    let mut results = Vec::new();

    for n_topics in topic_counts {
        let mut lda = LatentDirichletAllocation::with_n_topics(n_topics);
        lda.fit(&doc_term_matrix)?;

        let topics = lda.get_topics(5, &id_to_word)?;
        let coherence = coherence_calc.cv_coherence(&topics, &tokenized_docs)?;
        let diversity = TopicDiversity::calculate(&topics);

        results.push((n_topics, coherence, diversity));
        println!(
            "{} topics: coherence={:.4}, diversity={:.4}",
            n_topics, coherence, diversity
        );
    }

    // Find optimal number of topics
    let optimal = results
        .iter()
        .max_by(|a, b| {
            // Balance coherence and diversity
            let score_a = a.1 + 0.5 * a.2;
            let score_b = b.1 + 0.5 * b.2;
            score_a.partial_cmp(&score_b).unwrap()
        })
        .unwrap();

    println!(
        "\nOptimal number of topics: {} (coherence={:.4}, diversity={:.4})",
        optimal.0, optimal.1, optimal.2
    );

    // Manual topic example
    println!("\n5. Manual Topic Evaluation");
    println!("-------------------------");

    let manual_topics = vec![
        Topic {
            id: 0,
            top_words: vec![
                ("learning".to_string(), 0.15),
                ("machine".to_string(), 0.12),
                ("algorithm".to_string(), 0.10),
                ("data".to_string(), 0.08),
                ("model".to_string(), 0.07),
            ],
            coherence: None,
        },
        Topic {
            id: 1,
            top_words: vec![
                ("network".to_string(), 0.14),
                ("neural".to_string(), 0.13),
                ("deep".to_string(), 0.11),
                ("layer".to_string(), 0.09),
                ("process".to_string(), 0.08),
            ],
            coherence: None,
        },
    ];

    let manual_coherence = coherence_calc.cv_coherence(&manual_topics, &tokenized_docs)?;
    let manual_diversity = TopicDiversity::calculate(&manual_topics);

    println!("Manual topics coherence: {:.4}", manual_coherence);
    println!("Manual topics diversity: {:.4}", manual_diversity);

    Ok(())
}
