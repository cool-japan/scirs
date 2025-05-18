//! Parallel processing demonstration

use scirs2_text::{
    ParallelCorpusProcessor, ParallelTextProcessor, ParallelTokenizer, ParallelVectorizer,
    TfidfVectorizer, WordTokenizer,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Parallel Text Processing Demo");
    println!("============================\n");

    // Create test data with larger size to demonstrate parallelism
    println!("Creating test data...");
    let texts = create_test_texts(1000);

    // Create references to handle &[&str] requirements
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    println!("Total documents: {}", texts.len());
    println!("Example document: {}", texts[0]);

    // 1. Simple Parallel Text Processing
    println!("\n1. Basic Parallel Processing");
    println!("---------------------------");

    let processor = ParallelTextProcessor::new();

    let start = Instant::now();
    let word_counts = processor.process(&text_refs, |text| {
        // Count words in each document
        text.split_whitespace().count()
    });
    let duration = start.elapsed();

    println!("Processed {} documents in {:.2?}", texts.len(), duration);
    println!(
        "Average word count: {:.2}",
        word_counts.iter().sum::<usize>() as f64 / word_counts.len() as f64
    );

    // Sequential comparison
    let start = Instant::now();
    let _seq_word_counts: Vec<_> = texts
        .iter()
        .map(|text| text.split_whitespace().count())
        .collect();
    let seq_duration = start.elapsed();

    println!("Sequential processing took {:.2?}", seq_duration);
    println!(
        "Speedup factor: {:.2}x",
        seq_duration.as_secs_f64() / duration.as_secs_f64()
    );

    // 2. Parallel Tokenization
    println!("\n2. Parallel Tokenization");
    println!("----------------------");

    let tokenizer = ParallelTokenizer::new(WordTokenizer::new(true)); // Pass 'lowercase' parameter

    let start = Instant::now();
    let tokens = tokenizer.tokenize(&text_refs)?;
    let duration = start.elapsed();

    println!("Tokenized {} documents in {:.2?}", texts.len(), duration);
    println!(
        "Total tokens: {}",
        tokens.iter().map(|t| t.len()).sum::<usize>()
    );
    println!(
        "Sample tokens from first document: {:?}",
        tokens[0].iter().take(5).collect::<Vec<_>>()
    );

    // Custom token processing
    println!("\nCustom token processing...");
    let start = Instant::now();
    let token_stats = tokenizer.tokenize_and_map(&text_refs, |tokens| {
        // Calculate token statistics
        let count = tokens.len();
        let avg_len = if count > 0 {
            tokens.iter().map(|t| t.len()).sum::<usize>() as f64 / count as f64
        } else {
            0.0
        };
        (count, avg_len)
    })?;
    let duration = start.elapsed();

    println!("Processed token statistics in {:.2?}", duration);
    println!(
        "Average tokens per document: {:.2}",
        token_stats.iter().map(|(count, _)| *count).sum::<usize>() as f64
            / token_stats.len() as f64
    );
    println!(
        "Average token length: {:.2}",
        token_stats.iter().map(|(_, avg_len)| *avg_len).sum::<f64>() / token_stats.len() as f64
    );

    // 3. Parallel Vectorization
    println!("\n3. Parallel Vectorization");
    println!("------------------------");

    // First fit the vectorizer
    let mut vectorizer = TfidfVectorizer::default();
    let start = Instant::now();

    // Import the Vectorizer trait to use its methods
    use scirs2_text::Vectorizer;
    vectorizer.fit(&text_refs)?;
    let fit_duration = start.elapsed();

    println!("Fitted vectorizer in {:.2?}", fit_duration);

    // Now transform in parallel
    let parallel_vectorizer = ParallelVectorizer::new(vectorizer).with_chunk_size(100);

    let start = Instant::now();
    let vectors = parallel_vectorizer.transform(&text_refs)?;
    let transform_duration = start.elapsed();

    println!(
        "Transformed {} documents in {:.2?}",
        texts.len(),
        transform_duration
    );
    println!("Vector shape: {:?}", vectors.shape());
    println!(
        "Non-zero elements: {}",
        vectors.iter().filter(|&&x| x > 0.0).count()
    );

    // 4. Batch Processing with Progress
    println!("\n4. Batch Processing with Progress");
    println!("--------------------------------");

    let processor = ParallelCorpusProcessor::new(100).with_threads(num_cpus::get());

    println!("Processing with {} threads...", num_cpus::get());
    let start = Instant::now();

    let last_progress = std::sync::Mutex::new(0);
    let result = processor.process_with_progress(
        &text_refs,
        |batch| {
            // Analyze batch of documents
            let mut word_counts = Vec::new();
            let mut char_counts = Vec::new();

            for &text in batch {
                word_counts.push(text.split_whitespace().count());
                char_counts.push(text.chars().count());
            }

            Ok(word_counts.into_iter().zip(char_counts).collect::<Vec<_>>())
        },
        |current, total| {
            // Only print progress updates at 10% intervals
            let percent = current * 100 / total;
            let mut last = last_progress.lock().unwrap();
            if percent / 10 > *last / 10 {
                println!("  Progress: {}/{}  ({}%)", current, total, percent);
                *last = percent;
            }
        },
    )?;

    let duration = start.elapsed();

    println!("Processed {} documents in {:.2?}", texts.len(), duration);
    println!(
        "Average words per document: {:.2}",
        result.iter().map(|(words, _)| words).sum::<usize>() as f64 / result.len() as f64
    );
    println!(
        "Average characters per document: {:.2}",
        result.iter().map(|(_, chars)| chars).sum::<usize>() as f64 / result.len() as f64
    );

    // 5. Memory-efficient processing
    println!("\n5. Memory-Efficient Large Corpus Processing");
    println!("------------------------------------------");

    println!("Simulating processing of a large corpus...");
    let large_texts: Vec<&str> = text_refs.iter().cycle().take(5000).copied().collect();
    println!("Large corpus size: {} documents", large_texts.len());

    let processor = ParallelCorpusProcessor::new(250).with_max_memory(1024 * 1024 * 1024); // 1 GB limit

    let start = Instant::now();
    let summary = processor.process(&large_texts, |batch| {
        // Compute simple statistics for the batch
        let batch_size = batch.len();
        let total_words: usize = batch
            .iter()
            .map(|&text| text.split_whitespace().count())
            .sum();
        let total_chars: usize = batch.iter().map(|&text| text.chars().count()).sum();

        Ok(vec![(batch_size, total_words, total_chars)])
    })?;
    let duration = start.elapsed();

    let total_words: usize = summary.iter().map(|(_, words, _)| words).sum();
    let total_chars: usize = summary.iter().map(|(_, _, chars)| chars).sum();

    println!("Processed large corpus in {:.2?}", duration);
    println!("Total words: {}", total_words);
    println!("Total chars: {}", total_chars);
    println!(
        "Average processing speed: {:.2} documents/second",
        large_texts.len() as f64 / duration.as_secs_f64()
    );

    Ok(())
}

fn create_test_texts(size: usize) -> Vec<String> {
    // Sample text fragments to combine randomly
    let subjects = [
        "Machine learning",
        "Natural language processing",
        "Data science",
        "Artificial intelligence",
        "Statistical analysis",
        "Deep learning",
        "Text mining",
        "Information retrieval",
        "Computational linguistics",
    ];

    let verbs = [
        "transforms",
        "revolutionizes",
        "enhances",
        "analyzes",
        "processes",
        "interprets",
        "understands",
        "models",
        "extracts information from",
    ];

    let objects = [
        "text documents",
        "language patterns",
        "unstructured data",
        "communication systems",
        "research methodologies",
        "business decisions",
        "customer feedback",
        "social media content",
        "scientific literature",
    ];

    let adjectives = [
        "modern",
        "complex",
        "efficient",
        "intelligent",
        "advanced",
        "innovative",
        "powerful",
        "sophisticated",
        "state-of-the-art",
    ];

    let adverbs = [
        "dramatically",
        "significantly",
        "effectively",
        "precisely",
        "rapidly",
        "intelligently",
        "thoroughly",
        "fundamentally",
        "increasingly",
    ];

    let mut texts = Vec::with_capacity(size);
    let mut rng = rand::rng();

    for _ in 0..size {
        let subject = subjects[rng.next_u32() as usize % subjects.len()];
        let verb = verbs[rng.next_u32() as usize % verbs.len()];
        let object = objects[rng.next_u32() as usize % objects.len()];
        let adjective = adjectives[rng.next_u32() as usize % adjectives.len()];
        let adverb = adverbs[rng.next_u32() as usize % adverbs.len()];

        // Create 2-3 sentences per document
        let num_sentences = 2 + (rng.next_u32() % 2) as usize;
        let mut sentences = Vec::with_capacity(num_sentences);

        // First sentence
        sentences.push(format!("{} {} {} {}.", subject, adverb, verb, object));

        // Second sentence
        sentences.push(format!(
            "This {} approach enables {} applications in various domains.",
            adjective, adjective
        ));

        // Optional third sentence
        if num_sentences > 2 {
            let subject2 = subjects[rng.next_u32() as usize % subjects.len()];
            let adverb2 = adverbs[rng.next_u32() as usize % adverbs.len()];
            sentences.push(format!(
                "{} is {} improving with recent technological advances.",
                subject2, adverb2
            ));
        }

        texts.push(sentences.join(" "));
    }

    texts
}

trait Random {
    fn next_u32(&mut self) -> u32;
}

impl Random for rand::rngs::ThreadRng {
    fn next_u32(&mut self) -> u32 {
        use rand::Rng;
        self.random()
    }
}
