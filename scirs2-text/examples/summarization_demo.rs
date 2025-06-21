//! Text summarization demonstration

use scirs2_text::{CentroidSummarizer, KeywordExtractor, TextRank};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Text Summarization Demo");
    println!("======================\n");

    // Sample text for summarization
    let text = "Artificial intelligence (AI) is intelligence demonstrated by machines, \
        in contrast to the natural intelligence displayed by humans and animals. \
        Leading AI textbooks define the field as the study of intelligent agents: \
        any device that perceives its environment and takes actions that maximize \
        its chance of successfully achieving its goals. Colloquially, the term \
        artificial intelligence is often used to describe machines that mimic \
        cognitive functions that humans associate with the human mind, such as \
        learning and problem solving. As machines become increasingly capable, \
        tasks considered to require intelligence are often removed from the \
        definition of AI, a phenomenon known as the AI effect. A quip in \
        Tesler's Theorem states that AI is whatever hasn't been done yet. \
        For instance, optical character recognition is frequently excluded \
        from things considered to be AI, having become a routine technology. \
        Modern machine capabilities generally classified as AI include \
        successfully understanding human speech, competing at the highest \
        level in strategic game systems, autonomously operating cars, \
        intelligent routing in content delivery networks, and military simulations.";

    // TextRank summarization
    println!("1. TextRank Summarization");
    println!("------------------------");

    let textrank = TextRank::new(3); // Extract 3 sentences
    let summary = textrank.summarize(text)?;

    println!("Original text length: {} characters", text.len());
    println!("Summary length: {} characters", summary.len());
    println!("\nSummary:");
    println!("{}\n", summary);

    // Centroid-based summarization
    println!("2. Centroid-based Summarization");
    println!("------------------------------");

    let centroid = CentroidSummarizer::new(3);
    let centroid_summary = centroid.summarize(text)?;

    println!("Summary:");
    println!("{}\n", centroid_summary);

    // Keyword extraction
    println!("3. Keyword Extraction");
    println!("--------------------");

    let extractor = KeywordExtractor::new(10).with_ngram_range(1, 2)?;

    let keywords = extractor.extract_keywords(text)?;

    println!("Top 10 keywords/keyphrases:");
    for (i, (keyword, score)) in keywords.iter().enumerate() {
        println!("{:2}. {} (score: {:.4})", i + 1, keyword, score);
    }

    // Keywords with positions
    println!("\n4. Keyword Positions");
    println!("-------------------");

    let keywords_with_pos = extractor.extract_keywords_with_positions(text)?;

    for (keyword, _score, positions) in keywords_with_pos.iter().take(5) {
        println!("'{}' appears at positions: {:?}", keyword, positions);
    }

    // Multi-document summarization example
    println!("\n5. Multi-document Summarization");
    println!("-------------------------------");

    let docs = [
        "Machine learning is a subset of artificial intelligence. \
         It focuses on the development of computer programs that can learn from data.",
        "Deep learning is a subset of machine learning. \
         It uses neural networks with multiple layers to progressively extract features.",
        "Natural language processing enables computers to understand human language. \
         It combines computational linguistics with machine learning.",
        "Computer vision is a field of AI that trains computers to interpret visual information. \
         It uses deep learning models to process images and videos.",
    ];

    let combined_text = docs.join(" ");
    let multi_doc_summary = textrank.summarize(&combined_text)?;

    println!("Combined documents summary:");
    println!("{}\n", multi_doc_summary);

    // Comparative analysis
    println!("6. Comparative Analysis");
    println!("----------------------");

    let techniques = vec![
        ("TextRank", TextRank::new(2).summarize(text)?),
        ("Centroid", CentroidSummarizer::new(2).summarize(text)?),
    ];

    for (name, summary) in techniques {
        println!("{} (length: {} chars):", name, summary.len());
        println!("{}\n", summary);
    }

    Ok(())
}
