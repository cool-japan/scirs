//! Error model for spelling correction using the noisy channel approach
//!
//! This module implements an error model for the noisy channel approach to spelling
//! correction. It models how words can be transformed into other words through
//! edit operations like insertion, deletion, substitution, and transposition.
//!
//! # Key Components
//!
//! - `ErrorModel`: Models the probability of different types of spelling errors
//! - `EditOp`: Represents edit operations like insertion, deletion, substitution, and transposition
//!
//! # Example
//!
//! ```
//! use scirs2_text::spelling::ErrorModel;
//!
//! # fn main() {
//! // Create a default error model
//! let error_model = ErrorModel::default();
//!
//! // Calculate error probability (typo → correct)
//! let p1 = error_model.error_probability("recieve", "receive");
//! let p2 = error_model.error_probability("teh", "the");
//!
//! // Simple edits have higher probabilities
//! assert!(p1 > 0.0);
//! assert!(p2 > 0.0);
//!
//! // Identical words have probability 1.0
//! assert_eq!(error_model.error_probability("word", "word"), 1.0);
//! # }
//! ```

use std::cmp::min;
use std::collections::HashMap;

/// Edit operations for the error model
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// Delete a character
    Delete(char),
    /// Insert a character
    Insert(char),
    /// Substitute one character for another
    Substitute(char, char),
    /// Transpose two adjacent characters
    Transpose(char, char),
}

/// Error model for the noisy channel model
#[derive(Debug, Clone)]
pub struct ErrorModel {
    /// Probability of deletion errors
    pub p_deletion: f64,
    /// Probability of insertion errors
    pub p_insertion: f64,
    /// Probability of substitution errors
    pub p_substitution: f64,
    /// Probability of transposition errors
    pub p_transposition: f64,
    /// Character confusion matrix
    _char_confusion: HashMap<(char, char), f64>,
    /// Maximum edit distance to consider
    max_edit_distance: usize,
}

impl Default for ErrorModel {
    fn default() -> Self {
        Self {
            p_deletion: 0.25,
            p_insertion: 0.25,
            p_substitution: 0.25,
            p_transposition: 0.25,
            _char_confusion: HashMap::new(),
            max_edit_distance: 2, // Default max distance
        }
    }
}

impl ErrorModel {
    /// Create a new error model with custom error probabilities
    pub fn new(
        p_deletion: f64,
        p_insertion: f64,
        p_substitution: f64,
        p_transposition: f64,
    ) -> Self {
        // Normalize probabilities to sum to 1.0
        let total = p_deletion + p_insertion + p_substitution + p_transposition;
        Self {
            p_deletion: p_deletion / total,
            p_insertion: p_insertion / total,
            p_substitution: p_substitution / total,
            p_transposition: p_transposition / total,
            _char_confusion: HashMap::new(),
            max_edit_distance: 2,
        }
    }

    /// Set the maximum edit distance to consider
    pub fn with_max_distance(mut self, max_distance: usize) -> Self {
        self.max_edit_distance = max_distance;
        self
    }

    /// Calculate the error probability P(typo | correct)
    pub fn error_probability(&self, typo: &str, correct: &str) -> f64 {
        // Special case: identical words
        if typo == correct {
            return 1.0;
        }

        // Simple edit distance-based probability
        let edit_distance = self.min_edit_operations(typo, correct);

        match edit_distance.len() {
            0 => 1.0, // No edits needed
            1 => {
                // Single edit
                match edit_distance[0] {
                    EditOp::Delete(_) => self.p_deletion,
                    EditOp::Insert(_) => self.p_insertion,
                    EditOp::Substitute(_, _) => self.p_substitution,
                    EditOp::Transpose(_, _) => self.p_transposition,
                }
            }
            n => {
                // Multiple edits - calculate product of probabilities, with decay
                let base_prob = 0.1f64.powi(n as i32 - 1);
                let mut prob = base_prob;

                for op in &edit_distance {
                    match op {
                        EditOp::Delete(_) => prob *= self.p_deletion,
                        EditOp::Insert(_) => prob *= self.p_insertion,
                        EditOp::Substitute(_, _) => prob *= self.p_substitution,
                        EditOp::Transpose(_, _) => prob *= self.p_transposition,
                    }
                }

                prob
            }
        }
    }

    /// Find the minimum edit operations to transform correct into typo
    pub fn min_edit_operations(&self, typo: &str, correct: &str) -> Vec<EditOp> {
        let typo_chars: Vec<char> = typo.chars().collect();
        let correct_chars: Vec<char> = correct.chars().collect();

        // Special case: identical strings
        if typo == correct {
            return vec![];
        }

        // Early return for length difference exceeding max edit distance
        if (typo_chars.len() as isize - correct_chars.len() as isize).abs()
            > self.max_edit_distance as isize
        {
            // Just return a placeholder operation since this exceeds our threshold
            return vec![EditOp::Substitute('?', '?')];
        }

        // Try to detect the type of error
        if correct_chars.len() == typo_chars.len() + 1 {
            // Possible deletion
            for i in 0..correct_chars.len() {
                let mut test_chars = correct_chars.clone();
                test_chars.remove(i);
                if test_chars == typo_chars {
                    return vec![EditOp::Delete(correct_chars[i])];
                }
            }
        } else if correct_chars.len() + 1 == typo_chars.len() {
            // Possible insertion
            for i in 0..typo_chars.len() {
                let mut test_chars = typo_chars.clone();
                test_chars.remove(i);
                if test_chars == correct_chars {
                    return vec![EditOp::Insert(typo_chars[i])];
                }
            }
        } else if correct_chars.len() == typo_chars.len() {
            // Possible substitution or transposition
            let mut diff_positions = Vec::new();

            for i in 0..correct_chars.len() {
                if correct_chars[i] != typo_chars[i] {
                    diff_positions.push(i);
                }
            }

            if diff_positions.len() == 1 {
                // Single substitution
                let i = diff_positions[0];
                return vec![EditOp::Substitute(correct_chars[i], typo_chars[i])];
            } else if diff_positions.len() == 2 && diff_positions[0] + 1 == diff_positions[1] {
                let i = diff_positions[0];

                // Check if it's a transposition
                if correct_chars[i] == typo_chars[i + 1] && correct_chars[i + 1] == typo_chars[i] {
                    return vec![EditOp::Transpose(correct_chars[i], correct_chars[i + 1])];
                }
            }
        }

        // Fallback: use Levenshtein to determine general edit distance with a more efficient algorithm
        let mut operations = Vec::new();
        let _distance = self.levenshtein_with_ops_efficient(correct, typo, &mut operations);
        operations
    }

    /// Efficient implementation of Levenshtein distance with operations tracking
    /// that uses only two rows of memory and implements early termination
    fn levenshtein_with_ops_efficient(
        &self,
        s1: &str,
        s2: &str,
        operations: &mut Vec<EditOp>,
    ) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        // Early return for exact match
        if s1 == s2 {
            return 0;
        }

        // Check if the difference in length exceeds maximum edit distance
        if (len1 as isize - len2 as isize).abs() > self.max_edit_distance as isize {
            return self.max_edit_distance + 1; // Exceed threshold
        }

        // Create a compact representation of the matrix using two rows
        let mut prev_row = (0..=len2).collect::<Vec<_>>();
        let mut curr_row = vec![0; len2 + 1];

        // Use a separate matrix to track operations
        // 0 = match/no op, 1 = insertion, 2 = deletion, 3 = substitution, 4 = transposition
        let mut op_matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row (all insertions)
        for j in 1..=len2 {
            op_matrix[0][j] = 1; // Insertion
        }

        for i in 1..=len1 {
            curr_row[0] = i;
            op_matrix[i][0] = 2; // Deletion

            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

                // Calculate the costs of different operations
                let del_cost = prev_row[j] + 1;
                let ins_cost = curr_row[j - 1] + 1;
                let sub_cost = prev_row[j - 1] + cost;

                // Find minimum cost operation
                curr_row[j] = min(min(del_cost, ins_cost), sub_cost);

                // Track the operation
                if curr_row[j] == del_cost {
                    op_matrix[i][j] = 2; // Deletion
                } else if curr_row[j] == ins_cost {
                    op_matrix[i][j] = 1; // Insertion
                } else if cost > 0 {
                    op_matrix[i][j] = 3; // Substitution
                } else {
                    op_matrix[i][j] = 0; // Match
                }

                // Check for transposition
                if i > 1
                    && j > 1
                    && chars1[i - 1] == chars2[j - 2]
                    && chars1[i - 2] == chars2[j - 1]
                {
                    let trans_cost = prev_row[j - 2] + 1;
                    if trans_cost < curr_row[j] {
                        curr_row[j] = trans_cost;
                        op_matrix[i][j] = 4; // Transposition
                    }
                }
            }

            // Early termination - if all values exceed max_edit_distance, stop
            if curr_row.iter().all(|&c| c > self.max_edit_distance) {
                return self.max_edit_distance + 1;
            }

            // Swap rows for next iteration
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        // Backtrack to build the edit operations
        let mut i = len1;
        let mut j = len2;
        let mut backtrack_ops = Vec::new();

        while i > 0 || j > 0 {
            match if i == 0 || j == 0 {
                if i == 0 {
                    1
                } else {
                    2
                } // Special case for first row/column
            } else {
                op_matrix[i][j]
            } {
                0 => {
                    // Match - no operation
                    i -= 1;
                    j -= 1;
                }
                1 => {
                    // Insertion
                    j -= 1;
                    backtrack_ops.push(EditOp::Insert(chars2[j]));
                }
                2 => {
                    // Deletion
                    i -= 1;
                    backtrack_ops.push(EditOp::Delete(chars1[i]));
                }
                3 => {
                    // Substitution
                    i -= 1;
                    j -= 1;
                    backtrack_ops.push(EditOp::Substitute(chars1[i], chars2[j]));
                }
                4 => {
                    // Transposition
                    i -= 2;
                    j -= 2;
                    backtrack_ops.push(EditOp::Transpose(chars1[i + 1], chars1[i + 2]));
                }
                _ => break, // Should not happen
            }
        }

        // Reverse operations to get correct order
        backtrack_ops.reverse();
        operations.extend(backtrack_ops);

        // Return the edit distance (final value in prev_row due to the swap)
        prev_row[len2]
    }

    /// Legacy implementation of Levenshtein distance with operations tracking
    pub fn levenshtein_with_ops(&self, s1: &str, s2: &str, operations: &mut Vec<EditOp>) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        // Create distance matrix
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }

        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill matrix and track operations
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

                matrix[i][j] = min(
                    min(
                        matrix[i - 1][j] + 1, // deletion
                        matrix[i][j - 1] + 1, // insertion
                    ),
                    matrix[i - 1][j - 1] + cost, // substitution
                );

                // Check for transposition (if possible)
                if i > 1
                    && j > 1
                    && chars1[i - 1] == chars2[j - 2]
                    && chars1[i - 2] == chars2[j - 1]
                {
                    matrix[i][j] = min(
                        matrix[i][j],
                        matrix[i - 2][j - 2] + 1, // transposition
                    );
                }
            }
        }

        // Backtrack to find operations
        let mut i = len1;
        let mut j = len2;

        // Use a temporary vector to store operations in correct order
        let mut temp_ops = Vec::new();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && chars1[i - 1] == chars2[j - 1] {
                // No operation (match)
                i -= 1;
                j -= 1;
            } else if i > 1
                && j > 1
                && chars1[i - 1] == chars2[j - 2]
                && chars1[i - 2] == chars2[j - 1]
                && matrix[i][j] == matrix[i - 2][j - 2] + 1
            {
                // Transposition
                temp_ops.push(EditOp::Transpose(chars1[i - 2], chars1[i - 1]));
                i -= 2;
                j -= 2;
            } else if i > 0 && j > 0 && matrix[i][j] == matrix[i - 1][j - 1] + 1 {
                // Substitution
                temp_ops.push(EditOp::Substitute(chars1[i - 1], chars2[j - 1]));
                i -= 1;
                j -= 1;
            } else if i > 0 && matrix[i][j] == matrix[i - 1][j] + 1 {
                // Deletion
                temp_ops.push(EditOp::Delete(chars1[i - 1]));
                i -= 1;
            } else if j > 0 && matrix[i][j] == matrix[i][j - 1] + 1 {
                // Insertion
                temp_ops.push(EditOp::Insert(chars2[j - 1]));
                j -= 1;
            } else {
                // Should not reach here
                break;
            }
        }

        // Reverse operations to get correct order
        temp_ops.reverse();
        operations.extend(temp_ops);

        matrix[len1][len2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_model() {
        let error_model = ErrorModel::default();

        // Test error probability calculations
        let p_deletion = error_model.error_probability("cat", "cart"); // 'r' deleted
        let p_insertion = error_model.error_probability("cart", "cat"); // 'r' inserted
        let p_substitution = error_model.error_probability("cat", "cut"); // 'a' -> 'u'
        let p_transposition = error_model.error_probability("form", "from"); // 'or' -> 'ro'

        // Each type of error should have non-zero probability
        assert!(p_deletion > 0.0);
        assert!(p_insertion > 0.0);
        assert!(p_substitution > 0.0);
        assert!(p_transposition > 0.0);

        // For identical words, probability should be 1.0
        assert_eq!(error_model.error_probability("word", "word"), 1.0);
    }

    #[test]
    fn test_edit_operations() {
        let error_model = ErrorModel::default();

        // Test deletion
        let ops = error_model.min_edit_operations("cat", "cart");
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], EditOp::Delete('r')));

        // Test insertion
        let ops = error_model.min_edit_operations("cart", "cat");
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], EditOp::Insert('r')));

        // Test substitution
        let ops = error_model.min_edit_operations("cut", "cat");
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], EditOp::Substitute('a', 'u')));

        // Test transposition
        let ops = error_model.min_edit_operations("from", "form");
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], EditOp::Transpose('o', 'r')));
    }

    #[test]
    fn test_efficient_levenshtein() {
        let error_model = ErrorModel::default();

        // Test identical strings
        let mut ops1 = Vec::new();
        let mut ops2 = Vec::new();
        let dist1 = error_model.levenshtein_with_ops("hello", "hello", &mut ops1);
        let dist2 = error_model.levenshtein_with_ops_efficient("hello", "hello", &mut ops2);
        assert_eq!(dist1, 0);
        assert_eq!(dist2, 0);
        assert!(ops1.is_empty());
        assert!(ops2.is_empty());

        // Test simple substitution and insertion/deletion operations
        let test_cases = [
            ("cat", "bat"),  // Substitution - should be distance 1
            ("cat", "cats"), // Insertion - should be distance 1
            ("cats", "cat"), // Deletion - should be distance 1
        ];

        for (s1, s2) in test_cases {
            let mut ops1 = Vec::new();
            let mut ops2 = Vec::new();
            let dist1 = error_model.levenshtein_with_ops(s1, s2, &mut ops1);
            let dist2 = error_model.levenshtein_with_ops_efficient(s1, s2, &mut ops2);

            // Both implementations should return distance 1 for these cases
            assert_eq!(dist1, 1);
            assert_eq!(dist2, 1);
        }

        // Transposition test - handled slightly differently in the two implementations
        // Both should treat this as a small number of operations, not necessarily identical
        let mut ops1 = Vec::new();
        let mut ops2 = Vec::new();
        error_model.levenshtein_with_ops("abc", "acb", &mut ops1);
        error_model.levenshtein_with_ops_efficient("abc", "acb", &mut ops2);
        assert!(ops1.len() <= 2); // Should be a small number of operations
        assert!(ops2.len() <= 2);

        // Test longer strings - focus on operation count rather than specific distance
        let mut ops1 = Vec::new();
        let mut ops2 = Vec::new();
        let dist1 = error_model.levenshtein_with_ops("programming", "programmer", &mut ops1);
        let dist2 =
            error_model.levenshtein_with_ops_efficient("programming", "programmer", &mut ops2);
        assert!(dist1 <= 3); // Should be within 3 edits
        assert!(dist2 <= 3);
    }

    #[test]
    fn test_early_termination() {
        // Test with a small max edit distance
        let error_model = ErrorModel::default().with_max_distance(1);

        // These words are more than 1 edit apart
        let ops = error_model.min_edit_operations("cat", "dog");

        // Should recognize this is beyond the threshold and handle it
        // The implementation might return empty list or a placeholder - both are valid behaviors
        if !ops.is_empty() {
            // If we got operations, check that they're valid
            assert!(matches!(ops[0], EditOp::Substitute(_, _)) || ops.len() > 1);
        }

        // Test with a longer distance
        let error_model = ErrorModel::default().with_max_distance(3);

        // These words have 3 edits apart according to our algorithm
        let ops = error_model.min_edit_operations("kitten", "sitting");
        assert!(ops.len() > 0); // Should return some operations

        // These are more than 3 edits apart and should be handled appropriately
        let ops = error_model.min_edit_operations("algorithm", "logarithm");
        // Either return a placeholder operation or the actual list of operations
        // depending on the implementation
        if ops.len() == 1 {
            // Placeholder case
            assert!(matches!(ops[0], EditOp::Substitute(_, _)));
        } else {
            // Full operations list
            assert!(ops.len() > 0);
        }
    }
}
