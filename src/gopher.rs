use std::{collections::HashMap, str::FromStr};

use pyo3::prelude::*;

use crate::wimbd::tokens::tokenize;

static REQUIRED_WORDS: [&str; 8] = ["the", "be", "to", "of", "and", "that", "have", "with"];
static BULLET_POINTS: [&str; 3] = ["-", "*", "•"];

fn get_ngram_length(ngram: &Vec<String>) -> i32 {
    ngram.iter().map(|word| word.chars().count() as i32).sum()
}

#[pyfunction]
pub fn gopher_statistics(text: &str) -> PyResult<HashMap<String, f64>> {
    // will be used to store the results
    let mut result: HashMap<String, f64> = HashMap::new();

    // character statistics
    let text_length = text.chars().count() as f64;
    result.insert(String::from_str("character_count")?, text_length);

    // word statistics
    let words: Vec<&str> = tokenize(text).collect();
    result.insert(String::from_str("word_count")?, words.len() as f64);

    // median word length
    result.insert(String::from_str("median_word_length")?, {
        match words.len() {
            0 => 0.0,
            _ => {
                let mut word_lengths: Vec<usize> =
                    words.iter().map(|word| word.chars().count()).collect();
                word_lengths.sort();
                match word_lengths.len() % 2 {
                    0 => {
                        let mid = word_lengths.len() / 2;
                        (word_lengths[mid - 1] + word_lengths[mid]) as f64 / 2.0
                    }
                    _ => word_lengths[word_lengths.len() / 2] as f64,
                }
            }
        }
    });

    // get fraction of words with hashes
    result.insert(String::from_str("symbols_hashes_ratio")?, {
        match words.len() {
            0 => 0.0,
            _ => {
                let hashes_count: f64 = words
                    .iter()
                    .map(|word| word.matches("#").count() as f64)
                    .sum();
                hashes_count / (words.len() as f64)
            }
        }
    });

    // get fraction of words with ellipses
    result.insert(String::from_str("symbols_ellipses_ratio")?, {
        match words.len() {
            0 => 0.0,
            _ => {
                let hashes_count: f64 = words
                    .iter()
                    .map(|word| (word.matches("...").count() + word.matches("…").count()) as f64)
                    .sum();
                hashes_count / (words.len() as f64)
            }
        }
    });

    // symbol_to_word_ratio is fraction of lines with ellipses + fraction of lines with hashes
    result.insert(
        String::from_str("symbol_to_word_ratio")?,
        result.get("symbols_hashes_ratio").unwrap_or(&0.0)
            + result.get("symbols_ellipses_ratio").unwrap_or(&0.0),
    );

    // fraction of words with alpha character
    result.insert(
        String::from_str("fraction_of_words_with_alpha_character")?,
        {
            match words.len() {
                0 => 0.0,
                _ => {
                    let alpha_words: f64 = words
                        .iter()
                        .filter(|word| word.chars().any(|c| c.is_alphabetic()))
                        .count() as f64;
                    alpha_words / (words.len() as f64)
                }
            }
        },
    );

    // fraction of words with required words
    result.insert(
        String::from_str("fraction_of_words_with_required_words")?,
        {
            match words.len() {
                0 => 0.0,
                _ => {
                    let required_words_count: f64 = words
                        .iter()
                        .filter(|word| REQUIRED_WORDS.contains(&word.to_lowercase().as_str()))
                        .count() as f64;
                    required_words_count / (words.len() as f64)
                }
            }
        },
    );

    // lines in the document, excluding empty lines (note we don't trim because we want to capture
    // lines that are only whitespace characters).
    let lines = text
        .lines()
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>();

    // fraction of lines starting with bullet points
    result.insert(
        String::from_str("fraction_of_lines_starting_with_bullet_point")?,
        {
            match lines.len() {
                0 => 0.0,
                _ => {
                    let bullet_points_count: f64 = lines
                        .iter()
                        .filter(|line| {
                            BULLET_POINTS.contains(
                                &line
                                    .trim()
                                    .chars()
                                    .next()
                                    .unwrap_or(' ')
                                    .to_string()
                                    .as_str(),
                            )
                        })
                        .count() as f64;
                    bullet_points_count / (lines.len() as f64)
                }
            }
        },
    );

    // fraction of lines ending with ellipses
    result.insert(
        String::from_str("fraction_of_lines_ending_with_ellipsis")?,
        {
            match lines.len() {
                0 => 0.0,
                _ => {
                    let ellipses_count: f64 = lines
                        .iter()
                        .filter(|line| {
                            let trimmed_line = line.trim();
                            trimmed_line.ends_with("...") || trimmed_line.ends_with("…")
                        })
                        .count() as f64;
                    ellipses_count / (lines.len() as f64)
                }
            }
        },
    );

    let lines_counts = lines.iter().fold(HashMap::new(), |mut acc, line| {
        *acc.entry(line).or_insert(0) += 1;
        acc
    });
    let duplicated_lines: Vec<_> = lines_counts
        .iter()
        .filter(|&(_, &count)| count > 1)
        .map(|(&line, _)| line)
        .collect();

    // fraction of duplicate lines
    result.insert(String::from_str("fraction_of_duplicate_lines")?, {
        match duplicated_lines.len() {
            0 => 0.0,
            _ => duplicated_lines.len() as f64 / lines.len() as f64,
        }
    });

    // fraction of characters in duplicate lines
    result.insert(
        String::from_str("fraction_of_characters_in_duplicate_lines")?,
        {
            match duplicated_lines.len() {
                0 => 0.0,
                _ => {
                    let duplicated_characters: f64 = duplicated_lines
                        .iter()
                        .map(|line| line.chars().count() as f64)
                        .sum();
                    duplicated_characters / text_length
                }
            }
        },
    );

    for n in 2..11 {
        let ngrams = words.windows(n).map(|ngram| {
            ngram
                .iter()
                .map(|word| word.to_string())
                .collect::<Vec<String>>()
        });

        let ngram_counts = ngrams.clone().fold(HashMap::new(), |mut acc, line| {
            *acc.entry(line).or_insert(0) += 1;
            acc
        });

        if n < 5 {
            // count how many times the most common ngram appears in the text
            let most_common_ngram = ngram_counts.iter().max_by_key(|&(_, &count)| count);
            result.insert(
                String::from_str(&format!("fraction_of_characters_in_most_common_{}gram", n))?,
                match most_common_ngram {
                    Some((ngram, &count)) => {
                        let ngram_characters = get_ngram_length(ngram) as f64;
                        (count as f64) * ngram_characters / text_length
                    }
                    None => 0.0,
                },
            );
        } else {
            let duplicated_ngrams: Vec<_> = ngram_counts
                .iter()
                .filter(|&(_, &count)| count > 1)
                .collect();

            result.insert(
                String::from_str(&format!("fraction_of_characters_in_duplicate_{}grams", n))?,
                match duplicated_ngrams.len() {
                    // no duplicate ngrams; return 0
                    0 => 0.0,
                    // there are duplicate ngrams, let's calculate the fraction of characters that are duplicates
                    _ => {
                        let total_ngram_length: f64 =
                            ngrams.map(|ngram| get_ngram_length(&ngram) as f64).sum();
                        let duplicated_characters: f64 = duplicated_ngrams
                            .iter()
                            .map(|&(ngram, &count)| get_ngram_length(ngram) as f64 * (count as f64))
                            .sum();

                        // divide the total number of characters in duplicate ngrams by the total number of
                        // characters in all ngrams
                        duplicated_characters / total_ngram_length
                    }
                },
            );
        }
    }
    Ok(result)
}
