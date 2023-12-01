//! Code imported from github.com/allenai/wimbd/blob/main/src/io.rs
//! and modified by @soldni to integrate in dolma.
//!
//! Tokenizer classes and functions.

use anyhow::{anyhow, Result};
use tokenizers::tokenizer::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

/// Tokenize a string using a basic unicode tokenizer.
pub fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|w| {
        for c in w.chars() {
            if !c.is_whitespace() {
                return true;
            }
        }
        false
    })
}

/// A wrapper class for HuggingFace tokenizers.
#[derive(Debug, Clone)]
pub struct PretrainedTokenizer(Tokenizer);

impl PretrainedTokenizer {
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .0
            .encode(text, false)
            .map_err(|err| anyhow!("{}", err))?
            .get_tokens()
            .to_vec())
    }

    /// Initialize a new pretrained tokenizer from a path or identifier on HuggingFace.
    pub fn new(name: &str) -> Result<Self> {
        Ok(PretrainedTokenizer(
            Tokenizer::from_pretrained(name, None)
                .map_err(|err| anyhow!("Failed to load pretrained tokenizer {} - {}", name, err))?,
        ))
    }

    pub fn decode(&self, tokens: &[String]) -> Result<String> {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| self.0.token_to_id(t))
            .collect();
        self.0.decode(&ids, true).map_err(|err| anyhow!("{}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::tokenize;
    use crate::wimbd::ngrams::Ngram;

    #[test]
    fn test_tokenize_and_ngrams() {
        let s = "You can follow any responses to this entry through the RSS 2.0 feed";
        let tokens = tokenize(s).collect::<Vec<&str>>();
        assert_eq!(
            tokens,
            vec![
                "You",
                "can",
                "follow",
                "any",
                "responses",
                "to",
                "this",
                "entry",
                "through",
                "the",
                "RSS",
                "2.0",
                "feed"
            ]
        );

        let ngrams = tokenize(s).ngrams(10).collect::<Vec<Vec<&str>>>();
        assert_eq!(
            ngrams,
            vec![
                vec![
                    "You",
                    "can",
                    "follow",
                    "any",
                    "responses",
                    "to",
                    "this",
                    "entry",
                    "through",
                    "the",
                ],
                vec![
                    "can",
                    "follow",
                    "any",
                    "responses",
                    "to",
                    "this",
                    "entry",
                    "through",
                    "the",
                    "RSS",
                ],
                vec![
                    "follow",
                    "any",
                    "responses",
                    "to",
                    "this",
                    "entry",
                    "through",
                    "the",
                    "RSS",
                    "2.0",
                ],
                vec![
                    "any",
                    "responses",
                    "to",
                    "this",
                    "entry",
                    "through",
                    "the",
                    "RSS",
                    "2.0",
                    "feed",
                ],
            ]
        );
    }
}
