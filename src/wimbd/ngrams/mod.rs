//! Code imported from github.com/allenai/wimbd/blob/main/src/ngrams/mod.rs
//! and modified by @soldni to integrate in dolma.
//! Utilities for working with and counting ngrams.

use std::collections::VecDeque;
use std::fmt;

use anyhow::Result;

mod counter;
mod topk;

pub use counter::NgramCounter;
pub use topk::TopKNgrams;

use crate::wimbd::tokens::{tokenize, PretrainedTokenizer};

/// A helper function to quickly create an [`Ngram`] iterator given some text and a tokenizer.
pub fn ngrams<'a>(
    text: &'a str,
    num: usize,
    tokenizer: &Option<PretrainedTokenizer>,
) -> Result<Ngrams<'a, String>> {
    if let Some(tokenizer) = tokenizer {
        Ok(tokenizer.tokenize(text)?.into_iter().ngrams(num))
    } else {
        Ok(tokenize(text).map(|s| s.to_string()).ngrams(num))
    }
}

// Ngram code here adapted from https://docs.rs/ngrams/latest/ngrams/index.html, which has a bug.

/// A trait for iterators that gives an [`ngrams`] method.
pub trait Ngram<'a, T: 'a + fmt::Debug + Clone>: Iterator<Item = T>
where
    Self: Sized,
{
    fn ngrams(self, n: usize) -> Ngrams<'a, T>;
}

impl<'a, T: 'a + fmt::Debug + Clone, U: 'a + Iterator<Item = T>> Ngram<'a, T> for U {
    fn ngrams(self, n: usize) -> Ngrams<'a, T> {
        Ngrams::new(self, n)
    }
}

/// The iterator type created from [`Ngram::ngrams`].
pub struct Ngrams<'a, T: 'a + fmt::Debug + Clone> {
    source: Box<dyn Iterator<Item = T> + 'a>,
    num: usize,
    memsize: usize,
    memory: VecDeque<T>,
}

impl<'a, T: 'a + fmt::Debug + Clone> fmt::Debug for Ngrams<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ngrams(tokens, N)")
    }
}

impl<'a, T: 'a + fmt::Debug + Clone + Sized> Ngrams<'a, T> {
    /// The source for the `Ngrams` is expected to be pre-tokenized, this library
    /// does not make any decisions regarding how your input should be tokenized.
    pub(crate) fn new<V: 'a + Iterator<Item = T>>(source: V, n: usize) -> Ngrams<'a, T> {
        let memsize = n - 1;
        Ngrams {
            source: Box::new(source),
            num: n,
            memsize,
            memory: VecDeque::with_capacity(memsize),
        }
    }

    fn fill_memory(&mut self) {
        while self.memory.len() < self.memsize {
            if let Some(a) = self.source.next() {
                self.memory.push_back(a);
            } else {
                break;
            };
        }
    }
}

impl<'a, T: 'a + fmt::Debug + Clone> Iterator for Ngrams<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num > 1 {
            self.fill_memory();

            self.source.next().map(|n| {
                let mut result = Vec::with_capacity(self.num);

                for elem in &self.memory {
                    result.push(elem.clone());
                }

                result.push(n.clone());

                let _ = self.memory.pop_front();
                self.memory.push_back(n);

                result
            })
        } else {
            self.source.next().map(|n| {
                let mut result = Vec::with_capacity(self.num);
                result.push(n);
                result
            })
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{Ngram, Ngrams};
    use std::string::ToString;

    #[test]
    fn test_words_iter_adaptor() {
        let result: Vec<_> = "one two three four five".split(' ').ngrams(4).collect();
        assert_eq!(
            result,
            vec![
                vec!["one", "two", "three", "four"],
                vec!["two", "three", "four", "five"],
            ]
        );
    }

    #[test]
    fn test_words() {
        let seq = "one two three four".split(' ');
        let result: Vec<_> = Ngrams::new(seq, 2).collect();
        assert_eq!(
            result,
            vec![
                vec!["one", "two"],
                vec!["two", "three"],
                vec!["three", "four"],
            ]
        );
    }

    #[test]
    fn test_unigrams() {
        let seq = "one two three four".split(' ');
        let result: Vec<_> = Ngrams::new(seq, 1).collect();
        assert_eq!(
            result,
            vec![vec!["one"], vec!["two"], vec!["three"], vec!["four"],]
        );
    }

    #[test]
    fn test_chars() {
        let seq = "test string".chars().map(|c| c.to_string());
        let result: Vec<_> = Ngrams::new(seq, 4).collect();
        assert_eq!(
            result,
            vec![
                vec!["t", "e", "s", "t"],
                vec!["e", "s", "t", " "],
                vec!["s", "t", " ", "s"],
                vec!["t", " ", "s", "t"],
                vec![" ", "s", "t", "r"],
                vec!["s", "t", "r", "i"],
                vec!["t", "r", "i", "n"],
                vec!["r", "i", "n", "g"],
            ]
        );
    }
}
