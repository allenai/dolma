//! Code imported from github.com/allenai/wimbd/blob/main/src/ngrams/counter.rs
//! and modified by @soldni to integrate in dolma.

use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::Ordering;

use ahash::RandomState;
use anyhow::{Context, Result};
use atomic_traits::{Atomic, NumOps};
use num_traits::{Bounded, NumCast, One, SaturatingSub, Zero};

pub trait AsIterator<'a, T: 'a> {
    type Iterator: Iterator<Item = &'a T>;

    fn as_iter(&'a self) -> Self::Iterator;
}

impl<'a, T: 'a> AsIterator<'a, T> for [T] {
    type Iterator = std::slice::Iter<'a, T>;

    fn as_iter(&'a self) -> Self::Iterator {
        self.iter()
    }
}

// NOTE: this implementation conflicts with the below for VecDeque.
// impl<'a, U, T: 'a> AsIterator<'a, T> for U
// where
//     U: AsRef<[T]>,
// {
//     type Iterator = std::slice::Iter<'a, T>;

//     fn as_iter(&'a self) -> Self::Iterator {
//         self.as_ref().iter()
//     }
// }

impl<'a, T: 'a> AsIterator<'a, T> for std::collections::VecDeque<T> {
    type Iterator = std::collections::vec_deque::Iter<'a, T>;

    fn as_iter(&'a self) -> Self::Iterator {
        self.iter()
    }
}

/// A thread-safe counting Bloom filter for ngrams.
pub struct NgramCounter<A>
where
    A: Atomic + NumOps,
    <A as Atomic>::Type: Zero + One + Bounded + NumCast + Ord + SaturatingSub + Clone,
{
    size: usize,
    num_hash_functions: usize,
    hash_builders: Vec<RandomState>,
    count_array: Vec<A>,
}

impl<A> NgramCounter<A>
where
    A: Atomic + NumOps,
    <A as Atomic>::Type: Zero + One + Bounded + NumCast + Ord + SaturatingSub + Clone,
{
    /// Create a new counter with a hash table of `size` elements, initialized to `initial_value`.
    pub fn new(
        size: usize,
        num_hash_functions: usize,
        seed: Option<u64>,
        initial_value: <A as Atomic>::Type,
    ) -> Result<Self> {
        // Initialize count table
        let mut count_array = Vec::new();
        count_array.try_reserve_exact(size).with_context(|| {
            "Failed to allocate counts array. You may not have enough available memory.".to_string()
        })?;
        for _ in 0..size {
            count_array.push(A::new(initial_value.clone()));
        }

        // Initialize hash builders
        let mut hash_builders = Vec::with_capacity(num_hash_functions);
        for i in 0..num_hash_functions {
            let hash_builder = match seed {
                Some(seed) => RandomState::with_seed((seed as usize) + i),
                None => RandomState::new(),
            };
            hash_builders.push(hash_builder);
        }

        Ok(Self {
            size,
            num_hash_functions,
            hash_builders,
            count_array,
        })
    }

    /// Returns the number of non-zero elements in the hash table.
    pub fn nonzero(&self) -> u64 {
        let mut nonzero_count: u64 = 0;
        let zero = <A as Atomic>::Type::zero();
        for item in &self.count_array {
            if item.load(Ordering::Relaxed) > zero {
                nonzero_count += 1;
            }
        }
        nonzero_count
    }

    /// Increment the count for an ngram.
    pub fn increment<'a, N, I, T>(
        &self,
        ngram: &'a N,
        by: <A as Atomic>::Type,
    ) -> <A as Atomic>::Type
    where
        N: AsIterator<'a, T, Iterator = I> + ?Sized,
        I: Iterator<Item = &'a T>,
        T: 'a + Hash,
    {
        let mut min_count = <A as Atomic>::Type::max_value();
        for i in 0..self.num_hash_functions {
            let hash = self.hash(&mut ngram.as_iter(), i);
            let index = self.index_for_hash(hash);
            let old_count = self.count_array[index].fetch_add(by.clone(), Ordering::Relaxed);
            let count = if old_count > <A as Atomic>::Type::max_value() - by.clone() {
                // Catch overflows and just keep as MAX.
                self.count_array[index].store(<A as Atomic>::Type::max_value(), Ordering::Relaxed);
                <A as Atomic>::Type::max_value()
            } else {
                old_count + by.clone()
            };
            min_count = std::cmp::min(min_count, count);
        }
        min_count
    }

    /// Decrement the count for an ngram.
    pub fn decrement<'a, N, I, T>(
        &self,
        ngram: &'a N,
        by: <A as Atomic>::Type,
    ) -> <A as Atomic>::Type
    where
        N: AsIterator<'a, T, Iterator = I> + ?Sized,
        I: Iterator<Item = &'a T>,
        T: 'a + Hash,
    {
        let mut max_count = <A as Atomic>::Type::zero();
        for i in 0..self.num_hash_functions {
            let hash = self.hash(&mut ngram.as_iter(), i);
            let index = self.index_for_hash(hash);
            let old_count = self.count_array[index].fetch_sub(by.clone(), Ordering::Relaxed);
            let count = if old_count < by {
                // Catch underflows and just keep as 0.
                self.count_array[index].store(<A as Atomic>::Type::zero(), Ordering::Relaxed);
                <A as Atomic>::Type::zero()
            } else {
                old_count - by.clone()
            };
            max_count = std::cmp::max(max_count, count);
        }
        max_count
    }

    /// Get the max count for an ngram across all hash functions.
    pub fn max_count<'a, N, I, T>(&self, ngram: &'a N) -> <A as Atomic>::Type
    where
        N: AsIterator<'a, T, Iterator = I> + ?Sized,
        I: Iterator<Item = &'a T>,
        T: 'a + Hash,
    {
        let mut max_count = <A as Atomic>::Type::zero();
        for i in 0..self.num_hash_functions {
            let hash = self.hash(&mut ngram.as_iter(), i);
            let index = self.index_for_hash(hash);
            let count = self.count_array[index].load(Ordering::Relaxed);
            max_count = std::cmp::max(max_count, count);
        }
        max_count
    }

    fn hash<I, T>(&self, ngram: &mut I, hasher: usize) -> usize
    where
        I: Iterator<Item = T> + ?Sized,
        T: Hash,
    {
        let mut hasher = self.hash_builders[hasher].build_hasher();
        for token in ngram {
            token.hash(&mut hasher);
        }
        hasher.finish().try_into().unwrap()
    }

    fn index_for_hash(&self, hash: usize) -> usize {
        hash % self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_counter() {
        let counter = NgramCounter::<AtomicU32>::new(64, 4, Some(1), 0).unwrap();
        counter.increment(&["hi", "there"][..], 1);

        let deque = VecDeque::from(["hello", "world"]);
        counter.increment(&deque, 1);
    }
}
