//! Code imported from github.com/allenai/wimbd/blob/main/src/ngrams/topk.rs
//! and modified by @soldni to integrate in dolma.

use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use ahash::RandomState;
use atomic_traits::{Atomic, NumOps};
use num_traits::One;

/// A collection for tracking the top-k ngrams in a corpus.
pub struct TopKNgrams<T, A>
where
    T: Ord + Clone + Sized,
    A: Atomic + NumOps,
    <A as Atomic>::Type: One + Ord + Clone + Copy,
{
    k: usize,
    topk: BTreeSet<(<A as Atomic>::Type, Rc<Vec<T>>)>,
    ngrams: HashMap<Rc<Vec<T>>, <A as Atomic>::Type, RandomState>,
    pub(crate) min_count: <A as Atomic>::Type,
    min_count_atomic: Arc<A>,
}

impl<T, A> TopKNgrams<T, A>
where
    T: Ord + Clone + Sized + Hash,
    A: Atomic + NumOps,
    <A as Atomic>::Type: One + Ord + Clone + Copy,
{
    pub fn new(k: usize) -> Self {
        Self {
            k,
            topk: BTreeSet::new(),
            ngrams: HashMap::with_capacity_and_hasher(k + 1, RandomState::new()),
            min_count: <A as Atomic>::Type::one(),
            min_count_atomic: Arc::new(<A as Atomic>::new(<A as Atomic>::Type::one())),
        }
    }

    pub fn min_count(&self) -> Arc<A> {
        self.min_count_atomic.clone()
    }

    pub fn insert(&mut self, ngram: Vec<T>, count: <A as Atomic>::Type) {
        if count >= self.min_count {
            let ngram = Rc::new(ngram);

            if let Some(old_count) = self.ngrams.get_mut(&ngram) {
                if count <= *old_count {
                    // Nothing to do, return early
                    return;
                }

                // Update existing count for ngram.
                self.topk.remove(&(*old_count, ngram.clone()));
                *old_count = count;
            } else {
                self.ngrams.insert(ngram.clone(), count);
            }

            self.topk.insert((count, ngram.clone()));
        }

        // Update min count if needed.
        let mut update_min_count = false;
        while self.topk.len() > self.k {
            let (_, ngram) = self.topk.pop_first().unwrap();
            self.ngrams.remove(&ngram);
            update_min_count = true;
        }
        if update_min_count {
            if let Some((new_min_count, _)) = self.topk.first() {
                if *new_min_count != self.min_count {
                    self.min_count = *new_min_count;
                    self.min_count_atomic
                        .store(*new_min_count, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn drain(&mut self) -> Vec<(Rc<Vec<T>>, <A as Atomic>::Type)> {
        let mut out: Vec<(Rc<Vec<T>>, <A as Atomic>::Type)> = Vec::with_capacity(self.k);
        while let Some((count, ngram)) = self.topk.pop_last() {
            self.ngrams.remove(&ngram);
            out.push((ngram, count))
        }
        self.min_count = <A as Atomic>::Type::one();
        self.min_count_atomic
            .store(<A as Atomic>::Type::one(), Ordering::Relaxed);
        out
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::atomic::AtomicU32;

    use super::TopKNgrams;

    #[test]
    fn test_adding_same_ngram_multiple_times() {
        let mut topk: TopKNgrams<String, AtomicU32> = TopKNgrams::new(3);

        let ngram1 = vec!["foo".into(), "bar".into()];
        let ngram2 = vec!["bar".into(), "baz".into()];
        let ngram3 = vec!["baz".into(), "foo".into()];

        // Insert 3 unique ngrams.
        topk.insert(ngram1.clone(), 3);
        topk.insert(ngram2, 2);
        topk.insert(ngram3, 1);

        // Now try inserting a duplicate.
        topk.insert(ngram1.clone(), 3);
        assert_eq!(topk.ngrams.len(), 3);
        assert_eq!(topk.topk.len(), 3);
        assert_eq!(topk.ngrams.get(&Rc::new(ngram1.clone())), Some(&3));

        // And insert the same ngram with a new count.
        topk.insert(ngram1.clone(), 4);
        assert_eq!(topk.ngrams.len(), 3);
        assert_eq!(topk.topk.len(), 3);
        assert_eq!(topk.ngrams.get(&Rc::new(ngram1.clone())), Some(&4));

        // And insert the same ngram with a lower count.
        topk.insert(ngram1.clone(), 2);
        assert_eq!(topk.ngrams.len(), 3);
        assert_eq!(topk.topk.len(), 3);
        assert_eq!(topk.ngrams.get(&Rc::new(ngram1)), Some(&4));
    }
}
