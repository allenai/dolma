use ahash::RandomState;
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use human_bytes::human_bytes;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{create_dir_all, OpenOptions};
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::io::{BufReader, BufWriter, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use sysinfo::System;

mod bloom_test;
// A thread-safe bloom filter.
pub struct BloomFilter {
    bits: Vec<AtomicU32>,
    hash_builder_seeds: Vec<[u64; 4]>,
    // RandomState does not store its seeds, so we have to store them ourselves.
    hash_builders: Vec<RandomState>,
    pub read_only: bool,
}

impl BloomFilter {
    const MAGIC: u32 = 0x81F0F117;
    const VERSION: u32 = 1;

    pub fn optimal_number_of_hashers(size_in_bytes: usize, expected_elements: usize) -> usize {
        let expected_elements = expected_elements as f64;
        let size_in_bits = (size_in_bytes * 8) as f64;
        let k = (size_in_bits / expected_elements) * (2.0f64.ln());
        k.ceil() as usize
    }

    pub fn prob_of_false_positive(
        size_in_bytes: usize,
        expected_elements: usize,
        num_hashers: usize,
    ) -> f64 {
        let k = num_hashers as f64;
        let m = (size_in_bytes * 8) as f64;
        let n = expected_elements as f64;
        (1.0 - (1.0 - (1.0 / m)).powf(k * n)).powf(k)
    }

    pub fn suggest_size_in_bytes(
        expected_elements: usize,
        desired_false_positive_rate: f64,
    ) -> usize {
        let mut size_in_bytes = 1024 * 1024;
        while size_in_bytes < usize::MAX / 2
            && Self::prob_of_false_positive(
                size_in_bytes,
                expected_elements,
                Self::optimal_number_of_hashers(size_in_bytes, expected_elements),
            ) > desired_false_positive_rate
        {
            size_in_bytes *= 2;
        }
        size_in_bytes
    }

    pub fn compute_bloom_size_binsearch(
        expected_elements: usize,
        fp_rate: f64,
        sysram_limit: Option<f64>,
        num_hashers: usize,
    ) -> usize {
        /* Uses binary search to get a finer-grained bloom filter size.
           If limit_to_system: guarantees that no more than 90% of RAM gets allocated
           If num_hashers == 0: computes the optimal number of hashers on the fly
        */

        // Get 90% of System RAM and set binsearch bounds
        let mut sys = System::new_all();
        sys.refresh_all();
        let sysram_limit: f64 = sysram_limit.unwrap_or(0.0);
        let mut lo = 1 as usize;
        let mut hi = if sysram_limit > 0.0 {
            ((sys.total_memory() as f64) * sysram_limit) as usize
        } else {
            std::usize::MAX / 8
        };

        let compute_hashers = num_hashers == 0;
        let num_hashers = if num_hashers == 0 {
            BloomFilter::optimal_number_of_hashers(hi, expected_elements)
        } else {
            num_hashers
        };

        if (sysram_limit > 0.0)
            && BloomFilter::prob_of_false_positive(hi, expected_elements, num_hashers) > fp_rate
        {
            log::info!("WARNING: TO achieve desired false-positive rate, you'd need >90% of system RAM. Defaulting to {:?} SysRAM", sysram_limit);
            return hi;
        }

        // Do BinSearch
        while lo < hi - 1 {
            let mid = lo + (hi - lo) / 2;
            let num_hashers = if compute_hashers {
                BloomFilter::optimal_number_of_hashers(mid, expected_elements)
            } else {
                num_hashers
            };
            let computed_fp =
                BloomFilter::prob_of_false_positive(mid, expected_elements, num_hashers);
            if computed_fp > fp_rate {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        hi
    }

    #[allow(dead_code)]
    pub fn my_prob_of_false_positive(&self, expected_elements: usize) -> f64 {
        Self::prob_of_false_positive(
            self.size_in_bytes(),
            expected_elements,
            self.hash_builders.len(),
        )
    }

    pub fn calculate_sparsity(&self) -> f64 {
        let set_bits: usize = self
            .bits
            .par_iter()
            .map(|atomic| {
                let value = atomic.load(std::sync::atomic::Ordering::Relaxed);
                value.count_ones() as usize
            })
            .sum();
        let total_bits = self.size_in_bytes() * 8;
        (set_bits as f64) / (total_bits as f64)
    }

    #[allow(dead_code)]
    pub fn size_in_bytes(&self) -> usize {
        self.bits.len() * size_of::<AtomicU32>()
    }

    pub fn new(size_in_bytes: usize, num_hashers: usize, read_only: bool) -> Self {
        let mut rng = rand::thread_rng();
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers);
        let mut hash_builders = Vec::with_capacity(num_hashers);
        for _ in 0..num_hashers {
            let seeds = rng.gen::<[u64; 4]>();
            hash_builders.push(RandomState::with_seeds(
                seeds[0], seeds[1], seeds[2], seeds[3],
            ));
            hash_builder_seeds.push(seeds);
        }

        let number_of_u32 = size_in_bytes / size_of::<AtomicU32>();
        let bits = (0..number_of_u32)
            .into_par_iter()
            .map(|_| AtomicU32::default())
            .collect();
        Self {
            bits,
            hash_builder_seeds,
            hash_builders,
            read_only,
        }
    }

    pub fn from_file(path: &PathBuf, read_only: bool) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(path)?;
        let mut stream = BufReader::new(&mut file);

        let magic: u32 = stream.read_u32::<LittleEndian>()?;
        if magic != Self::MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
        }

        let version: u32 = stream.read_u32::<LittleEndian>()?;
        if version != Self::VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid version",
            ));
        }

        let num_hashers: u32 = stream.read_u32::<LittleEndian>()?;
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers as usize);
        let mut hash_builders = Vec::with_capacity(num_hashers as usize);
        for _ in 0..num_hashers {
            let seeds = [
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
            ];
            hash_builders.push(RandomState::with_seeds(
                seeds[0], seeds[1], seeds[2], seeds[3],
            ));
            hash_builder_seeds.push(seeds);
        }

        let number_of_elements = stream.read_u64::<LittleEndian>()?;
        let mut bits = Vec::with_capacity(number_of_elements as usize);
        for _ in 0..number_of_elements {
            bits.push(AtomicU32::new(stream.read_u32::<NativeEndian>()?));
        }

        Ok(Self {
            bits,
            hash_builder_seeds,
            hash_builders,
            read_only,
        })
    }

    pub fn write_to_file(&self, path: &PathBuf) -> io::Result<()> {
        create_dir_all(path.parent().unwrap())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let mut stream = BufWriter::new(&file);

        stream.write_u32::<LittleEndian>(Self::MAGIC)?;
        stream.write_u32::<LittleEndian>(Self::VERSION)?;
        stream.write_u32::<LittleEndian>(self.hash_builder_seeds.len() as u32)?;
        for hash_builder_seed in &self.hash_builder_seeds {
            for seed in hash_builder_seed {
                stream.write_u64::<LittleEndian>(*seed)?;
            }
        }

        stream.write_u64::<LittleEndian>(self.bits.len() as u64)?;
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                self.bits.as_ptr() as *const u8,
                self.bits.len() * size_of::<AtomicU32>(),
            );
            stream.write_all(bytes)?;
        };

        Ok(())
    }

    pub fn hashes(&self, s: &VecDeque<&str>) -> Vec<u64> {
        self.partial_hashes(s, Some(0), None)
    }

    pub fn first_hash(&self, s: &VecDeque<&str>) -> u64 {
        self.partial_hashes(s, Some(0), Some(1))[0]
    }
    pub fn remaining_hashes(&self, s: &VecDeque<&str>) -> Vec<u64> {
        self.partial_hashes(s, Some(1), None)
    }

    pub fn partial_hashes(
        &self,
        s: &VecDeque<&str>,
        start_index: Option<usize>,
        end_index: Option<usize>,
    ) -> Vec<u64> {
        let start = start_index.unwrap_or(0);
        let end = end_index.unwrap_or(self.hash_builders.len());

        self.hash_builders
            .iter()
            .skip(start)
            .take(end - start)
            .map(|hash_builder| {
                let mut hasher = hash_builder.build_hasher();
                s.hash(&mut hasher);
                hasher.finish()
            })
            .collect()
    }

    // No-op if read-only
    pub fn insert(&self, hashes: &Vec<u64>) {
        if !self.read_only {
            for hash in hashes {
                let hash = *hash as usize;
                let index = hash / 32 % self.bits.len();
                let bit = hash % 32;
                self.bits[index].fetch_or(1 << bit, Ordering::Relaxed);
            }
        }
    }

    pub fn contains(&self, hashes: &Vec<u64>) -> bool {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            if self.bits[index].load(Ordering::Relaxed) & (1 << bit) == 0 {
                return false;
            }
        }
        true
    }

    pub fn initialize(config: &BloomFilterConfig) -> Result<BloomFilter, io::Error> {
        let save_file = PathBuf::from(&config.file);
        let bloom_filter = if save_file.exists() {
            log::info!("Loading bloom filter from {:?}...", save_file.display());
            BloomFilter::from_file(&save_file, config.read_only).unwrap()
        } else {
            log::info!("Creating new bloom filter...");
            let mut bloom_filter_size: usize = config.size_in_bytes;
            if bloom_filter_size == 0 {
                bloom_filter_size = BloomFilter::compute_bloom_size_binsearch(
                    config.estimated_doc_count,
                    config.desired_false_positive_rate,
                    config.sysram_limit,
                    0,
                );
                log::info!("Creating bloom filter with size {} bytes to achieve false positive rate {} for {} elements", human_bytes(bloom_filter_size as f64), config.desired_false_positive_rate, config.estimated_doc_count);
            }
            let num_hashers = BloomFilter::optimal_number_of_hashers(
                bloom_filter_size,
                config.estimated_doc_count,
            );
            let p = BloomFilter::prob_of_false_positive(
                bloom_filter_size,
                config.estimated_doc_count,
                num_hashers,
            );
            log::info!(
                "Bloom filter will have size {}, {} hashers, false positive rate {}.",
                human_bytes(bloom_filter_size as f64),
                num_hashers,
                p
            );
            BloomFilter::new(bloom_filter_size, num_hashers, config.read_only)
        };

        Ok(bloom_filter)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BloomFilterConfig {
    pub file: String,
    pub size_in_bytes: usize,
    pub read_only: bool,
    pub estimated_doc_count: usize,
    pub desired_false_positive_rate: f64,
    pub save_to_disk: bool,
    pub sysram_limit: Option<f64>,
}
