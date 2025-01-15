use human_bytes::human_bytes;
use std::collections::VecDeque;
use std::io;
use std::io::{BufRead, Error, ErrorKind, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value};

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use once_cell::sync::OnceCell;
use crate::bloom_filter::BloomFilter;
use crate::io::MultiStream;
use crate::log_pbar::LogProgressBar;
use crate::s3_util;
use crate::shard::shard_config::{CompressionConfig, WorkDirConfig};
use crate::shard::{find_objects_matching_patterns, FileCache};
use crate::wimbd::tokens::tokenize;



use deduper_config::*;

static GLOBAL_POOL: OnceCell<()> = OnceCell::new();


fn build_pbar(num_items: usize) -> LogProgressBar {
    let mut pbar = LogProgressBar::new(num_items);
    pbar.inc(0);
    pbar
}

pub fn run(config: DeduperConfig) -> Result<u32, u32> {
    // Set global thread count for rayon parallelism
    let start_main = Instant::now();
    if config.processes > 0 {
        GLOBAL_POOL.get_or_init(|| {
            ThreadPoolBuilder::new()
                .num_threads(config.processes)
                .build_global()
                .expect("Failed to build global thread pool")
        });
    }

    let bloom_filter = BloomFilter::initialize(&config.bloom_filter).unwrap();
    let bloom_filter = Arc::new(bloom_filter);
    let paths = find_objects_matching_patterns(&config.documents)
        .unwrap()
        .clone();

    let docs_processed = AtomicUsize::new(0);
    let seen_bytes = AtomicUsize::new(0);
    let removed_bytes = AtomicUsize::new(0);
    let failed_shard_count = AtomicU32::new(0);
    let failed_shard_count_ref = Arc::new(failed_shard_count);
    let pbar = Arc::new(Mutex::new(build_pbar(paths.len())));

    println!("Starting par iter thing");
    paths.par_iter().for_each(|p| {
        let path = p.clone();
        let work_dirs = config.work_dir.clone();
        let dedupe = config.dedupe.clone();
        let compression = match config.compression.clone() {
            Some(c) => c,
            None => CompressionConfig::infer(),
        };
        let result = write_attributes(
            path,
            work_dirs,
            dedupe,
            compression,
            bloom_filter.clone(),
            !config.is_s3_volume.unwrap_or(false),
        );
        if let Err(e) = result {
            log::error!("Failed to process {:?}: {}", p, e);
            failed_shard_count_ref.fetch_add(1, Ordering::Relaxed);
        } else {
            let (path_docs_processed, path_seen_bytes, path_removed_bytes) = result.unwrap();
            docs_processed.fetch_add(path_docs_processed, Ordering::Relaxed);
            seen_bytes.fetch_add(path_seen_bytes, Ordering::Relaxed);
            removed_bytes.fetch_add(path_removed_bytes, Ordering::Relaxed);
        }
        pbar.lock().unwrap().inc(1);
    });

    if config.bloom_filter.save_to_disk {
        let bloom_filter_file = PathBuf::from(&config.bloom_filter.file);
        log::info!("Writing bloom filter to {:?}...", config.bloom_filter.file);
        match bloom_filter.write_to_file(&bloom_filter_file) {
            Ok(_) => log::info!("Bloom filter written."),
            Err(e) => {
                log::error!("Write failed: {}", e);
                panic!("Failed to write bloom filter");
            }
        }
    }

    // Log outputs
    let failure_count = failed_shard_count_ref.load(Ordering::Relaxed);
    let seen_bytes = seen_bytes.into_inner();
    let removed_bytes = removed_bytes.into_inner();
    log::info!("----------------------------------");
    log::info!(
        "Finished processing files in {:?} (s)",
        start_main.elapsed().as_secs()
    );
    log::info!(
        "Was successful on {:?}/{:?} of the paths",
        paths.len() - failure_count as usize,
        paths.len()
    );
    if failure_count > 0 {
        log::error!("FAILED ON {:?} PATHS", failure_count);
    }
    log::info!(
        "Bloom filter has sparsity {:?}",
        bloom_filter.calculate_sparsity()
    );
    log::info!(
        "Processed {:?} documents in total",
        docs_processed.into_inner()
    );
    log::info!(
        "Processed {} of data, removed {} of them | Removal rate of {:?}",
        human_bytes(seen_bytes as f32),
        human_bytes(removed_bytes as f32),
        if seen_bytes == 0 {
            0.0
        } else {
            removed_bytes as f32 / seen_bytes as f32
        }
    );


    if failure_count == 0 {
        Ok(failure_count)
    } else {
        Err(failure_count)
    }
}

//Use the first hash to check if this dedupe key belongs to the current partition.
//If it does, return all hashes. If it doesn't, return an empty list.
fn build_hashes(
    bloom_filter: &Arc<BloomFilter>,
    dedupe_key: &VecDeque<&str>,
    num_partitions: u64,
    partition_index: u64,
) -> Vec<u64> {
    let mut hashes = vec![bloom_filter.first_hash(&dedupe_key)];
    if hashes[0] % num_partitions == partition_index {
        hashes.extend(bloom_filter.remaining_hashes(&dedupe_key));
        return hashes;
    }
    return Vec::new();
}

// Write attributes for the documents in the given file:
// For doc-level deduping, check the Bloom filter for existence of the configured key and set the configured attribute to true.
// For paragraph-level deduping, check the Bloom filter for existence of a paragraph in the text and add a span to the configured attribute.
fn write_attributes(
    docs_location: String,
    work_dirs: WorkDirConfig,
    dedupe_config: DedupeConfig,
    compression: CompressionConfig,
    bloom_filter: Arc<BloomFilter>,
    label_temp: bool,
) -> Result<(usize, usize, usize), io::Error> {
    let cache = FileCache {
        s3_client: Box::new(s3_util::new_client(None)?),
        work: work_dirs.clone(),
    };

    let mut attr_key = dedupe_config.name.clone();
    if dedupe_config.num_partitions.unwrap_or(1) > 1 {
        attr_key = format!(
            "{}_{}",
            attr_key,
            dedupe_config.partition_index.unwrap_or(0)
        );
    }

    let attrs_location = {
        let attr_prefix = format!("/attributes/{}/", attr_key);
        docs_location.replace("/documents/", &attr_prefix)
    };
    if attrs_location == docs_location {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Malformed file location: no /documents/ in file path! Continuing would overwrite data!"));
    }
    let local_output = cache.prepare_output(&attrs_location, label_temp)?;
    let mut docs_processed = 0;
    let mut seen_bytes = 0;
    let mut removed_bytes = 0;
    if local_output.exists() {
        log::info!("Skipping {:?} because it already exists", attrs_location);
        return Ok((docs_processed, seen_bytes, removed_bytes));
    }

    std::fs::create_dir_all(local_output.parent().unwrap())?;
    log::info!(
        "Writing attributes for {} to {}",
        docs_location,
        local_output.display()
    );
    {
        let local_input = cache.prepare_input(&docs_location)?;

        // The input_compression is either provided by the user or inferred from the file extension.
        // We use `infer_compression_from_temp` to deal with local files potentially including `.tmp`
        // at the end when they are cached version of S3 files.
        let input_compression: String = match compression.input {
            Some(ref input) => input.clone(),
            None => MultiStream::infer_compression_from_temp(local_input.clone()),
        };

        // for the output_compression, it is either provided by the user or we use
        // the same compression type as the input.
        let output_compression = match compression.output {
            Some(ref output) => output.clone(),
            None => input_compression.clone(),
        };

        // let's open a stream to read the input file
        let reader = MultiStream::new(
            local_input.clone(),
            Some(input_compression),
            Some(1024 * 1024),
            None,
            None,
        )
        .reader()?;

        // this is the stream we use to write the output file
        let mut writer_stream = MultiStream::new(
            local_output.clone(),
            Some(output_compression),
            Some(1024 * 1024),
            None,
            None,
        )
        .writer()?;

        for (line_number, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    log::error!(
                        "Error reading line {} of {}: {}",
                        line_number,
                        &docs_location,
                        e
                    );
                    break;
                }
            };
            docs_processed += 1;
            let data: Value = serde_json::from_str(&line)?;
            let id = data["id"].clone();
            let (attributes, doc_seen_bytes, doc_removed_bytes) =
                if dedupe_config.dedupe_method == "documents" {
                    dedupe_documents(data, dedupe_config.clone(), &bloom_filter)
                } else if dedupe_config.dedupe_method == "paragraphs" {
                    dedupe_paragraphs(data, dedupe_config.clone(), &bloom_filter)
                } else if dedupe_config.dedupe_method == "dclm" {
                    dedupe_dclm(data, dedupe_config.clone(), &bloom_filter)
                } else {
                    (json!({}), 0, 0)
                };
            seen_bytes += doc_seen_bytes;
            removed_bytes += doc_removed_bytes;
            let mut output_object = json!({});
            output_object["id"] = id;
            output_object["attributes"] = attributes;
            serde_json::to_writer(&mut writer_stream, &output_object)?;
            writer_stream.write_all(b"\n")?;
        }

        // only remove the local_input file if it is different from docs_location
        // this is to prevent deleting the original file if docs_location is a local file
        let local_input_string = String::from(local_input.to_str().unwrap());
        if local_input_string != docs_location {
            log::info!(
                "Removing local temporary file {:?} (since != {:?})",
                local_input,
                docs_location
            );
            std::fs::remove_file(local_input)?;
        } else {
            log::info!("Keeping local file {:?} after deduping...", local_input);
        }
    }

    log::info!(
        "{:?} | Saw {:?} docs and {:?} bytes, removed {:?} of them",
        docs_location,
        docs_processed,
        seen_bytes,
        removed_bytes
    );
    if label_temp {
        //Finalize output performs a rename operation, which isn't implemented in mountpoint-s3 (https://github.com/awslabs/mountpoint-s3/issues/506)
        cache.finalize_output(&attrs_location)?;
    }
    Ok((docs_processed, seen_bytes, removed_bytes))
}

/*=================================================================
=                    DEDUP SINGLE-DOCUMENT METHODS                =
=================================================================*/
/* Methods to process a single document. Roughly shared signatures here:
Inputs:
    data: serde_json::Value
    key: str key to dedup on (usually this should be "text")
    config: the DedupeConfig struct
Output:
    the json with the duplicate spans
*/

pub fn dedupe_documents(
    data: Value,
    dedupe_config: DedupeConfig,
    bloom_filter: &Arc<BloomFilter>,
) -> (Value, usize, usize) {
    let mut attributes = json!({});
    let cfg = dedupe_config.documents.unwrap();
    let min_word_count = dedupe_config.min_words.unwrap_or(0);
    let min_content_length = dedupe_config.min_length.unwrap_or(0);
    // Get the thing we're trying to dedup as 'document_key'
    let document_key = {
        let mut finder = jsonpath_rust::JsonPathFinder::from_str("{}", &cfg.key)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
            .unwrap();
        finder.set_json(Box::new(data.clone()));
        finder
            .find()
            .as_array()
            .unwrap()
            .get(0)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    };

    let seen_bytes = document_key.len();
    let mut removed_bytes = 0;
    let attr_name_with_index;
    let attr_name = if dedupe_config.num_partitions.unwrap_or(1) > 1 {
        attr_name_with_index = format!(
            "{}_{}",
            cfg.attribute_name,
            dedupe_config.partition_index.unwrap_or(0)
        );
        &attr_name_with_index
    } else {
        &cfg.attribute_name
    };

    if min_word_count > 0 {
        // Split the text into words and check the number of words.
        let words = tokenize(&document_key);
        if words.count() < min_word_count {
            // skip documents with fewer than min_word_count words
            attributes[attr_name] = Value::Array(Vec::new());
        }
    } else if document_key.len() < min_content_length {
        // skip length 0 documents
        attributes[attr_name] = Value::Array(Vec::new());
    } else if dedupe_config.skip_empty.unwrap_or(false) && document_key.trim().is_empty() {
        // skip empty documents if dedupe_config.skip_empty is true
        // and the document key is empty after trimming (i.e., removing whitespace)
        attributes[attr_name] = Value::Array(Vec::new());
    } else {
        let dedupe_key = VecDeque::from([document_key.as_str()]);

        //Just compute the first hash to see if it matches the partition
        // num_observed += 1;
        let hashes = build_hashes(
            &bloom_filter,
            &dedupe_key,
            dedupe_config.num_partitions.unwrap_or(1),
            dedupe_config.partition_index.unwrap_or(0),
        );

        if !hashes.is_empty() {
            //num_processed += 1;
            //Compute the remaining hashes
            if bloom_filter.contains(&hashes) {
                // attributes[&cfg.attribute_name] = Value::Bool(true);

                let mut duplicate_docs_array = Vec::new();
                let attr = vec![
                    Value::from(0),
                    Value::Number(document_key.len().into()),
                    Value::from(1),
                ];
                duplicate_docs_array.push(Value::Array(attr));
                removed_bytes += document_key.len();
                attributes[attr_name] = Value::Array(duplicate_docs_array);
            } else if !bloom_filter.read_only {
                bloom_filter.insert(&hashes);
            }
        } else {
            //The dedupe key doesn't belong to this partition
            attributes[attr_name] = Value::Array(Vec::new());
        }
    }
    (attributes, seen_bytes, removed_bytes)
}

pub fn dedupe_paragraphs(
    data: Value,
    dedupe_config: DedupeConfig,
    bloom_filter: &Arc<BloomFilter>,
) -> (Value, usize, usize) {
    let mut attributes = json!({});
    let cfg = dedupe_config.paragraphs.unwrap();
    let min_content_length = dedupe_config.min_length.unwrap_or(0);
    let min_word_count = dedupe_config.min_words.unwrap_or(0);
    let text = data["text"].as_str().unwrap();
    let text_length = text.len();
    let mut seen_bytes = text_length;
    let mut removed_bytes = 0;
    let mut offset = 0;

    if text_length == 0 {
        return (attributes, seen_bytes, removed_bytes)
    }


    let paragraphs = text.split(cfg.paragraph_separator.as_deref().unwrap_or("\n"));
    let mut duplicate_paragraph_spans = Vec::new();

    for p in paragraphs {
        // Get start,end in half-open intervals like [start:end)
        let par_start = offset;
        let par_char_length = p.chars().count();
        offset += par_char_length;
        if offset < text_length - 1 {
            offset += 1;
        }
        let par_end = offset;

        // Skip degenerate cases
        if par_char_length < min_content_length { continue; } 
        if tokenize(&p).count() < min_word_count { continue; }
        if dedupe_config.skip_empty.unwrap_or(true) && p.trim().is_empty() { continue; }

        // If not doing ngrams, then the whole paragraph is "one ngram" (simulated) [for code simplicity]
        let (ngram_len, stride, threshold, skip_short) = if cfg.by_ngram.is_none() || cfg.by_ngram.as_ref().unwrap().ngram_length == 0 {
            (usize::MAX, 1, 1.0, false)
        } else {
            let by_ngram = cfg.by_ngram.clone().unwrap();
            (by_ngram.ngram_length, by_ngram.stride, by_ngram.overlap_threshold, by_ngram.skip_short_paragraphs.unwrap_or(false))
        };
        

        // And now iterate through the words/tokens

        let mut current_ngram = if ngram_len < usize::MAX {
            VecDeque::with_capacity(ngram_len)}
            else {
            VecDeque::new()
        };
        let mut hashes_to_add = Vec::new();
        let mut stride_status = 0;
        for token in tokenize(&p) {
            // Warmup phase 
            if hashes_to_add.len() == 0 && current_ngram.len() < ngram_len - 1 {
                current_ngram.push_back(token);
                continue;
            }

            // Once warm, if at stride check, make hash and store it
            current_ngram.push_back(token);
            if stride_status == 0 {
                let hashes = build_hashes(
                    &bloom_filter, 
                    &current_ngram.clone(),
                    dedupe_config.num_partitions.unwrap_or(1),
                    dedupe_config.partition_index.unwrap_or(1));
                hashes_to_add.push(hashes);
            }

            stride_status = (stride_status + 1) % stride;
            current_ngram.pop_front();
        }

        // If paragraph was too short...
        if hashes_to_add.len() == 0 && !skip_short{
            let hashes = build_hashes(
                &bloom_filter, 
                &current_ngram,
            dedupe_config.num_partitions.unwrap_or(1),
                dedupe_config.partition_index.unwrap_or(1));
            hashes_to_add.push(hashes);
        }


        // Get containment numbers:
        let total_hashes = hashes_to_add.len();
        let mut contain_count = 0;
        for hash in &hashes_to_add {
            if bloom_filter.contains(&hash) {
                contain_count += 1;
            }
        }

        // If containment matches threshold, set span and add hashes
        seen_bytes += par_end - par_start;
        if (contain_count as f64 / total_hashes as f64) >= threshold.into() {
            let span = vec![
                Value::Number(par_start.into()),
                Value::Number(par_end.into()),
                Value::from(contain_count as f64 / total_hashes as f64),
            ];
            // add span to duplicate_paragraph_spans
            removed_bytes += par_end - par_start;
            duplicate_paragraph_spans.push(Value::Array(span));
        } else if !bloom_filter.read_only {
            for hash in hashes_to_add {
                bloom_filter.insert(&hash);            
            }
        }

    }
    let attr_name_with_index;
    let attr_name = if dedupe_config.num_partitions.unwrap_or(1) > 1 {
        attr_name_with_index = format!(
            "{}_{}",
            cfg.attribute_name,
            dedupe_config.partition_index.unwrap_or(0)
        );
        &attr_name_with_index
    } else {
        &cfg.attribute_name
    };
    attributes[attr_name] = Value::Array(duplicate_paragraph_spans);
    (attributes, seen_bytes, removed_bytes)
}




pub fn dedupe_dclm(
    data: Value,
    dedupe_config: DedupeConfig,
    bloom_filter: &Arc<BloomFilter>,
) -> (Value, usize, usize) {
    // Setup/init for DCLM-style dedup
    // Break into paragraphs and skip the too-short paragraphs
    // For each paragraph: if >threshold of the ngrams are seen before, mark this paragraph as duplicate
    // For whole document: amongst long paragraphs only, if >threshold of ngrams have been seen before, mark whole document as duplicate
    // If whole document is duplicate: add nothing to the bloom filter
    // If only some paragraphs are duplicates, add just those hashes to the bloom filter


    // Set things up:
    let mut attributes = json!({});
    let cfg = dedupe_config.dclm.unwrap();
    let ngram_params = cfg.by_ngram;
    let min_content_length = dedupe_config.min_length.unwrap_or(0);
    let attr_name = if dedupe_config.num_partitions.unwrap_or(1) > 1 {
        &format!(
            "{}_{}",
            cfg.attribute_name,
            dedupe_config.partition_index.unwrap_or(0)
        )
    } else {
        &cfg.attribute_name
    };
    let text = data["text"].as_str().unwrap();
    let text_length = text.len();
    let splitter = cfg.paragraph_separator.as_deref().unwrap_or("\n");
    let paragraphs = text.split(splitter);
    let mut duplicate_paragraph_spans = Vec::new();
    let mut total_ngrams = 0;
    let mut total_contained_ngrams = 0;
    let seen_bytes = text.len();
    let mut removed_bytes = 0;
    let mut hashes_to_insert: Vec<Vec<u64>> = Vec::new();
    let mut offset = 0;
    let stride = ngram_params.stride;

    if text_length == 0 { return (attributes, seen_bytes, removed_bytes); } // degenerate empty case

    
    for p in paragraphs {
        // Get par start/end
        let par_start = offset;
        let par_char_length = p.chars().count();
        offset += par_char_length;
        if offset < text_length - 1 {
            offset += splitter.len();
        }
        let par_end = offset;

        // Skip degenerate cases
        if par_char_length < min_content_length { continue; } // Skip paragraph: too short (in chars)

        // Set more things up
        let mut hashes: Vec<Vec<u64>> = Vec::new();
        let mut ngram: VecDeque<&str> = VecDeque::with_capacity(ngram_params.ngram_length);
        let mut stride_status = 0;

        // Then loop over tokens/words

        for token in tokenize(p) {
            if hashes.len() == 0 && ngram.len() < ngram_params.ngram_length - 1 {
                ngram.push_back(token);
                continue;
            }            
            ngram.push_back(token);
            if stride_status == 0 {
                let this_hash = build_hashes(
                    &bloom_filter, 
                    &ngram.clone(),
                    dedupe_config.num_partitions.unwrap_or(1),
                    dedupe_config.partition_index.unwrap_or(1));
                hashes.push(this_hash);            
            }
            stride_status = (stride_status + 1) % stride;
            ngram.pop_front();

        }
        if hashes.len() == 0 {
            continue;
        } // Skip paragraph: too short (in tokens )


          // Check containment and keep track of whether we should keep/delete this para
        let contained_ngrams = hashes.iter().filter(|h| bloom_filter.contains(h)).count();
        total_ngrams += hashes.len();
        total_contained_ngrams += contained_ngrams;

        let paragraph_duplicate_score = contained_ngrams as f32 / hashes.len() as f32;
        let should_remove = paragraph_duplicate_score >= ngram_params.overlap_threshold;

        if should_remove {
            duplicate_paragraph_spans.push(Value::Array(vec![
                Value::from(par_start),
                Value::from(par_end),
                Value::from(paragraph_duplicate_score),
            ]));
            removed_bytes += par_end - par_start;
        } else {
            hashes_to_insert.extend(hashes);
        }
    }

    // If all paragraphs in aggregate are duplicates, then make adjustments
    if total_ngrams > 0 {
        let document_score = (total_contained_ngrams / total_ngrams) as f32;
        if (total_contained_ngrams / total_ngrams) as f32 >= ngram_params.overlap_threshold {
            duplicate_paragraph_spans.clear();
            duplicate_paragraph_spans.push(Value::Array(vec![
                Value::from(0),
                Value::from(text_length),
                Value::from(document_score),
            ]));
            hashes_to_insert.clear();
        }

        if !bloom_filter.read_only {
            for h in hashes_to_insert {
                bloom_filter.insert(&h);
            }
        }
    }
    attributes[attr_name] = Value::Array(duplicate_paragraph_spans);
    (attributes, seen_bytes, removed_bytes)
}

/*=================================================================
=                           CONFIG DEFINITIONS                    =
=================================================================*/

pub mod deduper_config {
    use serde::{Deserialize, Serialize};
    use std::io;
    use std::path::PathBuf;

    use crate::bloom_filter::BloomFilterConfig;
    use crate::io::MultiStream;
    use crate::shard::shard_config::*;

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DuplicateKeyConfig {
        // Remove duplicate paragraphs
        pub paragraphs: bool,
        // Use this key to dedupe whole documents
        pub document_key: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct DocumentDedupeConfig {
        pub attribute_name: String,
        pub key: String,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct ParagraphDedupeConfig {
        pub attribute_name: String,
        // If defined, remove paragraphs based on contained ngrams
        // Otherwise, hash the entire paragraph
        pub by_ngram: Option<NgramDedupeConfig>,

        // if not defined, we use '\n' as the paragraph separator
        pub paragraph_separator: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct NgramDedupeConfig {
        // Number of whitespace-delimited tokens per ngram
        pub ngram_length: usize,
        // Number of tokens to skip between ngrams
        pub stride: usize,
        // Treat as duplicate if more than this fraction of ngrams have been seen before
        pub overlap_threshold: f32,
        // If true, skip checking for duplicates if the paragraph is shorter ngram_length + stride
        pub skip_short_paragraphs: Option<bool>,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct DCLMDedupeConfig {
        // DCLMDedupeConfig does a hybrid of both fuzzy document and paragraph level deduplication
        pub attribute_name: String,
        pub by_ngram: NgramDedupeConfig, // NOT OPTIONAL
        pub paragraph_separator: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct DedupeConfig {
        pub name: String,
        pub dedupe_method: String,
        pub documents: Option<DocumentDedupeConfig>,
        pub paragraphs: Option<ParagraphDedupeConfig>,
        pub dclm: Option<DCLMDedupeConfig>,
        pub min_length: Option<usize>,
        pub min_words: Option<usize>,
        pub skip_empty: Option<bool>,
        pub num_partitions: Option<u64>,
        pub partition_index: Option<u64>,
    }

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct DeduperConfig {
        pub documents: Vec<String>,
        pub work_dir: WorkDirConfig,
        pub dedupe: DedupeConfig,
        pub bloom_filter: BloomFilterConfig,
        pub processes: usize,
        pub is_s3_volume: Option<bool>,
        pub compression: Option<CompressionConfig>,
    }

    impl DeduperConfig {
        pub fn read_from_file(path: &str) -> Result<DeduperConfig, io::Error> {
            let config_path = PathBuf::from(path);
            let reader = MultiStream::with_default(config_path).reader()?;
            let config: DeduperConfig = serde_json::from_reader(reader)?;
            Ok(config)
        }
        pub fn parse_from_string(s: &str) -> Result<DeduperConfig, io::Error> {
            let config: DeduperConfig = serde_json::from_str(s)?;
            Ok(config)
        }
    }
}
