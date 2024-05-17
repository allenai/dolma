use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::{json, Value};
use threadpool::ThreadPool;

use crate::bloom_filter::BloomFilter;
use crate::s3_util;
use crate::shard::shard_config::WorkDirConfig;
use crate::shard::{find_objects_matching_patterns, FileCache};
use crate::wimbd::tokens::tokenize;

use deduper_config::*;

pub fn run(config: DeduperConfig) -> Result<u32, u32> {
    let bloom_filter = BloomFilter::initialize(&config.bloom_filter).unwrap();
    let bloom_filter = Arc::new(bloom_filter);

    let paths = find_objects_matching_patterns(&config.documents)
        .unwrap()
        .clone();

    if !(config.dedupe.paragraphs.is_some() ^ config.dedupe.documents.is_some()) {
        log::error!("Must dedupe either paragraphs or documents");
        return Err(paths.len() as u32);
    }

    let threadpool = ThreadPool::new(config.processes);
    let failed_shard_count = AtomicU32::new(0);
    let failed_shard_count_ref = Arc::new(failed_shard_count);
    for p in paths {
        let path = p.clone();
        let work_dirs = config.work_dir.clone();
        let dedupe = config.dedupe.clone();
        let bloom_filter = bloom_filter.clone();
        let failed_shard_count_ref = failed_shard_count_ref.clone();
        threadpool.execute(move || {
            let result = write_attributes(path, work_dirs, dedupe, bloom_filter);
            if let Err(e) = result {
                log::error!("Failed to process {:?}: {}", p, e);
                failed_shard_count_ref.fetch_add(1, Ordering::Relaxed);
            }
        });
    }
    threadpool.join();

    let bloom_filter_file = PathBuf::from(&config.bloom_filter.file);
    log::info!("Writing bloom filter to {:?}...", config.bloom_filter.file);
    match bloom_filter.write_to_file(&bloom_filter_file) {
        Ok(_) => log::info!("Bloom filter written."),
        Err(e) => {
            log::error!("Write failed: {}", e);
            panic!("Failed to write bloom filter");
        }
    }

    let failure_count = failed_shard_count_ref.load(Ordering::Relaxed);
    if failure_count == 0 {
        log::info!("Done!");
        Ok(failure_count)
    } else {
        log::error!("{} shards failed to process.", failure_count);
        Err(failure_count)
    }
}

// Write attributes for the documents in the given file:
// For doc-level deduping, check the Bloom filter for existence of the configured key and set the configured attribute to true.
// For paragraph-level deduping, check the Bloom filter for existence of a paragraph in the text and add a span to the configured attribute.
fn write_attributes(
    docs_location: String,
    work_dirs: WorkDirConfig,
    dedupe_config: DedupeConfig,
    bloom_filter: Arc<BloomFilter>,
) -> Result<(), io::Error> {
    let cache = FileCache {
        s3_client: Box::new(s3_util::new_client(None)?),
        work: work_dirs.clone(),
    };

    let attrs_location = {
        let attr_prefix = format!("/attributes/{}/", &dedupe_config.name);
        docs_location.replace("/documents/", &attr_prefix)
    };
    let local_output = cache.prepare_output(&attrs_location)?;
    if local_output.exists() {
        log::info!("Skipping {:?} because it already exists", attrs_location);
        return Ok(());
    }
    log::info!(
        "Writing attributes for {} to {}",
        docs_location,
        local_output.display()
    );

    std::fs::create_dir_all(local_output.parent().unwrap())?;
    log::info!(
        "Writing attributes for {} to {}",
        docs_location,
        local_output.display()
    );
    {
        let local_input = cache.prepare_input(&docs_location)?;

        let input_file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(local_input.clone())?;
        let reader = BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(input_file));

        let tmp_output = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&local_output)?;

        let mut writer = BufWriter::with_capacity(
            1024 * 1024,
            GzEncoder::new(tmp_output, Compression::default()),
        );

        let min_content_length = dedupe_config.min_length.unwrap_or(0);
        let min_word_count = dedupe_config.min_words.unwrap_or(0);

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
            let data: Value = serde_json::from_str(&line)?;
            let mut attributes = json!({});

            if let Some(ref cfg) = dedupe_config.documents {
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

                if min_word_count > 0 {
                    // Split the text into words and check the number of words.
                    let words = tokenize(&document_key);
                    if words.count() < min_word_count {
                        // skip documents with fewer than min_word_count words
                        attributes[&cfg.attribute_name] = Value::Array(Vec::new());
                    }
                } else if document_key.len() < min_content_length {
                    // skip length 0 documents
                    attributes[&cfg.attribute_name] = Value::Array(Vec::new());
                } else if dedupe_config.skip_empty.unwrap_or(false)
                    && document_key.trim().is_empty()
                {
                    // skip empty documents if dedupe_config.skip_empty is true
                    // and the document key is empty after trimming (i.e., removing whitespace)
                    attributes[&cfg.attribute_name] = Value::Array(Vec::new());
                } else {
                    let dedupe_key = VecDeque::from([document_key.as_str()]);
                    let hashes = bloom_filter.hashes(&dedupe_key);
                    if bloom_filter.contains(&hashes) {
                        // attributes[&cfg.attribute_name] = Value::Bool(true);

                        let mut duplicate_docs_array = Vec::new();
                        let attr = vec![
                            Value::from(0),
                            Value::Number(document_key.len().into()),
                            Value::from(1),
                        ];
                        duplicate_docs_array.push(Value::Array(attr));
                        attributes[&cfg.attribute_name] = Value::Array(duplicate_docs_array);
                    } else if !bloom_filter.read_only {
                        bloom_filter.insert(&hashes);
                    }
                }
            }
            match dedupe_config.paragraphs {
                None => {}
                Some(ref cfg) => {
                    // Split the text into paragraphs and check each one.
                    let text = data["text"].as_str().unwrap();
                    let text_length = text.len();
                    let mut offset = 0;

                    if text_length > 0 {
                        let paragraphs =
                            text.split(cfg.paragraph_separator.as_deref().unwrap_or("\n"));
                        let mut duplicate_paragraph_spans = Vec::new();

                        // skip empty documents if text_length is 0
                        for p in paragraphs {
                            let par_start = offset;
                            offset += p.chars().count();
                            if offset < text_length - 1 {
                                offset += 1; // For the newline
                            }
                            let par_end = offset;

                            if offset < min_content_length {
                                // skip length 0 paragraphs
                                continue;
                            }
                            if min_word_count > 0 {
                                // Split the text into words and check the number of words.
                                let words = tokenize(&p);

                                if words.count() < min_word_count {
                                    // skip documents with fewer than min_words words
                                    continue;
                                }
                            } else if dedupe_config.skip_empty.unwrap_or(false)
                                && p.trim().is_empty()
                            {
                                // skip empty paragraphs if dedupe_config.skip_empty is true
                                // and the paragraph is empty after trimming (i.e., removing whitespace)
                                continue;
                            } else {
                                if cfg.by_ngram.is_none()
                                    || cfg.by_ngram.as_ref().unwrap().ngram_length == 0
                                {
                                    // Dedupe the entire paragraph
                                    let dedupe_key = VecDeque::from([p]);
                                    let hashes = bloom_filter.hashes(&dedupe_key);
                                    if bloom_filter.contains(&hashes) {
                                        let span = vec![
                                            Value::Number(par_start.into()),
                                            Value::Number(par_end.into()),
                                            Value::from(1),
                                        ];
                                        // add span to duplicate_paragraph_spans
                                        duplicate_paragraph_spans.push(Value::Array(span));
                                    } else if !bloom_filter.read_only {
                                        bloom_filter.insert(&hashes);
                                    }
                                } else {
                                    // Dedupe by ngram overlap
                                    let by_ngram = cfg.clone().by_ngram.unwrap();
                                    let ngram_length = by_ngram.ngram_length;
                                    let stride = by_ngram.stride;
                                    let mut ngram: VecDeque<&str> =
                                        VecDeque::with_capacity(ngram_length);
                                    let mut word_index = 0;
                                    let mut last_ngram_start = 0;
                                    let mut ngram_count = 0;
                                    let mut duplicate_ngram_count = 0;
                                    for token in tokenize(p) {
                                        ngram.push_back(token);
                                        if ngram.len() == ngram_length {
                                            let ngram_start = word_index - (ngram_length - 1);
                                            if last_ngram_start == 0
                                                || ngram_start - last_ngram_start >= stride
                                            {
                                                last_ngram_start = ngram_start;
                                                ngram_count += 1;
                                                let dedupe_key = VecDeque::from(ngram.clone());
                                                let hashes = bloom_filter.hashes(&dedupe_key);

                                                if bloom_filter.contains(&hashes) {
                                                    duplicate_ngram_count += 1;
                                                } else if !bloom_filter.read_only {
                                                    bloom_filter.insert(&hashes);
                                                }
                                            }
                                            ngram.pop_front();
                                        }
                                        word_index += 1;
                                    }
                                    if ngram_count < 2
                                        && !by_ngram.skip_short_paragraphs.unwrap_or(false)
                                    {
                                        // Too few ngrams to dedupe by overlap. Just compare the whole thing
                                        let dedupe_key = VecDeque::from([p]);
                                        let hashes = bloom_filter.hashes(&dedupe_key);

                                        let span_score = match bloom_filter.contains(&hashes) {
                                            // we found a match! score is 1.0
                                            true => 1.0,
                                            false => {
                                                // this is a new paragraph, push to bloom filter
                                                if !bloom_filter.read_only {
                                                    bloom_filter.insert(&hashes);
                                                }
                                                // score is 0.0 because it's not a duplicate
                                                0.0
                                            }
                                        };

                                        // we check if the score is above the threshold; note that
                                        // users can set the threshold to 0.0 to always include the span,
                                        // or 1.0 to only include spans that are exact duplicates.
                                        if span_score >= by_ngram.overlap_threshold {
                                            let span = vec![
                                                Value::Number(par_start.into()),
                                                Value::Number(par_end.into()),
                                                Value::from(span_score),
                                            ];
                                            // add span to duplicate_paragraph_spans
                                            duplicate_paragraph_spans.push(Value::Array(span));
                                        }
                                    } else {
                                        let overlap_fraction =
                                            duplicate_ngram_count as f32 / ngram_count as f32;
                                        if overlap_fraction >= by_ngram.overlap_threshold {
                                            let span = vec![
                                                Value::Number(par_start.into()),
                                                Value::Number(par_end.into()),
                                                Value::from(overlap_fraction),
                                            ];
                                            // add span to duplicate_paragraph_spans
                                            duplicate_paragraph_spans.push(Value::Array(span));
                                        }
                                    }
                                }
                            }
                        }
                        attributes[&cfg.attribute_name] = Value::Array(duplicate_paragraph_spans);
                    }
                }
            }
            let mut output_object = json!({});
            output_object["id"] = data["id"].clone();
            output_object["attributes"] = attributes;
            serde_json::to_writer(&mut writer, &output_object)?;
            writer.write_all(b"\n")?;
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
    cache.finalize_output(&attrs_location)?;
    Ok(())
}

pub mod deduper_config {
    use serde::{Deserialize, Serialize};
    use std::fs::File;
    use std::io;

    use crate::bloom_filter::BloomFilterConfig;
    use crate::shard::shard_config::*;

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DuplicateKeyConfig {
        // Remove duplicate paragraphs
        pub paragraphs: bool,
        // Use this key to dedupe whole documents
        pub document_key: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DocumentDedupeConfig {
        pub attribute_name: String,
        pub key: String,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct ParagraphDedupeConfig {
        pub attribute_name: String,
        // If defined, remove paragraphs based on contained ngrams
        // Otherwise, hash the entire paragraph
        pub by_ngram: Option<NgramDedupeConfig>,

        // if not defined, we use '\n' as the paragraph separator
        pub paragraph_separator: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone)]
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

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DedupeConfig {
        pub name: String,
        pub documents: Option<DocumentDedupeConfig>,
        pub paragraphs: Option<ParagraphDedupeConfig>,
        pub min_length: Option<usize>,
        pub min_words: Option<usize>,
        pub skip_empty: Option<bool>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DeduperConfig {
        pub documents: Vec<String>,
        pub work_dir: WorkDirConfig,
        pub dedupe: DedupeConfig,
        pub bloom_filter: BloomFilterConfig,
        pub processes: usize,
    }

    impl DeduperConfig {
        pub fn read_from_file(path: &str) -> Result<DeduperConfig, io::Error> {
            let file = File::open(path)?;
            let reader = io::BufReader::new(file);
            let config: DeduperConfig = serde_json::from_reader(reader)?;
            Ok(config)
        }
        pub fn parse_from_string(s: &str) -> Result<DeduperConfig, io::Error> {
            let config: DeduperConfig = serde_json::from_str(s)?;
            Ok(config)
        }
    }
}
