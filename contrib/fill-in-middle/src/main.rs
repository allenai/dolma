use clap::Parser;
use std::io::{BufRead, Error};
use std::path::PathBuf;
use std::fs::{File, create_dir_all};
use std::io::Read;

use glob::glob;
use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use rayon::prelude::*;
use serde_json;
use zstd::stream::Decoder;

mod fim;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input files to process
    #[arg(short, long, required = true)]
    inputs: Vec<PathBuf>,

    /// Destination output file
    #[arg(short, long, required = true)]
    output: PathBuf,

    /// Rate at which to perform FIM reordering
    #[arg(long, required = true)]
    fim_rate: f32,

    /// Rate at which to perform Prefix-Suffix-Middle vs Suffix-Prefix-Middle reordering
    #[arg(long, required = true)]
    psm_spm_split: f32,

    /// Value of the file separator token
    #[arg(long, default_value = "<|file_sep|>")]
    file_separator_token: String,

    /// Value of the fill-in-middle prefix sentinel token
    #[arg(long, required = true, default_value = "<|fim_prefix|>")]
    fim_prefix_token: String,

    /// Value of the fill-in-middle middle sentinel token
    #[arg(long, required = true, default_value = "<|fim_middle|>")]
    fim_middle_token: String,

    /// Value of the fill-in-middle suffix sentinel token
    #[arg(long, required = true, default_value = "<|fim_suffix|>")]
    fim_suffix_token: String,

    /// Optional replacement string for file separator token in output documents
    #[arg(long)]
    file_separator_replacement: Option<String>,
}

/// Compute the longest common prefix of a non-empty slice of paths.
fn compute_common_prefix(paths: &[PathBuf]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }

    // Split each path into its components.
    let components: Vec<Vec<&std::ffi::OsStr>> = paths
        .iter()
        .map(|p| p.components().map(|c| c.as_os_str()).collect())
        .collect();

    // Find the minimum number of components among all paths.
    let min_len = components.iter().map(|comp| comp.len()).min().unwrap_or(0);

    let mut common = PathBuf::new();
    // Iterate index-by-index comparing components.
    for i in 0..min_len {
        let candidate = components[0][i];
        if components.iter().all(|comp| comp[i] == candidate) {
            common.push(candidate);
        } else {
            break;
        }
    }
    common
}

/// Given a list of input paths and a destination prefix,
/// returns a vector where each file is the destination prefix plus
/// the input file’s path relative to the shared prefix.
fn map_paths_to_destination(inputs: &Vec<PathBuf>, dest_prefix: PathBuf) -> Vec<PathBuf> {
    if inputs.is_empty() {
        return Vec::new();
    }

    if inputs.len() == 1 {
        // get filename from single input path
        let src_filename = inputs[0].file_name().unwrap();
        let dst_filename = dest_prefix.join(src_filename);
        return vec![dst_filename];
    }

    let common_prefix = compute_common_prefix(&inputs);
    inputs
        .into_iter()
        .map(|input| {
            // Calculate the relative path from the common prefix.
            let relative = input
                .strip_prefix(&common_prefix)
                .expect("All inputs should share the common prefix");
            // Build the new destination path.
            let mut new_path = dest_prefix.clone();
            new_path.push(relative);
            new_path
        })
        .collect()
}

fn find_all_paths(inputs: Vec<PathBuf>) -> Vec<PathBuf> {
    let all_paths: Vec<PathBuf> = inputs
        .into_iter()
        .flat_map(|path| {
            let path_str = path.to_string_lossy();
            
            // Check if the path string contains wildcards (including recursive **)
            if path_str.contains('*') || path_str.contains('?') || path_str.contains('[') {
                // Handle wildcard patterns directly with glob
                match glob(&path_str) {
                    Ok(entries) => {
                        let mut found_paths: Vec<PathBuf> = entries.filter_map(Result::ok).collect();
                        // Sort to ensure consistent ordering
                        found_paths.sort();
                        found_paths
                    },
                    Err(e) => {
                        eprintln!("Glob pattern error for '{}': {}", path_str, e);
                        Vec::new()
                    }
                }
            } else if path.is_file() {
                // For a single file, just return it
                vec![path]
            } else if path.is_dir() {
                // Handle directory paths - expand recursively
                let manual_ext: Option<Vec<String>>;
                let input_paths: Vec<PathBuf>;

                match path.extension() {
                    Some(ext) => {
                        let ext_str = ext.to_string_lossy().into_owned();
                        manual_ext = Some(vec![ext_str]);
                        let mut trunk_path = path.clone();
                        trunk_path.pop();
                        input_paths = vec![trunk_path];
                    }
                    None => {
                        manual_ext = None;
                        input_paths = vec![path];
                    }
                }

                let manual_ext_refs: Option<Vec<&str>> = manual_ext
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect());
                
                expand_dirs(input_paths, manual_ext_refs.as_deref()).unwrap_or_default()
            } else {
                // Path doesn't exist - try as glob pattern anyway
                match glob(&path_str) {
                    Ok(entries) => {
                        let mut found_paths: Vec<PathBuf> = entries.filter_map(Result::ok).collect();
                        found_paths.sort();
                        found_paths
                    },
                    Err(_) => {
                        eprintln!("Warning: Path '{}' does not exist and is not a valid glob pattern", path_str);
                        Vec::new()
                    }
                }
            }
        })
        .collect();
    
    all_paths
}

// Function to read either normal or zstd-compressed file
fn read_file_with_zstd_support(path: &PathBuf) -> Result<Vec<u8>, Error> {
    // Check if file has .zst extension
    let is_zstd = path.extension().map_or(false, |ext| ext == "zst") || 
                 path.to_string_lossy().ends_with(".jsonl.zst");
    
    if is_zstd {
        // Handle zstd compressed file
        let file = File::open(path)?;
        let mut decoder = Decoder::new(file)?;
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;
        Ok(buffer)
    } else {
        // Convert from BufReader<Cursor<Vec<u8>>> to Vec<u8>
        match read_pathbuf_to_mem(path) {
            Ok(buf_reader) => {
                let mut buffer = Vec::new();
                for line in buf_reader.lines() {
                    let line = line?;
                    buffer.extend_from_slice(line.as_bytes());
                    buffer.push(b'\n');
                }
                Ok(buffer)
            },
            Err(e) => Err(Error::new(std::io::ErrorKind::Other, format!("Failed to read file: {:?}", e))),
        }
    }
}

fn process_single(
    src_path: &PathBuf,
    dst_path: &PathBuf,
    fim_rate: f32,
    psm_spm_split: f32,
    file_separator_token: &str,
    fim_prefix_token: &str,
    fim_middle_token: &str,
    fim_suffix_token: &str,
    file_separator_replacement: Option<&str>,
) -> Result<(), Error> {
    let mut fim = fim::FillInMiddle {
        fim_rate,
        psm_spm_split,
        file_separator_token,
        fim_prefix_token,
        fim_middle_token,
        fim_suffix_token,
    };

    println!("Processing {:?} -> {:?}", src_path, dst_path);
    
    // Ensure the output directory exists
    if let Some(parent_dir) = dst_path.parent() {
        create_dir_all(parent_dir)?;
    }
    
    let src_buf = read_file_with_zstd_support(src_path)?;
    let mut out_bytes: Vec<u8> = Vec::new();
    let newline: u8 = b'\n';

    for line in src_buf.lines() {
        let line = line.unwrap();
        let mut json_obj: serde_json::Value = serde_json::from_str(&line).unwrap();

        let src_text = json_obj.get("text").unwrap().as_str().unwrap();
        let new_text = fim.perform_on_document_text(src_text);
        
        json_obj["text"] = serde_json::Value::String(new_text);

        let mut json_bytes = serde_json::to_vec(&json_obj)?;
        
        // Apply file separator replacement after JSON serialization to avoid escaping
        if let Some(replacement) = file_separator_replacement {
            let json_str = String::from_utf8(json_bytes)
                .map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e))?;
            let separator_with_newlines = format!("\\n{}\\n", replacement);
            let modified_json = json_str.replace("<|file_sep|>", &separator_with_newlines);
            json_bytes = modified_json.into_bytes();
        }
        
        out_bytes.extend_from_slice(&json_bytes);
        out_bytes.push(newline);
    }

    write_mem_to_pathbuf(&out_bytes, dst_path).unwrap();
    Ok(())
}

fn main() {
    // parse command line arguments
    let args: Args = Args::parse();

    // for each prefix, we derive both
    let all_src: Vec<PathBuf> = find_all_paths(args.inputs);

    println!("Found {} paths to process", all_src.len());

    let all_dst = map_paths_to_destination(&all_src, args.output.clone());

    let pbar = build_pbar(all_src.len(), "Processing files");

    // here we can use rayon to parallelize the mapping operation
    all_src
        .into_iter()
        .zip(all_dst.into_iter())
        .collect::<Vec<_>>()
        .par_iter()
        .for_each(|(src_path, dst_path)| {
            process_single(
                src_path,
                dst_path,
                args.fim_rate,
                args.psm_spm_split,
                &args.file_separator_token,
                &args.fim_prefix_token,
                &args.fim_middle_token,
                &args.fim_suffix_token,
                args.file_separator_replacement.as_deref(),
            )
            .unwrap();
            pbar.inc(1);
        });
}
