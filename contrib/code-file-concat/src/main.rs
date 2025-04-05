use clap::Parser;
use std::io::{BufRead, Error};
use std::path::PathBuf;

use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use rayon::prelude::*;
use serde_json;

mod concat;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input files to process
    #[arg(short, long, required = true)]
    inputs: Vec<PathBuf>,

    /// Destination output file
    #[arg(short, long, required = true)]
    output: PathBuf,

    /// Whether to randomize the order during concatenation
    #[arg(long, default_value_t = false)]
    randomize_order: bool,

    /// Value of the file separator token
    #[arg(long, required = true, default_value = "<|file_sep|>")]
    file_separator_token: String,

    /// Which metadata field to find the repo name
    #[arg(long, required = true, default_value = "repo_name")]
    repo_field_name: String,

    /// Which metadata field to find the programming language
    #[arg(long, required = true, default_value = "language")]
    pl_field_name: String,
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
        .map(|path| {
            let manual_ext: Option<Vec<String>>; // Store Vec<String> instead of &[&str]
            let input_paths: Vec<PathBuf>;

            match path.extension() {
                Some(ext) => {
                    let ext_str = ext.to_string_lossy().into_owned(); // Convert to owned String
                    manual_ext = Some(vec![ext_str]); // Store as Vec<String>
                    let mut trunk_path = path.clone();
                    trunk_path.pop();
                    input_paths = vec![trunk_path];
                }
                None => {
                    manual_ext = None;
                    input_paths = vec![path];
                }
            }

            // Convert Vec<String> to Vec<&str> before passing it to expand_dirs
            let manual_ext_refs: Option<Vec<&str>> = manual_ext
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            expand_dirs(input_paths, manual_ext_refs.as_deref()).unwrap_or_default()
        })
        .flatten()
        .collect();
    return all_paths;
}

fn mk_serde_doc_reader<R: BufRead>(reader: R) -> impl Iterator<Item = serde_json::Value> {
    reader.lines().map(|line| {
        let line = line.unwrap();
        serde_json::from_str(&line).unwrap()
    })
}

fn process_single(
    src_path: &PathBuf,
    dst_path: &PathBuf,
    randomize_order: bool,
    file_separator_token: &str,
    repo_field_name: &str,
    pl_field_name: &str,
) -> Result<(), Error> {
    let concater = concat::CodeFileConcat {
        randomize_order,
        file_separator_token,
        repo_field_name,
        pl_field_name,
    };

    println!("Processing {:?} -> {:?}", src_path, dst_path);
    let src_buf = read_pathbuf_to_mem(src_path).unwrap();
    let mut out_bytes: Vec<u8> = Vec::new();
    let newline: u8 = b'\n';

    let mut file_documents = mk_serde_doc_reader(src_buf);
    let repo_documents = concater.perform_on_partition(&mut file_documents);

    for repo_document in repo_documents {
        out_bytes.extend_from_slice(&serde_json::to_vec(&repo_document)?);
        out_bytes.push(newline)
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
                args.randomize_order,
                &args.file_separator_token,
                &args.repo_field_name,
                &args.pl_field_name,
            )
            .unwrap();
            pbar.inc(1);
        });
}
