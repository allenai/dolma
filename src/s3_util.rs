use std::io;
use std::path::Path;
use std::str::FromStr;

use aws_sdk_s3::config::Region;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use regex::Regex;
use tokio::fs::File as TokioFile;

pub fn split_path(s3_prefix: &str) -> Result<(&str, &str), &'static str> {
    // use a regular expression to check if s3_prefix starts with s3://
    let re = Regex::new(r"^s3://").unwrap();
    if !re.is_match(s3_prefix) {
        return Err("s3_prefix must start with s3://");
    }

    // split the s3_prefix into parts
    let parts: Vec<&str> = s3_prefix.splitn(4, '/').collect();

    // if there are less than 3 parts, then the s3_prefix is invalid
    if parts.len() < 3 {
        return Err("s3_prefix must be in the form s3://bucket/path/to/object");
    }

    let bucket = parts[2];

    // if there are not 4 parts, then the object path is empty, so we set it to "/"
    let object_path = if parts.len() == 4 { parts[3] } else { "/" };

    Ok((bucket, object_path))
}

pub async fn download_to_file(
    s3_client: &S3Client,
    prefix: &str,
    path: &Path,
) -> Result<(), io::Error> {
    let (bucket, key) = split_path(prefix).unwrap();

    let result = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Error downloading {}: {}",
                    key,
                    e.message().unwrap_or_default()
                ),
            )
        })?;

    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut file = TokioFile::create(path).await?;
    let mut body = result.body.into_async_read();
    tokio::io::copy(&mut body, &mut file).await?;

    Ok(())
}

pub async fn upload_file(s3_client: &S3Client, prefix: &str, path: &Path) -> Result<(), io::Error> {
    let (bucket, key) = split_path(prefix).unwrap();

    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(ByteStream::from_path(path).await?)
        .send()
        .await
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Error uploading {}: {}",
                    key,
                    e.message().unwrap_or_default()
                ),
            )
        })?;

    Ok(())
}

pub async fn object_size(s3_client: &S3Client, prefix: &str) -> Result<usize, io::Error> {
    let (bucket, key) = split_path(prefix).unwrap();
    let resp = s3_client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e));
    match resp {
        Ok(resp) => Ok(resp.content_length as usize),
        Err(e) => Err(e),
    }
}

// Expand wildcard patterns into a list of object paths
// Only handles one wildcard per pattern
// e.g.: a/b/* -> a/b/1, a/b/2, a/b/3
// or:   a/*/b.txt -> a/1/b.txt, a/2/b.txt, a/3/b.txt
pub fn find_objects_matching_patterns(
    s3_client: &S3Client,
    patterns: &Vec<String>,
) -> Result<Vec<String>, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut stream_inputs: Vec<String> = Vec::new();
    for full_pattern in patterns.iter() {
        let start_size = stream_inputs.len();
        let (bucket, pattern) = split_path(full_pattern).unwrap();

        let mut output_prefix = String::from_str("s3://").unwrap();
        output_prefix.push_str(bucket);
        output_prefix.push_str("/");

        let mut prefix = pattern.clone().to_string();
        let mut suffix: Option<String> = Some("".to_owned());
        let maybe_index = pattern.chars().position(|c| c == '*');
        if let Some(index) = maybe_index {
            prefix = pattern[..index].to_string();
            suffix = None;
            if index < pattern.len() - 1 {
                suffix = Some(pattern[index + 2..].to_string());
            }
        }
        let mut has_more = true;
        let mut token: Option<String> = None;
        while has_more {
            let resp = if token.is_some() {
                rt.block_on(
                    s3_client
                        .list_objects_v2()
                        .bucket(bucket)
                        .prefix(&prefix)
                        .delimiter("/")
                        .continuation_token(token.unwrap())
                        .send(),
                )
                .unwrap()
            } else {
                rt.block_on(
                    s3_client
                        .list_objects_v2()
                        .bucket(bucket)
                        .prefix(&prefix)
                        .delimiter("/")
                        .send(),
                )
                .unwrap()
            };
            resp.contents().unwrap_or_default().iter().for_each(|obj| {
                let mut full_output_prefix = output_prefix.clone();
                full_output_prefix.push_str(obj.key().unwrap());
                stream_inputs.push(full_output_prefix);
            });
            suffix.iter().for_each(|s| {
                resp.common_prefixes()
                    .unwrap_or_default()
                    .iter()
                    .for_each(|sub_folder| {
                        let mut full_path = sub_folder.prefix().unwrap().to_owned();
                        full_path.push_str(s);
                        let mut full_output_prefix = output_prefix.clone();
                        full_output_prefix.push_str(&full_path);

                        stream_inputs.push(full_output_prefix);
                    });
            });
            token = resp.next_continuation_token().map(String::from);
            has_more = token.is_some();
        }
        log::info!(
            "Found {} objects for pattern \"{}\"",
            stream_inputs.len() - start_size,
            pattern
        );
    }
    stream_inputs.sort();
    Ok(stream_inputs)
}

pub fn new_client(region_name: Option<String>) -> Result<S3Client, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let region = Region::new(region_name.unwrap_or(String::from("us-east-1")));

    let config = rt.block_on(aws_config::from_env().region(region).load());
    let s3_client = S3Client::new(&config);
    Ok(s3_client)
}

#[cfg(test)]
mod test {
    use super::*;

    use std::collections::HashSet;
    use std::fs::read_dir;
    use std::fs::OpenOptions;
    use std::io;
    use std::io::{BufRead, BufReader};
    use std::path::Path;
    use std::str::FromStr;

    use flate2::read::MultiGzDecoder;

    fn compare_contents(expected: &str, actual: &str) {
        let expected_lines = BufReader::new(MultiGzDecoder::new(
            OpenOptions::new()
                .read(true)
                .write(false)
                .create(false)
                .open(expected)
                .unwrap(),
        ))
        .lines()
        .collect::<Vec<Result<String, io::Error>>>();
        let actual_lines = BufReader::new(MultiGzDecoder::new(
            OpenOptions::new()
                .read(true)
                .write(false)
                .create(false)
                .open(actual)
                .unwrap(),
        ))
        .lines()
        .collect::<Vec<Result<String, io::Error>>>();

        assert_eq!(
            expected_lines.len(),
            actual_lines.len(),
            "Wrong number of output documents"
        );

        for (actual, expected) in std::iter::zip(expected_lines, actual_lines) {
            let actual = actual.unwrap();
            let expected = expected.unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_split_path() -> Result<(), ()> {
        // test case when path is correct
        let prefix = "s3://my-bucket/my-key";
        let (bucket, key) = split_path(prefix).unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "my-key");

        // test case when path is incorrect
        let prefix = "s3:/my-bucket/my-key";
        let result = split_path(prefix);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_object_size() -> Result<(), io::Error> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let s3_client = new_client(None)?;

        let prefix =
            "s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz";
        let resp = rt.block_on(object_size(&s3_client, prefix));

        let size = resp.unwrap();
        assert_eq!(size, 25985);
        Ok(())
    }

    #[test]
    fn test_download_file() -> Result<(), io::Error> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let s3_client = new_client(None)?;

        let local_output_file =
            "tests/work/output/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz";
        let remote_output_file: &str =
            "s3://ai2-llm/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz";
        rt.block_on(download_to_file(
            &s3_client,
            remote_output_file,
            Path::new(local_output_file),
        ))?;

        compare_contents("tests/data/documents.json.gz", local_output_file);

        Ok(())
    }

    #[test]
    fn test_find_objects_matching_patterns() -> Result<(), io::Error> {
        let s3_client = new_client(None)?;

        let patterns =
            vec![
                String::from_str("s3://ai2-llm/pretraining-data/tests/mixer/expected/*.json.gz")
                    .unwrap(),
            ];

        let resp = find_objects_matching_patterns(&s3_client, &patterns).unwrap();
        let mut matches: HashSet<String> = HashSet::from_iter(resp.iter().map(|s| s.to_owned()));

        // list the contents of `tests/data/expected` and check that they match
        match read_dir("tests/data/expected") {
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let mut remote_path = String::from_str(
                                "s3://ai2-llm/pretraining-data/tests/mixer/expected/",
                            )
                            .unwrap();
                            remote_path.push_str(entry.file_name().to_str().unwrap());
                            matches.remove(&remote_path);
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
            }
            Err(err) => {
                return Err(err);
            }
        }

        assert_eq!(matches.len(), 0);
        Ok(())
    }
}
