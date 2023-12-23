use std::io;
use std::path::Path;

use aws_sdk_s3::config::Region;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use tokio::fs::File as TokioFile;
use tokio::time::Duration;

// Split an s3:// url into a bucket and key
pub fn split_url(s3_url: &str) -> Result<(&str, &str), &'static str> {
    // use a regular expression to check if s3_prefix starts with s3://
    if !s3_url.starts_with("s3://") {
        return Err("s3_prefix must start with s3://");
    }

    // split the s3_prefix into parts
    let parts: Vec<&str> = s3_url.splitn(4, '/').collect();

    // if there are less than 3 parts, then the s3_prefix is invalid
    if parts.len() < 3 {
        return Err("s3_prefix must be in the form s3://bucket/path/to/object");
    }

    let bucket = parts[2];

    // if there are not 4 parts, then the object path is empty, so we set it to "/"
    let key = if parts.len() == 4 { parts[3] } else { "/" };

    Ok((bucket, key))
}

pub async fn download_to_file(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    path: &Path,
    max_attempts: Option<u8>,
) -> Result<(), io::Error> {
    // Default to no retries if max_attempts is not provided
    let max_attempts: u8 = max_attempts.unwrap_or_else(|| 1);

    // Check that max_attempts is greater than 0
    if max_attempts == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "max_attempts must be greater than 0",
        ));
    }

    let remote_path = format!("s3://{}/{}", bucket, key);
    let local_path = path.to_str().unwrap_or_default();

    for _attempt in 1..(max_attempts + 1) {
        match s3_client.get_object().bucket(bucket).key(key).send().await {
            Ok(response) => {
                std::fs::create_dir_all(path.parent().unwrap())?;
                let mut file = TokioFile::create(path).await?;
                let mut body = response.body.into_async_read();
                tokio::io::copy(&mut body, &mut file).await?;
                return Ok(());
            }
            Err(error) => {
                let error_message = error.message().unwrap_or_default();
                if _attempt == max_attempts {
                    log::error!(
                        "Failed LAST attempt {}/{} to download '{}' to '{}': {} ('{}')",
                        _attempt,
                        max_attempts,
                        remote_path,
                        local_path,
                        error,
                        error_message
                    );
                    // This was the last attempt
                    break;
                } else {
                    // short wait (1s) before retrying
                    log::warn!(
                        "Failed attempt {}/{} to download '{}' to '{}': {} ('{}'); will retry...",
                        _attempt,
                        max_attempts,
                        remote_path,
                        local_path,
                        error,
                        error_message
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    // If we got here, all attempts failed
    return Err(io::Error::new(
        io::ErrorKind::Other,
        format!(
            "All {} attempts to download '{}' to '{}' failed",
            max_attempts, remote_path, local_path
        ),
    ));
}

pub async fn upload_file(
    s3_client: &S3Client,
    path: &Path,
    bucket: &str,
    key: &str,
    max_attempts: Option<u8>,
) -> Result<(), io::Error> {
    // Default to no retries if max_attempts is not provided
    let max_attempts: u8 = max_attempts.unwrap_or_else(|| 1);

    // Check that max_attempts is greater than 0
    if max_attempts == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "max_attempts must be greater than 0",
        ));
    }

    let remote_path = format!("s3://{}/{}", bucket, key);
    let local_path = path.to_str().unwrap_or_default();

    for _attempt in 1..(max_attempts + 1) {
        match s3_client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(ByteStream::from_path(path).await?)
            .send()
            .await
        {
            Ok(_) => return Ok(()),
            Err(error) => {
                let error_message = error.message().unwrap_or_default();
                if _attempt == max_attempts {
                    log::error!(
                        "Failed LAST attempt {}/{} to upload '{}' to '{}': {} ('{}')",
                        _attempt,
                        max_attempts,
                        local_path,
                        remote_path,
                        error,
                        error_message
                    );
                    // This was the last attempt
                    break;
                } else {
                    // short wait (1s) before retrying
                    log::warn!(
                        "Failed attempt {}/{} to upload '{}' to '{}': {} ('{}'); will retry...",
                        _attempt,
                        max_attempts,
                        local_path,
                        remote_path,
                        error,
                        error_message
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    // If we got here, all attempts failed
    return Err(io::Error::new(
        io::ErrorKind::Other,
        format!(
            "All {} attempts to upload '{}' to '{}' failed",
            max_attempts, remote_path, local_path
        ),
    ));
}

pub async fn object_size(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
) -> Result<usize, io::Error> {
    let resp = s3_client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e));
    Ok(resp?.content_length as usize)
}

// Expand wildcard patterns into a list of object paths
// Only handles one wildcard per pattern
// e.g.: a/b/* -> a/b/1, a/b/2, a/b/3
// or:   a/*/b.txt -> a/1/b.txt, a/2/b.txt, a/3/b.txt
pub fn find_objects_matching_patterns(
    s3_client: &S3Client,
    patterns: &[String],
) -> Result<Vec<String>, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut stream_inputs: Vec<String> = Vec::new();
    for pattern in patterns.iter() {
        let start_size = stream_inputs.len();
        let mut prefix = pattern.clone();
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
            let (bucket, key) = match split_url(&prefix) {
                Ok((bucket, key)) => (bucket, key),
                Err(e) => {
                    return Err(io::Error::new(io::ErrorKind::Other, e));
                }
            };
            let resp = if let Some(token_value) = token {
                log::info!("Listing objects in bucket={}, prefix={}", bucket, key);
                rt.block_on(
                    s3_client
                        .list_objects_v2()
                        .bucket(bucket)
                        .prefix(key)
                        .delimiter("/")
                        .continuation_token(token_value)
                        .send(),
                )
                .unwrap()
            } else {
                log::info!("Listing objects in bucket={}, prefix={}", bucket, key);
                rt.block_on(
                    s3_client
                        .list_objects_v2()
                        .bucket(bucket)
                        .prefix(key)
                        .delimiter("/")
                        .send(),
                )
                .unwrap()
            };

            // unrwrap resp.contents() and create a to_validate_stream_inputs vector
            // containing all prefixes that do NOT end with "/"
            let to_validate_stream_inputs: Vec<String> = resp
                .contents()
                .unwrap_or_default()
                .iter()
                .filter_map(|prefix| {
                    if !prefix.key().unwrap().ends_with("/") {
                        Some(format!("s3://{}/{}", bucket, prefix.key().unwrap()))
                    } else {
                        None
                    }
                })
                .collect();

            match suffix {
                None => {
                    // if suffix is none, push all the objects in to_validate_stream_inputs to stream_inputs
                    to_validate_stream_inputs.iter().for_each(|path| {
                        stream_inputs.push(path.to_owned());
                    });
                }
                _ => {
                    // push only the objects that match the suffix to stream_inputs
                    to_validate_stream_inputs.iter().for_each(|path| {
                        if path.ends_with(suffix.clone().unwrap().as_str()) {
                            stream_inputs.push(path.to_owned());
                        }
                    });
                }
            }
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

    use flate2::read::MultiGzDecoder;

    fn skip_dolma_aws_tests() -> bool {
        if std::env::var_os("DOLMA_TESTS_SKIP_AWS")
            .is_some_and(|var| var.eq_ignore_ascii_case("true"))
        {
            println!("Skipping test_download_file because DOLMA_TESTS_SKIP_AWS=True");
            return true;
        }
        false
    }

    fn get_dolma_test_prefix() -> String {
        let prefix = std::env::var_os("DOLMA_TESTS_S3_PREFIX")
            .map(|var| var.to_str().unwrap().to_string())
            .unwrap_or_else(|| "s3://dolma-tests".to_string());

        // remove any trailing slashes
        return prefix.strip_suffix("/").unwrap_or(&prefix).to_string();
    }

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
    fn test_split_url() -> Result<(), ()> {
        // test case when path is correct
        let prefix = "s3://my-bucket/my-key-dir/my-key";
        let (bucket, key) = split_url(prefix).unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "my-key-dir/my-key");

        // test case when path is incorrect
        let prefix = "s3:/my-bucket/my-key";
        let result = split_url(prefix);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_object_size() -> Result<(), io::Error> {
        if skip_dolma_aws_tests() {
            return Ok(());
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let s3_client = new_client(None)?;

        let key = "pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz";
        let resp = rt.block_on(object_size(&s3_client, "ai2-llm", key));

        let size = resp.unwrap();
        assert_eq!(size, 25985);
        Ok(())
    }

    #[test]
    fn test_download_file() -> Result<(), io::Error> {
        if skip_dolma_aws_tests() {
            return Ok(());
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let s3_client = new_client(None)?;

        let s3_prefix = get_dolma_test_prefix();
        let s3_dest = "/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz";
        let s3_path = s3_prefix + s3_dest;
        let (s3_bucket, s3_key) = split_url(s3_path.as_str()).unwrap();

        // upload a file to s3
        let local_source_file = "tests/data/provided/documents/000.json.gz";
        rt.block_on(upload_file(
            &s3_client,
            Path::new(local_source_file),
            s3_bucket,
            s3_key,
            Some(3), // number of attempts
        ))?;

        // download the file back from s3
        let local_output_file =
            "tests/work/output/pretraining-data/tests/mixer/inputs/v0/documents/head/0000.json.gz";
        rt.block_on(download_to_file(
            &s3_client,
            s3_bucket,
            // s3_path,
            s3_key,
            Path::new(local_output_file),
            Some(3), // number of attempts
        ))?;

        // compare the contents of the two files
        compare_contents(local_source_file, local_output_file);

        Ok(())
    }

    #[test]
    fn test_failed_download_file() -> Result<(), io::Error> {
        if skip_dolma_aws_tests() {
            return Ok(());
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let s3_client = new_client(None)?;

        let s3_prefix = get_dolma_test_prefix();
        let s3_dest = "/foo/bar/baz.json.gz";
        let s3_path = s3_prefix + s3_dest;
        let (s3_bucket, s3_key) = split_url(s3_path.as_str()).unwrap();

        // download the file back from s3
        let local_output_file = "tests/work/foo/bar/bz.json.gz";

        let resp_too_few_attempts: Result<(), io::Error> = rt.block_on(download_to_file(
            &s3_client,
            s3_bucket,
            s3_key,
            Path::new(local_output_file),
            Some(0), // number of attempts
        ));
        assert!(resp_too_few_attempts.is_err());
        assert_eq!(
            resp_too_few_attempts.unwrap_err().to_string(),
            "max_attempts must be greater than 0"
        );

        let resp_no_such_location: Result<(), io::Error> = rt.block_on(download_to_file(
            &s3_client,
            s3_bucket,
            s3_key,
            Path::new(local_output_file),
            Some(3), // number of attempts
        ));

        assert!(resp_no_such_location.is_err());
        let exp_msg = format!(
            "All 3 attempts to download '{}' to '{}' failed",
            s3_path, local_output_file
        );
        assert_eq!(resp_no_such_location.unwrap_err().to_string(), exp_msg);
        Ok(())
    }

    #[test]
    fn test_find_objects_matching_patterns() -> Result<(), io::Error> {
        if skip_dolma_aws_tests() {
            return Ok(());
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let s3_client = new_client(None)?;
        let s3_prefix = get_dolma_test_prefix();

        let local_source_dir = "tests/data/expected";
        // iterate over the files in `tests/data/expected` and upload them to s3
        let entries = read_dir(local_source_dir)?;
        for entry in entries {
            let local_source_file = entry?.path();

            // skip files not ending with .json.gz
            if !local_source_file.to_str().unwrap().ends_with(".json.gz") {
                continue;
            }

            let s3_url = format!(
                "{}/pretraining-data/tests/mixer/expected/{}",
                s3_prefix,
                local_source_file.file_name().unwrap().to_str().unwrap()
            );
            let (s3_bucket, s3_key) = split_url(s3_url.as_str()).unwrap();
            rt.block_on(upload_file(
                &s3_client,
                Path::new(local_source_file.to_str().unwrap()),
                s3_bucket,
                s3_key,
                Some(3), // number of attempts
            ))?;
        }

        // If we don't shutdown the runtime, the test will hang when running
        // find_objects_matching_patterns.
        // I'm not sure why this is the case. Need to read more. -@soldni
        rt.shutdown_background();

        let patterns = vec![format!(
            "{}/{}",
            s3_prefix, "pretraining-data/tests/mixer/expected/*.json.gz"
        )];

        let resp = find_objects_matching_patterns(&s3_client, &patterns).unwrap();
        let mut matches: HashSet<String> = HashSet::from_iter(resp.iter().map(|s| s.to_owned()));

        // list the contents of `tests/data/expected` and check that they match
        let entries = read_dir(local_source_dir)?;
        for entry in entries {
            let remote_path = format!(
                "{}/pretraining-data/tests/mixer/expected/{}",
                s3_prefix,
                entry?.file_name().to_str().unwrap()
            );
            matches.remove(&remote_path);
        }

        assert_eq!(matches.len(), 0);
        Ok(())
    }
}
