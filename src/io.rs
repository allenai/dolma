use std::fs::OpenOptions;
use flate2::{read::MultiGzDecoder, write::GzEncoder};
use flate2::Compression;
use zstd::stream::AutoFinishEncoder;
use zstd::{Encoder, Decoder};
use std::path::PathBuf;
use std::io::{BufReader, BufWriter, Write};
use std::fs::File;


pub struct GzFileStream {
    pub path: PathBuf,
    pub size: u64,
    pub compression: Compression,
}

impl GzFileStream {
    pub fn new(path: PathBuf, size: Option<u64>, compression: Option<Compression>) -> Self {
        let size = size.unwrap_or(1024 * 1024);
        let compression = compression.unwrap_or(Compression::default());
        Self { path, size, compression }
    }
    pub fn reader (&self) -> BufReader<MultiGzDecoder<File>> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)
            .unwrap();
        BufReader::with_capacity(self.size as usize, MultiGzDecoder::new(file))
    }

    pub fn writer (&self) -> BufWriter<GzEncoder<File>> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .unwrap();
        BufWriter::with_capacity(self.size as usize, GzEncoder::new(file, self.compression))
    }
}

pub struct ZstdStream {
    pub path: PathBuf,
    pub size: u64,
    pub level: i32,
}

impl ZstdStream {
    pub fn new(path: PathBuf, size: Option<u64>, level: Option<i32>) -> Self {
        let size = size.unwrap_or(1024 * 1024);
        let level = level.unwrap_or(3);
        Self { path, size, level }
    }
    pub fn reader (&self) -> BufReader<Decoder<'static, BufReader<File>>> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)
            .unwrap();
        let out = BufReader::with_capacity(self.size as usize, Decoder::new(file).unwrap());
        return out;
    }

    pub fn writer (&self) -> BufWriter<AutoFinishEncoder<File>> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .unwrap();
        let encoder = Encoder::new(file, self.level).unwrap();
        let auto_finish_encoder = encoder.auto_finish();
        BufWriter::with_capacity(self.size as usize, auto_finish_encoder)
    }
}

pub struct FileStream {
    pub path: PathBuf,
    pub size: u64,
}


impl FileStream {
    pub fn new(path: PathBuf, size: Option<u64>) -> Self {
        let size = size.unwrap_or(1024 * 1024);
        Self { path, size }
    }
    pub fn reader (&self) -> BufReader<File> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)
            .unwrap();
        BufReader::with_capacity(self.size as usize, file)
    }

    pub fn writer (&self) -> BufWriter<File> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .unwrap();
        BufWriter::with_capacity(self.size as usize, file)
    }
}


#[cfg(test)]
pub mod io_tests {

    use serde_json::json;
    use std::io::BufRead;
    use super::*;

    // rest of the code

    #[test]
    fn test_decompress_gz() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.gz");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = GzFileStream::new(path, None, None);
        let reader = stream.reader();

        // read each line, parse it and compare with the expected
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, expected[i]);
        }
    }

    #[test]
    fn test_decompress_zst() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.zst");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = ZstdStream::new(path, None, None);
        let reader = stream.reader();

        // read each line, parse it and compare with the expected
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
            println!("{:?}", line);
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, expected[i]);
        }
    }

    #[test]
    fn test_read_plain() {
        let path = PathBuf::from("tests/data/formats/test.jsonl");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = FileStream::new(path, None);
        let reader = stream.reader();

        // read each line, parse it and compare with the expected
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, expected[i]);
        }
    }

}
