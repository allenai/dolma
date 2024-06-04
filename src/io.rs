use std::fs::OpenOptions;
use flate2::{read::MultiGzDecoder, write::GzEncoder};
use flate2::Compression;
use zstd::stream::AutoFinishEncoder;
use zstd::{Encoder, Decoder};
use std::path::PathBuf;
use std::io::{BufReader, BufWriter, Error as IoError};
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
    pub fn reader (&self) -> Result<BufReader<MultiGzDecoder<File>>, IoError> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)?;
        let decoder = MultiGzDecoder::new(file);
        let reader = BufReader::with_capacity(self.size as usize, decoder);
        Ok(reader)
    }

    pub fn writer (&self) -> Result<BufWriter<GzEncoder<File>>, IoError> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;
        let encoder = GzEncoder::new(file, self.compression);
        let writer = BufWriter::with_capacity(self.size as usize, encoder);
        Ok(writer)
    }
}

pub struct ZstdFileStream {
    pub path: PathBuf,
    pub size: u64,
    pub level: i32,
}

impl ZstdFileStream {
    pub fn new(path: PathBuf, size: Option<u64>, level: Option<i32>) -> Self {
        let size = size.unwrap_or(1024 * 1024);
        let level = level.unwrap_or(3);
        Self { path, size, level }
    }

    pub fn reader(&self) -> Result<BufReader<Decoder<'static, BufReader<File>>>, IoError> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)?;
        let decoder = Decoder::new(file)?;
        let reader = BufReader::with_capacity(self.size as usize, decoder);
        // let reader = BufReader::with_capacity(self.size as usize, Decoder::new(file));
        Ok(reader)
    }

    pub fn writer(&self) -> Result<BufWriter<AutoFinishEncoder<'static, File>>, IoError> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;
        let encoder = Encoder::new(file, self.level)?;
        let writer = BufWriter::with_capacity(self.size as usize, encoder.auto_finish());
        Ok(writer)
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
    pub fn reader (&self) -> Result<BufReader<File>, IoError> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)?;
        let reader = BufReader::with_capacity(self.size as usize, file);
        Ok(reader)
    }

    pub fn writer (&self) -> Result<BufWriter<File>, IoError> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;
        let writer = BufWriter::with_capacity(self.size as usize, file);
        Ok(writer)
    }
}


#[cfg(test)]
pub mod io_tests {

    use tempfile::NamedTempFile;
    use serde_json::json;
    use std::io::{BufRead, Read, Write};
    use super::*;

    // rest of the code

    #[test]
    fn test_read_gz() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.gz");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = GzFileStream::new(path, None, None);
        let reader = stream.reader();

        // read each line, parse it and compare with the expected
        let lines = reader.unwrap().lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, expected[i]);
        }
    }

    #[test]
    fn test_read_zst() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.zst");
        // let path = PathBuf::from("temp/82f7fc2771d93a62edec3f826bf10019d1bc0939.jsonl.zst");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = ZstdFileStream::new(path, None, None);
        let reader = stream.reader().unwrap();

        // read each line, parse it and compare with the expected
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
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
        let lines = reader.unwrap().lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, expected[i]);
        }
    }

    #[test]
    fn test_write_gz() {
        let exp_path = PathBuf::from("tests/data/formats/test.jsonl.gz");
        let temp_path = NamedTempFile::new().unwrap().into_temp_path().to_path_buf();
        let got_path = temp_path.clone();
        // let path = PathBuf::from("temp/test.jsonl.gz");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and writer
        let stream = GzFileStream::new(temp_path, None, None);
        let mut writer = stream.writer().unwrap();

        // write each line
        for line in expected {
            let line = serde_json::to_string(&line).unwrap();
            serde_json::to_writer(&mut writer, &line).unwrap();
            writer.write_all(b"\n").unwrap();
        }
        writer.flush().unwrap();

        let mut exp_file = File::open(exp_path).expect("Failed to open expected path file");
        let mut got_file = File::open(got_path).expect("Failed to open produced path file");

        let mut exp_buf = Vec::new();
        let mut got_buf = Vec::new();

        exp_file.read_to_end(&mut exp_buf).expect("Failed to read expected file");
        got_file.read_to_end(&mut got_buf).expect("Failed to read produced file");

        assert_eq!(exp_buf, got_buf);
    }

    #[test]
    fn test_write_zstd() {
        let exp_path = PathBuf::from("tests/data/formats/test.jsonl");
        let temp_path = NamedTempFile::new().unwrap().into_temp_path().to_path_buf();
        let got_path = temp_path.clone();
        // let path = PathBuf::from("temp/test.jsonl.zst");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and writer
        let stream = ZstdFileStream::new(temp_path, None, None);
        let mut writer = stream.writer().unwrap();

        // write each line
        for line in expected {
            let line = serde_json::to_string(&line).unwrap();
            serde_json::to_writer(&mut writer, &line).unwrap();
            writer.write_all(b"\n").unwrap();
        }
        writer.flush().unwrap();

        let mut exp_file = File::open(exp_path).expect("Failed to open expected path file");
        let mut got_file = File::open(got_path).expect("Failed to open produced path file");

        let mut exp_buf = Vec::new();
        let mut got_buf = Vec::new();

        exp_file.read_to_end(&mut exp_buf).expect("Failed to read expected file");
        got_file.read_to_end(&mut got_buf).expect("Failed to read produced file");
    }

}
