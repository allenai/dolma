use flate2::Compression;
use flate2::{read::MultiGzDecoder, write::GzEncoder};
use std::ffi::OsStr;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Error as IoError, Write};
use std::path::PathBuf;
use zstd::stream::AutoFinishEncoder;
use zstd::{Decoder, Encoder};

pub struct GzFileStream {
    pub path: PathBuf,
    pub size: u64,
    pub compression: Compression,
}

impl GzFileStream {
    pub fn new(path: PathBuf, size: Option<u64>, compression: Option<Compression>) -> Self {
        let size = size.unwrap_or(1024 * 1024);
        let compression = compression.unwrap_or(Compression::default());
        Self {
            path,
            size,
            compression,
        }
    }
    pub fn reader(&self) -> Result<BufReader<MultiGzDecoder<File>>, IoError> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)?;
        let decoder: MultiGzDecoder<File> = MultiGzDecoder::new(file);
        let reader = BufReader::with_capacity(self.size as usize, decoder);
        Ok(reader)
    }

    pub fn writer(&self) -> Result<BufWriter<GzEncoder<File>>, IoError> {
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
    pub fn reader(&self) -> Result<BufReader<File>, IoError> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)?;
        let reader = BufReader::with_capacity(self.size as usize, file);
        Ok(reader)
    }

    pub fn writer(&self) -> Result<BufWriter<File>, IoError> {
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

pub enum MultiStream {
    Gz(GzFileStream),
    Zst(ZstdFileStream),
    Plain(FileStream),
}

impl MultiStream {
    pub fn new(
        path: PathBuf,
        extension: Option<String>,
        buffer_size: Option<u64>,
        gz_compression: Option<Compression>,
        zst_level: Option<i32>,
    ) -> Self {
        let extension = extension.unwrap_or(MultiStream::infer_compression(&path, None));
        match extension.as_str() {
            "gz" => MultiStream::Gz(GzFileStream::new(path, buffer_size, gz_compression)),
            "zst" => MultiStream::Zst(ZstdFileStream::new(path, buffer_size, zst_level)),
            _ => MultiStream::Plain(FileStream::new(path, buffer_size)),
        }
    }

    pub fn infer_compression_from_temp(path: PathBuf) -> String {
        MultiStream::infer_compression(&path, Some(".tmp"))
    }

    pub fn infer_compression(path: &PathBuf, suffix_to_ignore: Option<&str>) -> String {
        let file_name = path.file_name().unwrap_or(OsStr::new("")).to_str().unwrap();
        let extension = match file_name.split_once(".") {
            Some((_, ext)) => ext,
            None => "",
        };

        let extension = match suffix_to_ignore {
            Some(ignore) => extension.strip_suffix(ignore).unwrap_or(extension),
            None => extension,
        };

        match extension.rsplit_once(".") {
            Some((_, ext)) => ext.to_string(),
            None => extension.to_string(),
        }
    }

    pub fn get_compression(&self) -> String {
        match self {
            MultiStream::Gz(_) => String::from("gz"),
            MultiStream::Zst(_) => String::from("zst"),
            MultiStream::Plain(_) => String::from(""),
        }
    }

    pub fn with_default(path: PathBuf) -> Self {
        Self::new(path, None, None, None, None)
    }

    pub fn reader(&self) -> Result<Box<dyn BufRead>, IoError> {
        let reader = match self {
            MultiStream::Gz(stream) => Box::new(stream.reader()?) as Box<dyn BufRead>,
            MultiStream::Zst(stream) => Box::new(stream.reader()?) as Box<dyn BufRead>,
            MultiStream::Plain(stream) => Box::new(stream.reader()?) as Box<dyn BufRead>,
        };
        Ok(reader)
    }

    pub fn writer(&self) -> Result<Box<dyn Write>, IoError> {
        let writer = match self {
            MultiStream::Gz(stream) => Box::new(stream.writer()?) as Box<dyn Write>,
            MultiStream::Zst(stream) => Box::new(stream.writer()?) as Box<dyn Write>,
            MultiStream::Plain(stream) => Box::new(stream.writer()?) as Box<dyn Write>,
        };
        Ok(writer)
    }
}

#[cfg(test)]
pub mod io_tests {

    use super::*;
    use serde_json::json;
    use std::io::{BufRead, Read, Write};
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_infer_compression() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.gz");
        let expected = "gz";
        let got = MultiStream::infer_compression(&path, None);
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.jsonl.zst");
        let expected = "zst";
        let got = MultiStream::infer_compression(&path, None);
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.jsonl");
        let expected = "jsonl";
        let got = MultiStream::infer_compression(&path, None);
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.jsonl.tmp");
        let expected = "tmp";
        let got = MultiStream::infer_compression(&path, None);
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.jsonl.gz.tmp");
        let expected = "gz";
        let got = MultiStream::infer_compression(&path, Some(".tmp"));
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test");
        let expected = "";
        let got = MultiStream::infer_compression(&path, None);
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.tmp");
        let expected = "";
        let got = MultiStream::infer_compression(&path, Some("tmp"));
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.");
        let expected = "";
        let got = MultiStream::infer_compression(&path, Some("tmp"));
        assert_eq!(got, expected);

        let path = PathBuf::from("tests/data/formats/test.gz");
        let expected = "gz";
        let got = MultiStream::infer_compression(&path, Some(".tmp"));
        assert_eq!(got, expected);
    }

    #[test]
    fn test_read_gz() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.gz");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = GzFileStream::new(path, None, None);
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
    fn test_infer_read() {
        let path = PathBuf::from("tests/data/formats/test.jsonl.zst");
        let expected = vec![json!({"message": "this is a test"})];

        // create the stream and reader
        let stream = MultiStream::with_default(path);
        let reader = stream.reader().unwrap();

        // read each line, parse it and compare with the expected
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, expected[i]);
        }
    }

    fn _writer_gz(path: PathBuf, values: Vec<serde_json::Value>) {
        let stream = GzFileStream::new(path, None, None);
        let mut writer = stream.writer().unwrap();

        for line in values {
            serde_json::to_writer(&mut writer, &line).unwrap();
        }
        writer.flush().unwrap();
    }

    #[test]
    fn test_write_gz() {
        // let exp_path = PathBuf::from("tests/data/formats/test.jsonl.gz");
        let temp_path = NamedTempFile::new().unwrap().into_temp_path().to_path_buf();
        let got_path = temp_path.clone();
        // let path = PathBuf::from("temp/test.jsonl.gz");
        let to_write = vec![json!({"message": "this is a test"})];
        let expected = to_write.clone();

        // separate function to write the file ensures that the file is closed
        _writer_gz(temp_path, to_write);

        let got_file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(got_path)
            .unwrap();
        let mut got_stream = MultiGzDecoder::new(got_file);

        let mut got_string = String::new();
        got_stream
            .read_to_string(&mut got_string)
            .expect("Failed to read produced file");
        let got_data = serde_json::from_str::<serde_json::Value>(&got_string).unwrap();

        assert_eq!(got_data, expected[0]);
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
            serde_json::to_writer(&mut writer, &line).unwrap();
        }
        writer.flush().unwrap();

        let mut exp_file = File::open(exp_path).expect("Failed to open expected path file");
        let mut got_file = File::open(got_path).expect("Failed to open produced path file");

        let mut exp_buf = Vec::new();
        let mut got_buf = Vec::new();

        exp_file
            .read_to_end(&mut exp_buf)
            .expect("Failed to read expected file");
        got_file
            .read_to_end(&mut got_buf)
            .expect("Failed to read produced file");
    }

    #[test]
    fn test_write_plain() {
        let temp_path = NamedTempFile::new().unwrap().into_temp_path().to_path_buf();
        let got_path = temp_path.clone();
        // let path = PathBuf::from("temp/test.jsonl.gz");
        let to_write = vec![json!({"message": "this is a test"})];
        let expected = to_write.clone();

        let stream = FileStream::new(temp_path, None);
        let mut writer = stream.writer().unwrap();

        // write each line
        for line in to_write {
            serde_json::to_writer(&mut writer, &line).unwrap();
        }
        writer.flush().unwrap();

        let got_file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(got_path)
            .unwrap();
        let mut got_stream = BufReader::new(got_file);

        let mut got_string = String::new();
        got_stream
            .read_to_string(&mut got_string)
            .expect("Failed to read produced file");
        let got_data = serde_json::from_str::<serde_json::Value>(&got_string).unwrap();
        assert_eq!(got_data, expected[0]);
    }

    fn _write_multi(path: PathBuf, values: Vec<serde_json::Value>) {
        let stream = MultiStream::with_default(path);
        let mut writer = stream.writer().unwrap();

        // write each line
        for line in values {
            serde_json::to_writer(&mut writer, &line).unwrap();
        }
        writer.flush().unwrap();
    }

    #[test]
    fn test_multi_write() {
        let temp_dir = TempDir::new().unwrap();

        // test with a zst file
        let temp_path = temp_dir.path().join("test.jsonl.zst");
        let to_write = vec![json!({"message": "this is a test"})];

        // this function ensures that the file is closed
        _write_multi(temp_path.clone(), to_write.clone());

        let reader = ZstdFileStream::new(temp_path, None, None).reader().unwrap();
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, to_write[i]);
        }

        // same test but with a gz file
        let temp_path = temp_dir.path().join("test.jsonl.gz");
        _write_multi(temp_path.clone(), to_write.clone());

        let reader = GzFileStream::new(temp_path, None, None).reader().unwrap();
        let lines = reader.lines();
        for (i, line) in lines.enumerate() {
            let line = line.unwrap();
            let parsed = serde_json::from_str::<serde_json::Value>(&line).unwrap();
            assert_eq!(parsed, to_write[i]);
        }
    }
}
