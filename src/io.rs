use std::fs::OpenOptions;
use flate2::{read::MultiGzDecoder, write::GzEncoder};
use flate2::Compression;
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
    pub fn reader (&self) -> BufReader<Decoder> {
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&self.path)
            .unwrap();
        let out = BufReader::with_capacity(self.size as usize, Decoder::new(file).unwrap());
        return out;
    }

    pub fn writer (&self) -> BufWriter<Encoder<File>> {
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .unwrap();
        BufWriter::with_capacity(self.size as usize, Encoder::new(file, self.level).unwrap())
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


pub enum FileStream {
    Gz(GzFileStream),
    Zstd(ZstdStream),
    File(FileStream),
}
