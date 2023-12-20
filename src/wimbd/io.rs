//! Code imported from github.com/allenai/wimbd/blob/main/src/io.rs
//! and modified by @soldni to integrate in dolma.
//!
//! IO helpers.

use std::{
    fs::File,
    io::{self, prelude::*},
    rc::Rc,
};

use anyhow::Result;
use flate2::read::MultiGzDecoder;

/// A buffered reader for gzip files.
pub struct GzBufReader {
    reader: io::BufReader<MultiGzDecoder<File>>,
    buf: Rc<String>,
}

fn new_buf() -> Rc<String> {
    Rc::new(String::with_capacity(2048))
}

impl GzBufReader {
    // TODO: remove once open is used.
    #[allow(dead_code)]
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let reader = io::BufReader::new(MultiGzDecoder::new(File::open(path)?));
        let buf = new_buf();

        Ok(Self { reader, buf })
    }
}

type DataIteratorItem = io::Result<Rc<String>>;

impl Iterator for GzBufReader {
    type Item = DataIteratorItem;

    fn next(&mut self) -> Option<Self::Item> {
        let buf = match Rc::get_mut(&mut self.buf) {
            Some(buf) => {
                buf.clear();
                buf
            }
            None => {
                self.buf = new_buf();
                Rc::make_mut(&mut self.buf)
            }
        };

        self.reader
            .read_line(buf)
            .map(|u| {
                if u == 0 {
                    None
                } else {
                    Some(Rc::clone(&self.buf))
                }
            })
            .transpose()
    }
}
