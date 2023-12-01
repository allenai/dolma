//! Code imported from github.com/allenai/wimbd/blob/main/src/io.rs
//! and modified by @soldni to integrate in dolma.

use anyhow::{bail, Result};

use std::fs::{self, File};
use std::path::{Path, PathBuf};

pub(crate) fn get_output_file(path: impl AsRef<Path>, force: bool) -> Result<(File, PathBuf)> {
    let path = path.as_ref();

    if path.is_file() {
        if force {
            log::warn!("Overwriting output file {:?}", path);
        } else {
            bail!(
                "Output file {:?} already exists, use --force to overwrite",
                path
            );
        }
        Ok((File::options().write(true).open(path)?, path.into()))
    } else {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok((File::create(path)?, path.into()))
    }
}
