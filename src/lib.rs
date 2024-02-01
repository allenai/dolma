use pyo3::exceptions;
use pyo3::prelude::*;

pub mod bloom_filter;
pub mod deduper;
pub mod mixer;
pub mod s3_util;
pub mod shard;
pub mod wimbd;

use crate::deduper::deduper_config::DeduperConfig;
use crate::mixer::mixer_config::MixerConfig;
use std::env;

#[pyfunction]
fn deduper_entrypoint(config_str: &str) -> PyResult<()> {
    let config: DeduperConfig = DeduperConfig::parse_from_string(config_str).unwrap();

    if let Err(cnt) = deduper::run(config) {
        return Err(exceptions::PyRuntimeError::new_err(format!(
            "Failed with {} errors",
            cnt
        )));
    }
    Ok(())
}

#[pyfunction]
fn mixer_entrypoint(config_str: &str) -> PyResult<()> {
    //Result<u32, PyErr> {
    let config: MixerConfig = MixerConfig::parse_from_string(config_str).unwrap();
    if let Err(cnt) = mixer::run(config) {
        return Err(exceptions::PyRuntimeError::new_err(format!(
            "Failed with {} errors",
            cnt
        )));
    }
    Ok(())
}

// A Python module implemented in Rust. The name of this function must match
// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
// import the module.
#[pymodule]
fn dolma(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deduper_entrypoint, m)?)?;
    m.add_function(wrap_pyfunction!(mixer_entrypoint, m)?)?;

    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "dolma=info,deduper=info");
    }
    env_logger::init();

    Ok(())
}
