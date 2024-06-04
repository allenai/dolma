use pyo3::exceptions;
use pyo3::prelude::*;

use adblock::lists::ParseOptions;
use adblock::request::Request;
use adblock::Engine;

pub mod bloom_filter;
pub mod deduper;
pub mod filters;
pub mod io;
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

/// Adblocker class
/// Hold the adblocker engine loaded with the rules
///
/// input:
///     rules: List[str] -> list of strings that represent the rules to be applied
///
/// example:
///     braveblock.Adblocker(
///         rules=[
///             "-advertisement-icon.",
///             "-advertisement/script.",
///         ]
///     )
#[pyclass(unsendable)]
struct UrlBlocker {
    engine: Engine,
}

#[pymethods]
impl UrlBlocker {
    #[new]
    fn new(rules: Vec<String>) -> Self {
        UrlBlocker {
            engine: Engine::from_rules(&rules, ParseOptions::default()),
        }
    }
    /// The function that should tell whether a specific request should be blocked according to the loaded rules
    ///
    /// input:
    ///     url: str -> The inspected url that should be tested
    ///     source_url: str -> The source url that made the request to the inspected url
    ///     request_type: str -> The type of the resource that is being requested. Can be one of the following:
    ///         "beacon", "csp_report", "document", "font", "image", "imageset", "main_frame",
    ///         "media", "object_subrequest", "object", "other", "ping", "script", "speculative",
    ///         "stylesheet", "sub_frame", "subdocument", "web_manifest", "websocket", "xbl",
    ///         "xhr", "xml_dtd", "xmlhttprequest", "xslt"
    ///
    /// returns:
    ///     bool -> Whether the request should be blocked or not
    ///
    /// example:
    ///     adblocker.check_network_urls(
    ///         url="http://example.com/-advertisement-icon.",
    ///         source_url="http://example.com/",
    ///         request_type="image",
    ///     )
    fn check_network_urls(
        &mut self,
        url: &str,
        source_url: &str,
        request_type: &str,
    ) -> PyResult<bool> {
        match Request::new(url, source_url, request_type) {
            Ok(request) => {
                let blocker_result = self.engine.check_network_request(&request);
                Ok(blocker_result.matched)
            }
            Err(_) => {
                return Err(exceptions::PyValueError::new_err("Invalid request"));
            }
        }
    }
}

// A Python module implemented in Rust. The name of this function must match
// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
// import the module.
#[pymodule]
fn dolma(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deduper_entrypoint, m)?)?;
    m.add_function(wrap_pyfunction!(mixer_entrypoint, m)?)?;
    m.add_class::<UrlBlocker>()?;

    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "dolma=info,deduper=info");
    }
    env_logger::init();

    Ok(())
}
