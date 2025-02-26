use pyo3::exceptions;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::index::sample;

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

#[pyclass]
struct FillInMiddle {
    fim_rate: f32,
    psm_spm_split: f32,
    file_separator_token: String,
    fim_prefix_token: String,
    fim_middle_token: String,
    fim_suffix_token: String,
    rng: StdRng,
}

#[pymethods]
impl FillInMiddle {
    #[new]
    fn new(
        fim_rate: f32,
        psm_spm_split: f32,
        file_separator_token: String,
        fim_prefix_token: String,
        fim_middle_token: String,
        fim_suffix_token: String,
    ) -> Self {
        FillInMiddle {
            fim_rate,
            psm_spm_split,
            file_separator_token,
            fim_prefix_token,
            fim_middle_token,
            fim_suffix_token,
            rng: StdRng::from_entropy(),
        }
    }

    fn perform_on_document_text(&mut self, document_text: &str) -> PyResult<String> {
        let result: String = document_text
            .split(&self.file_separator_token)
            .map(|file_text| {
                // Decide whether we're applying FIM to this file text
                if &mut self.rng.gen::<f32>() < &mut self.fim_rate {
                    // Extract into unicode chars because of multi-byte characters
                    let file_chars: Vec<char> = file_text.chars().collect();

                    // Exclude front and rear character indices we don't want to split at
                    let front_offset = 1;
                    let rear_offset = 1;
                    let range_clip = front_offset + rear_offset + 1;

                    // Boundary condition: text is too short to rearrange
                    if range_clip > file_chars.len() || (file_chars.len() - range_clip) < 2 {
                        file_text.to_string()
                    } else {
                        let mut break_points: Vec<usize> =
                            sample(&mut self.rng, file_chars.len() - range_clip, 2)
                                .into_iter()
                                .map(|index| index + front_offset)
                                .collect();
                        break_points.sort();

                        // Slice out the chars and back to utf-8 strings
                        let prefix = file_chars[..break_points[0]].iter().collect::<String>();
                        let middle = file_chars[break_points[0]..break_points[1]]
                            .iter()
                            .collect::<String>();
                        let suffix = file_chars[break_points[1]..].iter().collect::<String>();

                        if &mut self.rng.gen::<f32>() < &mut self.psm_spm_split {
                            // Reorder into Prefix-Suffix-Middle
                            format!(
                                "{}{}{}{}{}{}",
                                self.fim_prefix_token,
                                prefix,
                                self.fim_suffix_token,
                                suffix,
                                self.fim_middle_token,
                                middle
                            )
                        } else {
                            // Reorder into Suffix-Prefix-Middle
                            format!(
                                "{}{}{}{}{}{}",
                                self.fim_suffix_token,
                                suffix,
                                self.fim_prefix_token,
                                prefix,
                                self.fim_middle_token,
                                middle
                            )
                        }
                    }
                } else {
                    file_text.to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(&self.file_separator_token);

        Ok(result)
    }
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
    m.add_class::<FillInMiddle>()?;

    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "dolma=info,deduper=info");
    }
    env_logger::init();

    Ok(())
}
