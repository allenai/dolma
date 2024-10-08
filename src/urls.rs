use pyo3::exceptions;
use pyo3::prelude::*;

use adblock::lists::ParseOptions;
use adblock::request::Request;
use adblock::Engine;

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
pub struct UrlBlocker {
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
