use std::io;

use crate::shard::shard_config::{FilterConfig, SpanReplacementConfig};
use jaq_interpret::{Ctx, Filter, FilterT, ParseCtx, RcIter, Val};
use jaq_std;
use jsonpath_rust::JsonPathFinder;
use serde_json::Value;

pub struct JqSelector {
    pub selector: Filter,
}

impl JqSelector {
    pub fn new(selector_string: &str) -> Result<JqSelector, io::Error> {
        let mut defs = ParseCtx::new(Vec::new());
        defs.insert_natives(jaq_core::core());
        defs.insert_defs(jaq_std::std());
        assert!(defs.errs.is_empty());

        let (selector, errs) = jaq_parse::parse(selector_string, jaq_parse::main());
        if !errs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Error parsing '{:?}' into filter: {:?}",
                    selector_string, errs
                ),
            ));
        }
        match selector {
            Some(selector) => {
                let selector: jaq_interpret::Filter = defs.compile(selector);
                if !defs.errs.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Error compiling '{:?}' into filter.", selector_string),
                    ));
                }

                Ok(JqSelector { selector: selector })
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Parsing '{:?}' resulted in no filter", selector_string),
                ));
            }
        }
    }

    // select returns array of results if the filter matches multiple elements,
    // or a single result if the filter matches a single element.
    // in case of no match, it returns null
    pub fn select(&self, json: &Value) -> Result<Value, io::Error> {
        let inputs: RcIter<std::iter::Empty<_>> = RcIter::new(core::iter::empty());
        let out: Vec<Result<jaq_interpret::Val, jaq_interpret::Error>> = self
            .selector
            .run((Ctx::new(Vec::new(), &inputs), Val::from(json.clone())))
            .collect();
        if out.is_empty() {
            return Ok(Value::Null);
        }
        let mut result = Vec::new();
        for resp in out {
            match resp {
                Ok(val) => result.push(val),
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Error evaluating filter: {:?}", e),
                    ))
                }
            }
        }

        match result.len() {
            0 => Ok(Value::Null),
            1 => Ok(Value::from(result[0].clone())),
            _ => Ok(Value::from(result)),
        }
    }
}

pub struct JsonPathSelector {
    pub path: String,
}

impl JsonPathSelector {
    pub fn new(path: &str) -> Result<JsonPathSelector, io::Error> {
        Ok(JsonPathSelector {
            path: path.to_string(),
        })
    }

    pub fn select(&self, json: &Value) -> Result<Value, io::Error> {
        match JsonPathFinder::from_str("{}", &self.path) {
            Ok(mut finder) => {
                finder.set_json(Box::new(json.clone()));
                match finder.find() {
                    Value::Array(arr) => match arr.len() {
                        0 => Ok(Value::Null),
                        1 => Ok(arr[0].clone()),
                        _ => Ok(Value::from(arr)),
                    },
                    Value::Null => Ok(Value::Null),
                    _ => Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Error evaluating filter: {:?}", self.path),
                    )),
                }
            }
            Err(e) => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Error evaluating filter: {:?}", e),
            )),
        }
    }
}

pub enum Selector {
    JqSelector(JqSelector),
    JsonPathSelector(JsonPathSelector),
}

impl Selector {
    pub fn new(selector_config: &SpanReplacementConfig) -> Result<Selector, io::Error> {
        match selector_config.syntax.as_deref() {
            Some("jq") => Ok(Selector::JqSelector(JqSelector::new(
                &selector_config.span,
            )?)),
            Some("jsonpath") | None => Ok(Selector::JsonPathSelector(JsonPathSelector::new(
                &selector_config.span,
            )?)),
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Unknown selector syntax: {:?}", selector_config.syntax),
            )),
        }
    }

    pub fn select(&self, json: &Value) -> Result<Value, io::Error> {
        match self {
            Selector::JqSelector(selector) => selector.select(json),
            Selector::JsonPathSelector(selector) => selector.select(json),
        }
    }
}

#[cfg(test)]
pub mod selector_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_select() {
        let doc = json!({
            "attributes": {
                "foo": "bar",
                "baz": "qux"
            }
        });
        let expected = json!("bar");

        let jq_selector = JqSelector::new(".attributes.foo").unwrap();
        assert_eq!(jq_selector.select(&doc).unwrap(), expected);

        let jsonpath_selector = JsonPathSelector::new("$.attributes.foo").unwrap();
        assert_eq!(jsonpath_selector.select(&doc).unwrap(), expected);
    }

    #[test]
    fn test_select_array() {
        let doc = json!({
            "attributes": {
                "foo": [1, 2, 3],
                "baz": "qux"
            }
        });
        let expected = json!([1, 2, 3]);

        let jq_selector = JqSelector::new(".attributes.foo").unwrap();
        assert_eq!(jq_selector.select(&doc).unwrap(), expected);

        let jsonpath_selector = JsonPathSelector::new("$.attributes.foo").unwrap();
        assert_eq!(jsonpath_selector.select(&doc).unwrap(), expected);
    }

    #[test]
    fn test_select_object() {
        let jq_selector = JqSelector::new(".attributes").unwrap();
        let doc = json!({
            "attributes": {
                "foo": "bar",
                "baz": "qux"
            }
        });
        assert_eq!(
            jq_selector.select(&doc).unwrap(),
            json!({"foo": "bar", "baz": "qux"})
        );
    }

    #[test]
    fn test_select_null() {
        let doc = json!({
            "attributes": {
                "baz": "qux"
            }
        });
        let expected = json!(null);

        let jq_selector = JqSelector::new(".attributes.foo").unwrap();
        assert_eq!(jq_selector.select(&doc).unwrap(), expected);

        let jsonpath_selector = JsonPathSelector::new("$.attributes.foo").unwrap();
        assert_eq!(jsonpath_selector.select(&doc).unwrap(), expected);
    }

    #[test]
    fn test_nested_select_null() {
        let doc = json!({
            "attributes": {
                "not_foo": {
                    "baz": "qux"
                }
            }
        });
        let expected = json!(null);

        let jq_selector = JqSelector::new(".attributes?.foo?.baz?").unwrap();
        assert_eq!(jq_selector.select(&doc).unwrap(), expected);

        let jsonpath_selector = JsonPathSelector::new("$.attributes.foo.baz").unwrap();
        assert_eq!(jsonpath_selector.select(&doc).unwrap(), expected);
    }

    #[test]
    fn test_select_error() {
        let doc = json!({
            "attributes": {
                "foo": ["water", " & ", "bread"],
            }
        });

        let jq_selector = JqSelector::new(".attributes.foo | add").unwrap();
        assert_eq!(jq_selector.select(&doc).unwrap(), json!("water & bread"));
    }
}

pub struct JqDocFilter {
    pub include: Vec<Filter>,
    pub exclude: Vec<Filter>,
}

pub struct JsonPathFilter {
    pub include: Vec<String>,
    pub exclude: Vec<String>,
}

impl JqDocFilter {
    fn parse_filters(filter_strs: Vec<String>) -> Result<Vec<Filter>, io::Error> {
        let mut defs = ParseCtx::new(Vec::new());
        defs.insert_natives(jaq_core::core());
        defs.insert_defs(jaq_std::std());
        assert!(defs.errs.is_empty());

        let mut filters: Vec<Filter> = Vec::new();
        for filter_str in filter_strs {
            let (filter, errs) = jaq_parse::parse(&filter_str, jaq_parse::main());
            if !errs.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Error parsing '{:?}' into filter: {:?}", filter_str, errs),
                ));
            }
            match filter {
                Some(filter) => {
                    let filter: jaq_interpret::Filter = defs.compile(filter);
                    if !defs.errs.is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("Error compiling '{:?}' into filter.", filter_str),
                        ));
                    }

                    filters.push(filter);
                }
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Parsing '{:?}' resulted in no filter", filter_str),
                    ));
                }
            }
        }
        Ok(filters)
    }

    fn evaluate_match(
        &self,
        result: &Result<Val, jaq_interpret::Error>,
    ) -> Result<bool, io::Error> {
        match result {
            Ok(jaq_interpret::Val::Bool(b)) => Ok(*b),
            Ok(jaq_interpret::Val::Null) => Ok(false),
            Ok(jaq_interpret::Val::Int(i)) => Ok(*i != 0),
            Ok(jaq_interpret::Val::Float(f)) => Ok(*f != 0.0),
            Ok(jaq_interpret::Val::Str(s)) => Ok(!s.is_empty()),
            Ok(jaq_interpret::Val::Arr(a)) => Ok(!a.is_empty()),
            Ok(jaq_interpret::Val::Obj(d)) => Ok(!d.is_empty()),
            Err(err) => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Error evaluating filter: {:?}", err),
            )),
            _ => Ok(true),
        }
    }

    pub fn new(filter_config: &FilterConfig) -> Result<JqDocFilter, io::Error> {
        let include_filters = JqDocFilter::parse_filters(filter_config.include.clone())?;
        let exclude_filters = JqDocFilter::parse_filters(filter_config.exclude.clone())?;
        Ok(JqDocFilter {
            include: include_filters,
            exclude: exclude_filters,
        })
    }
    pub fn should_keep(&self, json: &Value) -> Result<bool, io::Error> {
        let mut keep = self.include.is_empty();

        let inputs: RcIter<std::iter::Empty<_>> = RcIter::new(core::iter::empty());
        for filter in self.include.iter() {
            // exit early if keep is already true
            if keep {
                break;
            }

            let out: Vec<Result<jaq_interpret::Val, jaq_interpret::Error>> = filter
                .run((Ctx::new(Vec::new(), &inputs), Val::from(json.clone())))
                .collect();

            // if the filter returns something, evaluate each result and update keep;
            // keep will be true at the end of the loop if all results are true and the filter is not empty
            // if an any point an error is encountered, immediately return the error
            keep = match out.is_empty() {
                true => false,
                false => {
                    let mut partial_keep = true;
                    for result in out.iter() {
                        match self.evaluate_match(result) {
                            Ok(val) => partial_keep = partial_keep && val,
                            Err(e) => return Err(e),
                        }
                    }
                    partial_keep
                }
            };
        }

        for filter in self.exclude.iter() {
            // exit early if keep is already false
            if !keep {
                break;
            }
            let out: Vec<_> = filter
                .run((Ctx::new(Vec::new(), &inputs), Val::from(json.clone())))
                .collect();

            // if the filter returns nothing, we keep the document; otherwise, evaluate each result
            // and check if they all evaluate to false; if any result is true, we remove the document
            keep = match out.is_empty() {
                true => true,
                false => {
                    let mut partial_keep = true;
                    for result in out.iter() {
                        match self.evaluate_match(result) {
                            Ok(val) => partial_keep = partial_keep && !val,
                            Err(e) => return Err(e),
                        }
                    }
                    partial_keep
                }
            };
        }
        Ok(keep)
    }
}

impl JsonPathFilter {
    pub fn new(filter_config: &FilterConfig) -> Result<JsonPathFilter, io::Error> {
        Ok(JsonPathFilter {
            include: filter_config.include.clone(),
            exclude: filter_config.exclude.clone(),
        })
    }
    pub fn should_keep(&self, json: &Value) -> Result<bool, io::Error> {
        let mut keep = self.include.is_empty();
        for pattern in self.include.iter() {
            let mut finder = match JsonPathFinder::from_str("{}", pattern) {
                Ok(finder) => finder,
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "Error making include pattern {} into filter: {:?}",
                            pattern, e
                        ),
                    ))
                }
            };
            finder.set_json(Box::new(json.clone()));
            keep = finder.find() != Value::Null;
            if keep {
                break;
            }
        }
        if keep {
            for pattern in self.exclude.iter() {
                let mut finder = match JsonPathFinder::from_str("{}", pattern) {
                    Ok(finder) => finder,
                    Err(e) => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!(
                                "Error making exclude pattern {} into filter: {:?}",
                                pattern, e
                            ),
                        ))
                    }
                };
                finder.set_json(Box::new(json.clone()));
                keep = finder.find() == Value::Null;
                if !keep {
                    break;
                }
            }
        }
        Ok(keep)
    }
}

pub struct AllowAllFilter;

impl AllowAllFilter {
    pub fn new() -> Result<AllowAllFilter, io::Error> {
        Ok(AllowAllFilter)
    }
    pub fn should_keep(&self, _json: &Value) -> Result<bool, io::Error> {
        Ok(true)
    }
}

pub enum DocFilter {
    Jq(JqDocFilter),
    JsonPath(JsonPathFilter),
    AllowAll(AllowAllFilter),
}

impl DocFilter {
    pub fn new(filter_config: Option<&FilterConfig>) -> Result<DocFilter, io::Error> {
        match filter_config {
            Some(filter_config) => match filter_config.syntax.as_deref() {
                Some("jq") => Ok(DocFilter::Jq(JqDocFilter::new(filter_config)?)),
                Some("jsonpath") | None => {
                    Ok(DocFilter::JsonPath(JsonPathFilter::new(filter_config)?))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Unknown filter syntax: {:?}", filter_config.syntax),
                )),
            },
            None => Ok(DocFilter::AllowAll(AllowAllFilter::new()?)),
        }
    }
    pub fn should_keep(&self, json: &Value) -> Result<bool, io::Error> {
        match self {
            DocFilter::Jq(f) => f.should_keep(json),
            DocFilter::JsonPath(f) => f.should_keep(json),
            DocFilter::AllowAll(f) => f.should_keep(json),
        }
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_should_keep() {
        let filter_config = FilterConfig {
            include: vec![".attributes.foo".to_string()],
            exclude: vec![r#".attributes.baz == "quac""#.to_string()],
            syntax: Some("jq".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "attributes": {
                "foo": "bar",
                "baz": "qux"
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), true);
    }

    #[test]
    fn test_should_remove() {
        let filter_config = FilterConfig {
            include: vec![".attributes.foo".to_string()],
            exclude: vec![r#".attributes.baz == "qux""#.to_string()],
            syntax: Some("jq".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "attributes": {
                "foo": "bar",
                "baz": "qux"
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), false);
    }

    #[test]
    fn test_aggregate_filters() {
        let filter_config = FilterConfig {
            include: vec![".attributes.foo | length >= 3".to_string()],
            exclude: vec![],
            syntax: Some("jq".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "attributes": {
                "foo": [1.0, 2.0, 3.0],
                "baz": [4.0, 5.0]
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), true);
    }

    #[test]
    fn test_allow_all() {
        let filters = DocFilter::new(None).unwrap();
        let doc = json!({
            "attributes": {
                "foo": [1.0, 2.0, 3.0],
                "baz": [4.0, 5.0]
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), true);
    }

    #[test]
    fn test_jsonpath_allow() {
        let filter_config = FilterConfig {
            include: vec!["$..foo".to_string()],
            exclude: vec![],
            syntax: Some("jsonpath".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "foo": "bar",
            "baz": "qux"
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), true);
    }

    #[test]
    fn test_jsonpath_exclude() {
        let filter_config = FilterConfig {
            include: vec![],
            exclude: vec![
                "$@.attributes[?(@.value1 && @.value1[0] && @.value1[0][2] >= 1.0)]".to_string(),
            ],
            syntax: Some("jsonpath".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "attributes": {
                "value1": [[0, 30, 1.0], [30, 45, 0.5]],
                "value2": [[45, 60, 0.0], [60, 75, 0.5]],
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), false);

        let doc = json!({
            "attributes": {
                "value1": [[0, 30, 1], [30, 45, 0]],
                "value2": [[45, 60, 0], [60, 75, 0]],
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), true);
    }

    #[test]
    fn test_sum_jq_filter() {
        let filter_config = FilterConfig {
            include: vec![".attributes.foo | add >= 6".to_string()],
            exclude: vec![],
            syntax: Some("jq".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "attributes": {
                "foo": [1.0, 2.0, 3.0],
                "baz": [4.0, 5.0]
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), true);

        let doc = json!({
            "attributes": {
                "foo": [1.0, 2.0, 1.0],
                "baz": [4.0, 5.0]
            }
        });
        assert_eq!(filters.should_keep(&doc).unwrap(), false);
    }

    #[test]
    fn test_jq_raise_errror_compile() {
        let filter_config = FilterConfig {
            include: vec![".x | sum".to_string()],
            exclude: vec![],
            syntax: Some("jq".to_string()),
        };

        let result = DocFilter::new(Some(&filter_config));
        assert!(result.is_err());
    }

    #[test]
    fn test_jq_multiple_conditions() {
        let filter_config = FilterConfig {
            include: vec![
                "(.attributes.dedupe_para_ngrams_13_1 | length == 0) or ((.attributes.dedupe_para_ngrams_13_1 | map(.[2] * (.[1] - .[0])) | add) / (.text | length) <= 0.3)".to_string(),
            ],
            exclude: vec![
                ".attributes.paloma_documents != null".to_string(),
                "(.attributes.paloma_paragraphs | length) > 0".to_string(),
                "(.tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition != null) and (.tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition[0][-1] > 10)".to_string(),
                ".attributes.cc_multi_bin__cc_multi_bin__hq[0][-1] <= 0.01".to_string(),
                ".attributes.pii_regex_with_counts_fast_v2__pii_regex_with_counts_fast_v2__doc_count[0][-1] > 5".to_string(),
            ],
            syntax: Some("jq".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();

        let doc = json!({
            "attributes": {
                "cc_multi_bin__cc_multi_bin__lq": [[0, 1533, 0.99438]],
                "cc_multi_bin__cc_multi_bin__hq": [[0, 1533, 0.00564]],
                "dedupe_para_ngrams_13_1": [],
                "paloma_paragraphs": [],
                "pii_regex_with_counts_fast_v2__pii_regex_with_counts_fast_v2__doc_count": [[0, 1533, 0.0]],
                "pii_regex_with_counts_fast_v2__pii_regex_with_counts_fast_v2__doc_frac": [[0, 1533, 1.0]],
                "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__repetition": [[493, 533, 10.0]],
                "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition": [[0, 1533, 10.0]],
                "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_length_repetition": [[0, 1533, 40.0]],
                "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_frac_repetition": [[0, 1533, 0.02609]]
            }
        });

        assert_eq!(filters.should_keep(&doc).unwrap(), false);
    }

    #[test]
    fn test_jq_missing_attr() {
        let filter_config = FilterConfig {
            include: vec![".attributes.b.b != null".to_string()],
            exclude: vec![],
            syntax: Some("jq".to_string()),
        };
        let filters = DocFilter::new(Some(&filter_config)).unwrap();
        let doc = json!({
            "text": "test",
            "id": "0",
            "attributes": {"a": [[0, 3, 1]]},
            "source": "test"
        });
        let result = filters.should_keep(&doc);
        assert!(result.is_err());
    }
}
