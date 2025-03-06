use rand::prelude::*;
use rand::rng;
use std::iter;

use serde_json::json;

fn get_metadata_field(document: &serde_json::Value, field_name: &str) -> String {
    document
        .get("metadata")
        .and_then(|m| m.get(field_name))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

pub struct CodeFileConcat<'a> {
    pub randomize_order: bool,
    pub file_separator_token: &'a str,
    pub repo_field_name: &'a str,
    pub pl_field_name: &'a str,
}

impl CodeFileConcat<'_> {
    pub fn perform_on_partition<'a, T: Iterator<Item = serde_json::Value> + 'a>(
        &'a self,
        documents: &'a mut T,
    ) -> impl Iterator<Item = serde_json::Value> + 'a {
        let mut random = rng();
        let mut maybe_current_group_head: Option<serde_json::Value> = documents.next();

        iter::from_fn(move || {
            let current_group_head = maybe_current_group_head.take()?;

            let current_repo = get_metadata_field(&current_group_head, &self.repo_field_name);
            let current_pl = get_metadata_field(&current_group_head, &self.pl_field_name);

            let mut repo_texts: Vec<String> = vec![current_group_head
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()];

            while let Some(next) = documents.next() {
                let next_repo = get_metadata_field(&next, &self.repo_field_name);
                let next_pl = get_metadata_field(&next, &self.pl_field_name);
                if next_repo == current_repo && next_pl == current_pl {
                    repo_texts.push(
                        next.get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    );
                } else {
                    maybe_current_group_head = Some(next);
                    break;
                }
            }

            if self.randomize_order {
                repo_texts.shuffle(&mut random);
            }

            let repo_text = repo_texts.join(&self.file_separator_token);

            // We chose an arbitrary node from the current repo/pl group to
            // represent the concatenated document.
            // No attempt is made at this stage to coalesce attributes or ids
            // or other data from the different group members, but it perhaps
            // warrants future consideration.
            let mut repo_document = current_group_head.clone();

            if let Some(obj) = repo_document.as_object_mut() {
                obj.insert("text".to_string(), json!(repo_text));
                if let Some(metadata) = obj.get_mut("metadata") {
                    metadata["files_concatenated"] = json!(repo_texts.len());
                }
            }

            Some(repo_document)
        })
    }
}
