use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Document {
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Deserialize, Default)]
pub struct Metadata {
    #[serde(default)]
    pub repo_name: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct RepoStats {
    pub document_count: usize,
    pub total_score: f64,
    pub average_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageReport {
    pub repo_stats: HashMap<String, RepoStats>,
    pub total_documents: usize,
    pub total_repositories: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Report {
    pub languages: HashMap<String, LanguageReport>,
    pub summary: Summary,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Summary {
    pub total_languages: usize,
    pub total_repositories: usize,
    pub total_documents: usize,
}

#[derive(Serialize, Deserialize)]
pub struct StreamingScoreReport {
    pub summary: Summary,
    pub language_offsets: HashMap<String, (u64, u64)>, // language -> (start_offset, end_offset)
}

#[derive(Debug, Clone)]
pub struct RepoRecord {
    pub language: String,
    pub repo_name: String,
    pub document_count: u64,
    pub total_score: f64,
    pub average_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

impl RepoRecord {
    pub fn from_repo_stats(language: &str, repo_name: &str, stats: &RepoStats) -> Self {
        Self {
            language: language.to_string(),
            repo_name: repo_name.to_string(),
            document_count: stats.document_count as u64,
            total_score: stats.total_score,
            average_score: stats.average_score,
            min_score: stats.min_score,
            max_score: stats.max_score,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct BinReport {
    pub language_bins: HashMap<String, LanguageBinReport>,
    pub summary: BinSummary,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageBinReport {
    pub language: String,
    pub bins: Vec<ScoreBin>,
    pub summary: LanguageBinSummary,
}

#[derive(Serialize, Deserialize)]
pub struct ScoreBin {
    pub min_score: f64,
    pub max_score: f64,
    pub sample_repos: Vec<BinRepo>,
    pub total_repos_in_range: usize,
}

#[derive(Serialize, Deserialize)]
pub struct BinRepo {
    pub repo_name: String,
    pub language: String,
    pub average_score: f64,
    pub document_count: usize,
}

#[derive(Serialize, Deserialize)]
pub struct BinSummary {
    pub total_languages: usize,
    pub total_repositories: usize,
    pub total_documents: usize,
    pub num_bins: usize,
    pub sample_size_per_bin: usize,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageBinSummary {
    pub language: String,
    pub total_repositories: usize,
    pub total_documents: usize,
    pub num_bins: usize,
    pub sample_size_per_bin: usize,
}