//! Code imported from github.com/allenai/wimbd/blob/main/src/io.rs
//! and modified by @soldni to integrate in dolma.


use anyhow::Result;
use indicatif::{ProgressDrawTarget, ProgressStyle};

pub(crate) use indicatif::{MultiProgress, ProgressBar, ProgressIterator};

pub(crate) fn get_multi_progress_bar(hidden: bool) -> MultiProgress {
    if !hidden {
        MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(2))
    } else {
        MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
    }
}

pub(crate) fn get_file_progress_bar(
    msg: &'static str,
    n_files: usize,
    hidden: bool,
) -> Result<ProgressBar> {
    let progress = ProgressBar::new(n_files.try_into()?)
        .with_style(
            ProgressStyle::with_template(
                "{msg}: files {human_pos}/{human_len} [{elapsed_precise}] [{wide_bar:.cyan/blue}]",
            )?
            .progress_chars("#>-"),
        )
        .with_message(msg);
    if hidden {
        progress.set_draw_target(ProgressDrawTarget::hidden());
    } else {
        progress.set_draw_target(ProgressDrawTarget::stderr_with_hz(1));
        progress.enable_steady_tick(std::time::Duration::from_secs(1));
    }
    Ok(progress)
}

pub(crate) fn get_progress_bar(
    path: impl AsRef<std::path::Path>,
    limit: Option<usize>,
    hidden: bool,
) -> Result<ProgressBar> {
    let progress: ProgressBar = if let Some(limit) = limit {
        ProgressBar::new(limit.try_into()?).with_style(
            ProgressStyle::with_template(
                "{msg:<35!} {human_pos} [{wide_bar:.cyan/blue}] {per_sec:>12}, <{eta:<3} ",
            )?
            .progress_chars("#>-"),
        )
    } else {
        ProgressBar::new_spinner().with_style(ProgressStyle::with_template(
            "{msg:<35!} {spinner:.green} {human_pos} {per_sec:12}",
        )?)
    }
    .with_message(format!(
        "{}:",
        path.as_ref().file_name().unwrap().to_string_lossy()
    ));

    if hidden {
        progress.set_draw_target(ProgressDrawTarget::hidden());
    } else {
        progress.set_draw_target(ProgressDrawTarget::stderr_with_hz(1));
    }

    Ok(progress)
}
