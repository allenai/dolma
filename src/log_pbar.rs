use indicatif::{ProgressBar, ProgressStyle};
use log::info;

pub struct LogProgressBar {
    progress_bar: ProgressBar,
    last_logged_position: u64,
}

impl LogProgressBar {
    pub fn new(total: usize) -> Self {
        let progress_bar = ProgressBar::new(total.try_into().unwrap());
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{elapsed_precise} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        Self {
            progress_bar,
            last_logged_position: 0,
        }
    }

    pub fn inc(&mut self, delta: u64) {
        self.progress_bar.inc(delta);
        self.log_progress_if_needed();
    }

    fn finish_with_message(&mut self, msg: &str) {
        self.progress_bar.finish_with_message(msg.to_string());
        self.log_progress();
    }

    fn log_progress_if_needed(&mut self) {
        let current_position = self.progress_bar.position();
        if current_position - self.last_logged_position >= self.progress_bar.length().unwrap() / 10
        {
            self.log_progress();
        }
    }

    fn log_progress(&mut self) {
        let mut progress_message = String::new();
        self.progress_bar.println(&mut progress_message);
        info!("{}", progress_message.trim());
        self.last_logged_position = self.progress_bar.position();
    }
}
