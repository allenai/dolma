use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use threadpool::ThreadPool;

use crate::shard::Shard;

use mixer_config::*;

pub fn run(config: MixerConfig) -> Result<u32, u32> {
    let shards = Shard::split_streams(&config.streams).unwrap();

    let threadpool = ThreadPool::new(config.processes);
    let failed_shard_count_ref = Arc::new(AtomicU32::new(0));
    for shard in shards {
        let output_path = Path::new(&config.work_dir.output.clone()).join(&shard.output);
        if output_path.exists() {
            log::info!("Skipping {:?} because it already exists", shard.output);
            continue;
        }
        let shard = shard.clone();
        let work_dirs = config.work_dir.clone();
        let failed_shard_count_ref = failed_shard_count_ref.clone();

        threadpool.execute(move || {
            log::info!("Building output {:?}...", shard.output);
            if let Err(e) = shard.clone().process(work_dirs) {
                log::error!("Error processing {:?}: {}", shard.output, e);
                failed_shard_count_ref.fetch_add(1, Ordering::Relaxed);
            }
        });
    }
    threadpool.join();

    let failure_count = failed_shard_count_ref.load(Ordering::Relaxed);
    if failure_count == 0 {
        log::info!("Done!");
        Ok(failure_count)
    } else {
        log::error!("{} shards failed to process.", failure_count);
        Err(failure_count)
    }
}

pub mod mixer_config {
    use serde::{Deserialize, Serialize};
    use std::fs::File;
    use std::io;

    use crate::shard::shard_config::{StreamConfig, WorkDirConfig};

    #[derive(Serialize, Deserialize, Clone)]
    pub struct MixerConfig {
        pub streams: Vec<StreamConfig>,
        pub processes: usize,
        pub work_dir: WorkDirConfig,
    }

    impl MixerConfig {
        pub fn read_from_file(path: &str) -> Result<MixerConfig, io::Error> {
            let file = File::open(path)?;
            let reader = io::BufReader::new(file);
            let config: MixerConfig = serde_json::from_reader(reader)?;
            Ok(config)
        }
        pub fn parse_from_string(s: &str) -> Result<MixerConfig, io::Error> {
            let config: MixerConfig = serde_json::from_str(s)?;
            Ok(config)
        }
    }
}
