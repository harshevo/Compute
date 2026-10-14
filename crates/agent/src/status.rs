use std::path::PathBuf;

use async_trait::async_trait;
use compute_proto::message::{GpuInfo, NodeStatus};
use tokio::process::Command;

use crate::PauseSwitch;

#[async_trait]
pub trait StatusSource: Send + Sync {
    async fn status(&self) -> NodeStatus;
    async fn worker(&self, _request: compute_proto::message::WorkerRequest) -> String {
        r#"{"error":"worker execution is not configured"}"#.into()
    }
}

/// Reads the real machine: NVIDIA GPUs, free disk under the data directory, and the pause switch.
pub struct HostStatus {
    data_dir: PathBuf,
    pause: PauseSwitch,
}

impl HostStatus {
    pub fn new(data_dir: PathBuf, pause: PauseSwitch) -> Self {
        Self { data_dir, pause }
    }
}

#[async_trait]
impl StatusSource for HostStatus {
    async fn worker(&self, request: compute_proto::message::WorkerRequest) -> String {
        use tokio::io::AsyncWriteExt;
        let python = std::env::var("COMPUTE_PYTHON").unwrap_or_else(|_| {
            if cfg!(windows) {
                "python".into()
            } else {
                "python3".into()
            }
        });
        let child = Command::new(python)
            .args(["-m", "compute_sdk.worker", "--root"])
            .arg(&self.data_dir)
            .arg(&request.operation)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn();
        let Ok(mut child) = child else {
            return r#"{"error":"cannot start COMPUTE_PYTHON; install compute-sdk on the worker"}"#
                .into();
        };
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(120), async {
            if let Some(mut input) = child.stdin.take() {
                input.write_all(request.payload.as_bytes()).await?;
            }
            child.wait_with_output().await
        })
        .await;
        match outcome {
            Ok(Ok(output)) if output.status.success() && output.stdout.len() < 900_000 => {
                String::from_utf8_lossy(&output.stdout).into_owned()
            }
            _ => r#"{"error":"worker command failed or timed out; check Python installation"}"#
                .into(),
        }
    }
    async fn status(&self) -> NodeStatus {
        NodeStatus {
            gpus: {
                let mut gpus = query_nvidia_smi().await;
                if let Ok(text) = std::fs::read_to_string(self.data_dir.join("inventory.json")) {
                    if let Ok(serde_json::Value::Array(devices)) =
                        serde_json::from_str::<serde_json::Value>(&text)
                    {
                        for device in devices {
                            let backend = device["backend"].as_str().unwrap_or("");
                            if matches!(backend, "mps" | "xpu" | "openvino") {
                                gpus.push(GpuInfo {
                                    name: format!(
                                        "{} [{}]",
                                        device["name"].as_str().unwrap_or("GPU"),
                                        backend
                                    ),
                                    vram_total_mib: device["memory_bytes"].as_u64().unwrap_or(0)
                                        / (1 << 20),
                                    vram_free_mib: 0,
                                    utilization_percent: 0,
                                });
                            }
                        }
                    }
                }
                gpus
            },
            disk_free_bytes: fs4::available_space(&self.data_dir).unwrap_or(0),
            paused: self.pause.is_paused(),
        }
    }
}

const NVIDIA_SMI_FIELDS: &str = "name,memory.total,memory.free,utilization.gpu";

/// A machine without `nvidia-smi` simply reports no GPUs.
async fn query_nvidia_smi() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .arg(format!("--query-gpu={NVIDIA_SMI_FIELDS}"))
        .arg("--format=csv,noheader,nounits")
        .output()
        .await;
    match output {
        Ok(output) if output.status.success() => {
            parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout))
        }
        _ => Vec::new(),
    }
}

/// Parses `nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu
/// --format=csv,noheader,nounits`. Lines that do not fit are skipped.
pub fn parse_nvidia_smi(csv: &str) -> Vec<GpuInfo> {
    csv.lines().filter_map(parse_gpu_line).collect()
}

fn parse_gpu_line(line: &str) -> Option<GpuInfo> {
    let mut fields = line.split(',').map(str::trim);
    let gpu = GpuInfo {
        name: fields.next().filter(|name| !name.is_empty())?.to_owned(),
        vram_total_mib: fields.next()?.parse().ok()?,
        vram_free_mib: fields.next()?.parse().ok()?,
        utilization_percent: fields.next()?.parse().ok()?,
    };
    fields.next().is_none().then_some(gpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_one_line_per_gpu() {
        let csv =
            "NVIDIA GeForce RTX 4090, 24564, 23010, 3\nNVIDIA GeForce RTX 3060, 12288, 512, 97\n";

        let gpus = parse_nvidia_smi(csv);

        assert_eq!(
            gpus,
            [
                GpuInfo {
                    name: "NVIDIA GeForce RTX 4090".into(),
                    vram_total_mib: 24564,
                    vram_free_mib: 23010,
                    utilization_percent: 3,
                },
                GpuInfo {
                    name: "NVIDIA GeForce RTX 3060".into(),
                    vram_total_mib: 12288,
                    vram_free_mib: 512,
                    utilization_percent: 97,
                },
            ]
        );
    }

    #[test]
    fn skips_lines_it_cannot_understand() {
        let csv =
            "\nTesla T4, 15360, [N/A], 0\nonly, two\nA, 1, 2, 3, extra\nRTX A2000, 6138, 6000, 0\n";

        let gpus = parse_nvidia_smi(csv);

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "RTX A2000");
    }

    #[test]
    fn no_output_means_no_gpus() {
        assert_eq!(parse_nvidia_smi(""), []);
    }

    #[tokio::test]
    async fn host_status_reports_disk_space_and_the_pause_switch() {
        let dir = tempfile::tempdir().unwrap();
        let pause = PauseSwitch::new(dir.path().join("paused"));
        let host = HostStatus::new(dir.path().to_owned(), pause.clone());

        assert!(host.status().await.disk_free_bytes > 0);
        assert!(!host.status().await.paused);

        pause.pause().unwrap();
        assert!(host.status().await.paused);
    }
}
