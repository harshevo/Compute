//! Turns replies into text for the terminal.

use std::time::Duration;

use compute_proto::admin::{Liveness, NodeSummary, PathKind, ProbeReport};

const HEADINGS: [&str; 7] = [
    "NAME",
    "STATE",
    "GPU",
    "VRAM FREE",
    "UTIL",
    "DISK FREE",
    "ID",
];

pub fn nodes(nodes: &[NodeSummary]) -> String {
    if nodes.is_empty() {
        return "No machines have joined yet. Create a ticket with `compute invite <name>`.".into();
    }
    let rows: Vec<_> = nodes.iter().flat_map(node_rows).collect();
    table(&HEADINGS, &rows)
}

pub fn probe(report: &ProbeReport) -> String {
    let path = match report.path {
        PathKind::Direct => "direct",
        PathKind::Relayed => "relayed (slow; check the router's NAT settings)",
        PathKind::Unknown => "unknown",
    };
    format!(
        "round trip  {:.1} ms\nupload      {}\ndownload    {}\npath        {path}",
        report.rtt.as_secs_f64() * 1000.0,
        rate(report.upload_bps),
        rate(report.download_bps),
    )
}

/// One row per GPU; a machine without GPUs still gets a row.
fn node_rows(node: &NodeSummary) -> Vec<[String; 7]> {
    let identity = |gpu: [String; 3]| {
        let [name, vram, utilization] = gpu;
        [
            node.name.clone(),
            state(node),
            name,
            vram,
            utilization,
            bytes(node.status.disk_free_bytes),
            node.id.short(),
        ]
    };
    if node.status.gpus.is_empty() {
        return vec![identity(["-".into(), "-".into(), "-".into()])];
    }
    node.status
        .gpus
        .iter()
        .map(|gpu| {
            identity([
                gpu.name.clone(),
                format!("{} / {} MiB", gpu.vram_free_mib, gpu.vram_total_mib),
                format!("{}%", gpu.utilization_percent),
            ])
        })
        .collect()
}

fn state(node: &NodeSummary) -> String {
    match node.liveness {
        Liveness::Online if node.status.paused => "paused".into(),
        Liveness::Online => "online".into(),
        Liveness::Unreachable => format!("unreachable {}", ago(node.last_seen)),
        Liveness::Lost => format!("lost {}", ago(node.last_seen)),
    }
}

fn ago(elapsed: Duration) -> String {
    match elapsed.as_secs() {
        seconds if seconds < 120 => format!("{seconds}s"),
        seconds if seconds < 7200 => format!("{}m", seconds / 60),
        seconds => format!("{}h", seconds / 3600),
    }
}

fn rate(bits_per_second: u64) -> String {
    format!("{:.1} Mbit/s", bits_per_second as f64 / 1e6)
}

fn bytes(count: u64) -> String {
    format!("{:.1} GiB", count as f64 / (1u64 << 30) as f64)
}

fn table<const N: usize>(headings: &[&str; N], rows: &[[String; N]]) -> String {
    let width = |column: usize| {
        rows.iter()
            .map(|row| row[column].len())
            .chain([headings[column].len()])
            .max()
            .unwrap_or(0)
    };
    let widths: [usize; N] = std::array::from_fn(width);
    let line = |cells: Vec<&str>| {
        let padded: Vec<_> = cells
            .iter()
            .zip(widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect();
        padded.join("  ").trim_end().to_owned()
    };
    let mut lines = vec![line(headings.to_vec())];
    lines.extend(
        rows.iter()
            .map(|row| line(row.iter().map(String::as_str).collect())),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use compute_proto::NodeId;
    use compute_proto::message::{GpuInfo, NodeStatus};

    use super::*;

    fn node(name: &str, liveness: Liveness, gpus: Vec<GpuInfo>) -> NodeSummary {
        NodeSummary {
            id: NodeId::from_bytes([0xab; 32]),
            name: name.into(),
            liveness,
            status: NodeStatus {
                gpus,
                disk_free_bytes: 100 << 30,
                paused: false,
            },
            last_seen: Duration::from_secs(90),
        }
    }

    fn gpu(name: &str) -> GpuInfo {
        GpuInfo {
            name: name.into(),
            vram_total_mib: 24564,
            vram_free_mib: 20000,
            utilization_percent: 7,
        }
    }

    #[test]
    fn an_empty_pool_explains_what_to_do_next() {
        assert!(nodes(&[]).contains("compute invite"));
    }

    #[test]
    fn columns_line_up_across_rows() {
        let rendered = nodes(&[
            node("a", Liveness::Online, vec![gpu("RTX 4090")]),
            node("much-longer-name", Liveness::Online, vec![gpu("T4")]),
        ]);

        let state_columns: Vec<_> = rendered
            .lines()
            .skip(1)
            .map(|line| line.find("online").unwrap())
            .collect();
        assert_eq!(state_columns[0], state_columns[1]);
        assert!(rendered.starts_with("NAME"));
    }

    #[test]
    fn each_gpu_gets_its_own_row() {
        let rendered = nodes(&[node(
            "rig",
            Liveness::Online,
            vec![gpu("RTX 4090"), gpu("RTX 3060")],
        )]);

        assert_eq!(rendered.lines().count(), 3);
        assert!(rendered.contains("20000 / 24564 MiB"));
        assert!(rendered.contains("7%"));
    }

    #[test]
    fn a_machine_without_gpus_is_still_listed() {
        let rendered = nodes(&[node("laptop", Liveness::Online, vec![])]);

        assert!(rendered.lines().nth(1).unwrap().starts_with("laptop"));
        assert!(rendered.contains("100.0 GiB"));
    }

    #[test]
    fn state_shows_pause_and_how_long_a_node_has_been_silent() {
        let mut paused = node("a", Liveness::Online, vec![]);
        paused.status.paused = true;

        assert_eq!(state(&paused), "paused");
        assert_eq!(
            state(&node("a", Liveness::Unreachable, vec![])),
            "unreachable 90s"
        );
        assert_eq!(state(&node("a", Liveness::Lost, vec![])), "lost 90s");
    }

    #[test]
    fn silence_is_shown_in_the_largest_sensible_unit() {
        assert_eq!(ago(Duration::from_secs(5)), "5s");
        assert_eq!(ago(Duration::from_secs(600)), "10m");
        assert_eq!(ago(Duration::from_secs(3 * 3600)), "3h");
    }

    #[test]
    fn a_probe_report_is_readable() {
        let report = ProbeReport {
            rtt: Duration::from_micros(41_500),
            upload_bps: 18_400_000,
            download_bps: 94_000_000,
            path: PathKind::Direct,
        };

        let rendered = probe(&report);

        assert!(rendered.contains("41.5 ms"));
        assert!(rendered.contains("18.4 Mbit/s"));
        assert!(rendered.contains("94.0 Mbit/s"));
        assert!(rendered.contains("direct"));
    }

    #[test]
    fn a_relayed_path_is_called_out() {
        let report = ProbeReport {
            rtt: Duration::from_millis(80),
            upload_bps: 1,
            download_bps: 1,
            path: PathKind::Relayed,
        };

        assert!(probe(&report).contains("relayed (slow"));
    }
}
