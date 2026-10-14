//! Measures round-trip time and throughput over an established connection.

use std::time::Duration;

use compute_proto::admin::ProbeReport;
use compute_proto::message::{ProbeReply, ProbeRequest, StreamKind, Transfer};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::Instant;

use crate::stream::open_stream;
use crate::{BiStream, Connection, NetError, Result};

pub const MAX_PAYLOAD_BYTES: u64 = 256 << 20;
const CHUNK_BYTES: usize = 64 << 10;

#[derive(Clone, Copy, Debug)]
pub struct ProbeConfig {
    pub pings: u32,
    pub payload_bytes: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            pings: 5,
            payload_bytes: 8 << 20,
        }
    }
}

/// Measures the connection, once more if the path changed underneath the measurement
/// (a connection typically starts on a relay and then upgrades to a direct path).
pub async fn measure(connection: &dyn Connection, config: ProbeConfig) -> Result<ProbeReport> {
    let report = measure_once(connection, config).await?;
    if report.path == connection.path() {
        return Ok(report);
    }
    measure_once(connection, config).await
}

async fn measure_once(connection: &dyn Connection, config: ProbeConfig) -> Result<ProbeReport> {
    let path = connection.path();
    let mut stream = open_stream(connection, StreamKind::Probe).await?;
    Ok(ProbeReport {
        rtt: measure_rtt(&mut stream, config.pings).await?,
        upload_bps: upload(&mut stream, config.payload_bytes).await?,
        download_bps: download(&mut stream, config.payload_bytes).await?,
        path,
    })
}

/// Answers probe requests until the requester closes the stream.
pub async fn respond(mut stream: BiStream) -> Result<()> {
    loop {
        match stream.receive().await {
            Ok(ProbeRequest::Ping) => stream.send(&ProbeReply::Pong).await?,
            Ok(ProbeRequest::Sink { bytes }) => {
                let transfer = drain(&mut stream, checked(bytes)?).await?;
                stream.send(&ProbeReply::Sunk(transfer)).await?;
            }
            Ok(ProbeRequest::Source { bytes }) => write_zeros(&mut stream, checked(bytes)?).await?,
            Err(NetError::Closed) => return Ok(()),
            Err(error) => return Err(error),
        }
    }
}

async fn measure_rtt(stream: &mut BiStream, pings: u32) -> Result<Duration> {
    let mut best = Duration::MAX;
    for _ in 0..pings.max(1) {
        let started = Instant::now();
        stream.send(&ProbeRequest::Ping).await?;
        match stream.receive().await? {
            ProbeReply::Pong => best = best.min(started.elapsed()),
            ProbeReply::Sunk(_) => return Err(UNEXPECTED_REPLY),
        }
    }
    Ok(best)
}

async fn upload(stream: &mut BiStream, bytes: u64) -> Result<u64> {
    stream.send(&ProbeRequest::Sink { bytes }).await?;
    write_zeros(stream, bytes).await?;
    match stream.receive().await? {
        ProbeReply::Sunk(transfer) => Ok(transfer.bits_per_second()),
        ProbeReply::Pong => Err(UNEXPECTED_REPLY),
    }
}

async fn download(stream: &mut BiStream, bytes: u64) -> Result<u64> {
    stream.send(&ProbeRequest::Source { bytes }).await?;
    Ok(drain(stream, bytes).await?.bits_per_second())
}

const UNEXPECTED_REPLY: NetError = NetError::Protocol("unexpected probe reply");

fn checked(bytes: u64) -> Result<u64> {
    if bytes <= MAX_PAYLOAD_BYTES {
        Ok(bytes)
    } else {
        Err(NetError::Protocol("probe payload too large"))
    }
}

async fn write_zeros(stream: &mut BiStream, bytes: u64) -> Result<()> {
    let chunk = [0u8; CHUNK_BYTES];
    let mut remaining = bytes;
    while remaining > 0 {
        let len = remaining.min(CHUNK_BYTES as u64) as usize;
        stream.writer.write_all(&chunk[..len]).await?;
        remaining -= len as u64;
    }
    stream.writer.flush().await?;
    Ok(())
}

/// Reads and discards exactly `bytes`, timing everything after the first arrival.
async fn drain(stream: &mut BiStream, bytes: u64) -> Result<Transfer> {
    let mut buffer = vec![0u8; CHUNK_BYTES];
    let mut remaining = bytes;
    let mut first_arrival = None;
    let mut transfer = Transfer::default();
    while remaining > 0 {
        let wanted = remaining.min(CHUNK_BYTES as u64) as usize;
        let read = stream.reader.read(&mut buffer[..wanted]).await? as u64;
        if read == 0 {
            return Err(NetError::Closed);
        }
        remaining -= read;
        match first_arrival {
            None => first_arrival = Some(Instant::now()),
            Some(first) => {
                transfer.bytes_after_first_arrival += read;
                transfer.elapsed = first.elapsed();
            }
        }
    }
    Ok(transfer)
}
