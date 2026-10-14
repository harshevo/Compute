use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;

use crate::link::Link;

const BUFFER_BYTES: usize = 64 << 10;
const CHUNK_BYTES: usize = 16 << 10;
const CHUNKS_IN_FLIGHT: usize = 64;

struct Chunk {
    arrives_at: Instant,
    bytes: Vec<u8>,
}

/// A one-way byte pipe whose far end sees data after the link's transmit time and latency.
pub(crate) fn shaped(link: Arc<Link>, severed: CancellationToken) -> (DuplexStream, DuplexStream) {
    let (writer, wire_in) = tokio::io::duplex(BUFFER_BYTES);
    let (wire_out, reader) = tokio::io::duplex(BUFFER_BYTES);
    let (in_flight, arriving) = mpsc::channel(CHUNKS_IN_FLIGHT);

    tokio::spawn(severed.clone().run_until_cancelled_owned(transmit(
        wire_in,
        in_flight,
        link.clone(),
    )));
    tokio::spawn(severed.run_until_cancelled_owned(deliver(arriving, wire_out, link)));

    (writer, reader)
}

async fn transmit(mut wire_in: DuplexStream, in_flight: mpsc::Sender<Chunk>, link: Arc<Link>) {
    let mut buffer = vec![0; CHUNK_BYTES];
    let mut line_free_at = Instant::now();
    loop {
        let len = match wire_in.read(&mut buffer).await {
            Ok(0) | Err(_) => return,
            Ok(len) => len,
        };
        let profile = link.profile();
        line_free_at = line_free_at.max(Instant::now()) + profile.transmit_time(len);
        let chunk = Chunk {
            arrives_at: line_free_at + profile.latency,
            bytes: buffer[..len].to_vec(),
        };
        if in_flight.send(chunk).await.is_err() {
            return;
        }
    }
}

async fn deliver(mut arriving: mpsc::Receiver<Chunk>, mut wire_out: DuplexStream, link: Arc<Link>) {
    while let Some(chunk) = arriving.recv().await {
        sleep_until(chunk.arrives_at).await;
        link.passable().await;
        if wire_out.write_all(&chunk.bytes).await.is_err() {
            return;
        }
    }
}
