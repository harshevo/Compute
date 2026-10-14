//! Length-prefixed postcard messages.

use std::io::ErrorKind;

use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{NetError, Result};

pub const MAX_MESSAGE_BYTES: usize = 1 << 20;

pub async fn write_message<W, T>(writer: &mut W, message: &T) -> Result<()>
where
    W: AsyncWrite + Unpin + ?Sized,
    T: Serialize,
{
    let body = postcard::to_stdvec(message)?;
    if body.len() > MAX_MESSAGE_BYTES {
        return Err(NetError::TooLarge(body.len()));
    }
    writer.write_u32(body.len() as u32).await?;
    writer.write_all(&body).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_message<R, T>(reader: &mut R) -> Result<T>
where
    R: AsyncRead + Unpin + ?Sized,
    T: DeserializeOwned,
{
    let length = match reader.read_u32().await {
        Ok(length) => length as usize,
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Err(NetError::Closed),
        Err(error) => return Err(error.into()),
    };
    if length > MAX_MESSAGE_BYTES {
        return Err(NetError::TooLarge(length));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).await?;
    Ok(postcard::from_bytes(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn messages_round_trip_in_order() {
        let (mut near, mut far) = tokio::io::duplex(1024);
        write_message(&mut near, &("first", 1u32)).await.unwrap();
        write_message(&mut near, &("second", 2u32)).await.unwrap();

        let first: (String, u32) = read_message(&mut far).await.unwrap();
        let second: (String, u32) = read_message(&mut far).await.unwrap();

        assert_eq!(first, ("first".to_owned(), 1));
        assert_eq!(second, ("second".to_owned(), 2));
    }

    #[tokio::test]
    async fn clean_end_of_stream_is_reported_as_closed() {
        let (near, mut far) = tokio::io::duplex(1024);
        drop(near);

        let result: Result<u32> = read_message(&mut far).await;

        assert!(matches!(result, Err(NetError::Closed)));
    }

    #[tokio::test]
    async fn stream_cut_inside_a_message_is_an_error() {
        let (mut near, mut far) = tokio::io::duplex(1024);
        near.write_u32(100).await.unwrap();
        near.write_all(b"partial").await.unwrap();
        drop(near);

        let result: Result<String> = read_message(&mut far).await;

        assert!(matches!(result, Err(NetError::Io(_))));
    }

    #[tokio::test]
    async fn oversized_length_prefix_is_refused_before_allocating() {
        let (mut near, mut far) = tokio::io::duplex(1024);
        near.write_u32(u32::MAX).await.unwrap();

        let result: Result<String> = read_message(&mut far).await;

        assert!(matches!(result, Err(NetError::TooLarge(_))));
    }

    #[tokio::test]
    async fn oversized_message_is_refused_on_write() {
        let (mut near, _far) = tokio::io::duplex(1024);
        let huge = vec![0u8; MAX_MESSAGE_BYTES + 1];

        let result = write_message(&mut near, &huge).await;

        assert!(matches!(result, Err(NetError::TooLarge(_))));
    }

    #[tokio::test]
    async fn undecodable_body_is_an_error() {
        let (mut near, mut far) = tokio::io::duplex(1024);
        near.write_u32(1).await.unwrap();
        near.write_all(&[0xff]).await.unwrap();

        let result: Result<bool> = read_message(&mut far).await;

        assert!(matches!(result, Err(NetError::Decode(_))));
    }
}
