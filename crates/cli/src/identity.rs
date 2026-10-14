use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use compute_net::quic::{self, Secret};
use data_encoding::HEXLOWER;

/// Reads this machine's secret key, creating it on first use.
pub fn load_or_create(path: &Path) -> Result<Secret> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse(&text).with_context(|| format!("{} is corrupt", path.display())),
        Err(error) if error.kind() == ErrorKind::NotFound => create(path),
        Err(error) => Err(error).with_context(|| format!("cannot read {}", path.display())),
    }
}

fn parse(text: &str) -> Result<Secret> {
    let bytes = HEXLOWER.decode(text.trim().as_bytes())?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("a secret key is 32 bytes"))
}

fn create(path: &Path) -> Result<Secret> {
    let secret = quic::generate_secret();
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options
        .open(path)
        .and_then(|mut file| file.write_all(HEXLOWER.encode(&secret).as_bytes()))
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(secret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_identity_is_returned_on_every_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");

        let first = load_or_create(&path).unwrap();
        let second = load_or_create(&path).unwrap();

        assert_eq!(first, second);
    }

    #[cfg(unix)]
    #[test]
    fn the_key_file_is_private_to_its_owner() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");

        load_or_create(&path).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn a_corrupt_key_file_is_an_error_not_a_new_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, "not a key").unwrap();

        assert!(load_or_create(&path).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not a key");
    }
}
