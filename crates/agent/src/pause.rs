use std::io;
use std::path::PathBuf;

/// The owner's off switch. It is a file so that it survives restarts and needs no running agent.
#[derive(Clone, Debug)]
pub struct PauseSwitch {
    flag: PathBuf,
}

impl PauseSwitch {
    pub fn new(flag: PathBuf) -> Self {
        Self { flag }
    }

    pub fn is_paused(&self) -> bool {
        self.flag.exists()
    }

    pub fn pause(&self) -> io::Result<()> {
        std::fs::write(&self.flag, b"")
    }

    pub fn resume(&self) -> io::Result<()> {
        match std::fs::remove_file(&self.flag) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pausing_persists_until_resumed() {
        let dir = tempfile::tempdir().unwrap();
        let switch = PauseSwitch::new(dir.path().join("paused"));
        assert!(!switch.is_paused());

        switch.pause().unwrap();
        assert!(PauseSwitch::new(dir.path().join("paused")).is_paused());

        switch.resume().unwrap();
        assert!(!switch.is_paused());
    }

    #[test]
    fn pausing_and_resuming_twice_is_harmless() {
        let dir = tempfile::tempdir().unwrap();
        let switch = PauseSwitch::new(dir.path().join("paused"));

        switch.resume().unwrap();
        switch.pause().unwrap();
        switch.pause().unwrap();

        assert!(switch.is_paused());
    }
}
