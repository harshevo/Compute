use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const OVERRIDE_VARIABLE: &str = "COMPUTE_HOME";

/// Where this machine keeps its keys and state: `$COMPUTE_HOME`, or `~/.compute`.
pub struct Home {
    root: PathBuf,
}

impl Home {
    pub fn locate() -> Result<Self> {
        let root = match std::env::var_os(OVERRIDE_VARIABLE) {
            Some(root) => PathBuf::from(root),
            None => std::env::home_dir()
                .context("cannot find a home directory; set COMPUTE_HOME")?
                .join(".compute"),
        };
        Ok(Self { root })
    }

    #[cfg(test)]
    pub fn at(root: &Path) -> Self {
        Self {
            root: root.to_owned(),
        }
    }

    pub fn coordinator(&self) -> Result<CoordinatorDir> {
        Ok(CoordinatorDir(created(self.root.join("coordinator"))?))
    }

    pub fn agent(&self) -> Result<AgentDir> {
        Ok(AgentDir(created(self.root.join("agent"))?))
    }
}

fn created(dir: PathBuf) -> Result<PathBuf> {
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(dir)
}

pub struct CoordinatorDir(PathBuf);

impl CoordinatorDir {
    pub fn identity(&self) -> PathBuf {
        self.0.join("identity.key")
    }

    pub fn state(&self) -> PathBuf {
        self.0.join("state.db")
    }

    pub fn admin_socket(&self) -> PathBuf {
        self.0.join("admin.sock")
    }
}

pub struct AgentDir(PathBuf);

impl AgentDir {
    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn identity(&self) -> PathBuf {
        self.0.join("identity.key")
    }

    pub fn ticket(&self) -> PathBuf {
        self.0.join("ticket")
    }

    pub fn pause_flag(&self) -> PathBuf {
        self.0.join("paused")
    }
}
