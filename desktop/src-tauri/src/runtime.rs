use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Connection {
    pub base_url: String,
    pub token: String,
}

#[derive(Deserialize)]
struct Manifest {
    version: String,
    platform: String,
    machine: String,
}

fn normalized_arch(value: &str) -> &str {
    match value {
        "amd64" | "x86_64" => "x86_64",
        "arm64" | "aarch64" => "aarch64",
        other => other,
    }
}

pub fn packaged_executable(root: &Path, version: &str) -> Result<PathBuf, String> {
    let manifest_path = root.join("_internal/desktop_resources/runtime.json");
    let text = std::fs::read_to_string(&manifest_path).map_err(|_| format!("The bundled runtime is incomplete (missing {}). Reinstall the complete application, not just its executable.", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .map_err(|_| "The bundled runtime manifest is invalid. Reinstall NLP Suite.".to_string())?;
    let system = if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    };
    if manifest.platform != system
        || normalized_arch(&manifest.machine.to_lowercase())
            != normalized_arch(std::env::consts::ARCH)
        || manifest.version != version
    {
        return Err(format!("Runtime mismatch: this app needs {version} on {system}/{}; found {} on {}/{}. Install the matching package.", std::env::consts::ARCH, manifest.version, manifest.platform, manifest.machine));
    }
    let executable = root.join(if cfg!(windows) {
        "nlp-runtime.exe"
    } else {
        "nlp-runtime"
    });
    if !executable.is_file() {
        return Err("The bundled Python executable is missing. Reinstall NLP Suite.".into());
    }
    Ok(executable)
}

pub fn validate_connection(value: &Connection) -> Result<(), String> {
    let url = reqwest::Url::parse(&value.base_url).map_err(|_| "Invalid engine address")?;
    if url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || url.port().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
        || value.token.len() < 16
    {
        return Err("The engine returned an invalid local connection.".into());
    }
    Ok(())
}

pub fn start(mut command: Command, workspace: &Path) -> Result<(Connection, Child), String> {
    std::fs::create_dir_all(workspace)
        .map_err(|e| format!("Cannot create workspace {}: {e}", workspace.display()))?;
    let log_path = workspace.join("backend.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Cannot write {}: {e}", log_path.display()))?;
    command.arg("--desktop").arg("--data-dir").arg(workspace);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(log);
    #[cfg(windows)]
    command.creation_flags(0x08000000);
    let mut child = command.spawn().map_err(|e| format!("Cannot start the bundled engine: {e}. Reinstall the package for this computer. Log: {}", log_path.display()))?;
    let stdout = child.stdout.take().ok_or("No engine stdout")?;
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout)
            .take(8192)
            .read_line(&mut line)
            .map(|_| line);
        let _ = send.send(result);
    });
    let result = (|| -> Result<Connection, String> {
        let line = receive
            .recv_timeout(Duration::from_secs(60))
            .map_err(|_| "Engine startup timed out")?
            .map_err(|e| e.to_string())?;
        let value: Connection = serde_json::from_str(&line)
            .map_err(|_| "Engine exited before providing a connection")?;
        validate_connection(&value)?;
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(1))
            .build()
            .map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                return Err("Engine stopped during startup".into());
            }
            if let Ok(response) = client
                .get(format!("{}/api/health", value.base_url))
                .bearer_auth(&value.token)
                .send()
            {
                if response.status().is_success() {
                    return Ok(value);
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Err("Engine did not become ready".into())
    })();
    match result {
        Ok(connection) => Ok((connection, child)),
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            Err(format!("{error}. If a previous analysis is finishing after closing the app, wait for it to finish. Otherwise inspect {}. Your project files have not been deleted.", log_path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_accepts_local_engine_handshakes() {
        for address in [
            "https://127.0.0.1:1234",
            "http://example.org:1234",
            "http://127.0.0.1:1234/path",
            "http://user@127.0.0.1:1234",
            "http://127.0.0.1:1234/?q=x",
        ] {
            assert!(validate_connection(&Connection {
                base_url: address.into(),
                token: "a".repeat(32)
            })
            .is_err());
        }
        assert!(validate_connection(&Connection {
            base_url: "http://127.0.0.1:1234".into(),
            token: "a".repeat(32)
        })
        .is_ok());
    }
    #[test]
    fn missing_runtime_is_actionable() {
        let directory = tempfile::tempdir().unwrap();
        assert!(packaged_executable(directory.path(), "1.0")
            .unwrap_err()
            .contains("Reinstall"));
    }
    #[test]
    fn architecture_aliases_match() {
        assert_eq!(normalized_arch("amd64"), normalized_arch("x86_64"));
        assert_eq!(normalized_arch("arm64"), normalized_arch("aarch64"));
    }

    #[test]
    fn rejects_wrong_runtime_version_before_launch() {
        let directory = tempfile::tempdir().unwrap();
        let resources = directory.path().join("_internal/desktop_resources");
        std::fs::create_dir_all(&resources).unwrap();
        std::fs::write(
            resources.join("runtime.json"),
            r#"{"version":"old","platform":"win32","machine":"amd64"}"#,
        )
        .unwrap();
        assert!(packaged_executable(directory.path(), "new")
            .unwrap_err()
            .contains("Runtime mismatch"));
    }
}
