//! An artifact-only protocol. Never forwards the backend token or its CSP to a frame.
use crate::runtime::Connection;
use std::io::Read;
use std::time::Duration;
use tauri::http::{Request, Response};

const LIMIT: u64 = 32 * 1024 * 1024;
const POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; frame-src 'none'; worker-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; sandbox allow-scripts; frame-ancestors http://tauri.localhost https://tauri.localhost tauri://localhost http://127.0.0.1:1420 http://localhost:1420";

pub fn valid_path(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    let uuid = |v: &str| v.len() == 32 && v.bytes().all(|c| c.is_ascii_hexdigit());
    parts.len() == 7
        && parts[0].is_empty()
        && parts[1] == "projects"
        && uuid(parts[2])
        && parts[3] == "jobs"
        && uuid(parts[4])
        && parts[5] == "artifacts"
        && !parts[6].is_empty()
        && parts[6].len() <= 9
        && parts[6].bytes().all(|c| c.is_ascii_digit())
}

fn response(status: u16, mime: &str, body: Vec<u8>) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header("Content-Type", mime)
        .header("Content-Security-Policy", POLICY)
        .header("X-Content-Type-Options", "nosniff")
        .header("Referrer-Policy", "no-referrer")
        .header("Cache-Control", "no-store")
        .body(body)
        .expect("fixed preview headers")
}

pub fn serve(connection: Option<Connection>, request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    if request.method() != "GET"
        || request.uri().query().is_some()
        || !valid_path(request.uri().path())
    {
        return response(
            400,
            "text/plain",
            b"Invalid artifact preview request".to_vec(),
        );
    }
    let result = (|| -> Result<(String, Vec<u8>), String> {
        let connection = connection.ok_or("Engine is not ready")?;
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|_| "Cannot connect to local engine")?;
        let upstream = client
            .get(format!(
                "{}/api{}",
                connection.base_url,
                request.uri().path()
            ))
            .bearer_auth(connection.token)
            .send()
            .map_err(|_| "Cannot load artifact")?
            .error_for_status()
            .map_err(|_| "Artifact unavailable; reopen its run")?;
        let mime = upstream
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !matches!(
            mime.as_str(),
            "text/html"
                | "text/plain"
                | "image/svg+xml"
                | "application/pdf"
                | "application/vnd.google-earth.kml+xml"
        ) {
            return Err("This artifact is download-only".into());
        }
        let mut body = Vec::new();
        upstream
            .take(LIMIT + 1)
            .read_to_end(&mut body)
            .map_err(|_| "Cannot read artifact")?;
        if body.len() as u64 > LIMIT {
            return Err("Preview exceeds 32 MiB; use Download".into());
        }
        Ok((mime, body))
    })();
    match result {
        Ok((mime, body)) => response(200, &mime, body),
        Err(message) => response(400, "text/plain; charset=utf-8", message.into_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn proxies_only_the_artifact_and_replaces_upstream_policy() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let path = format!(
            "/projects/{}/jobs/{}/artifacts/0",
            "a".repeat(32),
            "b".repeat(32)
        );
        let expected_path = path.clone();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut headers = String::new();
            loop {
                let mut line = String::new();
                assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
                headers.push_str(&line);
            }
            assert!(headers.starts_with(&format!("GET /api{expected_path} HTTP/1.1\r\n")));
            assert!(headers
                .to_ascii_lowercase()
                .contains("authorization: bearer private-test-token\r\n"));
            let body = "<html><script>document.title='chart'</script></html>";
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Security-Policy: script-src 'none'\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        });
        let result = serve(
            Some(Connection {
                base_url,
                token: "private-test-token".into(),
            }),
            Request::builder()
                .uri(format!("http://nlp-viz.localhost{path}"))
                .body(vec![])
                .unwrap(),
        );
        server.join().unwrap();
        assert_eq!(result.status(), 200);
        assert_eq!(result.headers()["Content-Security-Policy"], POLICY);
        assert!(!String::from_utf8_lossy(result.body()).contains("private-test-token"));
        assert!(String::from_utf8_lossy(result.body()).contains("document.title='chart'"));
    }
    #[test]
    fn only_exact_artifact_routes_are_allowed() {
        let path = format!(
            "/projects/{}/jobs/{}/artifacts/0",
            "a".repeat(32),
            "b".repeat(32)
        );
        assert!(valid_path(&path));
        for bad in [
            "/health",
            "/projects/../jobs/x/artifacts/0",
            "/projects/%2e%2e/jobs/x/artifacts/0",
        ] {
            assert!(!valid_path(bad));
        }
        assert!(!valid_path(&(path.clone() + "/extra")));
        let request = Request::builder()
            .uri(path + "?download=true")
            .body(vec![])
            .unwrap();
        assert_eq!(serve(None, request).status(), 400);
    }
    #[test]
    fn errors_are_sandboxed_and_offline_too() {
        let result = serve(
            None,
            Request::builder().uri("/health").body(vec![]).unwrap(),
        );
        assert!(result.headers()["Content-Security-Policy"]
            .to_str()
            .unwrap()
            .contains("connect-src 'none'"));
        assert!(POLICY.contains("sandbox allow-scripts"));
        assert!(!POLICY.contains("allow-same-origin"));
    }
}
