#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod preview;
mod runtime;
use runtime::Connection;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

struct Bridge {
    connection: Connection,
    child: Mutex<Child>,
}

#[tauri::command]
fn connection(bridge: tauri::State<'_, Bridge>) -> Connection {
    bridge.connection.clone()
}

#[tauri::command]
async fn save_download(
    name: String,
    path: String,
    bridge: tauri::State<'_, Bridge>,
) -> Result<bool, String> {
    let connection = bridge.connection.clone();
    if !path.starts_with("/projects/") || path.contains("..") || path.contains('#') {
        return Err("Invalid export path".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        match rfd::FileDialog::new().set_file_name(&name).save_file() {
            Some(destination) => {
                let parent = destination.parent().ok_or("Invalid destination")?;
                let mut temporary =
                    tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
                let client = reqwest::blocking::Client::builder()
                    .no_proxy()
                    .redirect(reqwest::redirect::Policy::none())
                    .timeout(Duration::from_secs(600))
                    .build()
                    .map_err(|e| e.to_string())?;
                let mut response = client
                    .get(format!("{}/api{}", connection.base_url, path))
                    .bearer_auth(connection.token)
                    .send()
                    .map_err(|e| e.to_string())?
                    .error_for_status()
                    .map_err(|e| e.to_string())?;
                std::io::copy(&mut response, &mut temporary).map_err(|e| e.to_string())?;
                temporary.as_file().sync_all().map_err(|e| e.to_string())?;
                temporary.persist(destination).map_err(|e| e.to_string())?;
                Ok(true)
            }
            None => Ok(false),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

fn start_backend(app: &tauri::App) -> Result<Bridge, Box<dyn std::error::Error>> {
    let workspace = app.path().app_local_data_dir()?.join("workspace");
    std::fs::create_dir_all(&workspace)?;
    let mut command;
    if cfg!(debug_assertions) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        command =
            Command::new(std::env::var("NLP_SUITE_PYTHON").unwrap_or_else(|_| "python".into()));
        command
            .current_dir(root)
            .args(["-m", "desktop_backend.server"]);
    } else {
        command = Command::new(runtime::packaged_executable(
            &app.path().resource_dir()?.join("nlp-runtime"),
            env!("CARGO_PKG_VERSION"),
        )?);
    }
    let (connection, child) = runtime::start(command, &workspace)?;
    Ok(Bridge {
        connection,
        child: Mutex::new(child),
    })
}

/// Stop the engine before an update replaces its files.
///
/// On Windows the installer cannot overwrite a runtime that is still running,
/// and the engine only exits once accepted jobs finish. Ask it to stop (EOF on
/// stdin), give it 30 seconds, then end it: an update the reader asked for
/// should not wait on a long analysis indefinitely.
#[tauri::command]
async fn stop_engine(app: tauri::AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let bridge = app.state::<Bridge>();
        let mut child = bridge
            .child
            .lock()
            .map_err(|_| "Engine state unavailable".to_string())?;
        child.stdin.take();
        for _ in 0..300 {
            if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

fn main() {
    // Headless installed-payload validation for CI, without opening a webview.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--runtime-check") {
        let result = (|| -> Result<(), String> {
            if args.len() != 4 {
                return Err("Usage: --runtime-check RUNTIME_DIRECTORY TEST_WORKSPACE".into());
            }
            let executable = runtime::packaged_executable(
                std::path::Path::new(&args[2]),
                env!("CARGO_PKG_VERSION"),
            )?;
            let (_, mut child) =
                runtime::start(Command::new(executable), std::path::Path::new(&args[3]))?;
            child.stdin.take();
            for _ in 0..300 {
                if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                    return if status.success() {
                        Ok(())
                    } else {
                        Err("Engine shutdown failed".into())
                    };
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = child.kill();
            let _ = child.wait();
            Err("Engine shutdown timed out".into())
        })();
        match result {
            Ok(()) => {
                println!("Runtime check passed");
                return;
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
    }
    let result = tauri::Builder::default()
        .register_asynchronous_uri_scheme_protocol("nlp-viz", |ctx, request, responder| {
            let connection = ctx
                .app_handle()
                .try_state::<Bridge>()
                .map(|bridge| bridge.connection.clone());
            tauri::async_runtime::spawn_blocking(move || {
                responder.respond(preview::serve(connection, request))
            });
        })
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            app.manage(start_backend(app)?);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            connection,
            save_download,
            stop_engine
        ])
        .build(tauri::generate_context!());
    match result {
        Ok(app) => app.run(|handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(bridge) = handle.try_state::<Bridge>() {
                    if let Ok(mut child) = bridge.child.lock() {
                        // EOF requests graceful shutdown. Accepted jobs finish
                        // in the local backend before it releases the workspace.
                        child.stdin.take();
                    }
                }
            }
        }),
        Err(error) => {
            rfd::MessageDialog::new()
                .set_title("NLP Suite could not start")
                .set_description(error.to_string())
                .set_level(rfd::MessageLevel::Error)
                .show();
        }
    }
}
