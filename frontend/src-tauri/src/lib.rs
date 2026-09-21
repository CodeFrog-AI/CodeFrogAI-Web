mod repository;

/// Starts the CodeFrog desktop shell: a window that hosts the React UI.
///
/// Native surface, deliberately small:
/// - the official dialog plugin, used only to open a folder picker (permission `dialog:allow-open`);
/// - one command, `select_repository`, which validates a picked folder and reads Git metadata.
///
/// There is no filesystem plugin, no shell plugin, and no generic command execution.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![repository::select_repository])
        .run(tauri::generate_context!())
        .expect("error while running the CodeFrog desktop application");
}
