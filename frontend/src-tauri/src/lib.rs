mod repository;
mod repository_files;

/// Starts the CodeFrog desktop shell: a window that hosts the React UI.
///
/// Native surface, deliberately small:
/// - the official dialog plugin, used only to open a folder picker (permission `dialog:allow-open`);
/// - `select_repository`, which validates a picked folder, reads Git metadata, and records it as
///   the selected repository;
/// - `list_repository_tree` / `read_repository_file` / `clear_selected_repository`, which browse
///   only that selected repository by relative path (see `repository_files`).
///
/// There is no filesystem plugin, no shell plugin, and no generic command execution.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(repository_files::SelectedRepository::default())
        .invoke_handler(tauri::generate_handler![
            repository::select_repository,
            repository_files::list_repository_tree,
            repository_files::read_repository_file,
            repository_files::clear_selected_repository
        ])
        .run(tauri::generate_context!())
        .expect("error while running the CodeFrog desktop application");
}
