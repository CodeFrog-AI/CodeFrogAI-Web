/// Starts the CodeFrog desktop shell: a window that hosts the React UI. It registers no
/// commands and no plugins, so the UI has no native filesystem, shell, or Git access.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running the CodeFrog desktop application");
}
