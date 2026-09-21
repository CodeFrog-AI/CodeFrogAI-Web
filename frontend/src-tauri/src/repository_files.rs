//! Repository file browsing: list a repository's file tree and read one text file.
//!
//! Security boundary
//! -----------------
//! The frontend never supplies a root or an absolute path. The root is the repository the user
//! picked (recorded by `select_repository` in `SelectedRepository`); the frontend may only ask
//! for a repository-*relative* path, which is validated segment by segment and then checked
//! again after resolving the real location on disk:
//!
//! - empty paths, absolute paths, backslashes, `.`/`..`/empty segments, control characters
//!   (including NUL), and `:` (Windows drives and streams) are rejected;
//! - any segment naming an ignored directory (`.git`, `node_modules`, ...) is refused;
//! - the canonical (symlink-resolved) path must still be inside the canonical root, so a
//!   symlink that points outside the repository cannot be read through;
//! - the tree never lists or follows symlinks at all.
//!
//! Nothing here starts a process, writes a file, or logs file names or contents.

use serde::Serialize;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::State;

/// Deepest directory level that is listed (top-level folders are level 0). Deeper folders are
/// shown empty and the tree is marked `truncated`.
pub const MAX_DEPTH: usize = 12;
/// Most entries (files and folders) one tree may contain. Listing stops there, deterministically.
pub const MAX_ENTRIES: usize = 5_000;
/// Largest file `read_repository_file` will return: 1 MB of text.
pub const MAX_FILE_BYTES: u64 = 1_000_000;
const MAX_RELATIVE_PATH_LENGTH: usize = 1_024;

/// Names skipped in the tree (at any depth, case-insensitively) and refused when read.
pub const IGNORED_NAMES: [&str; 9] = [
    ".git",
    "node_modules",
    ".next",
    "dist",
    "build",
    "coverage",
    "target",
    ".venv",
    "__pycache__",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_depth: usize,
    pub max_entries: usize,
    pub max_file_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: MAX_DEPTH,
            max_entries: MAX_ENTRIES,
            max_file_bytes: MAX_FILE_BYTES,
        }
    }
}

// ------------------------------------------------------------------ errors and results

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FilesErrorCode {
    NoRepositorySelected,
    InvalidPath,
    PathNotAllowed,
    PathOutsideRepository,
    NotFound,
    NotAFile,
    FileTooLarge,
    NotUtf8Text,
    PermissionDenied,
    IoError,
}

#[derive(Debug, Serialize)]
pub struct FilesError {
    pub code: FilesErrorCode,
    pub message: String,
}

impl FilesError {
    fn new(code: FilesErrorCode, message: &str) -> Self {
        Self {
            code,
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    File,
    Directory,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct TreeNode {
    pub name: String,
    /// Repository-relative, `/`-separated.
    pub path: String,
    #[serde(rename = "type")]
    pub kind: NodeKind,
    /// Files only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Directories only (empty when the folder is empty, unreadable, or beyond the depth limit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<TreeNode>>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RepositoryTree {
    pub entries: Vec<TreeNode>,
    /// True when the entry or depth limit cut the listing short.
    pub truncated: bool,
    pub entry_count: usize,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct RepositoryFile {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub contents: String,
}

// ------------------------------------------------------------------ the selected repository

/// The root of the repository the user selected (canonical), held only in memory.
#[derive(Default)]
pub struct SelectedRepository(Mutex<Option<PathBuf>>);

impl SelectedRepository {
    pub fn set(&self, root: PathBuf) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = Some(root);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = None;
        }
    }

    fn get(&self) -> Result<PathBuf, FilesError> {
        self.0
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .ok_or_else(|| {
                FilesError::new(
                    FilesErrorCode::NoRepositorySelected,
                    "No repository is selected.",
                )
            })
    }
}

// ------------------------------------------------------------------ commands

/// Lists the selected repository's files. Takes no path: the root is the selected repository.
#[tauri::command]
pub async fn list_repository_tree(
    selection: State<'_, SelectedRepository>,
) -> Result<RepositoryTree, FilesError> {
    let root = selection.get()?;
    run_blocking(move || build_tree(&root, Limits::default())).await
}

/// Reads one text file of the selected repository by its repository-relative path.
#[tauri::command]
pub async fn read_repository_file(
    path: String,
    selection: State<'_, SelectedRepository>,
) -> Result<RepositoryFile, FilesError> {
    let root = selection.get()?;
    run_blocking(move || read_file(&root, &path, Limits::default())).await
}

/// Forgets the selected repository, so no file can be listed or read until one is selected again.
#[tauri::command]
pub fn clear_selected_repository(selection: State<'_, SelectedRepository>) {
    selection.clear();
}

async fn run_blocking<T, F>(job: F) -> Result<T, FilesError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, FilesError> + Send + 'static,
{
    match tauri::async_runtime::spawn_blocking(job).await {
        Ok(result) => result,
        Err(_) => Err(FilesError::new(
            FilesErrorCode::IoError,
            "The repository could not be read.",
        )),
    }
}

// ------------------------------------------------------------------ path validation

fn is_ignored(name: &str) -> bool {
    IGNORED_NAMES
        .iter()
        .any(|ignored| ignored.eq_ignore_ascii_case(name))
}

/// Validates a repository-relative path and returns its segments. The path must use `/` and
/// cannot climb out of the repository, be absolute, or name an ignored directory.
pub fn validate_relative_path(raw: &str) -> Result<Vec<&str>, FilesError> {
    let invalid = || {
        FilesError::new(
            FilesErrorCode::InvalidPath,
            "The path must be a relative path inside the repository.",
        )
    };
    if raw.is_empty()
        || raw.len() > MAX_RELATIVE_PATH_LENGTH
        || raw.chars().any(|character| character.is_control())
        || raw.starts_with('/')
        || raw.contains('\\')
    {
        return Err(invalid());
    }
    let segments: Vec<&str> = raw.split('/').collect();
    for segment in &segments {
        if segment.is_empty()
            || *segment == "."
            || *segment == ".."
            || segment.contains(':')
            || segment.ends_with('.')
            || segment.ends_with(' ')
        {
            return Err(invalid());
        }
        if is_ignored(segment) {
            return Err(FilesError::new(
                FilesErrorCode::PathNotAllowed,
                "This path is not available for browsing.",
            ));
        }
    }
    Ok(segments)
}

fn map_io_error(error: std::io::Error) -> FilesError {
    match error.kind() {
        ErrorKind::NotFound => FilesError::new(FilesErrorCode::NotFound, "The file was not found."),
        ErrorKind::PermissionDenied => FilesError::new(
            FilesErrorCode::PermissionDenied,
            "CodeFrog does not have permission to read this.",
        ),
        _ => FilesError::new(FilesErrorCode::IoError, "The repository could not be read."),
    }
}

// ------------------------------------------------------------------ the tree

/// Builds the tree below `root` (which must already be canonical). Directories come before
/// files, each group sorted case-insensitively by name, so the result is deterministic.
/// Symlinks are never listed or followed.
pub fn build_tree(root: &Path, limits: Limits) -> Result<RepositoryTree, FilesError> {
    let mut walker = Walker {
        limits,
        count: 0,
        truncated: false,
    };
    let entries = walker.read_directory(root, "", 0).map_err(map_io_error)?;
    Ok(RepositoryTree {
        entries,
        truncated: walker.truncated,
        entry_count: walker.count,
    })
}

struct Walker {
    limits: Limits,
    count: usize,
    truncated: bool,
}

impl Walker {
    fn read_directory(
        &mut self,
        directory: &Path,
        prefix: &str,
        depth: usize,
    ) -> std::io::Result<Vec<TreeNode>> {
        // (name, is_directory, size)
        let mut items: Vec<(String, bool, u64)> = Vec::new();
        for entry in fs::read_dir(directory)? {
            let Ok(entry) = entry else { continue };
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue; // names that are not valid Unicode cannot be addressed safely
            };
            if is_ignored(&name) {
                continue;
            }
            let Ok(file_type) = entry.file_type() else { continue };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                items.push((name, true, 0));
            } else if file_type.is_file() {
                let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
                items.push((name, false, size));
            }
        }
        items.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase()))
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut nodes = Vec::new();
        for (name, is_directory, size) in items {
            if self.count >= self.limits.max_entries {
                self.truncated = true;
                break;
            }
            self.count += 1;
            let path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}/{}", prefix, name)
            };
            if is_directory {
                let children = if depth < self.limits.max_depth {
                    // A folder that cannot be read is shown empty instead of failing the whole tree.
                    self.read_directory(&directory.join(&name), &path, depth + 1)
                        .unwrap_or_default()
                } else {
                    self.truncated = true;
                    Vec::new()
                };
                nodes.push(TreeNode {
                    name,
                    path,
                    kind: NodeKind::Directory,
                    size: None,
                    children: Some(children),
                });
            } else {
                nodes.push(TreeNode {
                    name,
                    path,
                    kind: NodeKind::File,
                    size: Some(size),
                    children: None,
                });
            }
        }
        Ok(nodes)
    }
}

// ------------------------------------------------------------------ reading a file

/// Reads a UTF-8 text file below `root` (which must already be canonical).
pub fn read_file(root: &Path, raw_path: &str, limits: Limits) -> Result<RepositoryFile, FilesError> {
    let segments = validate_relative_path(raw_path)?;

    let mut candidate = root.to_path_buf();
    for segment in &segments {
        candidate.push(segment);
    }
    // Resolve symlinks, then require the real location to still be inside the repository.
    let canonical = fs::canonicalize(&candidate).map_err(map_io_error)?;
    let relative = canonical.strip_prefix(root).map_err(|_| {
        FilesError::new(
            FilesErrorCode::PathOutsideRepository,
            "The path resolves outside the repository.",
        )
    })?;
    // A symlink inside the repository must not lead into an ignored directory either (.git!).
    if relative
        .components()
        .any(|component| component.as_os_str().to_str().map_or(true, is_ignored))
    {
        return Err(FilesError::new(
            FilesErrorCode::PathNotAllowed,
            "This path is not available for browsing.",
        ));
    }

    let metadata = fs::metadata(&canonical).map_err(map_io_error)?;
    if !metadata.is_file() {
        return Err(FilesError::new(
            FilesErrorCode::NotAFile,
            "The path is not a file.",
        ));
    }
    if metadata.len() > limits.max_file_bytes {
        return Err(too_large());
    }

    // Read at most one byte over the limit, in case the file grew after the check above.
    let mut bytes = Vec::new();
    File::open(&canonical)
        .map_err(map_io_error)?
        .take(limits.max_file_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(map_io_error)?;
    if bytes.len() as u64 > limits.max_file_bytes {
        return Err(too_large());
    }
    if bytes.contains(&0) {
        return Err(not_text());
    }
    let contents = String::from_utf8(bytes).map_err(|_| not_text())?;

    Ok(RepositoryFile {
        path: segments.join("/"),
        name: segments.last().map(|name| name.to_string()).unwrap_or_default(),
        size: contents.len() as u64,
        contents,
    })
}

fn too_large() -> FilesError {
    FilesError::new(
        FilesErrorCode::FileTooLarge,
        "The file is too large to display.",
    )
}

fn not_text() -> FilesError {
    FilesError::new(
        FilesErrorCode::NotUtf8Text,
        "The file is not UTF-8 text (it may be binary).",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, canonical directory under the system temp folder.
    fn temp_root(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "codefrog-files-{}-{}-{}",
            label,
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).unwrap();
        fs::canonicalize(path).unwrap()
    }

    fn write(root: &Path, relative: &str, contents: &[u8]) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn names(nodes: &[TreeNode]) -> Vec<&str> {
        nodes.iter().map(|node| node.name.as_str()).collect()
    }

    fn code<T: std::fmt::Debug>(result: Result<T, FilesError>) -> FilesErrorCode {
        result.unwrap_err().code
    }

    #[cfg(unix)]
    fn make_symlink(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).is_ok()
    }

    #[cfg(windows)]
    fn make_symlink(target: &Path, link: &Path) -> bool {
        // Needs Developer Mode or elevation; the test is skipped when it is not available.
        if target.is_dir() {
            std::os::windows::fs::symlink_dir(target, link).is_ok()
        } else {
            std::os::windows::fs::symlink_file(target, link).is_ok()
        }
    }

    // ---- boundary validation

    #[test]
    fn traversal_and_absolute_paths_are_rejected() {
        for path in [
            "",
            "../../.env",
            "..",
            "a/../b",
            "a/./b",
            "a//b",
            "/etc/passwd",
            "C:/Windows/win.ini",
            "C:win.ini",
            "a\\b",
            "..\\..\\.env",
            "a/b/",
            "file.txt:stream",
            "bad\0name",
            "bell\u{7}",
            "trailing.",
            "trailing ",
        ] {
            assert_eq!(
                code(validate_relative_path(path)),
                FilesErrorCode::InvalidPath,
                "{:?}",
                path
            );
        }
    }

    #[test]
    fn ignored_directories_cannot_be_addressed() {
        for path in [".git/config", "node_modules/x/index.js", "a/target/b", "DIST/x", ".next/x"] {
            assert_eq!(
                code(validate_relative_path(path)),
                FilesErrorCode::PathNotAllowed,
                "{:?}",
                path
            );
        }
    }

    #[test]
    fn a_normal_relative_path_is_accepted() {
        assert_eq!(
            validate_relative_path("src/app/page.tsx").unwrap(),
            vec!["src", "app", "page.tsx"]
        );
    }

    #[test]
    fn requests_that_escape_the_repository_are_rejected_when_reading() {
        let root = temp_root("escape");
        write(&root, "ok.txt", b"ok");
        for path in ["../../.env", "../x", "/etc/passwd", "C:/x", "..\\x"] {
            assert_eq!(
                code(read_file(&root, path, Limits::default())),
                FilesErrorCode::InvalidPath,
                "{:?}",
                path
            );
        }
    }

    // ---- the tree

    #[test]
    fn ignored_directories_and_files_are_left_out() {
        let root = temp_root("ignored");
        for name in IGNORED_NAMES {
            write(&root, &format!("{}/inside.txt", name), b"x");
        }
        write(&root, "src/keep.rs", b"x");
        write(&root, "src/node_modules/deep.js", b"x");
        write(&root, "README.md", b"x");

        let tree = build_tree(&root, Limits::default()).unwrap();

        assert_eq!(names(&tree.entries), vec!["src", "README.md"]);
        let src = tree.entries[0].children.as_ref().unwrap();
        assert_eq!(names(src), vec!["keep.rs"]);
        assert!(!tree.truncated);
    }

    #[test]
    fn directories_come_first_then_files_sorted_case_insensitively_and_deterministically() {
        let root = temp_root("sorted");
        for name in ["b.txt", "A.txt", "a_lower.txt", "Zeta.md", "m.md"] {
            write(&root, name, b"x");
        }
        for name in ["zdir", "Adir", "bdir"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        write(&root, "Adir/z.txt", b"x");
        write(&root, "Adir/B.txt", b"x");

        let first = build_tree(&root, Limits::default()).unwrap();
        let second = build_tree(&root, Limits::default()).unwrap();

        assert_eq!(
            names(&first.entries),
            vec!["Adir", "bdir", "zdir", "A.txt", "a_lower.txt", "b.txt", "m.md", "Zeta.md"]
        );
        assert_eq!(names(first.entries[0].children.as_ref().unwrap()), vec!["B.txt", "z.txt"]);
        assert_eq!(first, second);
    }

    #[test]
    fn nodes_carry_relative_paths_types_and_sizes() {
        let root = temp_root("shape");
        write(&root, "src/app/page.tsx", b"12345");

        let tree = build_tree(&root, Limits::default()).unwrap();

        let src = &tree.entries[0];
        assert_eq!((src.kind, src.path.as_str(), src.size), (NodeKind::Directory, "src", None));
        let page = &src.children.as_ref().unwrap()[0].children.as_ref().unwrap()[0];
        assert_eq!(
            (page.kind, page.path.as_str(), page.size, page.children.is_none()),
            (NodeKind::File, "src/app/page.tsx", Some(5), true)
        );
        assert_eq!(tree.entry_count, 3);
    }

    #[test]
    fn the_entry_limit_stops_the_listing_deterministically() {
        let root = temp_root("entries");
        for index in 0..20 {
            write(&root, &format!("file{:02}.txt", index), b"x");
        }
        let limits = Limits { max_entries: 5, ..Limits::default() };

        let tree = build_tree(&root, limits).unwrap();

        assert_eq!(tree.entry_count, 5);
        assert!(tree.truncated);
        assert_eq!(
            names(&tree.entries),
            vec!["file00.txt", "file01.txt", "file02.txt", "file03.txt", "file04.txt"]
        );
    }

    #[test]
    fn the_depth_limit_shows_deeper_folders_empty() {
        let root = temp_root("depth");
        write(&root, "a/b/c/deep.txt", b"x");
        let limits = Limits { max_depth: 1, ..Limits::default() };

        let tree = build_tree(&root, limits).unwrap();

        let a = &tree.entries[0];
        let b = &a.children.as_ref().unwrap()[0];
        assert_eq!(b.name, "b");
        assert!(b.children.as_ref().unwrap().is_empty());
        assert!(tree.truncated);
    }

    #[test]
    fn symlinks_are_never_listed() {
        let root = temp_root("tree-links");
        let outside = temp_root("tree-outside");
        write(&outside, "secret.txt", b"secret");
        write(&root, "real.txt", b"x");
        if !make_symlink(&outside.join("secret.txt"), &root.join("link.txt"))
            || !make_symlink(&outside, &root.join("linked-dir"))
        {
            return; // symlinks are not available here
        }

        let tree = build_tree(&root, Limits::default()).unwrap();

        assert_eq!(names(&tree.entries), vec!["real.txt"]);
    }

    #[test]
    fn a_missing_root_is_an_error() {
        let root = temp_root("gone");
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(code(build_tree(&root, Limits::default())), FilesErrorCode::NotFound);
    }

    // ---- reading files

    #[test]
    fn a_text_file_is_read() {
        let root = temp_root("read");
        write(&root, "src/main.rs", "fn main() {}\n  indented\n".as_bytes());

        let file = read_file(&root, "src/main.rs", Limits::default()).unwrap();

        assert_eq!(file.path, "src/main.rs");
        assert_eq!(file.name, "main.rs");
        assert_eq!(file.contents, "fn main() {}\n  indented\n");
        assert_eq!(file.size, file.contents.len() as u64);
    }

    #[test]
    fn a_missing_file_is_not_found() {
        let root = temp_root("missing");
        assert_eq!(code(read_file(&root, "nope.txt", Limits::default())), FilesErrorCode::NotFound);
    }

    #[test]
    fn a_directory_is_not_a_file() {
        let root = temp_root("dir");
        fs::create_dir_all(root.join("src")).unwrap();
        assert_eq!(code(read_file(&root, "src", Limits::default())), FilesErrorCode::NotAFile);
    }

    #[test]
    fn an_oversized_file_is_refused() {
        let root = temp_root("big");
        write(&root, "big.txt", &[b'x'; 11]);
        let limits = Limits { max_file_bytes: 10, ..Limits::default() };
        assert_eq!(code(read_file(&root, "big.txt", limits)), FilesErrorCode::FileTooLarge);
        write(&root, "ok.txt", &[b'x'; 10]);
        assert!(read_file(&root, "ok.txt", limits).is_ok());
    }

    #[test]
    fn invalid_utf8_and_binary_files_are_refused() {
        let root = temp_root("binary");
        write(&root, "latin1.txt", &[0xff, 0xfe, 0x41]);
        write(&root, "nul.bin", b"ab\0cd");
        assert_eq!(code(read_file(&root, "latin1.txt", Limits::default())), FilesErrorCode::NotUtf8Text);
        assert_eq!(code(read_file(&root, "nul.bin", Limits::default())), FilesErrorCode::NotUtf8Text);
    }

    #[test]
    fn a_symlink_that_escapes_the_repository_cannot_be_read() {
        let root = temp_root("read-links");
        let outside = temp_root("read-outside");
        write(&outside, "secret.txt", b"top secret");
        write(&outside, "dir/inner.txt", b"inner");
        if !make_symlink(&outside.join("secret.txt"), &root.join("leak.txt"))
            || !make_symlink(&outside.join("dir"), &root.join("leakdir"))
        {
            return; // symlinks are not available here
        }

        assert_eq!(code(read_file(&root, "leak.txt", Limits::default())), FilesErrorCode::PathOutsideRepository);
        assert_eq!(code(read_file(&root, "leakdir/inner.txt", Limits::default())), FilesErrorCode::PathOutsideRepository);
    }

    #[test]
    fn a_symlink_into_git_metadata_cannot_be_read() {
        let root = temp_root("git-link");
        write(&root, ".git/config", b"[remote \"origin\"] url = https://user:token@example.com/r.git");
        if !make_symlink(&root.join(".git/config"), &root.join("innocent.txt")) {
            return; // symlinks are not available here
        }
        assert_eq!(code(read_file(&root, "innocent.txt", Limits::default())), FilesErrorCode::PathNotAllowed);
    }

    #[test]
    fn errors_serialize_to_structured_codes() {
        let error = FilesError::new(FilesErrorCode::PathOutsideRepository, "x");
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"code\":\"PATH_OUTSIDE_REPOSITORY\""));
    }

    #[test]
    fn the_tree_serializes_with_a_type_field_and_only_the_relevant_members() {
        let root = temp_root("json");
        write(&root, "d/f.txt", b"abc");
        let tree = build_tree(&root, Limits::default()).unwrap();
        let json = serde_json::to_value(&tree).unwrap();
        assert_eq!(json["entries"][0]["type"], "directory");
        assert!(json["entries"][0].get("size").is_none());
        assert_eq!(json["entries"][0]["children"][0]["type"], "file");
        assert_eq!(json["entries"][0]["children"][0]["size"], 3);
        assert!(json["entries"][0]["children"][0].get("children").is_none());
        assert_eq!(json["entryCount"], 2);
    }

    // ---- the selected repository

    #[test]
    fn nothing_can_be_read_before_a_repository_is_selected_or_after_it_is_cleared() {
        let selection = SelectedRepository::default();
        assert_eq!(code(selection.get()), FilesErrorCode::NoRepositorySelected);
        selection.set(temp_root("selected"));
        assert!(selection.get().is_ok());
        selection.clear();
        assert_eq!(code(selection.get()), FilesErrorCode::NoRepositorySelected);
    }
}
