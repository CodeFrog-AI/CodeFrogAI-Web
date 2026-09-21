//! Local repository selection: validate a directory the user picked and read its Git metadata.
//!
//! Read-only and shallow: it never walks the repository's files and never runs anything the
//! frontend supplies. The only input is a directory path; the only programs started are `git`
//! with a fixed set of read-only subcommands, as an argument list (no shell). Errors are
//! structured codes with fixed messages, never raw operating-system or Git output.

use serde::Serialize;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const MAX_PATH_LENGTH: usize = 4096;
const MAX_REMOTE_URL_LENGTH: usize = 2048;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    InvalidPath,
    NotAGitRepository,
    PermissionDenied,
    GitUnavailable,
    GitError,
}

#[derive(Debug, Serialize)]
pub struct SelectRepositoryError {
    pub code: ErrorCode,
    pub message: String,
}

impl SelectRepositoryError {
    fn new(code: ErrorCode, message: &str) -> Self {
        Self {
            code,
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RepositoryInfo {
    pub name: String,
    pub path: String,
    /// `None` when HEAD is detached.
    pub branch: Option<String>,
    pub is_dirty: bool,
    /// The `origin` remote with any embedded credentials removed, if there is one.
    pub remote_url: Option<String>,
}

/// Validates the selected directory and returns its Git metadata.
///
/// Runs on a blocking thread so a slow disk or a large repository cannot freeze the window.
#[tauri::command]
pub async fn select_repository(path: String) -> Result<RepositoryInfo, SelectRepositoryError> {
    match tauri::async_runtime::spawn_blocking(move || inspect_repository(&path)).await {
        Ok(result) => result,
        Err(_) => Err(SelectRepositoryError::new(
            ErrorCode::GitError,
            "The repository could not be read.",
        )),
    }
}

pub fn inspect_repository(raw_path: &str) -> Result<RepositoryInfo, SelectRepositoryError> {
    let path = validate_directory(raw_path)?;

    let inside = run_git(&path, &["rev-parse", "--is-inside-work-tree"])?;
    if !inside.status.success() || stdout_text(&inside) != "true" {
        return Err(if inside.status.success() {
            SelectRepositoryError::new(
                ErrorCode::NotAGitRepository,
                "This folder is not a Git working tree.",
            )
        } else {
            git_failure()
        });
    }

    let branch = run_git(&path, &["symbolic-ref", "--short", "-q", "HEAD"])?;
    let branch = if branch.status.success() {
        non_empty(stdout_text(&branch))
    } else {
        None // detached HEAD
    };

    let status = run_git(&path, &["status", "--porcelain=v1", "--untracked-files=normal"])?;
    if !status.status.success() {
        return Err(git_failure());
    }
    let is_dirty = !status.stdout.is_empty();

    // A repository without an `origin` remote is normal, so a failure here is not an error.
    let remote_url = match run_git(&path, &["remote", "get-url", "origin"]) {
        Ok(output) if output.status.success() => sanitize_remote_url(&stdout_text(&output)),
        _ => None,
    };

    Ok(RepositoryInfo {
        name: repository_name(&path),
        path: path.to_string_lossy().into_owned(),
        branch,
        is_dirty,
        remote_url,
    })
}

/// The path must be an absolute, readable directory that contains `.git` (a folder, or the
/// file a linked worktree uses).
fn validate_directory(raw_path: &str) -> Result<PathBuf, SelectRepositoryError> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() || trimmed.len() > MAX_PATH_LENGTH || trimmed.contains('\0') {
        return Err(invalid_path());
    }
    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err(invalid_path());
    }

    let metadata = fs::metadata(&path).map_err(map_io_error)?;
    if !metadata.is_dir() {
        return Err(invalid_path());
    }
    fs::read_dir(&path).map_err(map_io_error)?;

    match fs::metadata(path.join(".git")) {
        Ok(_) => Ok(path),
        Err(error) if error.kind() == ErrorKind::NotFound => Err(SelectRepositoryError::new(
            ErrorCode::NotAGitRepository,
            "This folder is not a Git repository (no .git found).",
        )),
        Err(error) => Err(map_io_error(error)),
    }
}

fn map_io_error(error: std::io::Error) -> SelectRepositoryError {
    if error.kind() == ErrorKind::PermissionDenied {
        SelectRepositoryError::new(
            ErrorCode::PermissionDenied,
            "CodeFrog does not have permission to read this folder.",
        )
    } else {
        invalid_path()
    }
}

fn invalid_path() -> SelectRepositoryError {
    SelectRepositoryError::new(
        ErrorCode::InvalidPath,
        "The selected path is not an accessible folder.",
    )
}

fn git_failure() -> SelectRepositoryError {
    SelectRepositoryError::new(
        ErrorCode::GitError,
        "Git could not read this repository. If the folder belongs to another user, Git may refuse to open it.",
    )
}

/// Runs one fixed, read-only `git` subcommand in `directory`. No shell, no caller-supplied
/// arguments, no prompts, no optional index writes, and no repository-configured fsmonitor.
fn run_git(directory: &Path, arguments: &[&str]) -> Result<Output, SelectRepositoryError> {
    let mut command = Command::new("git");
    command
        .arg("--no-optional-locks")
        .args(["-c", "core.fsmonitor=false"])
        .args(arguments)
        .current_dir(directory)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null());
    hide_console_window(&mut command);
    command.output().map_err(|_| {
        SelectRepositoryError::new(
            ErrorCode::GitUnavailable,
            "Git is not installed or could not be started.",
        )
    })
}

#[cfg(windows)]
fn hide_console_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console_window(_command: &mut Command) {}

fn stdout_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn non_empty(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn repository_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// Removes credentials from a URL such as `https://user:token@host/org/repo.git`. SCP-style
/// remotes (`git@host:org/repo.git`) carry no secret and are returned as they are.
pub fn sanitize_remote_url(raw: &str) -> Option<String> {
    let url = raw.trim();
    if url.is_empty() || url.len() > MAX_REMOTE_URL_LENGTH || url.chars().any(|c| c.is_control()) {
        return None;
    }
    if let Some((scheme, rest)) = url.split_once("://") {
        let (authority, tail) = match rest.find('/') {
            Some(index) => rest.split_at(index),
            None => (rest, ""),
        };
        let host = authority.rsplit('@').next().unwrap_or(authority);
        return Some(format!("{}://{}{}", scheme, host, tail));
    }
    Some(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_directory(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "codefrog-test-{}-{}-{}",
            label,
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn credentials_are_removed_from_remote_urls() {
        assert_eq!(
            sanitize_remote_url("https://user:token@github.com/org/repo.git").as_deref(),
            Some("https://github.com/org/repo.git")
        );
        assert_eq!(
            sanitize_remote_url("https://github.com/org/repo.git").as_deref(),
            Some("https://github.com/org/repo.git")
        );
        assert_eq!(
            sanitize_remote_url("git@github.com:org/repo.git").as_deref(),
            Some("git@github.com:org/repo.git")
        );
        assert_eq!(sanitize_remote_url("   "), None);
    }

    #[test]
    fn invalid_paths_are_rejected() {
        for path in ["", "   ", "relative/dir", "bad\0path"] {
            let error = validate_directory(path).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidPath, "{:?}", path);
        }
        let missing = unique_directory("missing").join("does-not-exist");
        assert_eq!(
            validate_directory(&missing.to_string_lossy()).unwrap_err().code,
            ErrorCode::InvalidPath
        );
    }

    #[test]
    fn a_file_is_not_a_repository() {
        let directory = unique_directory("file");
        let file = directory.join("a.txt");
        fs::write(&file, "x").unwrap();
        assert_eq!(
            validate_directory(&file.to_string_lossy()).unwrap_err().code,
            ErrorCode::InvalidPath
        );
    }

    #[test]
    fn a_directory_without_dot_git_is_not_a_repository() {
        let directory = unique_directory("plain");
        assert_eq!(
            validate_directory(&directory.to_string_lossy()).unwrap_err().code,
            ErrorCode::NotAGitRepository
        );
    }

    #[test]
    fn errors_serialize_to_structured_codes() {
        let error = SelectRepositoryError::new(ErrorCode::NotAGitRepository, "x");
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"code\":\"NOT_A_GIT_REPOSITORY\""));
    }
}
