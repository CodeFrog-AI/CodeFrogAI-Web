"""Parse dependency manifests into small summaries that never retain file contents.

To support another manifest type, write a `_parse_*` function that returns a
`ManifestInfo` and register its filename in `parser_for`.
"""

import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

NPM = "npm"
PYPI = "pypi"
NPM_ENTRY_SCRIPTS = ("dev", "serve", "start")

_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_JS_PACKAGE_MANAGERS = {"npm", "yarn", "pnpm", "bun"}


class ManifestParseError(ValueError):
    """The manifest could not be parsed; the message never includes file content."""


@dataclass(frozen=True)
class Dependency:
    name: str
    ecosystem: str
    dev: bool


@dataclass(frozen=True)
class EntryPoint:
    kind: str
    name: str
    path: str


@dataclass
class ManifestInfo:
    path: str
    dependencies: list[Dependency] = field(default_factory=list)
    package_managers: set[str] = field(default_factory=set)
    entry_points: list[EntryPoint] = field(default_factory=list)
    frameworks: set[str] = field(default_factory=set)
    # Structural hints: workspaces, bin, scripts, library_fields, build_system.
    flags: set[str] = field(default_factory=set)


def _normalize_python_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str | None:
    """Extract the package name from a requirement line or PEP 508 string."""

    line = requirement.split("#", 1)[0].strip()
    if not line or line.startswith(("-", ".", "/", "git+", "http://", "https://")):
        return None
    match = _REQUIREMENT_NAME.match(line)
    return _normalize_python_name(match.group(1)) if match else None


def _parse_package_json(path: str, text: str) -> ManifestInfo:
    try:
        data = json.loads(text)
    except ValueError:
        raise ManifestParseError("Invalid JSON") from None
    if not isinstance(data, dict):
        raise ManifestParseError("Unexpected JSON structure")

    info = ManifestInfo(path=path)
    for section, dev in (
        ("dependencies", False),
        ("peerDependencies", False),
        ("optionalDependencies", False),
        ("devDependencies", True),
    ):
        names = data.get(section)
        if isinstance(names, dict):
            info.dependencies += [Dependency(name, NPM, dev) for name in names if isinstance(name, str)]

    scripts = data.get("scripts")
    if isinstance(scripts, dict):
        info.entry_points += [
            EntryPoint("npm_script", name, path) for name in NPM_ENTRY_SCRIPTS if name in scripts
        ]
    bin_field = data.get("bin")
    if isinstance(bin_field, dict):
        info.entry_points += [EntryPoint("npm_bin", name, path) for name in bin_field if isinstance(name, str)]
    elif isinstance(bin_field, str) and isinstance(data.get("name"), str):
        info.entry_points.append(EntryPoint("npm_bin", data["name"], path))
    if bin_field:
        info.flags.add("bin")
    if isinstance(data.get("main"), str):
        info.entry_points.append(EntryPoint("npm_main", data["main"], path))
    if any(key in data for key in ("main", "module", "exports", "types")):
        info.flags.add("library_fields")
    if "workspaces" in data:
        info.flags.add("workspaces")

    package_manager = data.get("packageManager")
    if isinstance(package_manager, str):
        candidate = package_manager.split("@", 1)[0]
        if candidate in _JS_PACKAGE_MANAGERS:
            info.package_managers.add(candidate)
    return info


def _parse_requirements(path: str, text: str) -> ManifestInfo:
    info = ManifestInfo(path=path, package_managers={"pip"})
    for line in text.splitlines():
        name = _requirement_name(line)
        if name:
            info.dependencies.append(Dependency(name, PYPI, False))
    return info


def _parse_pyproject(path: str, text: str) -> ManifestInfo:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        raise ManifestParseError("Invalid TOML") from None

    info = ManifestInfo(path=path)

    def add(names: Any, dev: bool) -> None:
        for entry in names if isinstance(names, list) else []:
            name = _requirement_name(entry) if isinstance(entry, str) else None
            if name:
                info.dependencies.append(Dependency(name, PYPI, dev))

    def add_keys(table: Any, dev: bool) -> None:
        for key in table if isinstance(table, dict) else []:
            if key.lower() != "python":
                info.dependencies.append(Dependency(_normalize_python_name(key), PYPI, dev))

    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    tool = data.get("tool") if isinstance(data.get("tool"), dict) else {}
    poetry = tool.get("poetry") if isinstance(tool.get("poetry"), dict) else None
    uv = tool.get("uv") if isinstance(tool.get("uv"), dict) else None

    add(project.get("dependencies"), False)
    for group in (project.get("optional-dependencies"), data.get("dependency-groups")):
        for names in group.values() if isinstance(group, dict) else []:
            add(names, True)

    script_names: list[str] = []
    if isinstance(project.get("scripts"), dict):
        script_names += list(project["scripts"])
    if poetry is not None:
        add_keys(poetry.get("dependencies"), False)
        add_keys(poetry.get("dev-dependencies"), True)
        for group in (poetry.get("group") or {}).values() if isinstance(poetry.get("group"), dict) else []:
            add_keys(group.get("dependencies") if isinstance(group, dict) else None, True)
        if isinstance(poetry.get("scripts"), dict):
            script_names += list(poetry["scripts"])
    if uv is not None:
        add(uv.get("dev-dependencies"), True)

    info.entry_points += [EntryPoint("python_script", name, path) for name in script_names]
    if script_names:
        info.flags.add("scripts")
    if "build-system" in data and project:
        info.flags.add("build_system")
    info.package_managers.add("poetry" if poetry is not None else "uv" if uv is not None else "pip")
    return info


def _parse_maven(path: str, text: str) -> ManifestInfo:
    info = ManifestInfo(path=path, package_managers={"maven"})
    if "spring-boot" in text:
        info.frameworks.add("Spring Boot")
    return info


def _parse_gradle(path: str, text: str) -> ManifestInfo:
    info = ManifestInfo(path=path, package_managers={"gradle"})
    if "org.springframework.boot" in text or "spring-boot" in text:
        info.frameworks.add("Spring Boot")
    return info


_PARSERS_BY_FILENAME: dict[str, Callable[[str, str], ManifestInfo]] = {
    "package.json": _parse_package_json,
    "pyproject.toml": _parse_pyproject,
    "pom.xml": _parse_maven,
    "build.gradle": _parse_gradle,
    "build.gradle.kts": _parse_gradle,
}


def parser_for(path: str) -> Callable[[str, str], ManifestInfo] | None:
    name = PurePosixPath(path).name.lower()
    if name.startswith("requirements") and name.endswith(".txt"):
        return _parse_requirements
    return _PARSERS_BY_FILENAME.get(name)


def is_manifest(path: str) -> bool:
    return parser_for(path) is not None


def parse_manifest(path: str, text: str) -> ManifestInfo:
    parser = parser_for(path)
    if parser is None:
        raise ManifestParseError("Unsupported manifest")
    return parser(path, text)
