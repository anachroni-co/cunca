#!/usr/bin/env python3
"""
Script to clean and deduplicate TODO files.

Author: Skydesk International Dev Team.

This script:
1. Parses all TODO entries from TODOs.md and TODOs_PRIORITIZED.md
2. Normalizes absolute paths to relative paths
3. Detects and removes duplicates (same file + same line = duplicate)
4. Generates clean, deduplicated TODO files organized by module
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TodoItem:
    """Represents a single TODO item."""
    line_num: Optional[int]
    content: str
    file_path: str  # The actual source file (e.g., agents/advanced_behaviors.py)
    priority: str = "High"

    def get_unique_key(self) -> str:
        """
        Generate a unique key for deduplication.
        Same file + same line = same TODO (regardless of content variations).
        """
        normalized_path = self.file_path.replace("\\", "/").lower().strip("/")
        return f"{normalized_path}:{self.line_num}"


def normalize_path(raw_path: str) -> str:
    """
    Normalize an absolute or relative path to a clean relative path.
    """
    path = raw_path.strip("`'\"").strip()

    # Handle "file.py:123" format - extract just the file
    if ":" in path:
        # Check if it's a Windows path (C:, D:, etc.)
        if not re.match(r"^[a-zA-Z]:", path):
            # Probably "file.py:123" format
            parts = path.rsplit(":", 1)
            if parts[-1].isdigit():
                path = parts[0]

    # Patterns for absolute paths to remove
    patterns = [
        r"[dD]:\\Escritorio\\Nueva carpeta \(3\)\\capibaraGPT_v3\\",
        r"[dD]:\\Escritorio\\Nueva folder \(3\)\\capibaraGPT_v3\\",
        r"[cC]:\\Users\\[^\\]+\\Documents\\GitHub\\capibaraGPT_v3\\",
    ]

    for pattern in patterns:
        path = re.sub(pattern, "", path, flags=re.IGNORECASE)

    # Normalize separators
    path = path.replace("\\", "/")

    # Clean up
    path = path.strip("/.")

    return path


def extract_source_file_and_line(line: str) -> tuple[Optional[str], Optional[int], str]:
    """
    Extract the actual source file, line number, and content from a TODO line.

    Returns: (source_file, line_num, content)

    Examples:
    - [ ] 249: # Comment - `d:\\...\\agents\\advanced_behaviors.py`
      -> ("agents/advanced_behaviors.py", 249, "# Comment")

    - [ ] 376: - [ ] # Comment - `agents\\file.py:249` - `agents/README.md`
      -> ("agents/file.py", 249, "# Comment")
    """
    line = line.strip()

    # Remove "- [ ]" prefix
    if line.startswith("- [ ]"):
        line = line[5:].strip()

    # Extract all paths in backticks
    paths_raw = re.findall(r"`([^`]+)`", line)
    paths = [normalize_path(p) for p in paths_raw]

    # Filter to source code files (not README.md or TODOs.md)
    source_paths = [
        p for p in paths
        if not p.lower().endswith(("readme.md", "todos.md"))
        and not p.lower().endswith(".md")  # Prefer actual code files
    ]

    # If no source paths, use all paths but prefer non-.md files
    if not source_paths:
        source_paths = [p for p in paths if not p.lower().endswith(".md")]

    if not source_paths and paths:
        source_paths = paths

    # Determine file path
    file_path = source_paths[0] if source_paths else ""

    # Extract line number from beginning of line (format: "123: content")
    line_num = None
    content = line

    match = re.match(r"^(\d+):\s*(.*)$", line)
    if match:
        line_num = int(match.group(1))
        content = match.group(2)

    # Also check if line number is embedded in path (file.py:123)
    for p in paths_raw:
        if ":" in p:
            parts = p.rsplit(":", 1)
            if parts[-1].isdigit():
                line_num = int(parts[-1])
                break

    # Clean content
    # Remove nested "- [ ]" patterns
    content = re.sub(r"- \[ \]\s*", "", content)
    # Remove path references
    for p in paths_raw:
        content = content.replace(f"`{p}`", "")
    content = content.strip()
    # Remove "source: ..." suffix
    content = re.sub(r"\s*\|\s*source:.*$", "", content)
    content = re.sub(r"\s*-\s*$", "", content)

    return file_path, line_num, content


def parse_todo_file(file_path: Path) -> list[TodoItem]:
    """Parse all TODO items from a file."""
    todos = []
    current_priority = "High"

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return todos

    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()

        # Check for priority headers
        if stripped.startswith("## "):
            priority = stripped[3:].strip()
            # Extract just the priority name (remove counts like "(676)")
            priority = re.sub(r"\s*\(\d+\)\s*$", "", priority)
            if priority in ["Critical", "High", "Medium", "Low"]:
                current_priority = priority
            continue

        # Skip non-TODO lines
        if not stripped.startswith("- [ ]"):
            continue

        # Skip completed items
        if "[x]" in stripped:
            continue

        # Parse the TODO
        file_path_str, line_num, content = extract_source_file_and_line(stripped)

        if not file_path_str or not content:
            continue

        todos.append(TodoItem(
            line_num=line_num,
            content=content,
            file_path=file_path_str,
            priority=current_priority
        ))

    return todos


def deduplicate_todos(todos: list[TodoItem]) -> list[TodoItem]:
    """
    Remove duplicate TODOs based on file path + line number.
    Keeps the first occurrence with the best content.
    """
    seen = {}

    for todo in todos:
        key = todo.get_unique_key()

        if key not in seen:
            seen[key] = todo
        else:
            # Keep the one with better content (longer, more descriptive)
            existing = seen[key]
            if len(todo.content) > len(existing.content):
                seen[key] = todo

    return list(seen.values())


def group_by_module(todos: list[TodoItem]) -> dict[str, list[TodoItem]]:
    """Group TODOs by their module (first path component)."""
    groups = {}

    for todo in todos:
        parts = todo.file_path.split("/")
        module = parts[0] if parts else "other"

        if module not in groups:
            groups[module] = []
        groups[module].append(todo)

    return dict(sorted(groups.items()))


def group_by_priority(todos: list[TodoItem]) -> dict[str, list[TodoItem]]:
    """Group TODOs by priority."""
    groups = {"Critical": [], "High": [], "Medium": [], "Low": []}

    for todo in todos:
        if todo.priority in groups:
            groups[todo.priority].append(todo)
        else:
            groups["High"].append(todo)

    return groups


def generate_todos_md(todos: list[TodoItem], output_path: Path) -> None:
    """Generate TODOs.md grouped by module."""
    grouped = group_by_module(todos)

    lines = [
        "# TODOs - Global (Cleaned)",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"Total unique TODOs: {len(todos)}",
        "",
        "_This file was auto-generated by scripts/clean_todos.py_",
        "",
    ]

    for module, module_todos in grouped.items():
        lines.append(f"## {module}")
        lines.append("")

        # Sort by file path, then line number
        sorted_todos = sorted(module_todos, key=lambda t: (t.file_path, t.line_num or 0))

        for todo in sorted_todos:
            line_ref = f":{todo.line_num}" if todo.line_num else ""
            lines.append(f"- [ ] `{todo.file_path}{line_ref}` - {todo.content}")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_todos_prioritized_md(todos: list[TodoItem], output_path: Path) -> None:
    """Generate TODOs_PRIORITIZED.md grouped by priority."""
    grouped = group_by_priority(todos)

    lines = [
        "# TODOs Prioritized (Cleaned)",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"Total pending: {len(todos)}",
        "",
        "_This file was auto-generated by scripts/clean_todos.py_",
        "",
    ]

    for priority in ["Critical", "High", "Medium", "Low"]:
        count = len(grouped[priority])
        lines.append(f"## {priority} ({count})")

        if count == 0:
            lines.append("- [x] No pending items at this priority.")
        else:
            sorted_todos = sorted(grouped[priority], key=lambda t: (t.file_path, t.line_num or 0))
            for todo in sorted_todos:
                line_ref = f":{todo.line_num}" if todo.line_num else ""
                lines.append(f"- [ ] `{todo.file_path}{line_ref}` - {todo.content}")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {output_path}")


def print_statistics(original_count: int, unique_todos: list[TodoItem]) -> None:
    """Print cleaning statistics."""
    unique_count = len(unique_todos)
    duplicates = original_count - unique_count
    reduction_pct = (duplicates / original_count * 100) if original_count > 0 else 0

    print("\n" + "=" * 50)
    print("CLEANING STATISTICS")
    print("=" * 50)
    print(f"Original TODO entries:  {original_count}")
    print(f"Unique TODOs:           {unique_count}")
    print(f"Duplicates removed:     {duplicates}")
    print(f"Reduction:              {reduction_pct:.1f}%")
    print("=" * 50)

    # Priority breakdown
    grouped = group_by_priority(unique_todos)
    print("\nBy Priority:")
    for priority, items in grouped.items():
        print(f"  {priority}: {len(items)}")

    # Module breakdown
    module_groups = group_by_module(unique_todos)
    sorted_modules = sorted(module_groups.items(), key=lambda x: len(x[1]), reverse=True)
    print("\nBy Module (top 15):")
    for module, items in sorted_modules[:15]:
        print(f"  {module}: {len(items)}")


def main() -> None:
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    todos_file = repo_root / "TODOs.md"
    todos_prioritized_file = repo_root / "TODOs_PRIORITIZED.md"

    print("Cleaning TODO files...")
    print(f"Repository root: {repo_root}")
    print()

    # Parse both files
    all_todos = []
    all_todos.extend(parse_todo_file(todos_file))
    all_todos.extend(parse_todo_file(todos_prioritized_file))

    original_count = len(all_todos)
    print(f"Parsed {original_count} TODO entries")

    # Deduplicate
    unique_todos = deduplicate_todos(all_todos)
    print(f"Found {len(unique_todos)} unique TODOs")

    # Generate clean files
    generate_todos_md(unique_todos, todos_file)
    generate_todos_prioritized_md(unique_todos, todos_prioritized_file)

    # Print statistics
    print_statistics(original_count, unique_todos)

    print("\nDone!")


if __name__ == "__main__":
    main()
