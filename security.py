# -*- coding: utf-8 -*-
"""
security.py — anvay production security helpers

Validation, input checking, job cleanup, and security logging.
Nothing in this file is application logic; it is purely defensive.
"""
from __future__ import annotations
import os
import re
import time
import shutil
import logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
)

# Folders that may appear in /results/<job_id>/<folder>/<file> URLs.
ALLOWED_VIZ_FOLDERS = frozenset({'plotly', 'bokeh', 'seaborn', 'html'})

# Only these filenames may be served via the download route.
ALLOWED_DOWNLOAD_FILES = frozenset({
    'topic_words.csv',
    'topic_words.txt',
    'doc_topic_weights.csv',
    'doc_topic_weights.txt',
    'topics.csv',
    'topics.txt',
})

# Per-file upload limit (bytes). 1 MB is generous for plain text.
MAX_FILE_BYTES = 1 * 1024 * 1024

# Maximum number of corpus files per submission.
MAX_FILES = 200

# Numeric hyperparameter safe ranges.
PARAM_BOUNDS = {
    'num_topics':  (2,   50),
    'iterations':  (10,  500),
    'passes':      (1,   50),
    'chunk_size':  (50,  2000),
    'no_below':    (1,   100),
}

# How long (seconds) to keep result directories on disk before cleanup.
RESULT_MAX_AGE = 2 * 3600  # 2 hours


# ---------------------------------------------------------------------------
# UUID validation
# ---------------------------------------------------------------------------

def is_valid_uuid(s: str) -> bool:
    """Return True only if s is a well-formed UUID4 string."""
    return bool(UUID_RE.match(str(s)))


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def validate_upload(file_obj, filename: str) -> tuple[bool, str | None]:
    """
    Validate a single uploaded file object before saving to disk.

    Checks performed:
      - .txt extension
      - Non-zero size
      - Per-file size limit
      - No null bytes (binary file check)
      - UTF-8 decodable header

    Returns (ok, error_message). error_message is None on success.
    The file_obj seek position is reset to 0 on return.
    """

    if not filename.lower().endswith('.txt'):
        return False, f"'{filename}' is not a .txt file."

    # Size check (seek to end)
    file_obj.seek(0, 2)
    size = file_obj.tell()
    file_obj.seek(0)

    if size == 0:
        return False, f"'{filename}' is empty."

    if size > MAX_FILE_BYTES:
        mb = MAX_FILE_BYTES // (1024 * 1024)
        return False, f"'{filename}' exceeds the {mb} MB per-file limit."

    # Content check: inspect the first 512 bytes
    header = file_obj.read(512)
    file_obj.seek(0)

    if b'\x00' in header:
        return False, f"'{filename}' contains null bytes and appears to be binary."
    
    try:
        header.decode('utf-8')
    except UnicodeDecodeError as e:
        # If the bad bytes start within the last 3 bytes, it's a buffer
        # boundary cut on a multibyte character, not actually corrupt data.
        # Bengali (and most Unicode scripts) use at most 3-byte sequences.
        if e.start < len(header) - 3:
            return False, f"'{filename}' does not appear to be valid UTF-8 text."
    return True, None

def validate_file_count(files: list) -> tuple[bool, str | None]:
    """Check that the submission does not exceed MAX_FILES."""
    if len(files) > MAX_FILES:
        return False, f"Too many files. Maximum is {MAX_FILES} per submission."
    return True, None


# ---------------------------------------------------------------------------
# Hyperparameter validation
# ---------------------------------------------------------------------------

def validate_hyperparams(form) -> list[str]:
    """
    Validate numeric hyperparameters against safe bounds.

    Returns a list of error strings. An empty list means all params are valid.
    Params not present in the form are silently skipped (defaults apply elsewhere).
    """
    errors = []
    for param, (lo, hi) in PARAM_BOUNDS.items():
        raw = form.get(param)
        if raw is None:
            continue
        try:
            n = int(raw)
        except (ValueError, TypeError):
            errors.append(f"'{param}' must be an integer.")
            continue
        if not (lo <= n <= hi):
            errors.append(f"'{param}' must be between {lo} and {hi}.")
    return errors


# ---------------------------------------------------------------------------
# Path / route safety
# ---------------------------------------------------------------------------

def safe_result_path(result_folder: str, job_id: str, *parts) -> str | None:
    """
    Build a path under result_folder/job_id and verify it does not escape
    the result_folder root (path traversal guard).

    Returns the resolved path on success, None if the path would escape.
    """
    if not is_valid_uuid(job_id):
        return None

    # os.path.basename on each part strips any directory component
    safe_parts = [os.path.basename(p) for p in parts]
    candidate = os.path.realpath(
        os.path.join(result_folder, job_id, *safe_parts)
    )
    root = os.path.realpath(result_folder)

    if not candidate.startswith(root + os.sep):
        return None
    return candidate


# ---------------------------------------------------------------------------
# Job cleanup
# ---------------------------------------------------------------------------

def cleanup_old_jobs(base_dir: str, max_age_seconds: int = RESULT_MAX_AGE) -> None:
    """
    Delete job subdirectories inside base_dir that are older than
    max_age_seconds. Only directories whose names are valid UUIDs are touched,
    so non-job content in base_dir is never affected.

    Call this at the top of the /process route so old jobs are swept up
    naturally without needing a separate cron job or background thread.
    """
    if not os.path.isdir(base_dir):
        return
    now = time.time()
    for entry in os.scandir(base_dir):
        if not entry.is_dir():
            continue
        if not is_valid_uuid(entry.name):
            continue
        try:
            age = now - entry.stat().st_mtime
            if age > max_age_seconds:
                shutil.rmtree(entry.path, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Security logger
# ---------------------------------------------------------------------------

def get_security_logger() -> logging.Logger:
    """Return the anvay security logger (anvay.security)."""
    return logging.getLogger('anvay.security')