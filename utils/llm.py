import re
from pathlib import Path

# Hugging Face repo ids are "org/name" -- exactly one slash, e.g.
# TinyLlama/TinyLlama-1.1B-Chat-v1.0. They always contain a slash, so an
# earlier "does it contain a slash" path test classified every repo id as a
# path and then died on the existence check. Only two-segment ids count here;
# a value with a backslash, a drive letter, or more than one slash is treated
# as a (missing) local path and reported as such.
_REPO_ID = re.compile(r"[A-Za-z0-9][\w.\-]*/[\w][\w.\-]*\Z")


def resolve_llm_ckp_dir(path_or_repo_id):
    """Return a --llm_ckp_dir usable by transformers.

    A local directory is returned as an absolute path. A Hugging Face repo id
    is returned unchanged so transformers downloads it into its own cache --
    this function does no network I/O itself.
    """
    text = str(path_or_repo_id)
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())

    if "\\" not in text and ":" not in text and not text.startswith(".") and _REPO_ID.match(text):
        return text

    raise FileNotFoundError(
        f"LLM path does not exist: {path}. "
        "Check --llm_ckp_dir, or download the model to that directory first."
    )
