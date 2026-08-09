#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON="${PROJECT_DIR}/.venv/bin/python"
readonly CA_NAME="${NOABOT_CA_NAME:-Zscaler Root CA}"
readonly CA_BUNDLE="${HOME}/.config/noabot/ca-bundle.pem"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "run_local.sh currently supports macOS only." >&2
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Virtual environment not found at ${PROJECT_DIR}/.venv" >&2
    exit 1
fi

"${PYTHON}" - "${CA_NAME}" "${CA_BUNDLE}" <<'PY'
from pathlib import Path
import subprocess
import sys

import certifi

ca_name = sys.argv[1]
target = Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)

corporate_root = subprocess.run(
    [
        "security",
        "find-certificate",
        "-c",
        ca_name,
        "-p",
        "/Library/Keychains/System.keychain",
    ],
    check=True,
    capture_output=True,
).stdout

if b"BEGIN CERTIFICATE" not in corporate_root:
    raise RuntimeError(f"{ca_name!r} was not exported from the System keychain")

public_roots = Path(certifi.where()).read_bytes()
target.write_bytes(public_roots.rstrip() + b"\n" + corporate_root.strip() + b"\n")
target.chmod(0o600)
print(f"Using local CA bundle: {target}")
PY

export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="${CA_BUNDLE}"
export SSL_CERT_FILE="${CA_BUNDLE}"
export REQUESTS_CA_BUNDLE="${CA_BUNDLE}"

cd "${PROJECT_DIR}"
exec "${PYTHON}" -m streamlit run streamlit_app.py "$@"
