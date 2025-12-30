#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAN="$ROOT/vsc/manifest.json"
SHA_FILE="$ROOT/vsc/manifest.sha256"

if [[ ! -f "$MAN" ]]; then
  echo "missing: $MAN" >&2
  exit 2
fi
if [[ ! -f "$SHA_FILE" ]]; then
  echo "missing: $SHA_FILE" >&2
  exit 2
fi

computed=$(sha256sum "$MAN" | awk '{print $1}')
file=$(tr -d ' \t\n\r' < "$SHA_FILE")

echo "verify_vsc_sha256"
echo "computed: $computed"
echo "file    : $file"

[[ "$computed" == "$file" ]]
