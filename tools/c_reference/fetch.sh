#!/usr/bin/env bash
# Clone the original catch22 C implementation at the pinned commit.
# The checkout lands in tools/c_reference/upstream/ and is gitignored.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sha="$(tr -d '[:space:]' < "$here/PINNED_SHA")"
dest="$here/upstream"

if [ -d "$dest/.git" ]; then
    current="$(git -C "$dest" rev-parse HEAD)"
    if [ "$current" = "$sha" ]; then
        echo "catch22 C reference already at $sha"
        exit 0
    fi
    echo "updating catch22 C reference to $sha"
    git -C "$dest" fetch --quiet origin "$sha"
else
    echo "cloning catch22 C reference"
    git clone --quiet https://github.com/DynamicsAndNeuralSystems/catch22.git "$dest"
fi

git -C "$dest" checkout --quiet "$sha"
echo "catch22 C reference at $(git -C "$dest" rev-parse HEAD)"
