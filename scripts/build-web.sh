#!/usr/bin/env bash
#
# Build the wasm demo and stage everything the page needs in crates/mapox-web/web/.
#
# Two artifacts have to stay in lockstep: the wasm binary, and miniquad's JS
# glue (mq_js_bundle.js). miniquad declares its GL/platform functions as plain
# wasm imports that the glue resolves at runtime, so a mismatched pair does not
# fail at link time -- it fails inside WebAssembly.instantiate with a LinkError
# and a blank canvas. Hence the glue is re-copied from the exact miniquad the
# lockfile resolved, on every build, rather than pinned by hand.
#
# Usage: scripts/build-web.sh [--debug] [--serve [PORT]]

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
web_dir="$repo_root/crates/mapox-web/web"
target=wasm32-unknown-unknown
bin=mapox-web-demo

profile=release
serve=0
port=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) profile=debug; shift ;;
        --release) profile=release; shift ;;
        --serve)
            serve=1; shift
            [[ ${1:-} =~ ^[0-9]+$ ]] && { port=$1; shift; }
            ;;
        -h|--help) sed -n '3,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# 1. toolchain
if ! rustup target list --installed | grep -qx "$target"; then
    echo "==> installing $target"
    rustup target add "$target"
fi

# 2. build
echo "==> building $bin ($profile)"
build_args=(--package mapox-web --bin "$bin" --target "$target")
[[ $profile == release ]] && build_args+=(--release)
cargo build "${build_args[@]}"

# 3. stage the wasm (gitignored; the page loads it by relative path)
artifact="$repo_root/target/$target/$profile/$bin.wasm"
cp "$artifact" "$web_dir/$bin.wasm"

if command -v wasm-opt >/dev/null && [[ $profile == release ]]; then
    echo "==> wasm-opt -Oz"
    wasm-opt -Oz "$web_dir/$bin.wasm" -o "$web_dir/$bin.wasm"
fi

# 4. sync miniquad's JS glue to whatever version Cargo.lock resolved
mq_version=$(awk '/^name = "miniquad"$/ { getline; gsub(/[",]/, "", $3); print $3; exit }' \
    "$repo_root/Cargo.lock")
if [[ -z $mq_version ]]; then
    echo "could not find miniquad in Cargo.lock -- is it still a dependency?" >&2
    exit 1
fi

cargo_home=${CARGO_HOME:-$HOME/.cargo}
glue=$(echo "$cargo_home"/registry/src/*/"miniquad-$mq_version"/js/gl.js)
if [[ ! -f $glue ]]; then
    # not unpacked yet (fresh clone, or a version bump that has not been fetched)
    cargo fetch --manifest-path "$repo_root/Cargo.toml" >/dev/null
    glue=$(echo "$cargo_home"/registry/src/*/"miniquad-$mq_version"/js/gl.js)
fi
if [[ ! -f $glue ]]; then
    echo "no gl.js for miniquad $mq_version under $cargo_home/registry/src" >&2
    exit 1
fi

dest="$web_dir/mq_js_bundle.js"
if [[ -f $dest ]] && cmp -s "$glue" "$dest"; then
    glue_state="unchanged"
else
    cp "$glue" "$dest"
    glue_state="updated"
fi
echo "==> glue: miniquad $mq_version ($glue_state)"

# 5. report, and optionally serve
printf '==> %s/\n' "$web_dir"
for f in index.html mq_js_bundle.js "$bin.wasm"; do
    printf '    %-24s %s\n' "$f" "$(du -h "$web_dir/$f" | cut -f1)"
done

if [[ $serve == 1 ]]; then
    echo "==> serving on http://127.0.0.1:$port/ (ctrl-c to stop)"
    exec python3 -m http.server "$port" --directory "$web_dir"
fi
