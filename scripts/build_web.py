#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build the wasm demo and stage everything the page needs in crates/mapox-web/web/.

Two artifacts have to stay in lockstep: the wasm binary, and miniquad's JS glue
(mq_js_bundle.js). miniquad declares its GL/platform functions as plain wasm
imports that the glue resolves at runtime, so a mismatched pair does not fail at
link time -- it fails inside WebAssembly.instantiate with a LinkError and a blank
canvas. Hence the glue is re-copied from the exact miniquad the lockfile
resolved, on every build, rather than pinned by hand.
"""

import argparse
import filecmp
import functools
import http.server
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "crates" / "mapox-web" / "web"
TARGET = "wasm32-unknown-unknown"
BIN = "mapox-web-demo"


def info(message: str) -> None:
    # flushed so these stay interleaved with cargo's stderr when piped to a log
    print(f"==> {message}", flush=True)


def die(message: str) -> NoReturn:
    sys.exit(f"{Path(__file__).name}: {message}")


def run(*cmd, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, raising on failure. stderr always passes through."""
    return subprocess.run([str(part) for part in cmd], check=True, **kwargs)


def ensure_target() -> None:
    installed = run(
        "rustup", "target", "list", "--installed", stdout=subprocess.PIPE, text=True
    ).stdout.split()
    if TARGET not in installed:
        info(f"installing {TARGET}")
        run("rustup", "target", "add", TARGET)


def build(release: bool) -> Path:
    profile = "release" if release else "debug"
    info(f"building {BIN} ({profile})")
    cmd = ["cargo", "build", "--package", "mapox-web", "--bin", BIN, "--target", TARGET]
    if release:
        cmd.append("--release")
    run(*cmd)
    return REPO_ROOT / "target" / TARGET / profile / f"{BIN}.wasm"


def stage_wasm(artifact: Path, release: bool) -> None:
    """Copy the binary into the web dir (gitignored; the page loads it by relative path)."""
    dest = WEB_DIR / f"{BIN}.wasm"
    shutil.copy2(artifact, dest)
    if release and shutil.which("wasm-opt"):
        info("wasm-opt -Oz")
        run("wasm-opt", "-Oz", dest, "-o", dest)


def sync_glue() -> None:
    """Copy gl.js out of the miniquad the dependency graph actually resolved.

    `cargo metadata` reports the unpacked source directory directly, so there is
    no version string to parse out of Cargo.lock and no registry path to guess.
    It also downloads and unpacks anything missing as a side effect of reading
    the manifests, which covers a fresh clone or an unfetched version bump.
    """
    meta = json.loads(
        run(
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            REPO_ROOT / "Cargo.toml",
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )

    packages = [pkg for pkg in meta["packages"] if pkg["name"] == "miniquad"]
    if not packages:
        die("no miniquad in the dependency graph -- is it still a dependency?")
    if len(packages) > 1:
        found = ", ".join(sorted(pkg["version"] for pkg in packages))
        die(f"expected one miniquad, found {len(packages)}: {found}")

    version = packages[0]["version"]
    glue = Path(packages[0]["manifest_path"]).parent / "js" / "gl.js"
    if not glue.is_file():
        die(f"miniquad {version} ships no {glue}")

    dest = WEB_DIR / "mq_js_bundle.js"
    if dest.is_file() and filecmp.cmp(glue, dest, shallow=False):
        state = "unchanged"
    else:
        shutil.copy2(glue, dest)
        state = "updated"
    info(f"glue: miniquad {version} ({state})")


def human(size: float) -> str:
    for unit in ("B", "K", "M"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


def report() -> None:
    info(f"{WEB_DIR}/")
    for name in ("index.html", "mq_js_bundle.js", f"{BIN}.wasm"):
        print(f"    {name:<24} {human((WEB_DIR / name).stat().st_size)}")


def serve(port: int) -> None:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR)
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        info(f"serving on http://127.0.0.1:{port}/ (ctrl-c to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        dest="release",
        action="store_false",
        help="build the debug profile",
    )
    parser.add_argument(
        "--release",
        dest="release",
        action="store_true",
        help="build the release profile, then wasm-opt -Oz it (default)",
    )
    parser.add_argument(
        "--serve",
        nargs="?",
        type=int,
        const=8000,
        metavar="PORT",
        help="serve the staged page on PORT (default 8000)",
    )
    parser.set_defaults(release=True)
    args = parser.parse_args()

    ensure_target()
    stage_wasm(build(args.release), args.release)
    sync_glue()
    report()
    if args.serve is not None:
        serve(args.serve)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        die(f"{exc.cmd[0]} failed (exit {exc.returncode})")
    except FileNotFoundError as exc:
        die(f"{exc.filename}: command not found")
    except KeyboardInterrupt:
        sys.exit(130)
