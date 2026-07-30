#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build the wasm demo and stage everything the page needs in crates/mapox-web/web/.

Two artifacts have to stay in lockstep: the wasm binary and the JS module that
loads it (mapox_web.js). Both come out of the wasm-bindgen CLI, but the CLI
bakes an ABI shared with the wasm-bindgen *crate* the binary was compiled
against, and a mismatched pair fails at instantiation time in the browser, not
at build time. Hence the CLI version is checked against the exact wasm-bindgen
the lockfile resolved, on every build.
"""

import argparse
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
# cargo package mapox-web -> cdylib mapox_web.wasm -> bindgen mapox_web.js
LIB = "mapox_web"


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
    info(f"building {LIB} ({profile})")
    cmd = ["cargo", "build", "--package", "mapox-web", "--lib", "--target", TARGET]
    if release:
        cmd.append("--release")
    run(*cmd)
    return REPO_ROOT / "target" / TARGET / profile / f"{LIB}.wasm"


def required_bindgen_version() -> str:
    """The wasm-bindgen version the dependency graph actually resolved.

    `cargo metadata` reads it out of the lockfile directly, so there is no
    version string to parse out of Cargo.lock by hand. It also downloads and
    unpacks anything missing as a side effect of reading the manifests, which
    covers a fresh clone or an unfetched version bump.
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

    versions = sorted(
        pkg["version"] for pkg in meta["packages"] if pkg["name"] == "wasm-bindgen"
    )
    if not versions:
        die("no wasm-bindgen in the dependency graph -- is it still a dependency?")
    if len(versions) > 1:
        die(f"expected one wasm-bindgen, found {len(versions)}: {', '.join(versions)}")
    return versions[0]


def bindgen(artifact: Path) -> None:
    """Generate mapox_web.js + mapox_web_bg.wasm into the web dir."""
    version = required_bindgen_version()
    if not shutil.which("wasm-bindgen"):
        die(
            "wasm-bindgen CLI not found -- install the version the lockfile "
            f"expects:\n  cargo install wasm-bindgen-cli --version {version} --locked"
        )
    installed = run(
        "wasm-bindgen", "--version", stdout=subprocess.PIPE, text=True
    ).stdout.split()[-1]
    if installed != version:
        die(
            f"wasm-bindgen CLI is {installed} but the crate in Cargo.lock is "
            f"{version}; the pair shares an ABI, so match them:\n"
            f"  cargo install wasm-bindgen-cli --version {version} --locked"
        )

    info(f"wasm-bindgen {version}")
    run(
        "wasm-bindgen",
        "--target",
        "web",
        "--no-typescript",
        "--out-dir",
        WEB_DIR,
        artifact,
    )


def optimize(release: bool) -> None:
    if release and shutil.which("wasm-opt"):
        info("wasm-opt -Oz")
        staged = WEB_DIR / f"{LIB}_bg.wasm"
        run("wasm-opt", "-Oz", staged, "-o", staged)


def human(size: float) -> str:
    for unit in ("B", "K", "M"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


def report() -> None:
    info(f"{WEB_DIR}/")
    for name in ("index.html", f"{LIB}.js", f"{LIB}_bg.wasm"):
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
    bindgen(build(args.release))
    optimize(args.release)
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
