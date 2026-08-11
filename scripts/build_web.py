import argparse
import functools
import http.server
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
# what index.html fetches; mapox-web has no fallback policy to run without it
POLICY = "policy.safetensors"


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


def build() -> Path:
    run("cargo", "build", "--package", "mapox-web", "--lib", "--target", TARGET, "--release")
    return REPO_ROOT / "target" / TARGET / "release" / f"{LIB}.wasm"


def bindgen(artifact: Path) -> None:
    """Generate mapox_web.js + mapox_web_bg.wasm into the web dir."""
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


def stage_policy(path: Path) -> None:
    if not path.is_file():
        die(f"{path}: no such policy file")
    info(f"staging {path.name} as {POLICY}")
    shutil.copyfile(path, WEB_DIR / POLICY)


def human(size: float) -> str:
    for unit in ("B", "K", "M"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


def report() -> None:
    info(f"{WEB_DIR}/")
    for name in ("index.html", f"{LIB}.js", f"{LIB}_bg.wasm", POLICY):
        staged = WEB_DIR / name
        if not staged.exists():
            print(f"    {name:<24} MISSING -- pass --policy <path>")
            continue
        print(f"    {name:<24} {human(staged.stat().st_size)}")

def serve(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        info(f"serving on http://127.0.0.1:{port}/ (ctrl-c to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serve",
        nargs="?",
        type=int,
        const=8000,
        metavar="PORT",
        help="serve the staged page on PORT (default 8000)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        metavar="PATH",
        help=f"copy a jaxrl policy export into the page as {POLICY}",
    )
    parser.set_defaults(release=True, autotune=True)
    args = parser.parse_args()

    ensure_target()
    bindgen(build())
    optimize(args.release)
    if args.policy is not None:
        stage_policy(args.policy)
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
