---
name: merge
description: >-
  Merge finished work into main for the mapox repository: commit, rebase onto
  the latest main, run the local checks, fast-forward main in the primary
  checkout, and push main to GitHub. Only invoke this when the user has
  explicitly requested merging or landing the change, not for review,
  preparation, running checks on their own, or skill installation.
disable-model-invocation: true
metadata:
  delta-action: land
---

# Merge into mapox `main`

Invoking this skill is the merge request: carry it through without asking for
confirmation again. If a blocker stops you, say plainly that the change has not
merged, or which destination has it and which does not.

## The process

There are no pull requests and no CI. `main` is the only long-lived branch, and
history stays linear: rebase and fast-forward, never a merge commit, never a
force push.

A merge lands in two places, in this order:

1. `main` in the primary checkout, via the working clone's `local` remote. The
   push updates that checkout's files in place, so it must be on `main` with no
   changes to tracked files. If it isn't, stop; that work is the user's.
2. `main` on GitHub, via `origin`.

If there is no `local` remote, you are already in the primary checkout and only
`origin` applies.

## Steps

1. Fetch both remotes. If `local/main` and `origin/main` have diverged, stop.
   Otherwise the base is whichever contains the other.
2. Commit the work that belongs to this change, in the style of `git log -5`:
   imperative subject, short body, `Co-Authored-By:` trailer for the assistant.
   Ask only if unrelated changes are mixed in and you can't tell them apart.
3. Rebase onto the base. Resolve conflicts yourself when the intended result is
   clear; if it is ambiguous or would drop someone else's work, abort the
   rebase and ask.
4. Run the checks below on the rebased commit. All that apply must pass.
5. `git push local HEAD:main`, then `git push origin HEAD:main`.
6. Confirm the primary checkout's `HEAD` and `git ls-remote origin
   refs/heads/main` both equal the merged commit, then report the hash and the
   checks that ran.

## Checks

These are the only gate, so a failing or unrunnable check blocks the merge. Fix
failures the change caused; never weaken a test to pass. Run what the changed
files call for:

| Changed | Run | Source |
|---|---|---|
| Any `.rs` file | `cargo fmt --all -- --check` | workspace in `Cargo.toml` |
| `crates/mapox-core/` | `cargo test -p mapox-core` | `crates/mapox-core/Cargo.toml` |
| `crates/mapox-burn/` | `cargo test -p mapox-burn` | `crates/mapox-burn/tests/` |
| `crates/mapox-core/`, `crates/mapox-py/`, `python/`, `tests/`, `solutions/` | `uv run maturin dev`, then `uv run pytest` | `scripts/run_survive.sh`; `[tool.pytest.ini_options]` in `pyproject.toml` |
| `crates/mapox-web/`, `crates/mapox-burn/` | `./scripts/build_web.py` (never with `--serve`) | `main()` in `scripts/build_web.py` |

Dependency or toolchain-config changes (`Cargo.toml`, `Cargo.lock`,
`.cargo/config.toml`, `pyproject.toml`, `uv.lock`) call for all of them.
Documentation-only changes need none. There is no ruff or clippy gate.
