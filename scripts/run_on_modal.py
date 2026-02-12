#!/usr/bin/env python3
"""Spin up an ephemeral Ollama server on Modal and run looselips or bench.

Launches an Ollama instance in a Modal Sandbox with a public HTTPS
tunnel, runs the chosen command against it, then tears down the sandbox.
No persistent state is kept (no volumes); models are baked into
the cached image so subsequent runs skip the download.

The --backend tag for bench is set automatically from the GPU type
(e.g. L4 -> modal/L4).

Requires: pip install modal && modal setup

Examples:

    # Scan a ChatGPT export
    python scripts/run_on_modal.py scan --config config.toml export.zip

    # Run benchmarks
    python scripts/run_on_modal.py bench \\
        --config config.toml --model ollama/qwen3:8b --db results.db

    # Use a bigger GPU for a larger model
    python scripts/run_on_modal.py --gpu L40S bench \\
        --config config.toml --model ollama/qwen3:32b --db results.db

    # Run all bench models in parallel (model gpu pairs)
    set -- \\
        qwen3:32b   L40S \\
        qwen3:14b   L4   \\
        qwen3:8b    L4   \\
        llama3.1:8b L4   \\
        qwen2.5:3b  L4   \\
        gemma2:2b   L4   \\
        phi3:mini   L4   \\
        qwen2.5:0.5b none
    while [ $# -ge 2 ]; do
        python scripts/run_on_modal.py --gpu "$2" bench \\
            --config config.toml --model "ollama/$1" --db bench.db &
        shift 2
    done
    wait

    # List/stop containers
    modal container list
    modal container stop CONTAINER_ID
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

OLLAMA_PORT = 11434
APP_NAME = "looselips-ollama"
OLLAMA_PREFIX = "ollama/"
OLLAMA_RELEASES_URL = "https://github.com/ollama/ollama/releases/latest"


def _latest_ollama_version() -> str:
    """Fetch the latest ollama version tag from GitHub."""
    req = urllib.request.Request(OLLAMA_RELEASES_URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=10) as r:
        # GitHub redirects /latest to /tag/vX.Y.Z
        tag = r.url.rsplit("/", 1)[-1]
    return tag.lstrip("v")


def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    parts = [f"{h}h"]
    if m:
        parts.append(f"{m}m")
    return "".join(parts)


def _wait_healthy(url: str, tries: int = 30, interval: float = 1.0) -> bool:
    """Poll the Ollama API until it responds or we run out of tries."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"{url}/api/version", timeout=3) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if i < tries - 1:
            time.sleep(interval)
    return False


def _models_from_config(config_path: str) -> list[str]:
    """Extract ollama model tags from a looselips config file.

    Returns bare model tags (no 'ollama/' prefix) for all models
    that use the ollama provider.
    """
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    models: set[str] = set()
    default = raw.get("model", "")
    if default.startswith(OLLAMA_PREFIX):
        models.add(default.removeprefix(OLLAMA_PREFIX))

    for m in raw.get("matcher", []):
        override = m.get("model", "")
        if override.startswith(OLLAMA_PREFIX):
            models.add(override.removeprefix(OLLAMA_PREFIX))

    return sorted(models)


def _build_image(models: list[str], ollama_version: str) -> object:  # modal.Image
    import modal

    pull_cmds = " && ".join(f"ollama pull {m}" for m in models)
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("curl", "ca-certificates", "zstd")
        .run_commands(
            "curl -fsSL https://ollama.com/install.sh"
            f" | OLLAMA_VERSION={ollama_version} sh",
            "ollama serve &"
            " for i in $(seq 30); do"
            " curl -sf http://localhost:11434/api/version > /dev/null"
            " && break; sleep 1; done"
            f" && {pull_cmds}; rc=$?; kill %1 2>/dev/null; exit $rc",
        )
        .env({"OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}"})
    )


def _create_sandbox(args: argparse.Namespace, models: list[str]) -> tuple[object, str]:
    import modal

    modal.enable_output()
    ollama_version = _latest_ollama_version()
    print(f"Ollama {ollama_version}, GPU {args.gpu}, models: {', '.join(models)}")

    app = modal.App.lookup(APP_NAME, create_if_missing=True)

    print("Building image...")
    image = _build_image(models, ollama_version)

    sandbox_kwargs: dict[str, object] = {
        "app": app,
        "image": image,
        "timeout": args.timeout,
        "idle_timeout": args.idle_timeout,
        "h2_ports": [OLLAMA_PORT],
    }
    if args.gpu.lower() != "none":
        sandbox_kwargs["gpu"] = args.gpu

    print("Starting sandbox...")
    sb = modal.Sandbox.create(
        "bash", "-lc", "ollama serve",
        **sandbox_kwargs,
    )

    tunnel = sb.tunnels()[OLLAMA_PORT]
    url = tunnel.url

    print("Waiting for Ollama to become healthy...")
    if not _wait_healthy(url):
        print("ERROR: Ollama did not become healthy after 30s", file=sys.stderr)
        sb.terminate()
        sys.exit(1)
    print("Ollama is up.")

    return sb, url


def _backend_tag(gpu: str) -> str:
    """Build a backend tag from the GPU type (e.g. 'L4' -> 'modal/L4')."""
    return f"modal/{gpu}"


def _build_scan_cmd(args: argparse.Namespace) -> list[str]:
    """Build the looselips CLI command for scanning."""
    cmd = ["looselips"]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.output:
        cmd.extend(["--output", args.output])
    verbosity = args.verbose or 1
    cmd.append("-" + "v" * verbosity)
    cmd.append(args.input)
    return cmd


def _build_bench_cmd(args: argparse.Namespace, backend: str) -> list[str]:
    """Build the bench.py command for benchmarking."""
    cmd = [sys.executable, "-m", "looselips.bench"]
    cmd.extend(["--config", args.config])
    cmd.extend(["--backend", backend])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.db:
        cmd.extend(["--db", args.db])
    if args.testcases:
        cmd.extend(["--testcases", args.testcases])
    if args.matchers:
        for m in args.matchers:
            cmd.extend(["--matcher", m])
    if args.force:
        cmd.append("--force")
    return cmd


def _run(args: argparse.Namespace) -> None:
    models = _models_from_config(args.config)

    # Include --model override so it gets baked into the image
    if args.command == "bench" and args.model:
        override = args.model
        if override.startswith(OLLAMA_PREFIX):
            tag = override.removeprefix(OLLAMA_PREFIX)
            if tag not in models:
                models.append(tag)
                models.sort()

    if not models:
        print("ERROR: no ollama models found in config", file=sys.stderr)
        sys.exit(1)

    sb, url = _create_sandbox(args, models)

    idle = _fmt_duration(args.idle_timeout)
    maxlife = _fmt_duration(args.timeout)
    print()
    print(f"  Sandbox   : {sb.object_id}")
    print(f"  URL       : {url}")
    print(f"  Models    : {', '.join(models)}")
    print(f"  GPU       : {args.gpu}")
    print(f"  Expires   : {idle} idle / {maxlife} max")
    print()
    print(f"  Dashboard : https://modal.com/apps/{APP_NAME}")
    print(f"  Logs      : modal container logs {sb.object_id}")
    print("  List      : modal container list")
    print(f"  Terminate : modal container stop {sb.object_id}")
    print()

    backend = _backend_tag(args.gpu)

    if args.command == "scan":
        cmd = _build_scan_cmd(args)
    else:
        cmd = _build_bench_cmd(args, backend)

    env = {**os.environ, "OLLAMA_API_BASE": url}
    print(f"  Running: {' '.join(cmd)}")
    print()

    try:
        rc = subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        rc = 130
    finally:
        print()
        try:
            print("Terminating sandbox... (Ctrl-C to skip)")
            sb.terminate()
            print("Terminated.")
        except:
            print()
            print("Sandbox possibly still running:")
            print(f"  modal container stop {sb.object_id}")
            raise

    sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run looselips or bench with an ephemeral Ollama server on Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model suggestions:\n"
            "  qwen3:32b    best quality (needs L40S)\n"
            "  qwen3:14b    best quality for L4\n"
            "  qwen3:8b     good balance of speed/quality\n"
            "  llama3.1:8b  Meta's best at 8B\n"
            "  qwen2.5:3b   fast, decent quality\n"
        ),
    )

    ap.add_argument(
        "--gpu", default="L4",
        help="GPU type: L4, L40S, A10G, A100, H100, or 'none' (%(default)s)",
    )
    ap.add_argument(
        "--idle-timeout", type=int, default=5 * 60, metavar="SEC",
        help="kill after N seconds idle (%(default)ss)",
    )
    ap.add_argument(
        "--timeout", type=int, default=8 * 60 * 60, metavar="SEC",
        help="hard max lifetime in seconds (%(default)ss)",
    )

    sub = ap.add_subparsers(dest="command", required=True)

    # -- scan subcommand --
    sp_scan = sub.add_parser("scan", help="Run looselips scan")
    sp_scan.add_argument("--config", required=True, help="looselips config file")
    sp_scan.add_argument("--output", default=None, help="output report path")
    sp_scan.add_argument("-v", "--verbose", action="count", default=0)
    sp_scan.add_argument("input", help="export file (.json or .zip)")

    # -- bench subcommand --
    sp_bench = sub.add_parser("bench", help="Run benchmark harness")
    sp_bench.add_argument("--config", required=True, help="looselips config file")
    sp_bench.add_argument(
        "--model", required=True,
        help="model to benchmark (e.g. ollama/qwen3:8b)",
    )
    sp_bench.add_argument("--db", required=True, help="SQLite database path")
    sp_bench.add_argument("--output", default=None, help="output HTML report path")
    sp_bench.add_argument("--testcases", default=None, help="testcase directory")
    sp_bench.add_argument(
        "-m", "--matcher", action="append", dest="matchers", metavar="NAME",
        help="only run these matchers (repeatable; default: all)",
    )
    sp_bench.add_argument(
        "--force", action="store_true",
        help="delete cached results for selected matchers before running",
    )

    args = ap.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
