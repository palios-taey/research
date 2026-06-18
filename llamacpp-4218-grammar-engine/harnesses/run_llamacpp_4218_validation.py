#!/usr/bin/env python3
"""Validation harness for llama.cpp #4218 grammar-stack blowups.

The harness is intentionally algorithm-agnostic. It only asks a built
llama.cpp tree to accept valid strings and run the existing grammar tests.
That makes it usable for before/after measurements regardless of whether the
candidate fix is GSS, GLL, Earley, derivatives, or another recognizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUMMARY_RE = re.compile(
    r"summary input_len=(?P<input_len>\d+) "
    r"final_stacks=(?P<final_stacks>\d+) "
    r"max_stacks=(?P<max_stacks>\d+) "
    r"total_accept_us=(?P<total_accept_us>\d+) "
    r"avg_accept_us=(?P<avg_accept_us>[0-9.]+) "
    r"max_accept_us=(?P<max_accept_us>\d+) "
    r"total_reject4_us=(?P<total_reject4_us>\d+) "
    r"avg_reject4_us=(?P<avg_reject4_us>[0-9.]+) "
    r"max_reject4_us=(?P<max_reject4_us>\d+) "
    r"accepts_eof=(?P<accepts_eof>true|false) "
    r"maxrss_kb=(?P<maxrss_kb>\d+)"
)


@dataclass
class ProbeResult:
    family: str
    depth: int
    input_len: int
    status: str
    returncode: int
    wall_s: float
    max_stacks: int | None
    total_accept_us: int | None
    max_accept_us: int | None
    total_reject4_us: int | None
    max_reject4_us: int | None
    accepts_eof: bool | None
    maxrss_kb: int | None
    raw_csv: Path
    raw_err: Path


@dataclass
class TestResult:
    name: str
    status: str
    returncode: int
    wall_s: float
    stdout_path: Path
    stderr_path: Path


def parse_range(spec: str) -> list[int]:
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise argparse.ArgumentTypeError(f"bad range {spec!r}")
        return list(range(start, end + 1))
    return [int(part) for part in spec.split(",") if part]


def synthetic_input(depth: int) -> str:
    return "a" * depth + "b" * depth


def workflow_node(depth: int) -> dict:
    if depth == 0:
        return {"title": "leaf-0", "children": [], "strategy": "sequential"}
    return {
        "title": f"group-{depth}",
        "children": [workflow_node(depth - 1)],
        "strategy": "sequential",
    }


def realistic_input(depth: int) -> str:
    return json.dumps({"root": workflow_node(depth)}, separators=(",", ":"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_process(
    argv: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int,
) -> tuple[int, str, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    run_argv = argv
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        run_argv = [stdbuf, "-oL", "-eL"] + argv
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as err:
            proc = subprocess.Popen(run_argv, cwd=str(cwd), stdout=out, stderr=err, text=True)
            try:
                rc = proc.wait(timeout=timeout_s)
                status = "PASS" if rc == 0 else "FAIL"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                err.write(f"\nTIMEOUT after {timeout_s}s\n")
                rc = 124
                status = "TIMEOUT"
    return rc, status, time.monotonic() - start


def parse_probe_output(csv_path: Path, err_path: Path) -> dict[str, object]:
    summary: dict[str, object] = {}
    err_text = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
    match = SUMMARY_RE.search(err_text)
    if match:
        data = match.groupdict()
        for key in [
            "input_len",
            "final_stacks",
            "max_stacks",
            "total_accept_us",
            "max_accept_us",
            "total_reject4_us",
            "max_reject4_us",
            "maxrss_kb",
        ]:
            summary[key] = int(data[key])
        summary["avg_accept_us"] = float(data["avg_accept_us"])
        summary["avg_reject4_us"] = float(data["avg_reject4_us"])
        summary["accepts_eof"] = data["accepts_eof"] == "true"

    max_stacks = int(summary.get("max_stacks", 0) or 0)
    total_accept_us = int(summary.get("total_accept_us", 0) or 0)
    max_accept_us = int(summary.get("max_accept_us", 0) or 0)
    total_reject4_us = int(summary.get("total_reject4_us", 0) or 0)
    max_reject4_us = int(summary.get("max_reject4_us", 0) or 0)

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    after = int(row.get("after_stacks") or 0)
                    accept = int(row.get("accept_us") or 0)
                    reject = int(row.get("reject4_us") or 0)
                except ValueError:
                    continue
                max_stacks = max(max_stacks, after)
                if not match:
                    total_accept_us += accept
                    max_accept_us = max(max_accept_us, accept)
                    total_reject4_us += reject
                    max_reject4_us = max(max_reject4_us, reject)

    if max_stacks:
        summary.setdefault("max_stacks", max_stacks)
    if total_accept_us:
        summary.setdefault("total_accept_us", total_accept_us)
    if max_accept_us:
        summary.setdefault("max_accept_us", max_accept_us)
    if total_reject4_us:
        summary.setdefault("total_reject4_us", total_reject4_us)
    if max_reject4_us:
        summary.setdefault("max_reject4_us", max_reject4_us)
    return summary


def ensure_probe(args: argparse.Namespace, bin_dir: Path, out_dir: Path) -> Path:
    if args.probe_bin:
        probe = Path(args.probe_bin).expanduser().resolve()
        if not probe.exists():
            raise SystemExit(f"probe binary not found: {probe}")
        return probe

    probe = bin_dir / "llamacpp_4218_probe"
    if probe.exists():
        return probe

    source = Path(args.probe_source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(
            "probe binary not found and probe source is missing. "
            f"Expected {probe} or pass --probe-source."
        )

    cxx = os.environ.get("CXX", "c++")
    built_probe = out_dir / "helper_bin" / "llamacpp_4218_probe"
    built_probe.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        str(source),
        f"-I{args.llama_dir}",
        f"-I{args.llama_dir / 'src'}",
        f"-I{args.llama_dir / 'include'}",
        f"-I{args.llama_dir / 'ggml' / 'include'}",
        f"-L{bin_dir}",
        "-lllama",
        f"-Wl,-rpath,{bin_dir}",
        "-o",
        str(built_probe),
    ]
    print("probe binary missing; compiling helper:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(args.llama_dir), check=True)
    return built_probe


def run_probe_case(
    probe_bin: Path,
    llama_dir: Path,
    out_dir: Path,
    family: str,
    depth: int,
    grammar_path: Path,
    input_text: str,
    timeout_s: int,
) -> ProbeResult:
    case_dir = out_dir / "raw" / family
    input_path = case_dir / f"input_{depth}.txt"
    stdout_path = case_dir / f"depth_{depth}.csv"
    stderr_path = case_dir / f"depth_{depth}.err"
    write_text(input_path, input_text)

    argv = [str(probe_bin), str(grammar_path), input_text]
    rc, status, wall_s = run_process(argv, llama_dir, stdout_path, stderr_path, timeout_s)
    parsed = parse_probe_output(stdout_path, stderr_path)
    if status == "PASS" and parsed.get("accepts_eof") is False:
        status = "FAIL"

    return ProbeResult(
        family=family,
        depth=depth,
        input_len=len(input_text),
        status=status,
        returncode=rc,
        wall_s=wall_s,
        max_stacks=parsed.get("max_stacks"),  # type: ignore[arg-type]
        total_accept_us=parsed.get("total_accept_us"),  # type: ignore[arg-type]
        max_accept_us=parsed.get("max_accept_us"),  # type: ignore[arg-type]
        total_reject4_us=parsed.get("total_reject4_us"),  # type: ignore[arg-type]
        max_reject4_us=parsed.get("max_reject4_us"),  # type: ignore[arg-type]
        accepts_eof=parsed.get("accepts_eof"),  # type: ignore[arg-type]
        maxrss_kb=parsed.get("maxrss_kb"),  # type: ignore[arg-type]
        raw_csv=stdout_path,
        raw_err=stderr_path,
    )


def discover_tests(bin_dir: Path) -> list[Path]:
    tests = sorted(bin_dir.glob("test-grammar-*"))
    json_schema_test = bin_dir / "test-json-schema-to-grammar"
    if json_schema_test.exists():
        tests.append(json_schema_test)
    seen: set[Path] = set()
    unique = []
    for test in tests:
        if test.is_file() and os.access(test, os.X_OK) and test not in seen:
            seen.add(test)
            unique.append(test)
    return unique


def run_correctness_tests(
    bin_dir: Path,
    llama_dir: Path,
    out_dir: Path,
    timeout_s: int,
) -> list[TestResult]:
    results: list[TestResult] = []
    test_dir = out_dir / "raw" / "correctness_tests"
    for test in discover_tests(bin_dir):
        stdout_path = test_dir / f"{test.name}.out"
        stderr_path = test_dir / f"{test.name}.err"
        rc, status, wall_s = run_process([str(test)], llama_dir, stdout_path, stderr_path, timeout_s)
        results.append(TestResult(test.name, status, rc, wall_s, stdout_path, stderr_path))
    return results


def fit_log2_slope(results: Iterable[ProbeResult]) -> tuple[float | None, float | None]:
    points = [
        (float(r.depth), math.log2(float(r.max_stacks)))
        for r in results
        if r.status == "PASS" and r.max_stacks and r.max_stacks > 0
    ]
    if len(points) < 3:
        return None, None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0:
        return None, None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / ss_xx
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r2 = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return slope, r2


def classify_growth(results: list[ProbeResult]) -> tuple[str, str]:
    timed_out = any(r.status == "TIMEOUT" for r in results)
    failed = any(r.status not in {"PASS", "TIMEOUT"} for r in results)
    slope, r2 = fit_log2_slope(results)
    if slope is None or r2 is None:
        return "INCONCLUSIVE_FAIL", "not enough completed points"
    if timed_out:
        return "EXPONENTIAL_FAIL", f"timeout plus log2_slope={slope:.3f}, r2={r2:.4f}"
    if failed:
        return "FAIL", f"probe failure plus log2_slope={slope:.3f}, r2={r2:.4f}"
    if slope >= 0.50 and r2 >= 0.95:
        return "EXPONENTIAL_FAIL", f"log2_slope={slope:.3f}, r2={r2:.4f}"
    if slope < 0.35:
        return "POLYNOMIAL_PASS", f"log2_slope={slope:.3f}, r2={r2:.4f}"
    return "INCONCLUSIVE_FAIL", f"log2_slope={slope:.3f}, r2={r2:.4f}"


def fmt_ms(us: int | None) -> str:
    if us is None:
        return "NA"
    return f"{us / 1000.0:.3f}"


def fmt_int(value: int | None) -> str:
    return "NA" if value is None else str(value)


def write_probe_csv(path: Path, rows: list[ProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "family",
            "depth",
            "input_len",
            "status",
            "returncode",
            "wall_s",
            "max_stacks",
            "total_accept_ms",
            "max_accept_ms",
            "total_reject4_ms",
            "max_reject4_ms",
            "accepts_eof",
            "maxrss_kb",
            "raw_csv",
            "raw_err",
        ])
        for r in rows:
            writer.writerow([
                r.family,
                r.depth,
                r.input_len,
                r.status,
                r.returncode,
                f"{r.wall_s:.3f}",
                fmt_int(r.max_stacks),
                fmt_ms(r.total_accept_us),
                fmt_ms(r.max_accept_us),
                fmt_ms(r.total_reject4_us),
                fmt_ms(r.max_reject4_us),
                r.accepts_eof,
                fmt_int(r.maxrss_kb),
                r.raw_csv,
                r.raw_err,
            ])


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    git_rev: str,
    probe_results: list[ProbeResult],
    test_results: list[TestResult],
) -> None:
    synthetic = [r for r in probe_results if r.family == "synthetic_ambiguous_wrappers"]
    realistic = [r for r in probe_results if r.family == "realistic_workflow_trailing_strategy"]
    synthetic_class, synthetic_detail = classify_growth(synthetic)
    realistic_class, realistic_detail = classify_growth(realistic)
    tests_pass = bool(test_results) and all(t.status == "PASS" for t in test_results)
    polynomial_pass = synthetic_class == "POLYNOMIAL_PASS" and realistic_class == "POLYNOMIAL_PASS"
    overall = "PASS" if polynomial_pass and tests_pass else "FAIL"

    lines: list[str] = []
    lines.append("# llama.cpp #4218 validation harness run")
    lines.append("")
    lines.append(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"llama.cpp dir: `{args.llama_dir}`")
    lines.append(f"build dir: `{args.build_dir}`")
    lines.append(f"git revision: `{git_rev}`")
    lines.append(f"probe timeout per case: `{args.timeout_sec}s`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"Overall validation result for a candidate fix: **{overall}**")
    lines.append("")
    lines.append("| Gate | Result | Detail |")
    lines.append("|---|---|---|")
    lines.append(f"| Synthetic polynomial-ness | {synthetic_class} | {synthetic_detail} |")
    lines.append(f"| Realistic polynomial-ness | {realistic_class} | {realistic_detail} |")
    lines.append(f"| Correctness regression tests | {'PASS' if tests_pass else 'FAIL'} | {sum(t.status == 'PASS' for t in test_results)}/{len(test_results)} test binaries passed |")
    lines.append("")
    lines.append("Interpretation: current master is expected to fail polynomial-ness. A proposed engine fix should turn the first two gates into POLYNOMIAL_PASS while keeping grammar tests passing.")
    lines.append("")
    lines.append("## Synthetic 3*2^n Family")
    lines.append("")
    lines.append('Grammar: `nest ::= "a" nest "b" | "a" nest "c" |`; input is `a^n b^n`.')
    lines.append("")
    lines.append("| n | status | input len | max stacks | total accept ms | max accept ms | wall s | max RSS KB |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in synthetic:
        lines.append(
            f"| {r.depth} | {r.status} | {r.input_len} | {fmt_int(r.max_stacks)} | "
            f"{fmt_ms(r.total_accept_us)} | {fmt_ms(r.max_accept_us)} | {r.wall_s:.3f} | {fmt_int(r.maxrss_kb)} |"
        )
    lines.append("")
    lines.append("## Realistic Recursive Workflow Schema")
    lines.append("")
    lines.append("Fixture: Pydantic-style recursive workflow schema with trailing `strategy` discriminator; generated GBNF is in `fixtures/workflow_trailing_strategy/grammar.gbnf`.")
    lines.append("")
    lines.append("| depth | status | input len | max stacks | total accept ms | max accept ms | wall s | max RSS KB |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in realistic:
        lines.append(
            f"| {r.depth} | {r.status} | {r.input_len} | {fmt_int(r.max_stacks)} | "
            f"{fmt_ms(r.total_accept_us)} | {fmt_ms(r.max_accept_us)} | {r.wall_s:.3f} | {fmt_int(r.maxrss_kb)} |"
        )
    lines.append("")
    lines.append("## Correctness Regression Tests")
    lines.append("")
    lines.append("| test binary | status | return code | wall s | stdout | stderr |")
    lines.append("|---|---|---:|---:|---|---|")
    for t in test_results:
        lines.append(
            f"| {t.name} | {t.status} | {t.returncode} | {t.wall_s:.3f} | `{t.stdout_path}` | `{t.stderr_path}` |"
        )
    lines.append("")
    lines.append("## Raw Artifacts")
    lines.append("")
    lines.append(f"- Machine-readable summary: `{out_dir / 'summary.csv'}`")
    lines.append(f"- Raw probe output root: `{out_dir / 'raw'}`")
    lines.append(f"- Synthetic grammar: `{out_dir / 'fixtures' / 'synthetic_ambiguous_wrappers.gbnf'}`")
    lines.append(f"- Realistic grammar: `{out_dir / 'fixtures' / 'workflow_trailing_strategy' / 'grammar.gbnf'}`")
    lines.append("")
    lines.append("## Reproducibility notes")
    lines.append("")
    lines.append("- Observed: every table row in this report was generated by running the supplied probe/test binary during this harness invocation.")
    lines.append("- Observed: TIMEOUT rows, if present, are real subprocess timeouts; partial max-stack data is parsed from the CSV emitted before timeout.")
    lines.append("- Unknown: this harness does not run a full model/vocabulary sampler loop; it validates grammar-engine behavior through the model-free probe and upstream grammar tests.")
    write_text(out_dir / "summary.md", "\n".join(lines) + "\n")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="llama.cpp build directory, e.g. build-cpu")
    parser.add_argument("--llama-dir", type=Path, default=None, help="llama.cpp source checkout")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for run artifacts")
    parser.add_argument("--probe-bin", type=Path, default=None, help="override path to built llamacpp_4218_probe")
    parser.add_argument("--probe-source", type=Path, default=script_dir / "helper_sources" / "llamacpp_4218_probe.cpp")
    parser.add_argument("--synthetic-depths", type=parse_range, default=parse_range("8:18"))
    parser.add_argument("--realistic-depths", type=parse_range, default=parse_range("1:13"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--test-timeout-sec", type=int, default=180)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    args.build_dir = args.build_dir.expanduser().resolve()
    if args.llama_dir is None:
        args.llama_dir = args.build_dir.parent
    args.llama_dir = args.llama_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()

    bin_dir = args.build_dir / "bin"
    if not bin_dir.exists():
        raise SystemExit(f"build bin dir not found: {bin_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_grammar = args.out_dir / "fixtures" / "synthetic_ambiguous_wrappers.gbnf"
    realistic_grammar = args.out_dir / "fixtures" / "workflow_trailing_strategy" / "grammar.gbnf"
    write_text(synthetic_grammar, 'root ::= nest\nnest ::= "a" nest "b" | "a" nest "c" |\n')
    bundled_realistic = script_dir / "fixtures" / "workflow_trailing_strategy" / "grammar.gbnf"
    if not realistic_grammar.exists():
        if bundled_realistic.exists():
            realistic_grammar.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_realistic, realistic_grammar)
        else:
            raise SystemExit(f"realistic grammar fixture missing: {bundled_realistic}")

    probe_bin = ensure_probe(args, bin_dir, args.out_dir)
    git_rev = "unknown"
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(args.llama_dir),
            text=True,
        ).strip()
    except Exception:
        pass

    print(f"llama_dir={args.llama_dir}")
    print(f"build_dir={args.build_dir}")
    print(f"out_dir={args.out_dir}")
    print(f"git_rev={git_rev}")
    print(f"probe_bin={probe_bin}")

    probe_results: list[ProbeResult] = []
    for depth in args.synthetic_depths:
        print(f"RUN synthetic depth={depth}", flush=True)
        probe_results.append(
            run_probe_case(
                probe_bin,
                args.llama_dir,
                args.out_dir,
                "synthetic_ambiguous_wrappers",
                depth,
                synthetic_grammar,
                synthetic_input(depth),
                args.timeout_sec,
            )
        )
        print(f"  -> {probe_results[-1].status} max_stacks={probe_results[-1].max_stacks} total_accept_us={probe_results[-1].total_accept_us}", flush=True)

    for depth in args.realistic_depths:
        print(f"RUN realistic depth={depth}", flush=True)
        probe_results.append(
            run_probe_case(
                probe_bin,
                args.llama_dir,
                args.out_dir,
                "realistic_workflow_trailing_strategy",
                depth,
                realistic_grammar,
                realistic_input(depth),
                args.timeout_sec,
            )
        )
        print(f"  -> {probe_results[-1].status} max_stacks={probe_results[-1].max_stacks} total_accept_us={probe_results[-1].total_accept_us}", flush=True)

    test_results: list[TestResult] = []
    if not args.skip_correctness:
        print("RUN correctness tests", flush=True)
        test_results = run_correctness_tests(bin_dir, args.llama_dir, args.out_dir, args.test_timeout_sec)
        for t in test_results:
            print(f"  -> {t.name}: {t.status} rc={t.returncode}", flush=True)

    write_probe_csv(args.out_dir / "summary.csv", probe_results)
    write_report(args.out_dir, args, git_rev, probe_results, test_results)

    synthetic_class, synthetic_detail = classify_growth([r for r in probe_results if r.family == "synthetic_ambiguous_wrappers"])
    realistic_class, realistic_detail = classify_growth([r for r in probe_results if r.family == "realistic_workflow_trailing_strategy"])
    tests_pass = bool(test_results) and all(t.status == "PASS" for t in test_results)
    overall_pass = synthetic_class == "POLYNOMIAL_PASS" and realistic_class == "POLYNOMIAL_PASS" and tests_pass

    print("")
    print("PASS/FAIL SUMMARY")
    print(f"synthetic: {synthetic_class} ({synthetic_detail})")
    print(f"realistic: {realistic_class} ({realistic_detail})")
    print(f"correctness: {'PASS' if tests_pass else 'FAIL'}")
    print(f"overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"report: {args.out_dir / 'summary.md'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
