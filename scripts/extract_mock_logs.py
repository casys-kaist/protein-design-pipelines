import argparse
import os
import re
from collections import defaultdict

EXEC_RE = re.compile(r"Executing in (\S+):")
PREFIX_RE = re.compile(r"^\[[^\]]*\]\[[^\]]*\]\[[^\]]*\] - ?")


def strip_prefix(line: str) -> str:
    """Remove timestamp and logger prefix from a log line."""
    return PREFIX_RE.sub("", line)


def extract_logs(path: str) -> dict[str, list[str]]:
    """Return mapping of container names to captured log lines."""
    logs: dict[str, list[str]] = defaultdict(list)
    current = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = EXEC_RE.search(line)
            if m:
                current = m.group(1)
                logs.setdefault(current, [])
                continue
            if current is None:
                continue
            if "Executing in" in line and EXEC_RE.search(line):
                current = EXEC_RE.search(line).group(1)
                logs.setdefault(current, [])
                continue
            if "urllib3.connectionpool" in line:
                continue
            if "Worker" in line or "Received" in line or "Experiment configuration" in line:
                continue
            logs[current].append(strip_prefix(line))
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract container logs from run.log")
    parser.add_argument("run_log", help="Path to run.log")
    parser.add_argument("--output-dir", default=os.path.join("tests", "logs"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logs = extract_logs(args.run_log)
    for container, lines in logs.items():
        short = container.split("_", 1)[-1]
        out_path = os.path.join(args.output_dir, f"{short}.log")
        with open(out_path, "w", encoding="utf-8") as out:
            for l in lines:
                out.write(l + "\n")
        print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
