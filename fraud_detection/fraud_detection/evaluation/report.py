"""Writes a per-run markdown + JSON metrics report, replacing the ad hoc
`print()`/`plt.show()` pattern in the original notebook (cell 15)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Union


def write_report(
    metrics: Dict,
    output_dir: Union[str, Path],
    run_name: str = "run",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    md_path = output_dir / f"{run_name}.md"
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [f"# {run_name}", "", f"Generated: {generated_at}", "", "| metric | value |", "|---|---|"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return md_path
