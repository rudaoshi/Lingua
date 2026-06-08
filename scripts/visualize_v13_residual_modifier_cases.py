#!/usr/bin/env python3
"""Visualize selected V13 residual modifier gold/pred graph pairs."""

from __future__ import annotations

import argparse
import json
import html
from pathlib import Path
from typing import Any

from lingua.structure.gpgraph import GPGraph, GPGraphVisualizer


TARGETS = [
    ("train", 138, "myeloid_cell"),
    ("train", 4259, "binding_sites"),
    ("train", 7719, "home_win"),
    ("test", 540, "most_men"),
    ("test", 938, "lymphocyte_activation"),
]


def load_record(prediction_dump_dir: Path, split: str, line_no: int) -> dict[str, Any]:
    path = prediction_dump_dir / split / "predictions.jsonl"
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("split") == split and int(record.get("line", -1)) == line_no:
                return record
    raise KeyError(f"Missing {split}:{line_no} in {path}")


def graph_from_record(record: dict[str, Any], key: str) -> GPGraph:
    graph_json = record.get(key)
    if graph_json is None:
        raise ValueError(f"Record {record.get('split')}:{record.get('line')} has no {key}")
    return GPGraph.parse(graph_json)


def metric_brief(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "metric: n/a"
    return (
        f"exact={metric.get('exact_same')} "
        f"node_f1={metric.get('node_f1'):.6f} "
        f"edge_f1={metric.get('edge_f1'):.6f}"
    )


def render_graph(
    visualizer: GPGraphVisualizer,
    record: dict[str, Any],
    graph_key: str,
    out_path: Path,
    fmt: str,
) -> None:
    graph = graph_from_record(record, graph_key)
    visualizer.visualize(graph, file_name=str(out_path), format=fmt)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_dump_dir",
        type=Path,
        default=Path("reports/v13_prediction_dump"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "reports/modification_scope_layer_inconsistency_after_simple_leaf_norm/"
            "gold_pred_visualizations"
        ),
    )
    parser.add_argument("--format", default="svg", choices=["svg", "png"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    visualizer = GPGraphVisualizer()
    manifest: list[dict[str, Any]] = []

    for split, line_no, slug in TARGETS:
        record = load_record(args.prediction_dump_dir, split, line_no)
        case_id = f"{split}_{line_no}_{slug}"
        case_dir = args.out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        graph_files: dict[str, str] = {}
        for key in [
            "gold_lingua_graph",
            "pred_lingua_graph",
            "gold_word_lingua_graph",
            "pred_word_lingua_graph",
        ]:
            out_path = case_dir / f"{key}.{args.format}"
            render_graph(visualizer, record, key, out_path, args.format)
            graph_files[key] = str(out_path.relative_to(args.out_dir))

        record_summary = {
            "case_id": case_id,
            "split": split,
            "line": line_no,
            "sentence": record.get("sentence"),
            "lingua_metric": record.get("lingua_metric"),
            "word_lingua_metric": record.get("word_lingua_metric"),
            "gold_word_lingua_error": record.get("gold_word_lingua_error"),
            "lingua_conversion_error": record.get("lingua_conversion_error"),
            "graph_files": graph_files,
        }
        write_json(case_dir / "summary.json", record_summary)
        manifest.append(record_summary)

    write_json(args.out_dir / "manifest.json", manifest)

    lines = [
        "# V13 residual modifier gold/pred 可视化",
        "",
        "来源：`reports/v13_prediction_dump/{train,dev,test}/predictions.jsonl`。",
        "",
    ]
    for item in manifest:
        case_id = item["case_id"]
        lines.extend(
            [
                f"## {case_id}",
                "",
                f"- sample: `{item['split']}:{item['line']}`",
                f"- sentence: {item['sentence']}",
                f"- Lingua: `{metric_brief(item['lingua_metric'])}`",
                f"- Word-Lingua: `{metric_brief(item['word_lingua_metric'])}`",
                f"- gold Lingua: [{case_id}/gold_lingua_graph.{args.format}]"
                f"({case_id}/gold_lingua_graph.{args.format})",
                f"- pred Lingua: [{case_id}/pred_lingua_graph.{args.format}]"
                f"({case_id}/pred_lingua_graph.{args.format})",
                f"- gold Word-Lingua: [{case_id}/gold_word_lingua_graph.{args.format}]"
                f"({case_id}/gold_word_lingua_graph.{args.format})",
                f"- pred Word-Lingua: [{case_id}/pred_word_lingua_graph.{args.format}]"
                f"({case_id}/pred_word_lingua_graph.{args.format})",
                f"- summary: [{case_id}/summary.json]({case_id}/summary.json)",
                "",
            ]
        )

    (args.out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    html_lines = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>V13 residual modifier gold/pred 可视化</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#202124;background:#fafafa}",
        "h1{font-size:24px;margin:0 0 16px}",
        "h2{font-size:18px;margin:32px 0 8px}",
        ".sentence{max-width:1100px;line-height:1.45;color:#3c4043}",
        ".metric{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#5f6368}",
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:14px;margin-top:12px}",
        ".panel{border:1px solid #dadce0;background:white;padding:10px;overflow:auto}",
        ".panel h3{font-size:13px;margin:0 0 8px;color:#3c4043}",
        ".panel img{max-width:none;background:white}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>V13 residual modifier gold/pred 可视化</h1>",
        "<p>来源：<code>reports/v13_prediction_dump/{train,dev,test}/predictions.jsonl</code></p>",
    ]
    panel_specs = [
        ("gold_lingua_graph", "gold Lingua"),
        ("pred_lingua_graph", "pred Lingua"),
        ("gold_word_lingua_graph", "gold Word-Lingua"),
        ("pred_word_lingua_graph", "pred Word-Lingua"),
    ]
    for item in manifest:
        case_id = html.escape(item["case_id"])
        html_lines.extend(
            [
                f"<section>",
                f"<h2>{case_id}</h2>",
                f"<p class=\"sentence\"><code>{html.escape(item['split'])}:{item['line']}</code> "
                f"{html.escape(item['sentence'] or '')}</p>",
                f"<p class=\"metric\">Lingua {html.escape(metric_brief(item['lingua_metric']))}<br>"
                f"Word-Lingua {html.escape(metric_brief(item['word_lingua_metric']))}</p>",
                '<div class="grid">',
            ]
        )
        for key, title in panel_specs:
            src = html.escape(item["graph_files"][key])
            html_lines.extend(
                [
                    '<div class="panel">',
                    f"<h3>{html.escape(title)}</h3>",
                    f'<img src="{src}" alt="{html.escape(title)}">',
                    "</div>",
                ]
            )
        html_lines.extend(["</div>", "</section>"])
    html_lines.extend(["</body>", "</html>"])
    (args.out_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "case_count": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
