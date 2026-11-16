#!/usr/bin/env python3
"""Fetch -> inspect SELF/DESC -> finalize an agent-reviewed Lingua candidate.

Python checks evidence, graph form and review completeness. It does NOT perform
linguistic judgment. A completed review is an attributed model judgment, not a
proof of semantic correctness.
"""
import argparse
from collections import Counter, deque
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

from lingua_service import SPACE, digest, fetch, strict_json

SKILL_DIR = Path(__file__).resolve().parents[1]
STANDARD = "docs/standard.md"
CHECKER = "skills/lingua-graph/scripts/validate_graph.py"
DIMENSIONS = ("semantic_fidelity", "predicate_arguments", "entity_structure",
              "modifier_and_negation_scope", "reference_and_coordination",
              "expression_and_punctuation", "fragments_and_accepted_variants")
VERDICTS = {"correct", "correct_variant", "minor", "major", "blocker", "needs_human_decision"}
ACCEPTED = {"correct", "correct_variant"}


def read(path):
    return strict_json(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkout_root():
    if SKILL_DIR.parent.name == "skills":
        root = SKILL_DIR.parent.parent
        if (root / STANDARD).is_file():
            return root
    return None


def standard_path(repo):
    return repo / STANDARD if repo is not None else SKILL_DIR / "references/lingua-standard.md"


def checker_path(repo):
    return repo / CHECKER if repo is not None else SKILL_DIR / "scripts/validate_graph.py"


def load_checker(repo):
    spec = importlib.util.spec_from_file_location("lingua_standard_checker", checker_path(repo))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokens(node, count):
    result = []
    spans = node.get("spans")
    if isinstance(spans, list):
        for span in spans:
            if (isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span)
                    and 0 <= span[0] <= span[1] < count):
                result.extend(range(span[0], span[1] + 1))
    return result


def inspect_graph(lingua, sentence, checker):
    """Bound input shapes before invoking the existing semantic-shape checker."""
    require(isinstance(lingua, dict), "Lingua must be an object")
    words, graph = lingua.get("words"), lingua.get("graph")
    require(isinstance(words, list) and 0 < len(words) <= 2048
            and all(isinstance(w, str) and w.strip() for w in words), "Invalid words array")
    require(isinstance(graph, dict), "Missing graph object")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    require(isinstance(nodes, list) and 0 < len(nodes) <= 512
            and all(isinstance(n, dict) for n in nodes), "Invalid nodes array (limit 512)")
    require(isinstance(edges, list) and len(edges) <= 4096
            and all(isinstance(e, dict) for e in edges), "Invalid edges array (limit 4096)")
    findings = []

    def issue(code, location, message):
        findings.append({"code": code, "location": location, "message": message, "severity": "error"})

    if re.sub(r"\s", "", "".join(words)) != re.sub(r"\s", "", sentence):
        issue("SENTENCE_ALIGNMENT", "words", "Non-whitespace characters differ from exact input")
    coverage = Counter()
    for i, node in enumerate(nodes):
        spans = node.get("spans")
        if not isinstance(node.get("type"), str) or not isinstance(spans, (str, list)):
            issue("NODE_SHAPE", f"nodes[{i}]", "Node requires string type and spans")
            continue
        if isinstance(spans, list):
            valid = tokens(node, len(words))
            if not spans or len(valid) != sum(
                s[1] - s[0] + 1 for s in spans if isinstance(s, list) and len(s) == 2
                and all(type(v) is int for v in s) and s[1] >= s[0]
            ) or any(not (isinstance(s, list) and len(s) == 2 and all(type(v) is int for v in s)
                          and 0 <= s[0] <= s[1] < len(words)) for s in spans):
                issue("SPAN_SHAPE", f"nodes[{i}]", "Invalid inclusive token span")
            coverage.update(valid)
    missing = [i for i in range(len(words)) if coverage[i] == 0]
    overlaps = [i for i in range(len(words)) if coverage[i] > 1]
    if missing or overlaps:
        issue("TOKEN_PARTITION", "nodes", f"Missing tokens {missing}; overlapping tokens {overlaps}")
    adjacency = [[] for _ in nodes]
    incoming = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    seen, valid_edges = set(), []
    for i, edge in enumerate(edges):
        s, t, label = edge.get("start"), edge.get("end"), edge.get("label")
        if not (type(s) is int and type(t) is int and 0 <= s < len(nodes)
                and 0 <= t < len(nodes) and isinstance(label, str)):
            issue("EDGE_SHAPE", f"edges[{i}]", "Invalid endpoint or label")
            continue
        key = (s, t, label)
        if key in seen:
            issue("DUPLICATE_EDGE", f"edges[{i}]", "Duplicate edge")
        seen.add(key)
        valid_edges.append((i, s, t))
        adjacency[s].append(t)
        incoming[t].append(i)
        indegree[t] += 1
    roots = [i for i, d in enumerate(indegree) if d == 0]
    if len(roots) != 1:
        issue("ROOT_COUNT", "graph", f"Expected one root, found {roots}")
    if "root" in graph and (len(roots) != 1 or graph["root"] != roots[0]):
        issue("ROOT_MISMATCH", "graph.root", "Declared root differs from inferred root")
    degrees, queue, processed = indegree[:], deque(roots), 0
    while queue:
        i = queue.popleft()
        processed += 1
        for child in adjacency[i]:
            degrees[child] -= 1
            if degrees[child] == 0:
                queue.append(child)
    if processed != len(nodes):
        issue("CYCLE", "graph", "Directed cycle prevents complete DAG traversal")
    # 'var' is an explicitly accepted alias (standard 3.4). Do not mutate input.
    normalized = deepcopy(lingua)
    for e in normalized["graph"]["edges"]:
        if e.get("label") == "var":
            e["label"] = "variable"
    if not findings:
        try:
            findings.extend(checker.validate_graph(None, {"sentence": sentence, "lingua": normalized}))
        except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
            issue("CHECKER_FAILURE", "graph", type(exc).__name__ + ": " + str(exc))
    for i, finding in enumerate(findings):
        finding["id"] = f"F{i}"
    views = []
    for i, node in enumerate(nodes):
        reachable, todo = set(), [i]
        while todo:
            j = todo.pop()
            if j not in reachable:
                reachable.add(j)
                todo.extend(adjacency[j])
        covered = sorted({t for j in reachable for t in tokens(nodes[j], len(words))})
        own = tokens(node, len(words))
        views.append({"index": i, "SELF": {"type": node.get("type"), "spans": node.get("spans"),
                      "text": "".join(words[t] for t in own).strip() if own else node.get("spans")},
                      "DESC": {"nodes": sorted(reachable),
                               "edges": [k for k, s, t in valid_edges if s in reachable and t in reachable],
                               "token_indexes": covered,
                               "surface_in_sentence_order": "".join(words[t] for t in covered).strip()},
                      "incoming_edges": incoming[i]})
    return {"graph_sha256": digest(lingua), "roots": roots, "nodes": views,
            "edges": [{"index": i, **e} for i, e in enumerate(edges)], "findings": findings,
            "program_checks_pass": not any(f.get("severity") != "warning" for f in findings),
            "semantic_reviewed": False}


def evidence(run, repo):
    manifest = read(run / "manifest.json")
    for filename, expected in manifest["files"].items():
        require(sha(run / filename) == expected, "Evidence changed: " + filename)
    require(sha(standard_path(repo)) == manifest["standard_sha256"], "Standard changed; start a new review run")
    require(sha(checker_path(repo)) == manifest["checker_sha256"], "Checker changed; start a new review run")
    receipt, original = read(run / "receipt.json"), read(run / "original.json")
    require(receipt.get("received_full_lingua"), "No full Lingua response; inspect receipt.json")
    require(digest(original) == receipt["graph_sha256"] == digest(receipt["graph"]), "Raw graph mismatch")
    return manifest, receipt, original


def differences(before, after):
    changes = []
    for field in ("nodes", "edges"):
        a, b = before["graph"][field], after["graph"][field]
        for i in range(max(len(a), len(b))):
            old, new = a[i] if i < len(a) else None, b[i] if i < len(b) else None
            if old != new:
                changes.append({"path": f"/graph/{field}/{i}", "before": old, "after": new})
    return changes


def packet(run, repo, candidate):
    manifest, receipt, original = evidence(run, repo)
    require(isinstance(candidate, dict), "Candidate must contain the full Lingua object")
    require({k: v for k, v in original.items() if k != "graph"} ==
            {k: v for k, v in candidate.items() if k != "graph"}, "Do not change words/meta or other source fields")
    require({k: v for k, v in original["graph"].items() if k not in ("nodes", "edges")} ==
            {k: v for k, v in candidate["graph"].items() if k not in ("nodes", "edges")},
            "Only graph nodes/edges may change")
    checker = load_checker(repo)
    return {"sentence": receipt["sentence"], "standard_sha256": manifest["standard_sha256"],
            "original": inspect_graph(original, receipt["sentence"], checker),
            "candidate": inspect_graph(candidate, receipt["sentence"], checker),
            "changes": differences(original, candidate)}


def template(p):
    def record(index):
        return {"index": index, "status": "", "assessment": "", "reason": "", "rules": []}
    result = {"standard_sha256": p["standard_sha256"], "original_sha256": p["original"]["graph_sha256"],
              "candidate_sha256": p["candidate"]["graph_sha256"], "reviewer": "", "summary": "",
              "issues": [], "unresolved": [], "edit_reasons": [], "reviews": {}}
    for side in ("original", "candidate"):
        result["reviews"][side] = {
            "nodes": [record(n["index"]) for n in p[side]["nodes"]],
            "edges": [record(e["index"]) for e in p[side]["edges"]],
            "dimensions": [{"name": name, "status": "", "reason": "", "rules": []} for name in DIMENSIONS],
            "findings": [{"id": f["id"], "status": "", "reason": "", "rules": []}
                         for f in p[side]["findings"]]}
    return result


def validate_review(review, p, standard_text):
    sections = set(re.findall(r"^#{1,4}\s+(\d+(?:\.\d+)*)(?:\.|\s)", standard_text, re.M))
    for k, expected in (("standard_sha256", p["standard_sha256"]),
                        ("original_sha256", p["original"]["graph_sha256"]),
                        ("candidate_sha256", p["candidate"]["graph_sha256"])):
        require(review.get(k) == expected, "Stale or missing review hash: " + k)
    for key in ("reviewer", "summary"):
        require(isinstance(review.get(key), str) and review[key].strip(), "Fill " + key)

    def rules(record):
        refs = record.get("rules")
        require(isinstance(refs, list) and refs and all(isinstance(r, str) and r in sections for r in refs),
                "Cite existing standard section numbers in rules")
        require(isinstance(record.get("reason"), str) and record["reason"].strip(), "Review reason required")

    candidate_accepted = True
    for side in ("original", "candidate"):
        reviews = review.get("reviews", {}).get(side, {})
        for kind, index_key in (("nodes", "index"), ("edges", "index"), ("dimensions", "name"), ("findings", "id")):
            records = reviews.get(kind)
            expected = (set(DIMENSIONS) if kind == "dimensions" else
                        {x[index_key] for x in p[side][kind]})
            require(isinstance(records, list) and len(records) == len(expected)
                    and {x.get(index_key) for x in records if isinstance(x, dict)} == expected,
                    f"Review every {side} {kind} exactly once")
            for record in records:
                require(record.get("status") in VERDICTS, "Invalid or blank review status")
                rules(record)
                if kind in ("nodes", "edges"):
                    require(isinstance(record.get("assessment"), str) and record["assessment"].strip(),
                            "Describe SELF/DESC meaning or required edge unit")
                if side == "candidate" and record["status"] not in ACCEPTED:
                    candidate_accepted = False
    issues = review.get("issues")
    require(isinstance(issues, list), "issues must be a list")
    issue_ids = set()
    for item in issues:
        require(isinstance(item.get("id"), str) and item["id"] and item["id"] not in issue_ids,
                "Issue IDs must be nonempty and unique")
        issue_ids.add(item["id"])
        require(item.get("severity") in ("minor", "major", "blocker"), "Invalid issue severity")
        for field in ("location", "explanation", "resolution"):
            require(isinstance(item.get(field), str) and item[field].strip(), "Issue requires " + field)
        rules(item)
    edits = review.get("edit_reasons")
    paths = {change["path"] for change in p["changes"]}
    require(isinstance(edits, list) and len(edits) == len(paths)
            and {e.get("path") for e in edits if isinstance(e, dict)} == paths,
            "Every changed node/edge must have one edit_reasons entry")
    for edit in edits:
        require(isinstance(edit.get("issue_ids"), list) and edit["issue_ids"]
                and set(edit["issue_ids"]) <= issue_ids, "Link each edit to a reported issue")
        rules(edit)
    unresolved = review.get("unresolved")
    require(isinstance(unresolved, list) and all(isinstance(u, str) and u.strip() for u in unresolved),
            "unresolved must be a list of explicit uncertainty descriptions")
    require(candidate_accepted or unresolved, "Explain all remaining candidate issues in unresolved")
    require(p["candidate"]["program_checks_pass"] or unresolved,
            "Explain blocking program findings in unresolved")
    return candidate_accepted and not unresolved


def finalize(run, repo, candidate, review):
    p = packet(run, repo, candidate)
    semantic_accepted = validate_review(review, p, (run / "standard.md").read_text())
    manifest, receipt, original = evidence(run, repo)
    accepted = semantic_accepted and p["candidate"]["program_checks_pass"]
    status = ("reviewed_repaired" if p["changes"] else "reviewed_unchanged") if accepted else "needs_human_decision"
    result = {"schema_version": "lingua-parse-validator.online/1", "status": status, "sentence": p["sentence"],
              "standard_sha256": p["standard_sha256"], "checker_sha256": manifest["checker_sha256"],
              "space_id": receipt["space_id"], "space_revision": receipt.get("space_revision"),
              "model_checkpoint_attested": False, "original_graph": original,
              "candidate_graph": candidate, "corrected_graph": candidate if accepted else None,
              "original_sha256": digest(original), "candidate_sha256": digest(candidate),
              "program_checks_pass": p["candidate"]["program_checks_pass"],
              "semantic_review": {"method": "agent_SELF_DESC_review", "reviewer": review["reviewer"],
                                  "accepted": semantic_accepted, "independent_certification": False},
              "changes": p["changes"], "inspection": p, "review": review}
    final = run / "result.json"
    if final.exists():
        require(read(final) == result, "Completed result exists with different contents; use a new run")
    write(final, result)
    if accepted:
        write(run / "corrected.json", {"sentence": p["sentence"], "lingua": candidate})
    report = ["# Lingua 在线结果校验与修正", "", p["sentence"], "", f"状态：`{status}`", "",
              review["summary"], "", f"程序检查：{p['candidate']['program_checks_pass']}；"
              f"逐项语义审查者：{review['reviewer']}。语义判断未作独立认证。", "",
              "## 修改与依据", ""]
    for item in review["issues"]:
        report.append(f"- {item['id']} [{item['severity']}] {item['location']}：{item['explanation']} "
                      f"处理：{item['resolution']}（标准 {', '.join(item['rules'])}）。")
    if not review["issues"]:
        report.append("无需修改；已保留可接受的结构变体。")
    report += ["", "## 未解决项", ""] + (review["unresolved"] or ["本次审查未发现待定项；不代表通用语义正确性证明。"])
    report += ["", "## 交付文件", "", "- `receipt.json`：真实在线调用与服务原始返回。",
               "- `original.json`：服务原图。", "- `result.json`：完整图、逐项审查、修改与缺口。",
               "- `standard.md`、`manifest.json`：本次标准快照与文件哈希。"]
    if accepted:
        report.append("- `corrected.json`：可继续使用的句子及完整 Lingua 图。")
    (run / "REPORT.md").write_text("\n".join(report) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=checkout_root(),
                        help="Use a Lingua checkout explicitly. Otherwise use the enclosing checkout or bundled files.")
    sub = parser.add_subparsers(dest="command", required=True)
    call = sub.add_parser("fetch")
    source = call.add_mutually_exclusive_group(required=True)
    source.add_argument("--sentence")
    source.add_argument("--sentence-file", type=Path)
    call.add_argument("--space", default=SPACE)
    call.add_argument("--timeout", type=float, default=150)
    call.add_argument("--out", type=Path, required=True)
    for name in ("inspect", "finalize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run", type=Path, required=True)
        cmd.add_argument("--candidate", type=Path)
        if name == "finalize":
            cmd.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            sentence = args.sentence if args.sentence is not None else args.sentence_file.read_text()
            require(sentence.strip(), "Empty sentence")
            source_standard = standard_path(args.repo)
            standard_bytes = source_standard.read_bytes()
            checker_hash = sha(checker_path(args.repo))
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "standard.md").write_bytes(standard_bytes)
            receipt = fetch(sentence, args.space, args.timeout)
            receipt["standard_sha256"] = sha(args.out / "standard.md")
            receipt["checker_sha256"] = checker_hash
            write(args.out / "receipt.json", receipt)
            files = ["receipt.json", "standard.md"]
            if receipt["received_full_lingua"]:
                write(args.out / "original.json", receipt["graph"])
                files.append("original.json")
            write(args.out / "manifest.json", {"standard_source": str(source_standard),
                  "standard_sha256": sha(args.out / "standard.md"), "checker_sha256": checker_hash,
                  "files": {f: sha(args.out / f) for f in files}})
            if not receipt["received_full_lingua"]:
                write(args.out / "result.json", {"status": "service_failed", "corrected_graph": None,
                      "sentence": sentence, "error": receipt.get("error"), "semantic_reviewed": False,
                      "standard_sha256": receipt["standard_sha256"]})
                print(json.dumps({"status": "service_failed", "receipt": str(args.out / "receipt.json")}))
                return 2
            print(json.dumps({"status": "fetched_pending_review", "run": str(args.out)}))
        else:
            original = read(args.run / "original.json")
            candidate = read(args.candidate) if args.candidate else original
            if args.command == "inspect":
                p = packet(args.run, args.repo, candidate)
                write(args.run / "inspection.json", p)
                write(args.run / "review.template.json", template(p))
                print(json.dumps({"status": "pending_agent_review", "changes": len(p["changes"]),
                                  "program_checks_pass": p["candidate"]["program_checks_pass"]}))
            else:
                result = finalize(args.run, args.repo, candidate, read(args.review))
                print(json.dumps({"status": result["status"], "result": str(args.run / "result.json")}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
