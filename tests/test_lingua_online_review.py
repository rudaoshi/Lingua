"""Offline contracts for the online + agent review skill. No semantic oracle claimed."""
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "skills/lingua-graph/scripts"
sys.path.insert(0, str(SCRIPTS))
import lingua_service as online
spec = importlib.util.spec_from_file_location("lingua_online_review", SCRIPTS / "review_online.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
sys.path.remove(str(SCRIPTS))


def test_bundled_standard_matches_checkout():
    assert review.standard_path(None).read_bytes() == review.standard_path(REPO).read_bytes()


def graph():
    return {"meta": {}, "words": ["John", " is", " happy", "."], "graph": {
        "nodes": [{"spans": [[i, i]], "type": t} for i, t in enumerate(
            ["NominalConstant", "FactualPredicator", "ModificationalConstant", "PunctuationalConstant"])],
        "edges": [{"start": 1, "end": 0, "label": "pred.arg.1"},
                  {"start": 1, "end": 2, "label": "pred.arg.2"},
                  {"start": 3, "end": 1, "label": "as:punctuation"}]}}


def check(g):
    return review.inspect_graph(g, "John is happy.", review.load_checker(REPO))


def test_desc_preserves_wrapper_and_internal_edges():
    p = check(graph())
    assert p["program_checks_pass"]
    assert p["nodes"][3]["DESC"]["nodes"] == [0, 1, 2, 3]
    assert p["nodes"][3]["DESC"]["edges"] == [0, 1, 2]
    assert p["nodes"][2]["DESC"]["token_indexes"] == [2]
    assert p["semantic_reviewed"] is False


@pytest.mark.parametrize("kind", ["cycle", "missing", "overlap", "bad_endpoint", "boolean", "label", "duplicate", "alignment"])
def test_bad_graph_never_passes(kind):
    g = graph()
    if kind == "cycle":
        g["graph"]["edges"].append({"start": 0, "end": 3, "label": "ref"})
    elif kind == "missing":
        g["graph"]["nodes"][0]["spans"] = []
    elif kind == "overlap":
        g["graph"]["nodes"][0]["spans"] = [[0, 1]]
    elif kind == "bad_endpoint":
        g["graph"]["edges"][0]["end"] = [999]
    elif kind == "boolean":
        g["graph"]["edges"][0]["end"] = False
    elif kind == "label":
        g["graph"]["edges"][0]["label"] = "unknown"
    elif kind == "duplicate":
        g["graph"]["edges"].append(deepcopy(g["graph"]["edges"][0]))
    else:
        g["words"][0] = "Mary"
    assert not check(g)["program_checks_pass"]


def run_dir(tmp_path):
    g = graph()
    receipt = {"sentence": "John is happy.", "space_id": online.SPACE,
               "received_full_lingua": True, "graph": g, "graph_sha256": online.digest(g)}
    review.write(tmp_path / "receipt.json", receipt)
    review.write(tmp_path / "original.json", g)
    (tmp_path / "standard.md").write_bytes((REPO / review.STANDARD).read_bytes())
    review.write(tmp_path / "manifest.json", {"standard_sha256": review.sha(REPO / review.STANDARD),
                 "checker_sha256": review.sha(REPO / review.CHECKER),
                 "files": {f: review.sha(tmp_path / f) for f in ["receipt.json", "original.json", "standard.md"]}})
    return g


def synthetic_review(p):
    """Test fixture only; live reviews must be sentence-specific model judgments."""
    r = review.template(p)
    r.update(reviewer="synthetic unit-test fixture, not linguistic review", summary="Contract test")
    for side in r["reviews"].values():
        for records in side.values():
            for item in records:
                item.update(status="correct", reason="Synthetic fixture", rules=["0.1"])
                if "assessment" in item:
                    item["assessment"] = "Synthetic fixture"
    return r


def test_empty_review_cannot_finalize(tmp_path):
    g = run_dir(tmp_path)
    p = review.packet(tmp_path, REPO, g)
    with pytest.raises(ValueError):
        review.finalize(tmp_path, REPO, g, review.template(p))
    assert not (tmp_path / "corrected.json").exists()


@pytest.mark.parametrize("tamper", ["omit_edge", "stale_hash", "fake_rule", "edit_without_reason", "words"])
def test_evidence_and_review_binding(tmp_path, tamper):
    g = run_dir(tmp_path)
    p = review.packet(tmp_path, REPO, g)
    r = synthetic_review(p)
    if tamper == "omit_edge":
        r["reviews"]["candidate"]["edges"].pop()
    elif tamper == "stale_hash":
        r["candidate_sha256"] = "0" * 64
    elif tamper == "fake_rule":
        r["reviews"]["candidate"]["nodes"][0]["rules"] = ["999"]
    elif tamper == "edit_without_reason":
        g["graph"]["edges"][0]["label"] = "pred.arg.2"
        r = synthetic_review(review.packet(tmp_path, REPO, g))
    else:
        g["words"][0] = "Mary"
    with pytest.raises(ValueError):
        review.finalize(tmp_path, REPO, g, r)


def test_raw_evidence_tamper_rejected(tmp_path):
    g = run_dir(tmp_path)
    (tmp_path / "original.json").write_text("{}")
    with pytest.raises(ValueError, match="Evidence changed"):
        review.packet(tmp_path, REPO, g)


def test_ambiguity_returns_candidate_but_no_corrected_graph(tmp_path):
    g = run_dir(tmp_path)
    r = synthetic_review(review.packet(tmp_path, REPO, g))
    r["unresolved"] = ["Synthetic ambiguity for status gate"]
    result = review.finalize(tmp_path, REPO, g, r)
    assert result["status"] == "needs_human_decision"
    assert result["corrected_graph"] is None
    assert result["candidate_graph"] == g
    assert not (tmp_path / "corrected.json").exists()


def test_blocking_program_error_cannot_be_accepted_by_reviewer(tmp_path):
    g = run_dir(tmp_path)
    # Frozen input is intentionally malformed, with a real schema error.
    g["graph"]["nodes"][2]["type"] = "NotALinguaType"
    receipt = review.read(tmp_path / "receipt.json")
    receipt.update(graph=g, graph_sha256=online.digest(g))
    review.write(tmp_path / "receipt.json", receipt)
    review.write(tmp_path / "original.json", g)
    manifest = review.read(tmp_path / "manifest.json")
    manifest["files"].update({f: review.sha(tmp_path / f) for f in ["receipt.json", "original.json"]})
    review.write(tmp_path / "manifest.json", manifest)
    r = synthetic_review(review.packet(tmp_path, REPO, g))
    with pytest.raises(ValueError, match="blocking program"):
        review.finalize(tmp_path, REPO, g, r)
    r["unresolved"] = ["Unknown type remains, irrespective of positive model judgments"]
    result = review.finalize(tmp_path, REPO, g, r)
    assert result["status"] == "needs_human_decision"
    assert result["corrected_graph"] is None


def test_finalize_idempotent_and_no_overwrite(tmp_path):
    g = run_dir(tmp_path)
    r = synthetic_review(review.packet(tmp_path, REPO, g))
    result = review.finalize(tmp_path, REPO, g, r)
    before = (tmp_path / "result.json").read_bytes()
    assert result["status"] == "reviewed_unchanged"
    review.finalize(tmp_path, REPO, g, r)
    assert before == (tmp_path / "result.json").read_bytes()
    r["summary"] = "Different review"
    with pytest.raises(ValueError, match="different contents"):
        review.finalize(tmp_path, REPO, g, r)


def test_sse_heartbeat_multiline_and_truncation():
    events = []
    data = b'event: heartbeat\ndata: null\n\nevent: complete\ndata: [null,\ndata: "OK", {}]\n\n'
    assert online.completion(io.BytesIO(data), time.monotonic() + 1, events) == [None, "OK", {}]
    assert [e["event"] for e in events] == ["heartbeat", "complete"]
    for bad in [b'event: complete\ndata: []\n', b'event: error\ndata: "failed"\n\n']:
        with pytest.raises(ValueError):
            online.completion(io.BytesIO(bad), time.monotonic() + 1, [])


class FakeOpener:
    def __init__(self, outputs, contract_changed=False, fail_submit=False):
        self.outputs, self.contract_changed, self.fail_submit = outputs, contract_changed, fail_submit
        self.posts = 0

    def open(self, request, timeout):
        url = request.full_url
        if url.startswith("https://huggingface.co/api/spaces/"):
            obj = {"id": online.SPACE, "sha": "a" * 40, "host": "https://rudaoshi-lang2logic.hf.space"}
        elif url.endswith("/config"):
            obj = {"api_prefix": "/changed" if self.contract_changed else "/gradio_api"}
        elif url.endswith("/info"):
            obj = {"named_endpoints": {"/process_sentence": {
                "parameters": [{"type": {"type": "string"}}],
                "returns": [{"component": c} for c in ["Image", "Textbox", "Json"]]}}}
        elif request.data:
            self.posts += 1
            assert online.strict_json(request.data) == {"data": ["John is happy."]}
            if self.fail_submit:
                raise TimeoutError("Submission response lost")
            obj = {"event_id": "job1"}
        else:
            return io.BytesIO(("event: complete\ndata: " + review.json.dumps(self.outputs) + "\n\n").encode())
        return io.BytesIO(review.json.dumps(obj).encode())


@pytest.mark.parametrize("status,expected", [("Graph generated successfully!", True),
    ("Failed to generate visualization.", True), ("Warning: Showing Word-Lingua graph.", False),
    ("Error: parse failed", False)])
def test_service_contract_preserves_raw_outputs(monkeypatch, status, expected):
    outputs = [None, status, graph()]
    opener = FakeOpener(outputs)
    monkeypatch.setattr(online, "build_opener", lambda *_: opener)
    result = online.fetch("John is happy.")
    assert result["received_full_lingua"] is expected
    assert result["raw_outputs"] == outputs
    assert result["semantic_reviewed"] is False
    assert opener.posts == 1


def test_no_post_retry_and_changed_contract(monkeypatch):
    for changed in [False, True]:
        opener = FakeOpener(None, contract_changed=changed, fail_submit=True)
        monkeypatch.setattr(online, "build_opener", lambda *_: opener)
        result = online.fetch("John is happy.")
        assert not result["received_full_lingua"]
        assert opener.posts == (0 if changed else 1)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        online.strict_json(text)
