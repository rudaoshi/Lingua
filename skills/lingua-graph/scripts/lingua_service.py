"""Public HF Lingua transport. No model inference or semantic repair locally.

POST /call + SSE protocol, adapted from ToLogicV3's lingua_service.py.
Keep a received malformed graph as evidence: the review stage diagnoses it.
"""
from datetime import datetime, timezone
import hashlib
import json
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

SPACE = "rudaoshi/lang2logic"
LIMIT = 8 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key: " + k)
            result[k] = v
        return result

    def constant(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Unexpected HTTP redirect; no redirect followed")


def completion(response, deadline, events):
    event, data, size = "", [], 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Inference deadline reached")
        raw = response.readline(LIMIT + 1)
        size += len(raw)
        if size > LIMIT:
            raise ValueError("SSE response too large")
        if not raw:
            raise ValueError("SSE ended before completion")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line:
            if event:
                events.append({"event": event, "data": "\n".join(data)})
                if event == "error":
                    raise ValueError("Service returned error event; see events")
                if event == "complete":
                    return strict_json("\n".join(data))
            event, data = "", []


def fetch(sentence, space=SPACE, timeout=150):
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Provide one nonempty exact sentence")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", space):
        raise ValueError("Space must be owner/name")
    if not 0 < timeout <= 600:
        raise ValueError("Timeout must be in (0, 600]")
    deadline = time.monotonic() + timeout
    opener = build_opener(NoRedirect())
    receipt = {"sentence": sentence, "space_id": space, "events": [],
               "requested_at_utc": datetime.now(timezone.utc).isoformat(),
               "received_full_lingua": False, "semantic_reviewed": False,
               "model_checkpoint_attested": False}

    def request(url, payload=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Request deadline reached")
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        headers = {"User-Agent": "Lingua-Parse-Validator/1"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        return opener.open(Request(url, data=data, headers=headers), timeout=min(30, remaining))

    def get_json(url, payload=None):
        with request(url, payload) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("JSON response too large")
        return strict_json(raw.decode("utf-8"))

    try:
        meta = get_json("https://huggingface.co/api/spaces/" + space)
        receipt["space_metadata"] = meta
        if meta.get("id") != space or not re.fullmatch(r"[0-9a-f]{40}", meta.get("sha", "")):
            raise ValueError("Space identity or app revision missing")
        host = meta.get("host", "").rstrip("/")
        url = urlsplit(host)
        if (url.scheme != "https" or not url.hostname or not url.hostname.endswith(".hf.space")
                or url.username or url.password or url.port or url.path or url.query or url.fragment):
            raise ValueError("Expected a plain HTTPS hf.space host")
        receipt.update(host=host, space_revision=meta["sha"])
        if meta.get("runtime", {}).get("stage") == "SLEEPING":
            with request(host + "/") as response:
                response.read(LIMIT + 1)
        config = get_json(host + "/config")
        receipt["config"] = config
        if config.get("api_prefix") != "/gradio_api":
            raise ValueError("Gradio API prefix changed")
        info = get_json(host + "/gradio_api/info")
        receipt["api_info"] = info
        api = info.get("named_endpoints", {}).get("/process_sentence", {})
        params, returns = api.get("parameters", []), api.get("returns", [])
        if (len(params) != 1 or params[0].get("type", {}).get("type") != "string"
                or [r.get("component", "").lower() for r in returns] != ["image", "textbox", "json"]):
            raise ValueError("process_sentence input/output contract changed")
        endpoint = host + "/gradio_api/call/process_sentence"
        # Never retry the submission automatically: it may have been accepted.
        job = get_json(endpoint, {"data": [sentence]})
        receipt["job"] = job
        event_id = job.get("event_id")
        if not isinstance(event_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", event_id):
            raise ValueError("Invalid Gradio event ID")
        with request(endpoint + "/" + event_id) as response:
            outputs = completion(response, deadline, receipt["events"])
        receipt["raw_outputs"] = outputs
        if not isinstance(outputs, list) or len(outputs) != 3 or not isinstance(outputs[1], str):
            raise ValueError("Expected image/status/graph output triple")
        receipt["service_status"] = outputs[1]
        if outputs[1].lower().startswith("error:") or "word-lingua" in outputs[1].lower():
            raise ValueError("Parser error or Word-Lingua fallback; not a full Lingua graph")
        graph = strict_json(outputs[2]) if isinstance(outputs[2], str) else outputs[2]
        if not isinstance(graph, dict):
            raise ValueError("Full Lingua graph absent")
        receipt.update(received_full_lingua=True, graph=graph, graph_sha256=digest(graph))
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return receipt
