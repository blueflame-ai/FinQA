"""Apply the reviewed DocFinQA v1 overlay to an upstream JSONL export."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "dataset/corrections/docfinqa-v1/corrections.json"
)
FIELDS = {"Context", "Question", "Program", "Answer"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    corrections = {}
    ids = set()
    for correction in manifest["corrections"]:
        cid = correction["id"]
        if cid in ids:
            raise ValueError("Duplicate correction ID in manifest")
        ids.add(cid)
        repair = correction["context"]
        if set(repair) not in (set(), {"replacement"}, {"insert"}):
            raise ValueError(f"{cid}: expected at most one context repair")
        if "insert" in repair:
            insertion = repair["insert"]
            if set(insertion) not in ({"after", "text"}, {"before", "text"}):
                raise ValueError(f"{cid}: expected exactly one insertion direction")
            if any(not isinstance(value, str) or not value for value in insertion.values()):
                raise ValueError(f"{cid}: insertion values must be nonempty strings")
        excerpts = correction["required_context"]
        if not isinstance(excerpts, list) or any(not isinstance(value, str) or not value for value in excerpts):
            raise ValueError(f"{cid}: required context must be a list of nonempty strings")
        question = correction["original"]["Question"]
        if question in corrections:
            raise ValueError("Duplicate original question in correction manifest")
        corrections[question] = correction
    return manifest, corrections


def apply_record(record: dict, correction: dict, manifest: dict, root: Path) -> dict:
    """Fail closed on upstream drift; do not mutate the caller's record."""
    original = correction["original"]
    if set(record) != FIELDS:
        raise ValueError("Expected exactly Context, Question, Program and Answer")
    for field in ("Question", "Program", "Answer"):
        if record[field] != original[field]:
            raise ValueError(f"{correction['id']}: original {field} differs")
    if sha256(record["Context"]) != original["context_sha256"]:
        raise ValueError(f"{correction['id']}: original Context SHA-256 differs")

    context = record["Context"]
    repair = correction["context"]
    if "replacement" in repair:
        source = manifest["sources"][repair["replacement"]]
        data = gzip.decompress((root / source["path"]).read_bytes())
        if hashlib.sha256(data).hexdigest() != source["text_sha256"]:
            raise ValueError(f"{correction['id']}: replacement source SHA-256 differs")
        context = data.decode("utf-8")
    elif "insert" in repair:
        insertion = repair["insert"]
        direction = "after" if "after" in insertion else "before"
        anchor = insertion[direction]
        if context.count(anchor) != 1:
            raise ValueError(f"{correction['id']}: source anchor is not unique")
        new = anchor + insertion["text"] if direction == "after" else insertion["text"] + anchor
        context = context.replace(anchor, new, 1)

    normalized = " ".join(context.replace("\\t", " ").replace("\\n", " ").split())
    for excerpt in correction["required_context"]:
        if excerpt not in normalized:
            raise ValueError(f"{correction['id']}: required source evidence is missing")

    return {**record, **correction["corrected"], "Context": context}


def apply_file(input_path: Path, output_path: Path, manifest_path: Path) -> dict:
    """Write atomically only after every manifest entry matched exactly once."""
    manifest, corrections = load_manifest(manifest_path)
    seen = set()
    count = 0
    temporary = None
    try:
        with input_path.open(encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_path.parent,
                prefix=f".{output_path.name}.", delete=False,
            ) as destination:
                temporary = Path(destination.name)
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict) or set(record) != FIELDS:
                        raise ValueError(f"Line {line_number}: expected the four DocFinQA fields")
                    correction = corrections.get(record["Question"])
                    if correction:
                        cid = correction["id"]
                        if cid in seen:
                            raise ValueError(f"{cid}: duplicate input record")
                        record = apply_record(record, correction, manifest, manifest_path.parent)
                        seen.add(cid)
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        missing = {entry["id"] for entry in corrections.values()} - seen
        if missing:
            raise ValueError(f"Input is missing {len(missing)} correction records: {sorted(missing)}")
        os.replace(temporary, output_path)
        return {"version": manifest["version"], "records": count, "corrected": len(seen)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Pinned upstream train export as JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Corrected JSONL; replaced only on success")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    print(json.dumps(apply_file(args.input, args.output, args.manifest)))


if __name__ == "__main__":
    main()
