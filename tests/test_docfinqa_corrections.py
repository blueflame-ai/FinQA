"""Regression checks for the opt-in corrected dataset; no model calls or network."""

import ast
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("corrections", ROOT / "code/apply_docfinqa_corrections.py")
corrections = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corrections)
MANIFEST_PATH = corrections.DEFAULT_MANIFEST
MANIFEST, PATCHES = corrections.load_manifest(MANIFEST_PATH)


class ManifestTests(unittest.TestCase):
    def test_provenance_matches_original_finqa(self) -> None:
        originals = {row["id"]: row for row in json.loads((ROOT / "dataset/train.json").read_text())}
        self.assertEqual(len(PATCHES), 109)
        self.assertEqual(len({patch["id"] for patch in PATCHES.values()}), 109)
        for patch in PATCHES.values():
            with self.subTest(case=patch["id"]):
                original = originals[patch["finqa_id"]]
                evidence = patch["finqa_evidence"]
                self.assertEqual(evidence["table"], original["table_ori"])
                self.assertEqual(evidence["gold_inds"], original["qa"]["gold_inds"])
                self.assertEqual(evidence["original_answer"], original["qa"]["answer"])
                self.assertEqual(evidence["original_execution_answer"], original["qa"]["exe_ans"])
                self.assertEqual(set(patch["corrected"]), {"Question", "Answer", "Program"})
                self.assertEqual(len(patch["original"]["context_sha256"]), 64)

    def test_programs_compute_canonical_answers(self) -> None:
        for patch in PATCHES.values():
            with self.subTest(case=patch["id"]):
                tree = ast.parse(patch["corrected"]["Program"])
                # Only numeric arithmetic, comparison and round are allowed.
                allowed = (ast.Module, ast.Assign, ast.Name, ast.Store, ast.Load,
                           ast.Call, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare,
                           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.Gt)
                self.assertTrue(all(isinstance(node, allowed) for node in ast.walk(tree)))
                self.assertTrue(all(node.id in {"answer", "round"} for node in ast.walk(tree) if isinstance(node, ast.Name)))
                namespace = {}
                exec(compile(tree, "reviewed-reference", "exec"), {"__builtins__": {"round": round}}, namespace)
                value = namespace["answer"]
                answer = patch["corrected"]["Answer"]
                expression = ast.parse(patch["calculation"], mode="eval").body
                program_expression = tree.body[0].value
                self.assertEqual(
                    ast.dump(expression),
                    ast.dump(program_expression if patch["unit"] == "boolean" else program_expression.args[0]),
                )
                if patch["unit"] == "boolean":
                    self.assertEqual(answer, "yes" if value else "no")
                else:
                    self.assertEqual(answer.endswith("%"), patch["unit"] == "%")
                    self.assertEqual(program_expression.args[1].value, patch["decimal_places"])
                    self.assertEqual(answer.rstrip("%"), f"{value:.{patch['decimal_places']}f}")
                    self.assertAlmostEqual(float(answer.rstrip("%")), value, places=8)
                    unrounded = eval(compile(ast.Expression(expression), "reviewed-calculation", "eval"), {"__builtins__": {}})
                    if abs(unrounded - value) > 1e-9:
                        self.assertRegex(patch["corrected"]["Question"], r"(?i)\bround\b[^.?!]*\bto\b|\bnearest\b")

    def test_replacement_reports_and_coverage(self) -> None:
        self.assertEqual(sum("replacement" in p["context"] for p in PATCHES.values()), 13)
        self.assertEqual(sum("insert" in p["context"] for p in PATCHES.values()), 25)
        for name, source in MANIFEST["sources"].items():
            with self.subTest(source=name):
                data = gzip.decompress((MANIFEST_PATH.parent / source["path"]).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), source["text_sha256"])
                self.assertGreater(len(data), 100000)
                self.assertIn("\f", data.decode("utf-8"))
                self.assertTrue(source["url"].startswith("https://"))
        text = gzip.decompress((MANIFEST_PATH.parent / MANIFEST["sources"]["GS-2014"]["path"]).read_bytes()).decode()
        self.assertIn("35,764", text)
        self.assertIn("73,597", text)
        text = gzip.decompress((MANIFEST_PATH.parent / MANIFEST["sources"]["NTAP-2014"]["path"]).read_bytes()).decode()
        self.assertIn("197.58", text)
        self.assertIn("236.12", text)

    def test_known_reference_regressions(self) -> None:
        answers = {patch["id"]: patch["corrected"]["Answer"] for patch in PATCHES.values()}
        expected = {
            "docfin_01f7b28ff978f900": "152",
            "docfin_886496abadadb680": "2.2",
            "docfin_c9b033272177530e": "5.9",
            "docfin_eaf9d77c5e8ae58f": "1.103",
            "docfin_38f474542292d2b7": "30146",
            "docfin_cf59a45f3e23c122": "4538.7",
            "docfin_b2776d525dab44e0": "1171400",
            "docfin_e18ea9da36bb4fbf": "0.022441",
        }
        for cid, answer in expected.items():
            self.assertEqual(answers[cid], answer)
        # Genuine wrong-table/entity model failures receive no answer patch.
        for prefix in ("118159", "66180", "ad563", "e1b668", "d51f", "d879", "f19dd"):
            self.assertFalse(any(cid.startswith("docfin_" + prefix) for cid in answers))


class ApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patch = deepcopy(next(p for p in PATCHES.values() if "insert" in p["context"]))
        self.anchor = self.patch["context"]["insert"].get("after") or self.patch["context"]["insert"]["before"]
        self.context = "Previous unrelated disclosure.\n" + self.anchor + "Next unrelated disclosure.\n"
        self.patch["original"]["context_sha256"] = corrections.sha256(self.context)
        self.record = {field: self.patch["original"][field] for field in ("Question", "Answer", "Program")}
        self.record["Context"] = self.context

    def test_insertion_preserves_other_content_and_input(self) -> None:
        original = deepcopy(self.record)
        result = corrections.apply_record(self.record, self.patch, MANIFEST, MANIFEST_PATH.parent)
        self.assertEqual(self.record, original)
        self.assertEqual(result["Answer"], self.patch["corrected"]["Answer"])
        self.assertIn(self.patch["context"]["insert"]["text"], result["Context"])
        self.assertIn("Previous unrelated disclosure.", result["Context"])
        self.assertIn("Next unrelated disclosure.", result["Context"])

    def test_rejects_reference_and_context_drift(self) -> None:
        for field in ("Context", "Question", "Answer", "Program"):
            with self.subTest(field=field):
                bad = {**self.record, field: self.record[field] + " drift"}
                with self.assertRaises(ValueError):
                    corrections.apply_record(bad, self.patch, MANIFEST, MANIFEST_PATH.parent)

    def test_rejects_duplicate_or_missing_anchor(self) -> None:
        for context in ("no anchor", self.context + self.anchor):
            patch = deepcopy(self.patch)
            patch["original"]["context_sha256"] = corrections.sha256(context)
            with self.assertRaises(ValueError):
                corrections.apply_record({**self.record, "Context": context}, patch, MANIFEST, MANIFEST_PATH.parent)

    def test_required_evidence_is_checked_after_repair(self) -> None:
        patch = deepcopy(self.patch)
        patch["required_context"] = ["disclosure that is not in the document"]
        with self.assertRaisesRegex(ValueError, "required source evidence"):
            corrections.apply_record(self.record, patch, MANIFEST, MANIFEST_PATH.parent)
        patch["required_context"] = [" ".join(patch["context"]["insert"]["text"].split())]
        corrections.apply_record(self.record, patch, MANIFEST, MANIFEST_PATH.parent)

    def test_rejects_conflicting_manifest_repairs(self) -> None:
        invalid_repairs = [
            {"replacement": "GS-2014", "insert": self.patch["context"]["insert"]},
            {"insert": {"after": "a", "before": "b", "text": "c"}},
            {"insert": {"after": "", "text": "c"}},
            {"unknown": "value"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            for repair in invalid_repairs:
                patch = {**self.patch, "context": repair}
                path.write_text(json.dumps({**MANIFEST, "corrections": [patch]}))
                with self.assertRaises(ValueError):
                    corrections.load_manifest(path)

    def test_replacement_uses_full_verified_source(self) -> None:
        patch = deepcopy(next(p for p in PATCHES.values() if "replacement" in p["context"]))
        record = {field: patch["original"][field] for field in ("Question", "Answer", "Program")}
        record["Context"] = "Wrong-year source"
        patch["original"]["context_sha256"] = corrections.sha256(record["Context"])
        result = corrections.apply_record(record, patch, MANIFEST, MANIFEST_PATH.parent)
        source = MANIFEST["sources"][patch["context"]["replacement"]]
        self.assertEqual(corrections.sha256(result["Context"]), source["text_sha256"])
        bad = deepcopy(MANIFEST)
        bad["sources"][patch["context"]["replacement"]]["text_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            corrections.apply_record(record, patch, bad, MANIFEST_PATH.parent)

    def test_file_application_is_complete_atomic_and_preserves_controls(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({**MANIFEST, "corrections": [self.patch]}))
            source, output = root / "in.jsonl", root / "out.jsonl"
            control = {"Question": "Unchanged control", "Answer": "7", "Program": "", "Context": "Other filing"}
            source.write_text(json.dumps(self.record) + "\n" + json.dumps(control) + "\n")
            counts = corrections.apply_file(source, output, manifest)
            self.assertEqual(counts["corrected"], 1)
            self.assertEqual(json.loads(output.read_text().splitlines()[1]), control)
            good = output.read_bytes()
            for records in ([control], [self.record, self.record], [{**self.record, "Context": "drift"}]):
                source.write_text("\n".join(json.dumps(row) for row in records))
                with self.assertRaises(ValueError):
                    corrections.apply_file(source, output, manifest)
                self.assertEqual(output.read_bytes(), good)
                self.assertEqual(list(root.glob(".out.jsonl.*")), [])


if __name__ == "__main__":
    unittest.main()
