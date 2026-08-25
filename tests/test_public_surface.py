import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicSurfaceTests(unittest.TestCase):
    def test_expected_public_files_exist(self):
        for relative in (
            "README.md",
            "LICENSE",
            "NOTICE",
            "CITATION.cff",
            "SECURITY.md",
            "schemas/work-order.schema.json",
            "schemas/result.schema.json",
            "schemas/review-bundle.schema.json",
            "scripts/run_static_qa.py",
            "scripts/run_synthetic_benchmark.py",
            "scripts/build_review_bundle.py",
            "scripts/scan_publication_boundary.py",
            "scripts/fresh_clone_smoke.py",
            "publication-manifest.json",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        github = ROOT / ".github"
        issue_templates = github / "ISSUE_TEMPLATE"
        for relative in (
            github / "workflows" / "qa.yml",
            github / "labels.yml",
            issue_templates / "config.yml",
            issue_templates / "bug_report.yml",
            issue_templates / "docs.yml",
            issue_templates / "inbound_feedback.yml",
        ):
            self.assertTrue(relative.is_file(), relative.relative_to(ROOT).as_posix())

    def test_manifest_is_self_excluding_and_safe(self):
        manifest = json.loads((ROOT / "publication-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["self_hash_policy"], "excluded-from-entries")
        self.assertIsNone(manifest["manifest_sha256"])
        self.assertNotIn("publication-manifest.json", {item["path"] for item in manifest["entries"]})
        self.assertNotIn("review-bundle.json", {item["path"] for item in manifest["entries"]})
        self.assertEqual(manifest["excluded_generated_outputs"], ["review-bundle.json"])
        self.assertTrue(all(item["public_safe_decision"] == "pass-deterministic-only" for item in manifest["entries"]))
