from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scan_publication_boundary import _find_issues, build_manifest  # noqa: E402


class PublicationScanTests(unittest.TestCase):
    def test_high_entropy_candidate_is_reported(self):
        parts = ("Ab7Kz2Qp9Lm4Xr8V", "t1Nc6Yw3Gh0Jf5Ds")
        candidate_value = "".join(parts)
        issues = _find_issues("candidate.txt", candidate_value.encode("utf-8"))
        self.assertIn("high-entropy credential candidate", issues)

    def test_sha256_hex_is_not_reported_as_high_entropy_credential(self):
        digest = "0123456789abcdef" * 4
        issues = _find_issues("hash.txt", digest.encode("ascii"))
        self.assertNotIn("high-entropy credential candidate", issues)

    def test_obviously_private_repository_url_is_reported_without_a_denylist(self):
        private_url = "https://" + "github.com/example/" + "private-project"
        issues = _find_issues("candidate.txt", private_url.encode("utf-8"))
        self.assertIn("private repository URL", issues)

    def test_default_review_bundle_is_excluded_from_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "review-bundle.json").write_text("{}\n", encoding="utf-8")
            (root / "README.md").write_text("public\n", encoding="utf-8")
            manifest, _ = build_manifest(root)
            paths = {entry["path"] for entry in manifest["entries"]}
            self.assertNotIn("review-bundle.json", paths)
            self.assertEqual(manifest["excluded_generated_outputs"], ["review-bundle.json"])
