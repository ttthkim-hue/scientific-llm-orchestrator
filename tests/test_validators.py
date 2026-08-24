from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scientific_llm_orchestrator.validators import (  # noqa: E402
    check_path_scope,
    check_protected_literals,
    check_validation_allowlist,
)


class ValidatorTests(unittest.TestCase):
    def test_path_scope_accepts_relative_and_rejects_escape(self):
        self.assertTrue(check_path_scope(["src/example.py", "docs/guide.md"]).passed)
        self.assertFalse(check_path_scope(["../outside.py"]).passed)
        self.assertFalse(check_path_scope(["C" + ":/outside.py"]).passed)

    def test_protected_literal_is_exact(self):
        self.assertTrue(check_protected_literals(["F_DEP", "f_dep"], "F_DEP and f_dep").passed)
        self.assertFalse(check_protected_literals(["F_DEP"], "f_dep").passed)

    def test_validation_allowlist_is_fail_closed(self):
        self.assertTrue(check_validation_allowlist(["schema", "unit"]).passed)
        self.assertFalse(check_validation_allowlist(["execute_shell"]).passed)
