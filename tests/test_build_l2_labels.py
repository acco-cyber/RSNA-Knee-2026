"""Critical combiner invariants; all examples are synthetic, not competition data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("l2", Path(__file__).parents[1] / "scripts/build_l2_labels.py")
l2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2)


class L2InvariantTests(unittest.TestCase):
    def table(self, value, ids=("weak",)):
        return pd.DataFrame(value, index=pd.Index(ids, name=l2.UID), columns=l2.LABELS)

    def test_unmentioned_and_missing_annotation_are_exact_abstentions(self):
        b, teacher, family = self.table(.8), self.table(.2), self.table(.4)
        mp, mq = self.table(np.nan), self.table(0.0)
        target, _, _ = l2.combine(b, teacher, [family], mp, mq)
        alpha = np.array([l2.REPORT_SHARE[c] for c in l2.LABELS])
        np.testing.assert_allclose(target[0], alpha*.8+(1-alpha)*.3)

    def test_new_annotation_is_one_confidence_weighted_seventh_vote(self):
        b, teacher, family = self.table(.8), self.table(.2), self.table(.4)
        target, _, audit = l2.combine(b, teacher, [family], self.table(.95), self.table(.9))
        report = (6*.8+.9*.95)/(6+.9)
        np.testing.assert_allclose(audit["report"], report)
        self.assertAlmostEqual(target[0, 0], .6*report+.4*.3)

    def test_missing_targets_get_zero_loss_weight(self):
        missing = self.table(np.nan)
        target, weight, _ = l2.combine(missing, missing, [], missing, self.table(0.0))
        np.testing.assert_array_equal(target, .5)
        np.testing.assert_array_equal(weight, 0.0)

    def test_teacher_failure_sentinel_cannot_be_training_evidence(self):
        missing = self.table(np.nan)
        _, weight, audit = l2.combine(missing, self.table(.5), [], missing, self.table(0.0))
        self.assertTrue(audit["teacher_failed"][0])
        np.testing.assert_array_equal(weight, 0.0)

    def test_correlated_folds_are_not_duplicate_oof_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            tr = self.table(np.nan)
            df = self.table(.4).assign(set="val")
            df.to_csv(directory / "oof_r2_synthetic_f0.csv")
            with self.assertRaisesRegex(ValueError, "frozen A1 fold"):
                l2.load_round2(directory, tr, pd.Series(1, index=tr.index))

    def test_evidence_and_uncertainty_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b000.json"
            tr = self.table(np.nan).assign(Report="ACL intact. No fracture.")
            findings = {c: {"state": "unmentioned", "probability": None,
                            "confidence": 0.0, "evidence": ""} for c in l2.LABELS}
            findings["ACL"] = {"state": "negative", "probability": .05,
                                "confidence": .9, "evidence": "ACL intact."}
            payload = {"batch_id": "b000", "annotations": [{l2.UID: "weak", "findings": findings}]}
            path.write_text(json.dumps(payload))
            self.assertEqual(len(l2.annotation_file(path, tr, None)), 12)
            findings["ACL"]["evidence"] = "ACL completely torn"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "literal report span"):
                l2.annotation_file(path, tr, None)
            findings["ACL"] = {"state": "uncertain", "probability": .5,
                                "confidence": .9, "evidence": "ACL intact."}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "confidence <=.40"):
                l2.annotation_file(path, tr, None)

    def test_gold_studies_cannot_enter_report_annotation_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b000.json"
            tr = self.table(0, ids=("gold",)).assign(Report="Normal examination.")
            path.write_text(json.dumps({"batch_id": "b000", "annotations": [{l2.UID: "gold", "findings": {}}]}))
            with self.assertRaisesRegex(ValueError, "report-only training pool"):
                l2.annotation_file(path, tr, None)


if __name__ == "__main__":
    unittest.main()
