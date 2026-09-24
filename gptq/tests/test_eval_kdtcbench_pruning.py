import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_kdtcbench


class KdtcbenchPruningTest(unittest.TestCase):
    def test_resolve_drop_layers_uses_recorded_order(self):
        self.assertEqual(eval_kdtcbench.resolve_drop_layers(0), [])
        self.assertEqual(eval_kdtcbench.resolve_drop_layers(2), [31, 30])
        self.assertEqual(eval_kdtcbench.resolve_drop_layers(4), [31, 30, 34, 32])

    def test_default_out_tag_includes_model_and_prune_k(self):
        self.assertEqual(eval_kdtcbench.default_out_tag("gptq", 0), "gptq")
        self.assertEqual(eval_kdtcbench.default_out_tag("gptq", 2), "gptq_prune_k2")
        self.assertEqual(eval_kdtcbench.default_out_tag("fp16", 4), "fp16_prune_k4")


if __name__ == "__main__":
    unittest.main()
