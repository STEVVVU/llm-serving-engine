"""Dependency-free workload/oracle checks; native generation still needs GPU preflight."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "benchmark_serving.py"
spec = importlib.util.spec_from_file_location("benchmark_serving", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkContractTests(unittest.TestCase):
    def test_rotating_orders_cover_each_arm_once(self):
        orders = [benchmark.trial_order(repeat) for repeat in range(5)]
        self.assertEqual(len(set(orders)), 5)
        for order in orders:
            self.assertCountEqual(order, benchmark.SIDES)
        for side in benchmark.SIDES:
            positions = [order.index(side) for order in orders]
            self.assertEqual(set(positions), {0, 1, 2})
            self.assertLessEqual(max(positions.count(i) for i in range(3)), 2)

    def test_eos_stopping_is_disabled_without_suppressing_eos_logits(self):
        options = benchmark.generation_options(32, 151643)
        self.assertIsNone(options["eos_token_id"])
        self.assertIsNone(options["forced_eos_token_id"])
        self.assertFalse(options["do_sample"])
        self.assertEqual(options["num_beams"], 1)
        self.assertEqual(options["min_new_tokens"], options["max_new_tokens"])
        self.assertEqual(options["cache_implementation"], "dynamic")
        # EOS-valued tokens are ordinary oracle tokens under the ignore-EOS policy.
        benchmark.assert_tokens([[151645, 7]], [[151645, 7]], 2)
        with self.assertRaisesRegex(AssertionError, "length/token mismatch"):
            benchmark.assert_tokens([[151645]], [[151645, 7]], 2)

    def test_every_request_and_token_is_checked(self):
        for actual in ([[1, 2]], [[1, 2], [3, 5]], [[1, 2], [3]]):
            with self.assertRaises(AssertionError):
                benchmark.assert_tokens(actual, [[1, 2], [3, 4]], 2)


if __name__ == "__main__":
    unittest.main()
