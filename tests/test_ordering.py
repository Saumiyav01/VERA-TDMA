import unittest

from vera_tdma.ordering import natural_sort_key, natural_sorted


class NaturalOrderingTests(unittest.TestCase):
    def test_sorts_numeric_segments_naturally(self):
        self.assertEqual(
            natural_sorted(["Node10", "Node1", "Node2", "Node20"]),
            ["Node1", "Node2", "Node10", "Node20"],
        )

    def test_order_is_case_sensitive_and_deterministic(self):
        values = ["node2", "Node2", "NODE2", "Node10", "Node02"]
        self.assertEqual(
            natural_sorted(values),
            ["NODE2", "Node2", "Node02", "Node10", "node2"],
        )
        self.assertEqual(natural_sorted(values), natural_sorted(values))

    def test_rejects_non_string_identifiers(self):
        with self.assertRaisesRegex(TypeError, "must be strings"):
            natural_sort_key(2)


if __name__ == "__main__":
    unittest.main()
