import tempfile
import unittest
import warnings
from pathlib import Path

from vera_tdma.coordinates import (
    CoLocatedNodesWarning,
    CoordinateLoadError,
    load_coordinates,
)


class CoordinateLoaderTests(unittest.TestCase):
    def write_file(self, directory, name, content):
        path = Path(directory) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_json_and_returns_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "nodes.json",
                '{"Node10": [10, 20], "Node2": [2.5, -3], "node1": [1, 1]}',
            )
            nodes = load_coordinates(path)

        self.assertEqual([node.node_id for node in nodes], ["Node2", "Node10", "node1"])
        self.assertEqual(nodes[0].coordinates_m, (2.5, -3.0))

    def test_loads_csv_and_parses_numeric_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(
                directory,
                "nodes.csv",
                "node_id,x_coordinate,y_coordinate\nNode10,10.5,20\nNode2,2,-3.25\n",
            )
            nodes = load_coordinates(path)

        self.assertEqual([node.node_id for node in nodes], ["Node2", "Node10"])
        self.assertEqual(nodes[1].coordinates_m, (10.5, 20.0))

    def test_can_select_format_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(directory, "nodes.data", '{"n1": [1, 2]}')
            self.assertEqual(load_coordinates(path, "JSON")[0].node_id, "n1")
            with self.assertRaisesRegex(CoordinateLoadError, "format must be"):
                load_coordinates(path, "xml")

    def test_warns_for_co_located_nodes_and_keeps_both(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_file(directory, "nodes.json", '{"n1": [1, 2], "n2": [1, 2]}')
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                nodes = load_coordinates(path)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(emitted), 1)
        self.assertIs(emitted[0].category, CoLocatedNodesWarning)
        self.assertIn("n1, n2", str(emitted[0].message))

    def test_rejects_malformed_or_invalid_json_nodes(self):
        invalid_documents = [
            "[]",
            '{"n1": [1]}',
            '{"n1": [1, 2, 3]}',
            '{"n1": ["1", 2]}',
            '{"n1": [true, 2]}',
            '{"n1": [1, NaN]}',
            '{"": [1, 2]}',
            '{"n1": [1, 2], "n1": [3, 4]}',
            "{",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, document in enumerate(invalid_documents):
                with self.subTest(document=document):
                    path = self.write_file(directory, f"invalid{index}.json", document)
                    with self.assertRaises(CoordinateLoadError):
                        load_coordinates(path)

    def test_rejects_invalid_csv_headers_rows_values_and_duplicate_ids(self):
        invalid_documents = [
            "id,x,y\nn1,1,2\n",
            "node_id,x_coordinate,y_coordinate\nn1,1\n",
            "node_id,x_coordinate,y_coordinate\nn1,nope,2\n",
            "node_id,x_coordinate,y_coordinate\nn1,1,2\nn1,3,4\n",
            "node_id,x_coordinate,y_coordinate\n,1,2\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, document in enumerate(invalid_documents):
                with self.subTest(document=document):
                    path = self.write_file(directory, f"invalid{index}.csv", document)
                    with self.assertRaises(CoordinateLoadError):
                        load_coordinates(path)

    def test_rejects_unsupported_extension_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CoordinateLoadError, "format must be"):
                load_coordinates(Path(directory) / "nodes.txt")
            with self.assertRaisesRegex(CoordinateLoadError, "could not read coordinate file"):
                load_coordinates(Path(directory) / "missing.json")


if __name__ == "__main__":
    unittest.main()
