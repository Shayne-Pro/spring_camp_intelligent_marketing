import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


def load_preprocessor():
    """Load only the actual notebook function, without executing training cells."""
    notebook = Path(__file__).resolve().parents[1] / '智慧营销_baseline.ipynb'
    for cell in json.loads(notebook.read_text())['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and 'def preprocess_data' in source:
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_data')
            namespace = {'np': np, 'pd': pd}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(notebook), 'exec'), namespace)
            return namespace['preprocess_data']
    raise AssertionError('Notebook preprocessor not found')


class CountTests(unittest.TestCase):
    def test_counts_preserve_numbers_and_handle_buckets_and_unknowns(self):
        frame = pd.DataFrame({
            'publish_time': ['20250101'] * 3,
            'update_time': ['20250102'] * 3,
            'fans_cnt': [500, '小于100', None],
            'coin_cnt': ['200', '小于100', 'unknown'],
            'video_cnt': [2, 3, 1],
        })
        actual = load_preprocessor()(frame)
        self.assertEqual(actual['fans_cnt'].tolist(), [500, 50, 0])
        self.assertEqual(actual['coin_cnt'].tolist(), [200, 50, 0])


if __name__ == '__main__':
    unittest.main()
