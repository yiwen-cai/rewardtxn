"""JSONL records retain valid Unicode separators within string values."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
with patch.dict(sys.modules, {'torch': object()}):
    from ablation_evaluate import read_jsonl


class JsonlTests(unittest.TestCase):
    def test_unicode_separators_are_content(self):
        rows=[{'text':'a\u2028b\u2029c\u0085d'}, {'text':'next'}]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'data.jsonl'
            path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
            self.assertEqual(read_jsonl(path),rows)

    def test_malformed_record_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'data.jsonl'
            path.write_text('{"text":"unterminated\n')
            with self.assertRaises(json.JSONDecodeError): read_jsonl(path)


if __name__=='__main__': unittest.main()
