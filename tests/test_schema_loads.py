from pathlib import Path

import yaml


def test_canonical_schema_loads_and_has_constraints():
    schema_path = Path('config/canonical_schema.yaml')
    data = yaml.safe_load(schema_path.read_text(encoding='utf-8'))

    fields = data.get('fields', [])
    assert isinstance(fields, list)
    assert 80 <= len(fields) <= 140

    field_ids = [f['field_id'] for f in fields]
    assert len(field_ids) == len(set(field_ids)), 'field_id must be unique'

    required_core_count = sum(1 for f in fields if f.get('required_core') is True)
    assert required_core_count >= 40

    sectors = {f.get('sector') for f in fields}
    assert {'all', 'bank', 'insurance'}.issubset(sectors)
