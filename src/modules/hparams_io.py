"""Write changed training / model parameters back into the `hyper_params` dict of hparams.py.

Only the source text of the changed values is replaced (located via the AST), so comments, formatting and all
other entries stay untouched. Expressions such as `d_base * 2` are only replaced when the evaluated value
differs. Changing `d_model` (which is `d_base`) updates `d_base` instead, so the derived values stay consistent.

Before writing, a backup `hparams.py.bak` is created and the result is validated (re-parse + exec + value check).
"""
import ast
import os
import shutil
from pathlib import Path

from modules.train_params import ALLOWED_KEYS, coerce_param

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'hparams.py'


class HparamsSaveError(ValueError):
    """Raised with a message (and optional per-field errors); the file is left unchanged."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or {}


def _line_offsets(source_bytes: bytes):
    offsets = [0]
    for line in source_bytes.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _span(node, offsets):
    return (offsets[node.lineno - 1] + node.col_offset, offsets[node.end_lineno - 1] + node.end_col_offset)


def _exec_namespace(source: str, path) -> dict:
    ns = {'__name__': 'hparams_probe', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), ns)  # noqa: S102 - the user's own hparams file
    return ns


def read_hparams(path=None) -> dict:
    """Evaluate hparams.py from disk and return a copy of its `hyper_params` dict (always the current file content)."""
    path = Path(path) if path is not None else DEFAULT_PATH
    ns = _exec_namespace(path.read_text(encoding='utf-8'), path)
    return dict(ns['hyper_params'])


def _format_scalar(value):
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    return repr(value)


def _format_value(value, original_text: str, col_offset: int) -> str:
    if isinstance(value, list):
        rows = [f"[{_format_scalar(a)}, {_format_scalar(b)}]" for a, b in value]
        if '\n' in original_text and rows:
            indent = ' ' * (col_offset + 1)
            return '[' + (',\n' + indent).join(rows) + ',\n' + indent + ']'
        return '[' + ', '.join(rows) + ']'
    return _format_scalar(value)


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def save_hparams(changes: dict, path=None, make_backup: bool = True) -> dict:
    """Apply `changes` ({key: value}) to hparams.py. Returns {'changed': {...}, 'unchanged': [...], 'backup': str|None}.

    Raises HparamsSaveError (a ValueError) on unknown keys, invalid values or an invalid result.
    """
    path = Path(path) if path is not None else DEFAULT_PATH

    errors = {}
    new_values = {}
    for key, raw in (changes or {}).items():
        if key not in ALLOWED_KEYS:
            errors[key] = 'this parameter cannot be saved to hparams.py'
            continue
        try:
            new_values[key] = coerce_param(key, raw)
        except ValueError as exc:
            errors[key] = str(exc)
    if errors:
        raise HparamsSaveError('Invalid parameters: ' + '; '.join(f'{k}: {v}' for k, v in errors.items()), errors)

    original_text = path.read_text(encoding='utf-8')
    source_bytes = original_text.encode('utf-8')
    tree = ast.parse(original_text)
    offsets = _line_offsets(source_bytes)

    dict_node = None
    d_base_node = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if stmt.targets[0].id == 'hyper_params' and isinstance(stmt.value, ast.Dict):
                dict_node = stmt.value
            elif stmt.targets[0].id == 'd_base' and isinstance(stmt.value, ast.Constant):
                d_base_node = stmt.value
    if dict_node is None:
        raise HparamsSaveError('hyper_params dict not found in hparams.py')

    value_nodes = {}
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            value_nodes[key_node.value] = value_node
    missing = [k for k in new_values if k not in value_nodes]
    if missing:
        raise HparamsSaveError(
            'Keys not present in hyper_params: ' + ', '.join(missing), {k: 'not found in hparams.py' for k in missing}
        )

    ns = _exec_namespace(original_text, path)
    current = ns['hyper_params']
    probe_ns = dict(ns)

    def segment(node):
        start, end = _span(node, offsets)
        return source_bytes[start:end].decode('utf-8'), start, end

    edits = []  # (start, end, replacement)
    changed = {}
    unchanged = []

    # d_model is `d_base` in the default file: change d_base so derived values (latent_dim, ff_dim) follow.
    if 'd_model' in new_values and d_base_node is not None:
        text, _, _ = segment(value_nodes['d_model'])
        if text.strip() == 'd_base' and not _same(current['d_model'], new_values['d_model']):
            start, end = _span(d_base_node, offsets)
            edits.append((start, end, str(int(new_values['d_model']))))
            probe_ns['d_base'] = int(new_values['d_model'])
            changed['d_model'] = new_values['d_model']

    for key, new_value in new_values.items():
        if key in changed:
            continue
        node = value_nodes[key]
        text, start, end = segment(node)
        try:
            evaluated = eval(compile(ast.Expression(body=node), str(path), 'eval'), probe_ns)  # noqa: S307
        except Exception:
            evaluated = current[key]
        if _same(evaluated, new_value):
            unchanged.append(key)
            continue
        edits.append((start, end, _format_value(new_value, text, node.col_offset)))
        changed[key] = new_value

    # Keys whose expression follows a changed d_base but were not requested: report as derived.
    derived = {}
    if 'd_base' in probe_ns and probe_ns['d_base'] != ns.get('d_base'):
        for key, node in value_nodes.items():
            if key in new_values or key not in ALLOWED_KEYS:
                continue
            text, _, _ = segment(node)
            if 'd_base' in text:
                derived[key] = eval(compile(ast.Expression(body=node), str(path), 'eval'), probe_ns)  # noqa: S307

    if not edits:
        return {'changed': {}, 'unchanged': sorted(unchanged), 'derived': {}, 'backup': None}

    new_bytes = source_bytes
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        new_bytes = new_bytes[:start] + replacement.encode('utf-8') + new_bytes[end:]
    new_text = new_bytes.decode('utf-8')

    # Validate before touching the file.
    try:
        ast.parse(new_text)
        new_ns = _exec_namespace(new_text, path)
    except Exception as exc:
        raise HparamsSaveError(f'The modified hparams.py would be invalid ({exc}); nothing was written')
    result = new_ns['hyper_params']
    for key, expected in new_values.items():
        if not _same(result[key], expected):
            raise HparamsSaveError(
                f"Value check failed for '{key}': got {result[key]!r}, expected {expected!r}; nothing was written",
                {key: 'value check failed'},
            )
    if int(result['d_model']) % int(result['n_heads']) != 0:
        raise HparamsSaveError(
            f"d_model ({result['d_model']}) must be divisible by n_heads ({result['n_heads']}); nothing was written",
            {'d_model': 'must be divisible by n_heads', 'n_heads': 'must divide d_model'},
        )

    backup = None
    if make_backup:
        backup = str(path) + '.bak'
        shutil.copy2(path, backup)
    tmp_path = str(path) + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
        f.write(new_text)
    os.replace(tmp_path, path)
    return {'changed': changed, 'unchanged': sorted(unchanged), 'derived': derived, 'backup': backup}
