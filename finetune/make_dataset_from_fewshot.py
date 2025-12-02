#!/usr/bin/env python
"""
Extract DEFAULT_EXAMPLES from server.py and build JSONL datasets.

Outputs by default:
  - data/fewshot.jsonl  (all examples)
  - data/train.jsonl    (90%)
  - data/val.jsonl      (10%)

Usage:
  python finetune/make_dataset_from_fewshot.py \
    --server_file server.py \
    --out_dir data --val_ratio 0.1 --seed 42

Options:
  --filter_dir ko-en|en-ko|any    (default any)
"""

import argparse
import ast
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def node_to_obj(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):  # legacy
        return node.s
    if isinstance(node, ast.Num):  # legacy
        return node.n
    if isinstance(node, ast.List):
        return [node_to_obj(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(node_to_obj(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {node_to_obj(k): node_to_obj(v) for k, v in zip(node.keys, node.values)}
    # Names like True/False/None in very old ASTs
    if isinstance(node, ast.NameConstant):
        return node.value
    raise ValueError(f"Unsupported AST node: {type(node)}")


def extract_default_examples(py_path: str) -> List[Dict[str, str]]:
    src = open(py_path, 'r', encoding='utf-8').read()
    mod = ast.parse(src, filename=py_path)
    for n in ast.walk(mod):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == 'DEFAULT_EXAMPLES':
                    data = node_to_obj(n.value)
                    # sanity: list of dicts with input/output
                    out = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        if 'input' in item and 'output' in item:
                            out.append({
                                'source': item.get('source'),
                                'target': item.get('target'),
                                'input': item['input'],
                                'output': item['output'],
                            })
                    return out
    raise RuntimeError('DEFAULT_EXAMPLES not found in server.py')


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server_file', default='server.py')
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--filter_dir', choices=['ko-en','en-ko','any'], default='any')
    args = ap.parse_args()

    examples = extract_default_examples(args.server_file)
    if args.filter_dir != 'any':
        src, tgt = args.filter_dir.split('-')
        examples = [e for e in examples if (e.get('source') == src and e.get('target') == tgt)]

    # dedupe
    seen = set()
    uniq: List[Dict[str, str]] = []
    for e in examples:
        key = (e.get('source'), e.get('target'), e['input'], e['output'])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    # write all
    out_all = os.path.join(args.out_dir, 'fewshot.jsonl')
    write_jsonl(out_all, uniq)

    # split
    random.seed(args.seed)
    idx = list(range(len(uniq)))
    random.shuffle(idx)
    cut = int(len(idx) * (1.0 - args.val_ratio))
    train_rows = [uniq[i] for i in idx[:cut]]
    val_rows = [uniq[i] for i in idx[cut:]]

    write_jsonl(os.path.join(args.out_dir, 'train.jsonl'), train_rows)
    write_jsonl(os.path.join(args.out_dir, 'val.jsonl'), val_rows)

    print(f"Wrote {len(uniq)} rows to {out_all}")
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)} -> {args.out_dir}")


if __name__ == '__main__':
    main()

