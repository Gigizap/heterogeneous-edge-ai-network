#!/usr/bin/env python3
"""
dump_templates.py — print the chat template baked into every GGUF model.

Loads each *.gguf under --models-dir and prints its
tokenizer.chat_template metadata (the Jinja template the generic
llama-cpp handler uses to render messages).

Usage:
  python dump_templates.py --models-dir models/
  python dump_templates.py --models-dir models/ --save templates/
"""

from __future__ import annotations
import argparse
from pathlib import Path

from llama_cpp import Llama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--save", default=None,
                    help="optional dir to also write each template to a .jinja file")
    a = ap.parse_args()

    md = Path(a.models_dir)
    ggufs = sorted(md.glob("**/*.gguf"))
    if not ggufs:
        print(f"No .gguf files found under {md}")
        return

    save_dir = Path(a.save) if a.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for g in ggufs:
        print("\n" + "=" * 70)
        print(f"  {g.name}")
        print("=" * 70)
        try:
            # vocab_only avoids loading weights — fast, low memory; metadata
            # (incl. the chat template) is still fully available.
            llm = Llama(model_path=str(g), vocab_only=True, verbose=False)
        except Exception as e:
            print(f"  [ERROR loading] {e}")
            continue

        md_dict = getattr(llm, "metadata", {}) or {}
        tmpl = md_dict.get("tokenizer.chat_template")

        # Some GGUFs ship multiple named templates as
        # tokenizer.chat_template.<name>; collect those too.
        named = {k: v for k, v in md_dict.items()
                 if k.startswith("tokenizer.chat_template.")}

        if not tmpl and not named:
            print("  [no chat template in metadata]")
        else:
            if tmpl:
                print(tmpl)
                if save_dir:
                    (save_dir / f"{g.stem}.jinja").write_text(tmpl)
            for name, body in named.items():
                short = name.split("tokenizer.chat_template.")[-1]
                print(f"\n  --- named template: {short} ---")
                print(body)
                if save_dir:
                    (save_dir / f"{g.stem}.{short}.jinja").write_text(body)

        del llm

    if save_dir:
        print(f"\nSaved templates to {save_dir}")


if __name__ == "__main__":
    main()