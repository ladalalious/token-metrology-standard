# -*- coding: utf-8 -*-
"""用 TMS 参考实现对 13 个分词器生成标准校准报告。"""
import json, argparse
import tms

SPECS = [
    ("GPT-2", {"type": "tiktoken", "name": "gpt2"}),
    ("GPT-3.5/4", {"type": "tiktoken", "name": "cl100k_base"}),
    ("GPT-4o", {"type": "tiktoken", "name": "o200k_base"}),
    ("LLaMA-2", {"type": "json", "path": r"..\tokfiles\llama2.json"}),
    ("LLaMA-3", {"type": "json", "path": r"..\tokfiles\LLaMA_3.json"}),
    ("Mistral-7B", {"type": "json", "path": r"..\tokfiles\mistral.json"}),
    ("Phi-3", {"type": "json", "path": r"..\tokfiles\phi3.json"}),
    ("Qwen2.5", {"type": "json", "path": r"..\tokfiles\Qwen2.5.json"}),
    ("DeepSeek-V3", {"type": "json", "path": r"..\tokfiles\DeepSeek_V3.json"}),
    ("Yi-6B", {"type": "json", "path": r"..\tokfiles\yi6b.json"}),
    ("Baichuan2", {"type": "sentencepiece", "path": r"..\tokfiles\baichuan2.model"}),
    ("GLM-4", {"type": "glm4", "path": r"..\tokfiles\glm4.model"}),
    ("InternLM2", {"type": "sentencepiece", "path": r"..\tokfiles\internlm2.model"}),
]

corpus = json.load(open("tms_corpus_v1.json", encoding="utf-8"))
results = {}
for label, spec in SPECS:
    args = argparse.Namespace(type=spec["type"], name=spec.get("name"), path=spec.get("path"))
    tok, meta = tms.load_tokenizer(args)
    rows = tms.measure(tok, corpus)
    meta["label"] = label
    results[label] = {"tokenizer": meta, "measurements": rows,
                      "aggregates": tms.aggregate(rows)}
    zh = next(r for r in rows if r["id"] == "zh")
    print(f"{label:14s} vocab={meta['vocabulary_size']:>7}  中文token={zh['tokens']:>3}")

out = {
    "standard": "Token Metrology Standard",
    "standard_version": tms.TMS_VERSION,
    "corpus": {"name": corpus["name"], "version": corpus["version"]},
    "n_tokenizers": len(results),
    "tokenizers": results,
}
json.dump(out, open("tms_results_13tokenizers.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n汇总已写 tms_results_13tokenizers.json")
