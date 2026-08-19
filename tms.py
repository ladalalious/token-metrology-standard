#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Metrology Standard (TMS) v1.0 — reference implementation.

A reference implementation of the TMS measurement protocol: given any tokenizer
and the TMS reference corpus, it produces a standardized, comparable metrology
report (token counts, byte/char metrics, token inflation factor, CV) together
with the mandatory measurement conditions.

Usage:
    python tms.py --type tiktoken --name o200k_base --corpus tms_corpus_v1.json --out report.json
    python tms.py --type json        --path tok.json    --corpus tms_corpus_v1.json --out report.json
    python tms.py --type sentencepiece --path tok.model --corpus tms_corpus_v1.json --out report.json
    python tms.py --type glm4        --path tok.model   --corpus tms_corpus_v1.json --out report.json
"""
import json, struct, base64, argparse, statistics
import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.normalizers import NFKC

TMS_VERSION = "1.0"
CORPUS_NAME = "TMS-Corpus"

# ============ SentencePiece protobuf parser (no external dependency) ============
def _read_varint(data, pos):
    r, sh = 0, 0
    while True:
        b = data[pos]; pos += 1
        r |= (b & 0x7F) << sh
        if not (b & 0x80):
            return r, pos
        sh += 7


def _parse_piece(data):
    piece, score, ptype = "", 0.0, 0
    pos, n = 0, len(data)
    while pos < n:
        tag, pos = _read_varint(data, pos)
        f, wt = tag >> 3, tag & 7
        if wt == 0:
            v, pos = _read_varint(data, pos)
            if f == 3:
                ptype = v
        elif wt == 1:
            pos += 8
        elif wt == 2:
            ln, pos = _read_varint(data, pos)
            pl = data[pos:pos + ln]; pos += ln
            if f == 1:
                piece = pl.decode("utf-8", "replace")
        elif wt == 5:
            if f == 2:
                score = struct.unpack("<f", data[pos:pos + 4])[0]
            pos += 4
        else:
            break
    return piece, score, ptype


def _parse_sp_model(data):
    pieces, pos, n = [], 0, len(data)
    while pos < n:
        tag, pos = _read_varint(data, pos)
        f, wt = tag >> 3, tag & 7
        if wt == 0:
            _, pos = _read_varint(data, pos)
        elif wt == 1:
            pos += 8
        elif wt == 2:
            ln, pos = _read_varint(data, pos)
            pl = data[pos:pos + ln]; pos += ln
            if f == 1:
                pieces.append(_parse_piece(pl))
        elif wt == 5:
            pos += 4
        else:
            break
    return pieces


def build_sp_tokenizer(path):
    pieces = _parse_sp_model(open(path, "rb").read())
    vocab = {p: i for i, (p, s, t) in enumerate(pieces)}
    normal = [(p, s) for (p, s, t) in pieces if t in (0, 1) and len(p) > 1]
    normal.sort(key=lambda x: -x[1])
    merges = [(p[:-1], p[-1]) for (p, s) in normal]
    merges = [(a, b) for (a, b) in merges if a in vocab and b in vocab]
    tok = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))
    tok.pre_tokenizer = Metaspace(replacement="\u2581", prepend_scheme="always")
    tok.normalizer = NFKC()
    return tok


# ============ GLM-4 tiktoken-style loader ============
GLM4_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def load_glm4(path):
    ranks = {}
    for line in open(path, "rb"):
        line = line.strip()
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return tiktoken.Encoding(name="glm4", pat_str=GLM4_PAT,
                             mergeable_ranks=ranks, special_tokens={})


# ============ Unified loader ============
def load_tokenizer(args):
    t = args.type
    if t == "tiktoken":
        tok = tiktoken.get_encoding(args.name)
        meta = {"type": "tiktoken", "name": args.name,
                "vocabulary_size": tok.n_vocab,
                "conditions": {"normalization": "none (raw bytes)", "special_tokens": "excluded"}}
        return tok, meta
    if t == "json":
        tok = Tokenizer.from_file(args.path)
        meta = {"type": "tokenizers-json", "path": args.path,
                "vocabulary_size": tok.get_vocab_size(),
                "conditions": {"normalization": "as defined in tokenizer.json",
                               "special_tokens": "excluded"}}
        return tok, meta
    if t == "sentencepiece":
        tok = build_sp_tokenizer(args.path)
        meta = {"type": "sentencepiece-reconstructed", "path": args.path,
                "vocabulary_size": tok.get_vocab_size(),
                "conditions": {"normalization": "NFKC", "pre_tokenizer": "Metaspace(\u2581)"}}
        return tok, meta
    if t == "glm4":
        tok = load_glm4(args.path)
        meta = {"type": "tiktoken-style (glm4)", "path": args.path,
                "vocabulary_size": tok.n_vocab,
                "conditions": {"pat_str": "GPT-4 pattern", "special_tokens": "excluded"}}
        return tok, meta
    raise ValueError(f"unknown tokenizer type: {t}")


# ============ Measurement ============
def count(tok, text):
    nb, nc = len(text.encode("utf-8")), len(text)
    if isinstance(tok, tiktoken.Encoding):
        nt = len(tok.encode(text))
    else:
        nt = len(tok.encode(text).ids)
    return nt, nb, nc


def measure(tok, corpus):
    rows = []
    for sec in corpus["sections"]:
        nt, nb, nc = count(tok, sec["text"])
        rows.append({"id": sec["id"], "language": sec["language"],
                     "domain": sec["domain"],
                     "tokens": nt, "bytes": nb, "chars": nc,
                     "token_per_byte": round(nt / nb, 4),
                     "token_per_char": round(nt / nc, 4)})
    return rows


def aggregate(rows):
    # token inflation factor relative to English (legal-prose sections only)
    en = next(r for r in rows if r["id"] == "en")
    ratios = {}
    for r in rows:
        if r["domain"] == "legal prose":
            ratios[r["language"]] = r["tokens"] / en["tokens"]
    return {"token_inflation_factor": {k: round(v, 3) for k, v in ratios.items()}}


# ============ Report ============
def run(args):
    corpus = json.load(open(args.corpus, encoding="utf-8"))
    tok, meta = load_tokenizer(args)
    rows = measure(tok, corpus)
    report = {
        "standard": "Token Metrology Standard",
        "standard_version": TMS_VERSION,
        "corpus": {"name": corpus["name"], "version": corpus["version"]},
        "tokenizer": meta,
        "measurements": rows,
        "aggregates": aggregate(rows),
    }
    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"report written to {args.out}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Token Metrology Standard reference implementation")
    p.add_argument("--type", required=True, choices=["tiktoken", "json", "sentencepiece", "glm4"])
    p.add_argument("--name", help="tiktoken encoding name (e.g. o200k_base)")
    p.add_argument("--path", help="path to tokenizer file (json/model)")
    p.add_argument("--corpus", required=True, help="path to TMS corpus JSON")
    p.add_argument("--out", help="output report JSON path")
    run(p.parse_args())
