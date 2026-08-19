# Token Metrology Standard (TMS)

The **Token Metrology Standard (TMS)** is a versioned, language- and domain-diverse
reference corpus, a reference implementation, and calibration data for making token
counts of large language models (LLMs) **definable, comparable, and auditable**.

It operationalizes the "report the measurement conditions" protocol proposed in the
accompanying paper:

> Liufu, Y. (2026). *Token Metrology for Large Language Models: A Multilingual
> Tokenizer Benchmark, Reference Corpus, and Reporting Protocol.* (submitted)

The token is the de facto unit of measurement of the LLM ecosystem (pricing, context
capacity, throughput), yet it is neither unique nor commensurable across tokenizers.
TMS addresses this by fixing a common reference corpus and requiring every reported
token count to state its tokenizer identity and measurement conditions.

---

## Repository contents

| File | Description |
|---|---|
| `tms_spec.md` | The TMS v1.0 specification: purpose, core principle, corpus, protocol, report format, conformance criteria. |
| `tms_corpus_v1.json` | **TMS-Corpus v1.0** — the versioned reference corpus: Article 1 of the UDHR in 12 languages, plus technical, code, mathematical, and symbolic samples (18 sections). |
| `tms.py` | The **reference implementation**. Loads tiktoken / tokenizers-JSON / SentencePiece / GLM-4 tokenizers and emits a TMS-conformant report. |
| `run_all.py` | Reproduction script that runs the 13-tokenizer calibration (see "Reproduction" below). |
| `tms_results_13tokenizers.json` | **Calibration data** — TMS reports for 13 widely used tokenizers. |
| `benchmark_results.csv` | Raw benchmark data: 13 tokenizers × 18 corpus sections (234 rows), with token/byte/char counts and per-byte/per-char densities. |
| `benchmark_summary.csv` | Per-section summary statistics (min/max/mean/std/CV across tokenizers). |
| `benchmark_tax.csv` | Token inflation factor per language relative to English (parallel UDHR text). |
| `LICENSE` | MIT (code). Corpus data terms are stated in `tms_corpus_v1.json`. |

## Quick start

```bash
# install dependencies
pip install tiktoken tokenizers

# produce a TMS report for a single tokenizer
python tms.py --type tiktoken --name o200k_base \
              --corpus tms_corpus_v1.json --out report.json

# also supported: --type json | sentencepiece | glm4
python tms.py --type json --path tok.json --corpus tms_corpus_v1.json --out report.json
```

A TMS report records, for each corpus section: `tokens`, `bytes`, `chars`,
`token_per_byte`, `token_per_char`, plus per-language **token inflation factors** and
their coefficient of variation (CV).

## Reproduction

`tms_results_13tokenizers.json` and the benchmark CSVs are the **precomputed** results
reported in the paper. To regenerate them end-to-end:

1. Obtain the tokenizer vocabulary/model files for the 13 tokenizers from their
   official releases (tiktoken encodings are bundled with `tiktoken`; the others are
   the public `tokenizer.json` / SentencePiece models of LLaMA-2/3, Mistral-7B, Phi-3,
   Qwen2.5, DeepSeek-V3, Yi-6B, Baichuan2, GLM-4, InternLM2).
2. Point the `SPECS` paths in `run_all.py` at those files and run:
   ```bash
   python run_all.py
   ```

## Citation

If you use the TMS corpus, implementation, or calibration data, please cite the
accompanying paper and the TMS release:

```bibtex
@article{liufu2026tokenmetrology,
  title   = {Token Metrology for Large Language Models: A Multilingual Tokenizer
             Benchmark, Reference Corpus, and Reporting Protocol},
  author  = {Liufu, Yujie},
  year    = {2026},
  note    = {Token Metrology Standard v1.0, https://github.com/ladalalious/token-metrology-standard}
}
```

## Data availability

The corpus, specification, reference implementation, calibration data, and reproduction
scripts in this repository are released under the terms below. The repository is the
canonical source; a persistent archive (Zenodo DOI) will be linked here upon acceptance.

The public repository is https://github.com/ladalalious/token-metrology-standard; a
persistent Zenodo DOI will be added here upon acceptance.

## License

Code (`tms.py`, `run_all.py`): **MIT** — see `LICENSE`.
Corpus (`tms_corpus_v1.json`): UDHR text is public domain; the technical/code/symbol
samples were authored by the TMS contributors and are released under CC-BY-4.0.
