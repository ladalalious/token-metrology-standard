# Token Metrology Standard (TMS) — Specification v1.0

## 1. Purpose

The Token Metrology Standard (TMS) defines a minimal, reproducible protocol for
*measuring* and *reporting* token counts of large language models (LLMs). Its goal
is to make token counts **definable, comparable, and auditable** by requiring that
every reported token count be accompanied by (a) its measurement conditions and
(b) a set of standardized metrics computed on a common reference corpus.

TMS operationalizes the "report the measurement conditions" norm proposed in the
accompanying paper, in the same spirit that the GUM requires measurement results
to state their measurement conditions and uncertainty.

## 2. Core principle

> A token count is not a property of a text alone; it is a function of
> (text × tokenizer × vocabulary × normalization × special-token rule).

Therefore, any TMS-conformant report MUST include the tokenizer identity and the
measurement conditions, and SHOULD include the standardized metrics below.

## 3. Reference corpus — TMS-Corpus v1.0

A versioned, language- and domain-diverse reference corpus (file
`tms_corpus_v1.json`), comprising:

- Article 1 of the Universal Declaration of Human Rights (UDHR) as parallel text
  in 12 languages (English, Chinese, Japanese, Korean, French, German, Spanish,
  Russian, Arabic, Hindi, Vietnamese, Thai);
- two technical descriptions (English, Chinese);
- two code samples (Python, JavaScript);
- one mathematical formula sample;
- one symbol/emoji sample.

Total: 18 sections. The corpus is versioned so that measurements are reproducible
and comparable across time and vendors.

## 4. Measurement protocol

For each corpus section, measure and report:

| Metric | Definition |
|---|---|
| `tokens` | number of tokens produced by the tokenizer for the raw text |
| `bytes` | length of the UTF-8 encoding of the text |
| `chars` | number of Unicode code points |
| `token_per_byte` | tokens / bytes |
| `token_per_char` | tokens / chars |

### Aggregates

- **Token inflation factor** (per language): the ratio of the language's token
  count to English's token count on the parallel UDHR section, averaged over the
  tokenizers under comparison.
- **Coefficient of variation (CV)**: the standard deviation divided by the mean
  of the inflation factor across tokenizers, expressing cross-tokenizer dispersion.

## 5. Report format

A TMS report is a JSON document with the following top-level keys:

```json
{
  "standard": "Token Metrology Standard",
  "standard_version": "1.0",
  "corpus": {"name": "TMS-Corpus", "version": "1.0"},
  "tokenizer": {
    "name": "...", "type": "...", "vocabulary_size": N,
    "conditions": {"normalization": "...", "special_tokens": "...", "pre_tokenizer": "..."}
  },
  "measurements": [
    {"id": "zh", "language": "Chinese", "domain": "legal prose",
     "tokens": 27, "bytes": 211, "chars": 43, "token_per_byte": 0.128, "token_per_char": 0.628}
  ],
  "aggregates": {"token_inflation_factor": {"English": 1.0, "Chinese": 1.17, "..."}}
}
```

The `conditions` block is MANDATORY: it records the tokenizer's normalization,
special-token treatment, and pre-tokenization, so that a third party can reproduce
the measurement.

## 6. Reference implementation

`tms.py` is the reference implementation. It loads tokenizers of four kinds
(tiktoken encodings, tokenizers JSON, sentencepiece models, and the GLM-4
tiktoken-style format) and produces a TMS report:

```bash
python tms.py --type tiktoken --name o200k_base --corpus tms_corpus_v1.json --out report.json
```

The first batch of calibration data for 13 widely used tokenizers is provided in
`tms_results_13tokenizers.json`.

## 7. Conformance

A token-counting implementation is *TMS-conformant* if it:

1. reports the tokenizer identity and measurement conditions (Section 2);
2. reports the five per-section metrics on the TMS-Corpus (Section 4);
3. reports the token inflation factor per language (Section 4).

Conformance does not prescribe *how* to tokenize; it prescribes *what to report*
so that token counts become comparable and auditable.
