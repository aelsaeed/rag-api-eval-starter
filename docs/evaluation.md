# Evaluation methodology

The evaluator answers one practical question: **did a change make the application's retrieval and
evidence contract worse on known cases?** It is a deterministic regression suite, not a claim that
12 examples represent production traffic.

## Evaluation boundary

The default `application` adapter exercises the same path as the service:

1. create an isolated in-memory Qdrant store;
2. ingest `data/sample_docs/` through the application chunking and embedding code;
3. retrieve with dense, lexical, or hybrid strategy;
4. apply answer/abstention behavior; and
5. convert the full ranked retrieval list, extractive answer, and answer citations to the evaluator
   contract.

The adapter passes every setting explicitly and uses deterministic hashed embeddings. Ambient
`RAG_*` variables cannot redirect the run to a developer's database, change the embedding model, or
pollute persistent data. This makes the result suitable for a fast CI gate.

An independent `offline` lexical adapter remains available as a scorer/reference seam, but it is not
the adapter used by the default Make targets or CI quality gate.

## Golden dataset

`data/eval.jsonl` currently contains 12 human-reviewed cases: 10 answerable questions and two
unanswerable hard negatives. Each line follows this shape:

```json
{
  "id": "hybrid-benefit-001",
  "question": "How are dense and lexical retrieval results combined?",
  "answerable": true,
  "gold_contexts": [
    {
      "source": "platform_overview.md",
      "anchor": "Hybrid mode combines those rankings with weighted reciprocal-rank fusion (RRF), allowing an exact lexical match to recover a chunk that dense retrieval missed.",
      "relevance": 2
    }
  ],
  "required_fact_groups": [
    ["reciprocal rank fusion", "RRF"],
    ["exact lexical match"],
    ["dense retrieval missed", "dense retrieval misses"]
  ],
  "tags": ["retrieval", "exact-term"]
}
```

The loader rejects unknown fields, duplicate IDs or contexts, invalid relevance grades, missing
answerable-case judgments, and hard negatives that accidentally define answer evidence. For the
offline adapter, every source and normalized anchor is also checked against the corpus. The raw
JSONL bytes are SHA-256 hashed into every report so results can be tied to an exact fixture revision.

## Metrics

Metrics for answerable cases are macro-averaged so each question has equal weight:

- **Context recall@1/3/5:** fraction of a case's gold contexts found in the full ranked retrieval list
  by depth 1, 3, or 5. A hit must match both the reviewed source and normalized anchor text.
- **MRR@5:** reciprocal rank of the first relevant context within the first five results. It rewards
  putting at least one useful context early.
- **nDCG@5:** quality of the full ranked top five using the 1–3 relevance grade, discounted by rank.
  Duplicate matches to the same gold context do not earn repeated credit.
- **Answer fact coverage:** fraction of required fact groups represented in the extractive answer.
  Any normalized alias within a group satisfies that fact.
- **Citation precision:** fraction of citations attached to the answer that match a reviewed
  source/anchor pair.
- **Abstention accuracy:** fraction of all cases where the system answers an answerable question and
  abstains on an unanswerable one.

This separation is intentional: retrieving a gold chunk at rank five should improve Recall@5 even
when the extractive answer cites only its first two hits, while an irrelevant answer citation should
still reduce citation precision. Before scoring, the evaluator validates the adapter contract:
retrieval hits are bounded by `k`; hits and citations are complete, finite, and unique; every citation
must be present in the retrieval list; abstentions have no answer or citations; and non-abstentions
must cite evidence. Retrieval hits remain available on abstentions for ranking diagnostics.

## Quality gates

CI applies every threshold to the `hybrid` application run. Dense and lexical strategies are
diagnostic baselines in the ablation report.

| Metric | Minimum |
|---|---:|
| Context recall@1 | 0.700 |
| Context recall@3 | 0.900 |
| Context recall@5 | 0.950 |
| MRR@5 | 0.750 |
| nDCG@5 | 0.800 |
| Answer fact coverage | 0.800 |
| Citation precision | 0.500 |
| Abstention accuracy | 1.000 |

`eval.run` writes its report and then exits non-zero with each metric below threshold. `eval.compare`
evaluates all three strategies and gates hybrid before writing the ablation report. Threshold changes
therefore need the same review as behavioral changes; lowering one should include evidence that the
old target was invalid.

## Current deterministic result

Dataset SHA-256:
`9fa78a7a37fef0b330a0e8bf79071c3d46eeb4108d09bde5821d4cd6acf2926d`

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | Fact coverage | Citation precision | Abstention accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.300 | 1.000 | 1.000 | 0.600 | 0.702 | 0.033 | 0.000 | 0.417 |
| Lexical | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.850 | 1.000 |
| Hybrid | 0.900 | 1.000 | 1.000 | 0.950 | 0.963 | 1.000 | 0.850 | 1.000 |

These values come from the default application adapter with fake 64-dimensional hashed embeddings,
the sample corpus, `top_k=5`, and the exact dataset hash shown above the table. They measure regression
behavior for this fixture, not sentence-transformer quality or real-world answer accuracy.

Hybrid clears every configured gate and substantially improves on dense. Lexical is perfect on this
small, literal corpus and matches hybrid's answer, citation, and abstention metrics. The dense answer
metrics are intentionally poor because hashed-vector similarities rarely clear the shared evidence
threshold; this is a diagnostic baseline, not a real embedding result. The honest conclusion is that
lexical retrieval is the strongest included ranking strategy for this fixture; the result does not
establish that lexical or hybrid will dominate on a broader semantic dataset.

## Run it locally

```bash
make eval
make eval-ablation
```

Or run the same sequence CI uses:

```bash
make eval-ci
```

The commands generate:

- `reports/latest.md` and `reports/latest.json` for the hybrid application run; and
- `reports/ablation.md` and `reports/ablation.json` for the strategy comparison.

`reports/` is intentionally gitignored because the output belongs to the evaluated commit and
dataset. When generated, the CI workflow appends the ablation Markdown to the job summary and uploads
available Markdown, JSON, and `coverage.xml` files as a `quality-evidence-<commit SHA>` artifact. The
upload step runs even after an earlier failure, but only files produced before that failure can be
included. Consult the Actions run for the commit under review; this documentation does not claim that
a remote run is green.

The package also installs two console scripts:

```bash
rag-eval --help
rag-eval-compare --help
```

Use explicit paths and thresholds for experiments, and write experimental reports outside the
repository's default `reports/` directory if several variants must be retained.

## Interpreting an ablation

The comparison holds the corpus, dataset, chunking, embeddings, and retrieval depth constant. Only
the retrieval strategy changes:

- `dense` ranks the independent vector candidates by bounded cosine-derived confidence;
- `lexical` ranks token-overlap candidates; and
- `hybrid` combines both rank lists with weighted RRF, calibrated by each component's bounded score.

For a candidate present in either list, the hybrid score is the sum of
`weight × component_confidence / (RRF_K + rank)` terms, normalized by `1 / (RRF_K + 1)` and clamped
to `[0, 1]`. Defaults are `RRF_K=60`, lexical weight `0.65`, dense weight `0.35`, and minimum answer
confidence `0.40`. Missing component ranks contribute nothing.

The fusion score ranks candidates; it is not used directly as answer confidence. Evidence confidence
is the strongest available non-negative cosine-similarity or lexical-query-coverage signal for each
hit. The answer keeps the first two threshold-clearing hits in fused-rank order and reports the
strongest returned evidence as confidence. This lets strong dense-only semantic evidence clear the
threshold while an orthogonal dense result scores zero evidence; it also avoids diluting an exact
lexical match with a weaker dense score. The API exposes both the ranking `score` and
`evidence_score` so that decision remains auditable.

A stronger hybrid result supports the fusion choice on this fixture. It does not establish that the
same weights are optimal for another corpus. Look at per-case output as well as aggregates: one hard
negative or one context split can expose a material behavior even when most summary metrics remain
high.

## Valid and invalid conclusions

The benchmark can support statements such as:

- a specific commit passes or fails the declared deterministic gates;
- hybrid outperforms one or both included baselines on the same fixture; and
- a named case regressed in ranking, evidence, citation, or abstention behavior.

It cannot establish:

- statistically representative production quality;
- semantic embedding quality, because the default vectors are deterministic token hashes;
- LLM faithfulness, because the answer path is extractive and has no generator;
- throughput, tail latency, or capacity under load; or
- optimal thresholds for a new domain.

The next credible step is a larger held-out dataset plus a separately recorded real-embedding run,
not weaker thresholds on the development fixture.
