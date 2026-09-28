# 14 · Decisions

[← Run it yourself](13-run-it-yourself.md) · [Index](README.md) · [Next: Glossary →](15-glossary.md)

Every design choice, what we rejected, and why. One page, for when someone asks
"but why did you…".

## Data

| Decision | Alternative rejected | Why |
|---|---|---|
| Keep 6 of CUAD's 41 categories | All 41 | Commercially material and frequent enough to learn from. Some of the 41 have too few examples to evaluate honestly. |
| Read the category from the question **id** | Parse the question text | The id suffix *is* the category. Parsing the text is what produced zero labels in v1 ([ch 5](05-data-pipeline.md#the-bug-that-made-everything-meaningless)). |
| Keep `answer_start` character spans | Document-level labels | Labels are locations. Discarding offsets makes window-level supervision impossible. **The root cause of both v1 bugs.** |
| Do **not** merge `No-Solicit Of Employees` into Non-Compete | Merge them | They are legally distinct. Defensible either way, but it should be a decision, not an accident. |

## Chunking

| Decision | Alternative rejected | Why |
|---|---|---|
| Windows measured in **words** | Model tokens | Tokenizer-agnostic, so all four models see identical boundaries. Otherwise a comparison measures tokenizer differences mixed with model differences. |
| 300-word windows, 100-word overlap | No overlap | Overlap means a boundary-straddling clause still appears whole somewhere. |
| Label a window at **≥0.5 overlap of the shorter range** | Any overlap; full containment | Handles both a short clause in a long window and a long clause containing a window. Exposed as `min_overlap_ratio` because it is a judgement call. |
| Chunk **at load time** | Persist chunked datasets | BERT needs 300 words, Longformer 2,600. Persisting means four files that drift. |

## Splitting

| Decision | Alternative rejected | Why |
|---|---|---|
| Split by **document** | Split by window | Windows overlap by 100 words — a random split puts near-duplicate text on both sides. Pure leakage. |
| **Persist** the assignment to disk | Recompute per run | Every model must be scored on the identical test set, or model differences and split differences are indistinguishable. |
| 70/15/15 | 80/10/10 | 76 test contracts is already small for per-class metrics on rare labels. |

## Models

| Decision | Alternative rejected | Why |
|---|---|---|
| CNN as baseline | Start with BERT | Without a floor, a score is uninterpretable. Also the fastest check the pipeline carries signal. |
| Kernels 3, 4, 5 in **parallel** | Deeper stacked convolutions | Clause-announcing phrases are 3–5 words. This is the Kim (2014) standard. |
| **Max-over-time** pooling | Average pooling | The question is "did this pattern appear anywhere", not "how often". |
| BERT **and** Legal-BERT | Only Legal-BERT | Without the general-domain control, Problem 1 cannot be answered. |
| RoBERTa/DeBERTa rejected | Add a stronger encoder | Changes architecture *and* pretraining at once. Would improve a leaderboard and ruin the experiment. |
| Longformer for long context | BigBird, LED | Widely used, well-supported, same encoder family — keeps the comparison clean. |
| Global attention on `[CLS]` only | None; all tokens | Local attention alone cannot pool the window. All-global would forfeit the efficiency that makes 4,096 possible. |
| Hierarchical BERT **deferred** | Build it now | Only worth it if pooling proves to be the bottleneck. Evaluation has not shown that. |

## Training

| Decision | Alternative rejected | Why |
|---|---|---|
| **One shared loop** for all models | Per-model scripts | Identical treatment makes score differences attributable to models. Separate loops drift. |
| BCE with `pos_weight` | Plain BCE; focal loss | At 1.5–4.2% positives, unweighted BCE collapses to predicting nothing. `pos_weight` is the simplest correct fix; focal loss adds a hyperparameter we have no budget to tune. |
| Six **sigmoids** | Softmax | 338 of 510 contracts carry 2+ labels. Softmax would be wrong by construction. |
| Weights from **train split only** | Whole corpus | Corpus-wide statistics leak test information. |
| Vocabulary from **train split only** | Whole corpus | Same reason. |
| Early stop on validation macro F1 | Fixed epochs; stop on loss | Loss can improve while the metric you care about degrades. |
| Accept ±0.03 non-determinism | Force deterministic kernels | Determinism costs significant speed. Better to **measure and report** the variance. |

## Evaluation

| Decision | Alternative rejected | Why |
|---|---|---|
| **Never report accuracy** | Report it alongside | An all-negative model scores >95%. It would be flattering and meaningless. |
| Per-label thresholds tuned on **validation** | Fixed 0.5 | Each label has its own positive rate. Ours span 0.22–0.83. |
| Report **both** window and document level | Pick one | They answer different questions. Quoting one misrepresents the system. |
| **Max** pooling | Mean; top-k | Existential task. Mean buries a true positive (0.95 → <0.03 across 60 windows). Top-k scores marginally higher on F1 but inside the noise floor. |
| Document thresholds tuned **separately**, on validation | Reuse window thresholds; tune on test | A pooled max is systematically higher. Tuning on test is leakage — it cost us 0.067 of fake score ([ch 9](09-evaluation.md#the-leak-we-caught)). |
| Report **average precision** too | F1 only | Threshold-free, so it is the stable number when thresholds are in question. |
| **Generate** the comparison table | Write it by hand | No number reaches the write-up by being copied. |

## Interpretability and deployment

| Decision | Alternative rejected | Why |
|---|---|---|
| CNN **filter attribution** | Attention heatmaps only | A convolution's receptive field is uncontested; attention as explanation is disputed. Attribution returns a literal phrase. |
| Demo reuses the **evaluation runtime** | Separate inference path | A demo that chunks differently shows numbers that do not match the report — worse than no demo. |
| Show the **threshold** next to each score | Score alone | Otherwise a 0.223 flag beside an unflagged 0.341 looks like a bug, and someone "fixes" it. |
| Embed contracts the model gets **wrong** | Only successes | A page of only successes is a page nobody believes. |
| **Hugging Face Space** for hosting | GitHub Pages | A user-level custom domain makes every project repo a subpath of it, with no per-repo opt-out ([ch 11](11-deployment.md#why-not-github-pages)). |
| Export models to **standard HF layout** | Upload `model.pt` | The raw checkpoint has `backbone.`-prefixed keys that `from_pretrained` cannot load. |
| Space `app.py` **installs from the repo** | Vendor a copy | A vendored copy drifts from the evaluated pipeline. |

## Engineering

| Decision | Alternative rejected | Why |
|---|---|---|
| Delete the FastAPI/Postgres/Redis/Next.js scaffold | Keep and maintain it | It served no deliverable and no grading criterion, and every file needed maintaining. |
| Plain CLIs; notebooks call them | Logic in notebooks | Notebooks cannot drift from the code, and results do not depend on cell order. |
| Script-style tests, no pytest | pytest | Two suites run with **zero dependencies installed**. `python tests/x.py` works anywhere. |
| Tests encode **reasoning** | Test only outputs | `test_mean_pooling_buries_that_same_positive` means the default cannot be silently reversed. |
| Data layer is **standard library only** | Import numpy freely | Preparing data and running its 21 tests needs nothing installed. |

---

[← Run it yourself](13-run-it-yourself.md) · [Index](README.md) · [Next: Glossary →](15-glossary.md)
