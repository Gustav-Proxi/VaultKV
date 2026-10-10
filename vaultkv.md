# VaultKV Research Plan — Version- and Permission-Consistent Chunk-KV Reuse for RAG Serving on vLLM V1

Build **VaultKV**: a vLLM V1 KV-connector plus control plane that lets RAG serving reuse precomputed KV caches for retrieved chunks (CacheBlend/EPIC-style, position-independent) while guaranteeing that a cached chunk is never served after its document was edited/deleted or after the requesting tenant lost access, and that cross-tenant sharing does not open the SpliceLeak/HijackKV attack surface. A targeted search found no 2025–2026 paper or open-source system that combines these three properties, whereas LateLens-style late-interaction serving is now crowded and should be dropped as a standalone project.

## TL;DR

- **Recommendation: VaultKV.** Chunk-level KV reuse is the biggest serving win for RAG (CacheBlend: 2.2–3.3× lower TTFT; EPIC: up to 8× TTFT), but every existing system assumes chunks are immutable and globally shareable. Recent work shows each half of the problem separately (stale-KV repair after edits, tenant salting for prefix caches, SpliceLeak/HijackKV attacks on non-prefix reuse); nobody has shipped the combined, enterprise-correct version. It extends an agent-serving control plane into a coherent research line.
- **Drop LateLens as a standalone project.** WARP (SIGIR'25: 41× over XTR, 3× over PLAID), MUVERA, PLAID reproducibility studies, ColBERTSaR, TileMaxSim (GPU MaxSim kernels), and a stack of ColPali patch-pruning/merging papers (Light-ColPali, DocPruner, AdaMerge, MarginMerge) have already covered the quality-vs-storage curve; there is now a dedicated ECIR 2026 workshop on late interaction. You would be the 10th entrant.
- **Headline result to aim for:** "Within X% of global-sharing CacheBlend TTFT/throughput, with 0 stale-version or post-revocation serves (vs. N for LMCache CacheBlend) and a SpliceLeak timing signal reduced to chance, on Llama-3.1-8B/Qwen on a single A100."

## Key Findings: What Production RAG Looks Like in 2025–2026

**Retrieval quality is largely "solved enough" with known recipes; serving cost and correctness are not.**
- Anthropic's "Introducing Contextual Retrieval" (Sept 2024) reports that reranked contextual embeddings plus contextual BM25 "reduced the top-20-chunk retrieval failure rate by 67% (5.7% → 1.9%)," and it is now a standard RAG recipe. Hybrid BM25/dense + RRF + rerank is already well explored, so retrieval quality alone has limited novelty.
- Visual-document RAG has matured into a benchmark race. ViDoRe V3 (Loison et al., ACL 2026; arXiv:2601.08620) has 26,000 pages and 3,099 human-verified queries in 6 languages,\[1\] and top late-interaction models still score under 65% nDCG@10.\[2\] That is an open problem, but it is a model-training race dominated by ILLUIN/NVIDIA/Nomic/Jina-scale teams.
- Agentic search (Search-R1, R1-Searcher, Agentic-R, AutoSearch) is the dominant research direction for multi-hop.\[3\]\[4\] An ICLR 2026 study of search-agent efficiency shows both overly high and overly low recall hurt end-to-end efficiency, and a 2026 analysis of Search-R1 traces found 30.5% of post-first-turn queries are byte-identical repeats (Semantics Delivery Network, arXiv:2609.22486).\[5\]\[6\]

**The serving layer is where the money and the open problems are.**
- Prefix caching (vLLM APC, SGLang RadixAttention) rarely hits in RAG because retrieved chunks appear after user-specific prefixes and in varying order.\[7\]\[8\] Non-prefix chunk reuse is the fix: CacheBlend (Yao et al., EuroSys'25 Best Paper; DOI 10.1145/3790254) recomputes 15% of tokens by default (the paper notes "an update fraction of less than 15% can typically generate same-quality responses") for 2.2–3.3× TTFT reduction and 2.8–5× throughput; EPIC/LegoLink (Hu et al., ICML 2025; DOI 10.5555/3780338.3781294) reports up to 8× TTFT; follow-ups include RAGCache, TurboRAG, Cache-Craft, KVLink, CacheClip (arXiv:2510.10129), ProphetKV (ICML 2026), SpecCache (ACL 2026; DOI 10.18653/v1/2026.acl-long.859), and Tessera (arXiv:2609.32999), which makes retrieval the control plane for KV reuse and cuts mean TTFT by up to 3.6× over SGLang and LMCache+EPIC.
- LMCache is the production vehicle (per the LMCache tech report, arXiv:2510.09665, the vLLM KV-connector API was "initiated by LMCache team" and is implemented and maintained jointly by the LMCache and vLLM teams), but its CacheBlend path is fragile in vLLM V1: LMCache issue #3238 reports that non-prefix reuse with the V1 connector hit only 24 of 2,451 tokens (the system-prompt prefix) because the scheduler-side lookup does prefix-only matching.\[9\] A maintainer noted in 2026 that the hashing code has since changed but the fix has not been re-verified. This is a concrete upstream contribution opportunity.
- Industry routing is going KV-aware (Red Hat Developer reports that llm-d KV-cache-aware routing over vLLM pods raised the KV-cache hit rate from 62% under naive load balancing to approximately 90%), and access control is moving to query time. The AWS Machine Learning Blog describes Amazon Quick/Bedrock Knowledge Bases verifying permissions with the source system at query time because a source like Confluence "doesn't emit an event when group membership changes" and ACLs sync every 24 hours by default, so between syncs a user whose access was revoked "might still receive AI answers from documents they should no longer see."

**Practitioner pain points (blogs, vendor docs, 2026 papers) converge on freshness and multi-tenancy.**
- Freshness: stale chunks surviving updates, deletes not propagating, and caches not invalidated are described as the failure "teams solve last" (the Oracle Developers blog on RAG index drift, Jul 16, 2026, says a production system "should prove that deleted content no longer retrieves, updated content replaces stale chunks"). Answer/retrieval caches are getting version keys (FinCacheServe, REVA, Grounded Cache Routing), but KV caches are not.\[10\]\[11\]\[12\]
- Multi-tenant leakage: PromptPeek (NDSS 2025) showed prefix-cache timing attacks;\[13\] SpliceLeak (arXiv:2606.21842, June 2026) extends this to non-prefix fusion on vLLM+LMCache with up to 100% extraction success in bounded-entropy settings;\[14\] HijackKV (arXiv:2607.19957, USENIX Security 2026) shows a poisoned prefix can make a benign chunk's cached KV hijack other users' outputs with 94% single-attempt success,\[14\]\[15\]\[16\] and that standard partial recomputation (Random/EPIC/CacheBlend) at current ratios does not mitigate it.\[16\]\[17\]

## Candidate Comparison

| # | Candidate | Closest prior / concurrent work | Novelty & remaining delta | Who pays | Feasibility (1 student, 6–10 wks, Colab/A100/Apple Silicon) | Benchmarks & baselines | Verdict |
|---|---|---|---|---|---|---|---|
| A | **VaultKV**: version- and ACL/tenant-consistent chunk-KV reuse in vLLM V1 | CacheBlend, EPIC, LMCache; stale-KV repair "Contiguity, Not Importance" (arXiv:2609.17983); Leyline (arXiv:2606.01065); KVGov (arXiv:2608.09225); SafeKV, CachePrune; SpliceLeak, HijackKV; FinCacheServe/REVA (answer/text caches only) | **High.** No system combines position-independent chunk KV + edit/delete invalidation or repair + tenant/ACL isolation with attack-resistant sharing | Any enterprise RAG vendor or platform team (Glean-style search, Bedrock/Vertex-style KB services, LMCache/llm-d users), regulated sectors (health, finance) | **High.** 1.5B–8B models; single A100/L4; no training required; builds on LMCache and vLLM | HotpotQA/2WikiMQA/MuSiQue/LongBench (CacheBlend's datasets) + synthetic edit/revoke streams; baselines: full recompute, vLLM APC, per-tenant `cache_salt`, LMCache CacheBlend, EPIC-style | **Recommend** |
| B | Agentic multi-hop KV reuse for search-agent loops | Tessera, AgentKVShift (arXiv:2607.21604), StepKV (arXiv:2609.22158), SpecCache, Semantics Delivery Network, ICLR'26 search-agent efficiency study | **Low–medium.** Tessera already covers RAG + agent memory with retrieval-driven KV management | LLM platforms running deep-research agents | Medium; needs agent traces + multi-node to beat Tessera | MuSiQue, HotpotQA, BrowseComp-Plus; Search-R1 policies | Crowded; fold the idea into VaultKV's eval |
| C | Visual-document RAG with page-level VLM KV reuse (ColQwen retrieval + Qwen-VL PIC) | QV-PIC (arXiv:2608.12121, 83.8% online prefill reduction), VLCache (arXiv:2512.12977, SGLang), CONDUIT (arXiv:2609.05821), Kamera (arXiv:2606.23581), LMCache multimodal | **Low–medium.** Four 2025–2026 systems in 9 months | Document-AI vendors | Medium; VLM KV is memory-heavy on T4/L4 | ViDoRe V2/V3, MM-NIAH, MMLongBench-Doc | Scooped; reuse as VaultKV stretch goal only |
| D | Retrieval-augmented 3D CT report generation | CPR-RAG (ACL 2026), MICCAI 2026 "Balancing Retrieved Evidence for 3D CT Report Generation", CT-RetrievR (SPIE), MARCH (arXiv:2604.16175), MedScribe, MvKeTR, C-MIR (ColBERT-style 3D CBIR) | **Medium.** Crowded on CT-RATE; MICCAI 2026 found prompt-only RAG reduces hallucinated labels but raises omissions, with significant CE macro-F1 drops | Radiology AI vendors (long sales cycle) | Medium; 3D encoders + CT-RATE are heavy | CT-RATE, RadBERT 18-label CE-F1; CT-CHAT, CT-CLIP retrieval | Useful follow-up, but not the strongest standalone RAG project |
| E | LateLens: ColBERTv2/PLAID serving and quality-vs-storage curve | PLAID, WARP (DOI 10.1145/3726302.3729904), MUVERA, XTR, PyLate, ColBERTSaR (arXiv:2606.05568), TileMaxSim (arXiv:2606.26439), Light-ColPali (arXiv:2506.04997), DocPruner, AdaMerge, MarginMerge; LIR @ ECIR 2026 workshop | **Low.** The curve exists across text and visual retrieval; GPU MaxSim kernels exist | Vector-DB vendors (Vespa, Qdrant, LanceDB) already ship it | High, but the result would be a reproduction | BEIR, LoTTE, ViDoRe | **Scooped; drop** |
| F | Retrieval-aware GPU scheduling / prefetch (CPU-GPU IVF, pipelining) | TeleRAG (MLSys'26; 3.8× on 4×H200), RAGO (ISCA'25), RAGDoll, PipeRAG, VectorLiteRAG | **Low–medium.** Strong systems groups own it; gains show mostly at multi-GPU scale | Cloud inference providers | Low on modest GPUs | NQ, HotpotQA, TriviaQA; vLLM+FAISS | Lower priority |

## Recommendation: VaultKV

### Pitch
Chunk-level KV reuse is how RAG serving gets 2–8× faster, but today's implementations treat a document chunk's KV as an immutable, globally shareable blob keyed by token hash. Real enterprise corpora change (policies get revised, records get corrected, documents get deleted)\[12\] and are permissioned (users lose access, tenants must not share side channels). VaultKV is a vLLM V1 KV-connector and control plane that keys every cached chunk-KV by (content hash, document version, ACL scope/epoch, tenant namespace), invalidates or repairs cached KV in place when the source changes, enforces revocation before any KV is loaded, and shares KV across tenants only when it is provably safe against timing (SpliceLeak) and contamination (HijackKV) attacks. The thesis: **correctness and isolation are achievable at near-global-sharing speed.**

### Architecture sketch
1. **Ingestion/CDC layer.** Document store → change-data-capture events (create/update/delete/ACL-change). Each chunk gets `chunk_id`, `content_hash`, `doc_version`, `acl_scope`, `tenant_ns`. Uses LangChain/LlamaIndex-style hash-based change detection to avoid redundant work.\[18\]
2. **KV Ledger (the core).** A metadata service (in-memory + SQLite/Redis) mapping `chunk_id@version` → KV location (GPU/CPU/disk via LMCache), plus a reverse index `doc_id → {chunk KVs, dependent in-flight requests}`. Every entry carries a **policy epoch**; any ACL or membership change bumps it.\[19\]
3. **Admission gate (scheduler side).** Hooks the vLLM V1 `KVConnectorBase_V1` scheduler methods (`get_num_new_matched_tokens`): a chunk-KV counts as a "hit" only if version matches the retriever's current version, the requester's ACL epoch authorizes it, and the tenant namespace allows sharing. Otherwise it is a miss → recompute. This is also where the content-based non-prefix lookup problem from LMCache issue #3238 is addressed.
4. **Edit-repair path.** For small in-place edits, apply budgeted edit-local recomputation (the 2609.17983 finding: a contiguous edit-local window recovers ≥0.94 of answer margin, 13–21× faster than full re-prefill) instead of evicting;\[20\] for length-changing edits or deletes, invalidate (Leyline-style replace/remove).\[12\]
5. **Isolation policy.** Three sharing tiers per chunk: *public* (global sharing), *tenant-shared* (namespace-salted keys, KVGov/SpliceDefense-style), *private* (per-principal). Defenses: namespace partitioning, TTFT padding/quantization for hits, and **provenance-pinned KV**, where a chunk's KV is only reused if it was computed from a trusted "canonical" context (e.g., position-0, system-template-only prefill), which directly blocks HijackKV-style poisoned-prefix KV.
6. **Agent OS integration.** Expose the ledger as a resource-control-plane module: per-agent/per-tenant KV budgets, eviction priorities, and isolation tier become Agent OS policies. AgentTraceDAG can visualize which chunk-KV versions each agent step consumed (useful for the demo, not a new verification product).

### MVP scope (what must work by the end of week 6)
- vLLM V1 + LMCache (CacheBlend or EPIC-style blending) on Qwen2.5-1.5B/7B or Llama-3.1-8B, single GPU.
- Ledger with version + ACL epoch + tenant namespace; scheduler-side admission gate; invalidate-on-edit/delete/revoke.
- Workload generator: HotpotQA/2WikiMQA RAG traffic with Zipfian chunk popularity, multiple tenants, and an interleaved stream of edits, deletes, and revocations.
- Reproduce one attack (SpliceLeak timing signal or HijackKV) against vanilla LMCache, then show it fails against VaultKV.

Out of scope for MVP: multi-node KV pool, learned token selection, multimodal KV.

### Original week-by-week plan (Oct 12 – Dec 13, 2026)
| Week | Dates | Deliverable |
|---|---|---|
| 1 | Oct 12–18 | Reproduce vLLM APC vs LMCache CacheBlend TTFT on 2WikiMQA (512-token chunks); confirm or fix the V1 non-prefix hit issue (#3238); open an upstream issue comment/PR |
| 2 | Oct 19–25 | Workload generator: tenants, ACL groups, Zipf chunk popularity, edit/delete/revoke event streams; stale-serve and post-revocation-serve detectors (ground truth from ledger) |
| 3 | Oct 26–Nov 1 | KV Ledger + version/epoch keys; scheduler-side admission gate in the KV connector; invalidation path |
| 4 | Nov 2–8 | Edit-repair path (edit-local budgeted recompute) vs invalidate-and-recompute; quality (F1/EM) vs latency curve |
| 5 | Nov 9–15 | Security: reproduce SpliceLeak-style TTFT probe and/or HijackKV on vanilla LMCache; implement namespace salting, TTFT padding, provenance-pinned KV |
| 6 | Nov 16–22 | Full evaluation matrix across baselines; ablations (each defense on/off, edit rate, revocation rate, tenant overlap) |
| 7 | Nov 23–29 | Agent OS integration (per-agent KV budgets + isolation tiers); multi-step agent workload (Search-R1-style traces) to show reuse across agent turns |
| 8 | Nov 30–Dec 6 | arXiv preprint (workshop-format, 6–8 pages), README with one-command benchmark, blog post with plots; PyPI/GitHub release |
| 9–10 (buffer) | Dec 7–20 | Stretch: multimodal page-KV (ColQwen + Qwen2.5-VL) or medical tenant demo; submit to a systems/ML workshop (e.g., MLSys/EuroMLSys/ICLR workshop cycle) |

### Evaluation plan
**Workloads:** HotpotQA, 2WikiMQA, MuSiQue (multi-hop RAG, matching CacheBlend/EPIC evaluation), LongBench subsets; synthetic mutation streams at 0.1–10% of chunks edited per hour; revocation streams; 2–16 tenants with controlled corpus overlap; optionally Search-R1 agent traces for repeat-retrieval patterns.

**Baselines:** full recompute; vLLM automatic prefix caching; per-tenant isolation via vLLM `cache_salt` (safe but slow); LMCache CacheBlend with global sharing (fast but unsafe); EPIC/LegoLink-style blending; TTL-based invalidation (what most teams do).

**Metrics:**
- *Speed/cost:* TTFT (mean, p50, p99), throughput (req/s at SLO), chunk-KV hit rate, recompute ratio, GPU-seconds per 1,000 queries, KV storage footprint.
- *Quality:* F1/EM vs full recompute (target: within 1–2 points, matching CacheBlend's "no quality loss" claim).
- *Correctness:* stale-version serve rate, post-revocation serve rate (target: exactly 0 for VaultKV), revocation-to-effect latency, invalidation latency at 100k entries.
- *Security:* SpliceLeak-style hit/miss distinguishability (AUC of TTFT classifier → target ≈0.5), HijackKV attack success rate (vs 94% reported on undefended reuse), throughput cost of each defense.

**Headline figure:** a Pareto plot of TTFT vs "unsafe serves," where vanilla CacheBlend sits fast-but-unsafe, per-tenant isolation sits safe-but-slow, and VaultKV sits near the fast corner with zero unsafe serves.

### Why this is the best choice for both tracks
- **Research value:** It is a crisp systems question: can position-independent KV reuse be made consistent and isolated without losing its speedup? It has a measurable Pareto result and a security angle relevant to KV-cache and serving research.
- **Industry interviews:** It touches exactly what infra teams are hiring for in 2026: vLLM V1 internals, KV connectors, LMCache, cache invalidation/CDC, multi-tenancy, SLOs, and security reviews. "I found and fixed why non-prefix reuse wasn't hitting in vLLM V1, then made it safe for multi-tenant enterprise RAG" is a strong ML Engineer / Research Engineer / MLOps story.
- **No overlap:** It does not touch retrieval-confidence self-correction (Self-Healing RAG), groundedness/attribution (VectorLens), tutoring (UnMask), or clinical provenance across agent traces (ClinTrace). Its correctness notion is cache consistency and access control, not answer verification.

### Optional differentiation
- **Medical angle (low-cost, high-signal):** Run a demo where tenants are hospitals and ACL scopes are patient/care-team permissions over de-identified CT-RATE radiology reports; show that a revoked clinician's next query never touches the patient's cached report KV. This links VaultKV to FOVEA-Net/medical AI without turning it into a CT-RAG paper (Candidate D's crowded space).
- **Multimodal stretch:** Extend the ledger to page-image KV for a VLM (ColQwen retrieval + Qwen2.5-VL generation), positioning against QV-PIC/VLCache, which do not address versions or tenants.

## Caveats
- Many cited 2026 papers are fresh arXiv preprints (SpliceLeak, HijackKV, Tessera, 2609.17983, KVGov, REVA, FinCacheServe); some numbers are single-group, simulation-based, or not yet peer reviewed (one independent review called SpliceLeak's evidence under continuous batching "thin").\[21\] Re-measure everything you build on rather than quoting it.
- The "no combined system exists" claim rests on a focused literature/open-source search (around a dozen targeted queries plus fetches), which is strong but not exhaustive; re-check arXiv weekly through November, because the space moves monthly. If someone publishes the combination, pivot the contribution to the vLLM V1 implementation + attack-defense evaluation, which remains valuable.
- LMCache/vLLM V1 APIs change quickly; the #3238 behavior may already be fixed on `dev`.\[9\] Pin versions early.
- Security results depend on threat model; state the model explicitly (client-only attacker, same privileges as ordinary users) as SpliceLeak and CachePrune do.\[22\]
- Edit-local repair is validated only for small, length-preserving edits with adjacent dependent text (quality drops to 0.01–0.09 margin recovery when the answer-bearing text is 250 tokens downstream); default to invalidation outside that regime.\[20\]

## Sources

1. [ViDoRe V3: A Comprehensive Evaluation of Retrieval ...](https://aclanthology.org/2026.acl-long.755.pdf)
2. [ViDoRe Benchmarks for Visual Retrieval](https://www.emergentmind.com/topics/vidore-benchmarks)
3. [Agentic-R: Learning to Retrieve for Agentic Search](https://aclanthology.org/2026.findings-acl.785.pdf)
4. [Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges](https://arxiv.org/pdf/2506.10408)
5. [DEMYSTIFYING AND ENHANCING THE EFFICIENCY OF ...](https://proceedings.iclr.cc/paper_files/paper/2026/file/44bed3ea6eb269aa71ef1f9630c0139b-Paper-Conference.pdf)
6. [Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents](https://arxiv.org/html/2609.22486v1)
7. [Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG](https://arxiv.org/pdf/2606.21842)
8. [ICML Poster EPIC: Efficient Position-Independent Caching for Serving Large Language Models](https://icml.cc/virtual/2025/poster/43926)
9. [\[Bug\] CacheBlend non-prefix KV cache reuse broken with vLLM V1 connector · Issue #3238 · LMCache/LMCache](https://github.com/LMCache/LMCache/issues/3238)
10. <https://arxiv.org/html/2607.26076>
11. [REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving](https://arxiv.org/html/2609.11209)
12. [Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits](https://arxiv.org/pdf/2609.17983)
13. [Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference](https://arxiv.org/pdf/2608.09225)
14. [\[2606.21842\] Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG](https://arxiv.org/abs/2606.21842)
15. [Prompt Leakage via KV-Cache Sharing in Multi-Tenant ...](https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf)
16. [HijackKV: New Threat in Position-Independent KV Cache Reuse](https://arxiv.org/abs/2607.19957)
17. [HijackKV: New Threat in Position-Independent KV Cache Reuse](https://arxiv.org/pdf/2607.19957)
18. [RAG Knowledge Base Freshness: The Staleness Problem Teams Solve Last - TianPan.co](https://tianpan.co/blog/2026-04-20-rag-knowledge-base-freshness-index-rot)
19. [Access Revoked. Why Can Your RAG App Still Answer? - DEV Community](https://dev.to/naveen_alavilli/access-revoked-why-can-your-rag-app-still-answer-dfh)
20. [Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits](https://arxiv.org/html/2609.17983)
21. [Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG · Pith Review](https://pith.science/paper/2606.21842)
22. [CachePrune: Privacy-Aware and Fine-Grained KV Cache Sharing for Efficient LLM Inference](https://arxiv.org/pdf/2605.23640)
