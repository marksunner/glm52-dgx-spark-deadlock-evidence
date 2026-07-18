# PR draft — [Spec Decode] TP-consensus guard for the MTP drafter gate (rank-divergent `input_fits_in_drafter` at the context ceiling)

**Status: DRAFT — staged for Mark's coordinated wave. Do not push before gates 2+4 are green (ledger, publication gates). Ref: vllm-project/vllm RFC #48720 §4/§5.**

## Title (proposed)
`[Bugfix][Spec Decode] Reach TP consensus on the drafter-pass gate before launching drafter collectives (fixes rank-divergent wedge at max_model_len boundary)`

## Summary
With MTP speculative decoding under TP>1, each rank independently evaluates the drafter gate (`input_fits_in_drafter`-style conditions) in `sample_tokens`. The decision depends on per-rank state that is only guaranteed identical if per-rank compute is bit-identical. Near the served context ceiling (the gate can only flip at `max_seq_len ≥ max_model_len − spec_tokens + 1`; with spec=4 that is ceiling−3), a bit-divergent rejection count is enough to make rank A run the drafter's collectives while rank B skips them. The TP group's collective sequences misalign; NCCL pairs by issue order, so either the next collective blocks forever (shape-unequal pairing) or the group keeps running misaligned until it wedges at a shape boundary. The second regime is not hypothetical — incident A's per-rank launched-op counts (27911/27890/27872/27860, spread 51; Evidence below) are a direct receipt that roughly 50 collectives launched past the divergence point before the wedge, i.e. an **observed** corrupt-and-continue window. That the shape-equal all-reduces inside that window completed with mixed activations (silent output corruption) is the **inference** NCCL's issue-order pairing forces; direct activation-level proof was not captured. On sm_121 there is no post-launch recovery (`ncclCommAbort` does not unwedge under graph replay); prevention must be **pre-launch**.

There is an existing dummy-run guard for the DP case; the TP case is unguarded (the divergence site we captured: the MTP MoE final all-reduce, intra-pass op #2).

## Evidence
- **Live capture (R1, 2026-07-16 12:59:25Z):** boundary prompt (199,872 tok + 128 decode) on a warm 4×-DGX-Spark TP=4 cluster → the gate guard in observe mode logged the four ranks' would-pass totals as **123/122/122/122** (each rank's decision independently journaled) → uncorrected divergence → RAS per-communicator launched-op counts frozen at 19500/19500/19500/19499 → cluster wedge. Hardware/driver exonerated in-window.
- **Incident A (2026-07-15):** same class from production traffic — TP comm frozen MISMATCH, per-rank launched-op counts 27911/27890/27872/27860 (spread 51 — the observed ~50-collective corrupt-and-continue window discussed in the Summary), step counters frozen equal.
- Geometry: with native `max_position_embeddings` ≫ served `max_model_len`, `effective_drafter_max_model_len` = served ceiling, so the gate is only exercisable within spec_tokens of the ceiling — quiet mid-context traffic proves nothing about this bug.

## Fix (deployed and validated on 4× DGX Spark, TP=4, GLM-5.2 + MTP spec=4)
In `sample_tokens`, **before any drafter collective is launched**, all-gather a small consensus tuple over the TP **gloo cpu_group** (the same group used by our fingerprint guard; CPU transport so it works under graph replay and never touches NCCL pre-consensus):

```
(proposed_pass_count, sched_count, fits, md_none)
```

and adopt **MAX** across ranks:
- `{0, N} → all ranks run the drafter pass.` Safe because the drafter clamps out-of-window positions and voids their KV writes — the receipt, in-tree: `vllm/v1/spec_decode/utils.py`, function `compute_new_slot_mapping()` (defined at line 241): `clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)` at line 258 prevents OOB block-table indexing, and positions `>= max_model_len` are masked to `PADDING_SLOT_ID = -1` at lines 266–267 (constant at line 11), so no KV-cache slot is ever written for them. Verified twice: read in the container-extracted deployed tree when the fix was built (2026-07-15), and re-confirmed by a read-only grep inside the running deployment's container (vLLM 0.23.1rc1.dev893+gd3a66aa7e, 2026-07-18 — the line numbers above are from that tree). A rank forced to run a pass it would have skipped therefore produces clamped, discarded proposals — never OOB.
- All-skip carve-outs: if any rank reports `md_none` (no metadata) or the scheduled-count split disagrees, all ranks skip — degenerate states converge to the safe side.
- Env-gated rollout: `GLM52_DRAFTER_GATE=off|observe|enforce` (default `off`). `observe` journals per-rank decisions + would-enforce totals without intervening (this mode produced the live capture above); `enforce` adopts the consensus.

## Validation (receipts available)
- **Safety: proven at the boundary.** 30 consecutive ceiling sessions under `enforce` at seq ceiling−3..ceiling: no crash, no OOB, no quality collapse, no false enforcement, engine steps lockstep across ranks. Extended overnight to **500 consecutive enforce ceiling sessions, zero incidents**. Gate jurisdiction demonstrably exercised — scoped to where the numbers come from: in each of the 120K-ceiling 30-session rounds (observe, enforce, and off), 20 `GLM52_DRAFTER skip` events per rank, lockstep-identical across all four ranks; the single 200K-ceiling probe session (2026-07-18) showed 3 skips per rank, again lockstep 4/4, consistent with its session count.
- **Mode sweep: 90 consecutive ceiling sessions clean across all three gate modes** (30 observe / 30 enforce / 30 off) — the guard neither helps nor harms when the underlying per-rank compute is deterministic.
- **Efficacy: honestly marked NOT EXERCISED.** After the attention-backend route change that removed our platform's kernel nondeterminism (routing to the sm12x Triton sparse-MLA drop-ins, whose Triton kernels are @jasl's, from the jasl/vllm deepseek_v4 path — see companion sm12x PR), zero divergence events occurred to enforce against. The R1 observe capture (123/122/122/122 on the prior stack) remains the live specimen of the disease. The guard is defense-in-depth: any future source of per-rank nondeterminism (kernel upgrade, autotune change, different hardware) re-arms the trap at the ceiling, and this gate is what stands in front of it.

## Design notes / known caveats
- **Transport cost:** the consensus is a synchronous gloo all-gather on the hot path, once per spec-decode step. Measured overhead was noise-level on our fleet (gloo over a dedicated NIC), but the synchronous-gather-on-hot-path pattern is the documented landmine of this v1 design; async or piggybacked-sync-point transports are the redesign direction if the cost matters at scale.
- **MAX vs MIN:** MAX ("all run") was ruled over MIN ("all skip") because the drafter's clamp makes run-side convergence provably safe, while skip-side convergence silently changes accept-length statistics at the boundary; ruling + criteria pre-registered before the verdict rounds.
- Prevention is pre-launch **only**: once a mismatched collective launches, ranks are in device spin with no host control point (twice-proven on sm_121; do not design for mid-step graceful degradation).
- Naming: `GLM52_*` env vars are our deployment's; upstream should rename to a vLLM-appropriate flag (e.g. `VLLM_SPEC_GATE_CONSENSUS=off|observe|enforce`) — semantics as above.

## Refs
- RFC vllm-project/vllm #48720 — §4 (mechanism #1), §5 (prevention-not-recovery platform facts), §10; results update posted alongside this PR.
- Companion PR: sm12x Triton sparse-MLA selector (removes this platform's nondeterminism seed; this PR is the generic guard).
- Evidence pack (sanitized public receipts, staged alongside this PR): `evidence-pack/r1-capture/` (R1 boundary-capture summary + per-rank gate-line counts), `evidence-pack/ras-dumps/` (NCCL RAS per-rank op-count excerpts for R1 and incident A), `evidence-pack/soak-logs/` (R-series 30+30+30 rounds, the 500-session statistics run, and the 200K probe driver logs). Full raw bundles retained privately.
