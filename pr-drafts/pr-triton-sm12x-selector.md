# PR draft — [Platform/Attention] Expose Triton sparse-MLA as a selectable fallback for sm12x (GB10) main-model attention

**Status: DRAFT — staged for Mark's coordinated wave. Do not push before gates 2+4 are green (ledger, publication gates). Ref: vllm-project/vllm RFC #48720.**

## Title (proposed)
`[Bugfix][Platform] sm12x: add FLASHMLA_SPARSE (Triton sparse-MLA) to the MLA-sparse backend candidate list — flashinfer sparse_mla_sm120 livelocks on GB10`

## Summary
On sm_121 (GB10 / DGX Spark) the flashinfer `sparse_mla_sm120` attention kernels can livelock the device during cold (non-prefix-cached) prefill. The livelock is probabilistic and scales with cold-prefill tile count; once it engages, the GPU reports ~96 % utilization at near-idle power (~18 W, 0 % memory throughput), the launch queue never drains, and the whole TP group wedges behind the stuck rank's next collective. `ncclCommAbort` does not recover sm_121 under graph replay; the only recovery is process exit (twice-proven; the receipts live in RFC #48720 §5, the prevention-not-recovery platform facts — not re-argued here).

On capability-12 devices the MLA-sparse candidate list currently resolves to `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` (`vllm/platforms/cuda.py:130` in 0.23.1rc1.dev893+gd3a66aa7e); for sparse-MLA models (e.g. GLM-5.2 / DeepSeek-lineage MTP configs) the flashinfer sm120 path is the only viable pick — boot log receipt: `Using FLASHINFER_MLA_SPARSE_SM120 … out of potential backends: ['FLASHINFER_MLA_SPARSE_SM120']` — so affected hardware has no escape hatch.

Enum-to-implementation map, for reviewers cold to the sparse-MLA backend zoo: `FLASHINFER_MLA_SPARSE_SM120` is the flashinfer `sparse_mla_sm120` CUDA kernel family (the livelocking one). `FLASHMLA_SPARSE` upstream binds the native FlashMLA CUDA extension (`vllm._flashmla_C`), which does not build/exist on sm12x; on our deployment the sm12x Triton drop-ins rebind its two sparse ops (`flash_mla_sparse_fwd`, `flash_mla_with_kvcache`) to portable Triton implementations — the Triton kernels come verbatim from the deepseek_v4 path of jasl's vLLM fork (credit @jasl) — so selecting `FLASHMLA_SPARSE` routes to Triton there. `TRITON_MLA` is the dense-MLA Triton backend, not a sparse-MLA implementation, so it is not viable for this model class — which is why the flashinfer path is today's sole effective candidate.

**Ask:** make the Triton sparse-MLA implementation (`FLASHMLA_SPARSE` backend enum) a member of the sm12x candidate list (ranked above the flashinfer sm120 path until the kernel race is fixed), or at minimum honor `--attention-backend FLASHMLA_SPARSE` for the **main model** on sm12x. The Triton kernels have no inter-block mbarrier/TMA dependencies and are structurally immune to this livelock class.

## Evidence (cuda-gdb on a live wedge, 2026-07-17)
- Spinning device kernel: `sparse_mla_prefill_kernel<(ModelType)2,(ComputeMode)0,16,2048,64>` (flashinfer `sparse_mla_sm120` family, `csrc/sparse_mla_sm120_prefill.cu`), param type `PrefillColdParams`.
- Grid (120,1,1) with **one resident block**, warps 8–11 all lanes active, non-divergent, spinning at `SYNCS.PHASECHK.TRANS64.TRYWAIT P0[UR8+0x14980]` → `@P0 BRA` loop — an **mbarrier phase TRYWAIT spin: TMA/expect-tx arrive-wait race**.
- Device signature during wedge: 96 % GPU util at ~18 W, 0 % memory utilization, clocks P0; host threads piled in `cuLaunchKernel` behind the jammed queue (making innocent kernels — marlin GEMMs, Triton ops — look guilty in host stacks; they were bystanders).
- Exonerations by receipt: RoCE fabric hw_counters all zero during a live wedge; driver/kernel identical pre/post-reboot (NVRM 580.159.03, kernel 6.17.0-1026); memory headroom falsified as cause (froze with 4.4–7.1 GB free); chunk size (8192→2048), max_num_seqs (6→3), served ceiling (200K→120K) all falsified as remedies, n≥2 each.
- Blast radius: 8 distinct freeze incidents across 2 days, on fresh boots, all previously misattributed (marlin/UMA, memory exhaustion, single-node hardware). One incident wedged on the FP8 decode-kernel route as well (MIN_HEADS reroute) — the race lives in shared sm120-family machinery, not one kernel.
- Full dossier (attach or link): `nvidia-addendum/sparse-mla-sm120-mbarrier-livelock-20260717.md`; NVIDIA bug filed separately.

## Fix validated (route swap = the only variable)
Deployment fact, stated precisely: the sm12x Triton drop-ins were installed and exercised by the **drafter** throughout the campaign — a constant, present in every round on both sides of the swap — and the flipped variable was the **main model's backend selection only**. Routing the main model to the Triton sparse-MLA stack (`--attention-backend FLASHMLA_SPARSE`) with everything else pinned:
- The failing regime (cold staged prefill) went from 8/8 wedges to clean on the first attempt: 64-stage climb through 119,856 tok, every stage OK.
- 90 consecutive context-ceiling sessions (seq = ceiling−3..ceiling, temp 1.2) across all three drafter-gate modes, zero wedges; extended to **500 consecutive ceiling sessions** overnight, zero wedges.
- Determinism spot-check (not a general correctness proof): the smoke completion — fixed prompt `"The capital of France is"`, temperature 0, max_tokens 16 — returned byte-identical text across every healthy boot pre- and post-route-swap; the comparison scope is that single 16-token greedy completion.
- The drafter had run the same Triton route at every decode step of the campaign — long-baked-in evidence of stability.

## Proposed change (shape)
1. `vllm/platforms/cuda.py` (capability-12 MLA-sparse selection): add `FLASHMLA_SPARSE`'s Triton implementation to the candidate list for sm12x, preferred over `FLASHINFER_MLA_SPARSE_SM120` until the mbarrier race is resolved upstream; keep flashinfer selectable explicitly.
2. Ensure `--attention-backend FLASHMLA_SPARSE` is honored for the main model on sm12x (today it is effectively reachable only via spec-config `attention_backend` for the drafter).
3. Log the fallback reason when flashinfer sm120 is skipped/deprioritized on sm12x so users can correlate with this issue.

## Caveats / honest scope
- The livelock is probabilistic per cold-prefill tile count: quiet short-context or warm-cache serving will NOT reproduce it; staged cold prefill at ≥2.4K-token deltas reproduced it within seconds on our fleet.
- Perf: Triton route throughput vs the flashinfer baseline is measured in our write-up (the cure's cost is a number, stated there); selector preference may want a perf note.
- Generalization, softened to what we actually know: the race is in kernel-internal barrier logic and plausibly family-wide across sm12x; our validation covers only one model family (GLM-5.2 int4-int8mix, TP=4, MTP spec=4) on GB10 4× DGX Spark; other sm12x targets are untested.

## Refs
- RFC vllm-project/vllm #48720 (this campaign's RFC; §4/§5 mechanism material, results update posted alongside this PR).
- Issue #41725 (sm_12x sampler event-sync hang — separate disease, same platform; comment cross-posted).
- NVIDIA dossier: sparse-mla-sm120 mbarrier livelock (filed separately; kernel fix is theirs, this PR is the vLLM-side escape hatch).
