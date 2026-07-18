# GB10 / sm_121: flashinfer sparse_mla_sm120 kernels livelock in mbarrier
# phase TRYWAIT under cold-prefill load (2026-07-17, castle cluster)

## Summary
On a 4-node DGX Spark (GB10, NVRM 580.159.03, kernel 6.17.0-1026-nvidia,
vLLM 0.23.1rc1.dev893, flashinfer JIT sparse_mla_sm120 kernels), serving
GLM-5.2 (DeepSeek-V4-class sparse MLA, fp8_ds_mla cache, TP=4), any
cold-prefill-heavy request probabilistically hard-wedges one rank GPU:
the device spins forever inside a sparse-MLA kernel, the host launch queue
fills, and every host thread ends blocked in cuLaunchKernel. Probability
scales with cold-prefill size: >=60K-token prefills wedged ~always (8/8
observed over one day, two served-context configs 200K and 120K); 120-token
COLD PREFILL requests (fresh prompts computed from scratch, no prefix-cache
hit) wedged occasionally (1/3); per-step decode work (M<=24 tokens per
engine step) never wedged. Basis for the decode claim: the journaled
per-rank engine-step totals across the campaign — the gate's periodic
self-report milestones in the rank logs (`GLM52_GATE OK ... steps=20000
div=0 forced=0` per rank in the overnight statistics window alone) plus the
per-round step-counter receipts, cumulatively ~30,000+ engine steps per rank
across boots (>120,000 rank-steps fleet-wide), journaled in the soak driver
logs and gate milestones in the evidence pack. Prefix-cache-warm prefills
(small computed suffix) never wedged.

## The device-side receipt (cuda-gdb attach on a live wedge, 2026-07-17
## 12:10Z, rank 0; bundle climb-stall7-cudagdb-20260717T1210Z)
Attach: docker exec -u root --privileged, /usr/local/cuda/bin/cuda-gdb -p.
  info cuda kernels ->
    Kernel 0, Dev 0, Grid 337207, Status Active, SMs Mask 0x000080000000,
    GridDim (120,1,1), BlockDim (384,1,1):
    sparse_mla_prefill_kernel<(ModelType)2,(ComputeMode)0,16,2048,64>
    (params type PrefillColdParams)
  Focus block (60,0,0): warps 8-11, ALL 32 lanes active, divergent mask 0,
  identical Active PC on all warps:
    => @P0 BRA 0x...f670                (loop back)
       BSYNC.RECONVERGENT B0
       EXIT
       SYNCS.PHASECHK.TRANS64.TRYWAIT P0[UR8+0x14980],R4
  I.e. a single resident block spin-looping on an mbarrier phase check
  (TRANS64 = transaction-count/arrive-wait barrier) whose expected phase
  never arrives — consistent with a TMA/cp.async.bulk expect-tx accounting
  race or a producer warp/block that already exited.

## Corroborating signatures (sanitized excerpts in `evidence-pack/`)
- nvidia-smi during wedge: utilization.gpu 96 %, power ~18 W,
  utilization.memory 0 %, clocks P0/2535 MHz, persistence on — a spin loop,
  not compute.
- All four ranks' host py-spy natives: MainThread sched_yield <- 10 libcuda
  frames <- cuLaunchKernel (queue full behind the spinning kernel); the
  Python-level frame is arbitrary (marlin_gemm, MoE workspace, triton
  launcher — bystanders).
- NCCL RAS per-rank collective counts freeze at/near the pre-request value
  (the pending TP all-reduce never launches/completes); spread 0 or 1
  depending on victim timing.
- RoCE exonerated: /sys/class/infiniband/*/ports/1/hw_counters ALL ZERO
  (out_of_sequence, packet_seq_err, local_ack_timeout_err, req/resp_cqe_
  error, ECN/CNP) on every port of every node DURING a live wedge.
- Driver/kernel regression excluded: NVRM 580.159.03 and kernel identical
  across pre/post-reboot boots; last apt upgrade 07-10.
- Independent of: free UMA (wedges with 5–7 GB avail and with ~1 GB),
  max_num_batched_tokens (8192 and 2048), served ceiling (200K and 120K),
  max_num_seqs (6 and 3), victim rank (rank 2 x3, rank 1, fleet-wide,
  rank 0 — rank-agnostic), our drafter-gate instrumentation
  (off/observe/enforce).
- A second wedge occurred with prefill routed through the FP8 *decode*
  kernel of the same family (mixed-batch mode) — the race is not exclusive
  to the BF16 prefill kernel. NOTE: cuda-gdb attach HANGS on that wedge
  state (attach succeeded on the prefill-kernel wedge).

## Bundles (public excerpts: `evidence-pack/cuda-gdb/` +
## `evidence-pack/stall-dossiers/`; full raw bundles retained privately)
climb-stall2-20260717T100720Z (py-spy x4 plain+native, nvidia-smi, freem,
0x51 counts, README-ANALYSIS.md), climb-stall4-20260717T1048Z,
climb-stall7-cudagdb-20260717T1210Z (CUDA-GDB-SUMMARY + observe-round log),
plus earlier same-class captures: climb-stall-20260717T070242Z,
climb-stall2-20260717T0902Z (then misattributed to marlin/UMA), and R2a
bundle R2a-void-TypeB-20260716T144754Z (pre-reboot; "rank 2 frozen inside
flashinfer sparse-MLA kernel" — same bug).

## Workaround in production here
Route the main model's attention to the portable Triton sparse-MLA
implementations (vLLM --attention-backend FLASHMLA_SPARSE + the sm12x Triton
drop-in patch stack): zero wedges since. Validation as of 2026-07-18:
560+ context-ceiling sessions clean (including 500 consecutive at seq
119,997–120,000, client-logged, plus a ~15 h unattended overnight window),
and a cold staged climb to 199,872 tokens (64 stages, zero stalls) followed
by a clean boundary completion at exactly 200,000 tokens — i.e. the same
cold-prefill workload that wedged 8/8 on the flashinfer route completes
routinely on Triton, at no decode-throughput cost (~25–27 tok/s at the 120K
boundary, ~26 tok/s at the 200K boundary, vs the ~23 tok/s flashinfer
baseline). Structural claim, qualified: by inspection of the Triton
implementations (the sm12x drop-in stack — `sparse_mla_kernels.py` +
`sm12x_sparse_mla_attn.py`, Triton sparse-MLA kernels from the jasl/vllm
deepseek_v4 path), the Triton kernels have no inter-block mbarrier/TMA
dependencies — each program is self-contained — so this livelock class has
no mechanism there; the empirical record above is the primary evidence.

## Asks
1. Review sparse_mla_sm120 prefill/decode mbarrier expect-tx logic for
   sm_121/GB10 (source ships in flashinfer: data/csrc/sparse_mla_sm120*.cu,
   include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh).
2. The separately-filed GB10 0x51 UMA memdesc leak (see the 0x51 addendum
   files in this directory) remains unfixed and independent.
