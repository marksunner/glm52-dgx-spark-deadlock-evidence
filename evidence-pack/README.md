# Evidence pack — GLM-5.2 TP=4 deadlock campaign on 4× DGX Spark (GB10/sm_121)

Sanitized public receipts for the claims made in:
- the RFC vllm-project/vllm #48720 results update (2026-07-17/18),
- the two companion PR drafts (sm12x Triton sparse-MLA selector; MTP drafter-gate TP consensus guard),
- the vllm #41725 comment,
- the NVIDIA-facing sparse_mla_sm120 mbarrier-livelock dossier.

**Sanitization policy:** internal hostnames, LAN/cluster IPs, usernames, and
filesystem paths are replaced with bracketed placeholders (`<node-rank0>`,
`<head-node-ip>`, `<home>`, `<deploy-dir>`, …). Nothing else was altered; log
lines, timestamps, counters, and stack contents are verbatim. Full raw bundles
are retained privately and can be shared with maintainers on a trusted channel.

**Node labels:** capture filenames and pack text refer to nodes by TP rank
(`rank0`..`rank3`); `rank0` is the head node.
GLM-5.2 int4-int8mix, TP=4, MTP spec=4; vLLM 0.23.1rc1.dev893;
NVRM 580.159.03; flashinfer JIT sparse_mla_sm120 kernels (pre-cure route).

## cuda-gdb/ — the sparse_mla_sm120 livelock receipt (2026-07-17 12:10Z wedge)
- `CUDA-GDB-SUMMARY.txt` — cuda-gdb attach on the live wedge, rank 0:
  `sparse_mla_prefill_kernel<ModelType2,ComputeMode0,16,2048,64>`, grid (120,1,1),
  single resident block (60,0,0), warps 8–11 spinning at
  `SYNCS.PHASECHK.TRANS64.TRYWAIT` (mbarrier expect-tx race).
- `nvidia-smi-during-wedge.txt` — the spin signature: 96 % GPU util, 0 % memory
  utilization, ~18.4 W.
- `power-during-wedge.txt` — power draw sample during a same-class wedge (~21 W).

## r1-capture/ — mechanism #1 live capture (R1, 2026-07-16 12:59:25Z, 200K flashinfer stack)
Boundary prompt 199,872 tok + 128 decode on a warm cluster; drafter-gate guard in
observe mode. Per rank (`rank0..rank3`):
- `rank*-gate-lines.txt` — the guard's full journal: `GLM52_GATE ARMED`,
  telemetry, and the 12:59:25Z `WOULD-ENFORCE` / `GLM52_RAS_TRIGGER_SIG` /
  `GLM52_RAS_DUMP` lines (per-rank would-enforce totals 123/122/122/122 across
  the four files; NCCL RAS frozen at 19500/19500/19500/19499).
- `rank*-docker-tail.txt` — server log tails in the wedge window.
- `rank*-memlog.txt` — free-memory samples (0x51 exoneration context).
- `R1-verdict.txt` — the one-line capture verdict as journaled.

## stall-dossiers/ — the sub-boundary freeze class (n=8, later root-caused to cuda-gdb/)
- `stall1-20260717T0702Z-README.md` — stall #1 (100K cold prefill, RAS spread 1,
  host spin in marlin_gemm — the initial, later-retracted marlin/UMA read).
- `stall2-20260717T0902Z-README.md` — stall #2 (fleet-wide equal-freeze, spread 0,
  all four hosts in fused_marlin_moe; falsified the seqs-cut remedy and the
  single-node-hardware theory).
- `stall3-20260717T1007Z-README.md` — stall #3 (froze WITH multiple GB free —
  memory-as-sole-cause falsified; device 96 % util at 18.4 W = spin, not compute).
- `ras-final-stall-*.txt` — frozen NCCL RAS per-rank collective-count snapshots
  for all eight stalls (timestamps in filenames), showing spread 0/1 — victim-
  timing artifacts, not consensus divergence.
- `R2a-void-TypeB-20260716T1447Z-verdict.txt` — the pre-reboot R2a freeze, later
  unified with this class ("rank 2 frozen inside flashinfer sparse-MLA kernel").

Note: the READMEs are verbatim as written mid-campaign; stalls #1–#2 carry the
marlin/UMA interpretation that the cuda-gdb receipt later retracted (marlin was a
bystander behind a jammed launch queue). They are included unedited for honesty;
read them with `cuda-gdb/CUDA-GDB-SUMMARY.txt` and the dossier as the final word.

## soak-logs/ — R-series, overnight statistics, and the 200K probe (Triton route, post-cure)
- `R1-observe-climb-soak30-20260717T1247Z.log` / `R2-enforce-climb-soak30-20260717T1310Z.log`
  / `R3-control-climb-soak30-20260717T1333Z.log` — driver logs for the three
  30-session ceiling rounds at 120K (observe/enforce/off): 64-stage staged climb,
  then 30 boundary sessions each, `SOAK DONE rc=0` at 12:57:24Z / 13:20:19Z /
  13:43:51Z. 90/90 clean.
- `R1-observe-gate-lines-rank*.txt` — per-rank `GLM52_DRAFTER skip n=1..20`
  reachability receipts (lockstep across ranks) from the observe round.
- `R2-enforce-30x-20260717T1320Z.log`, `R3-control-30x-20260717T1343Z.log` —
  server-side round readouts banked at round end.
- `soak-statistics-470-20260717T1355Z.log` — the overnight 470-session statistics
  run (with the R2 enforce round: 500 consecutive enforce ceiling sessions, zero
  wedges). Session 470's incoherent tail (temp-1.2 sampling artifact, 11 s vs
  4–5 s norm) is visible at the end — disclosed in the write-ups.
- `climb-200k-probe-20260718T0506Z.log` + `-part2.log` — the 200K probe: staged
  cold climb to 199,872 tok, zero stalls (driver pacing swap between parts is
  client-side only).
- `boundary-session-200k-20260718.log` — the boundary session at seq
  199,997–200,000: 128 tokens, finish=length, ~26 tok/s.

## ras-dumps/ — NCCL RAS per-rank launched-op-count excerpts
- `incident-A-20260715T0636Z-ras.txt` / `.json` / `-reason.txt` — incident A
  (production traffic, 2026-07-15): TP comm frozen MISMATCH, per-rank op counts
  27911/27890/27872/27860 (spread 51), step counters frozen equal — the observed
  ~50-collective corrupt-and-continue window cited in the gate PR.
- `R1-ras-dump-excerpt-rank0.txt` — R1's trigger signature + full RAS dump
  (frozen 19500/19500/19500/19499, rank 3 one AllReduce short).
- `R2a-typeB-contrast-ras.json` — a Type B (equal-freeze, spread 0) specimen for
  contrast: what a NON-divergence wedge looks like in the same instrument.
