# GLM-5.2 on 4x DGX Spark: FlashInfer sparse-MLA mbarrier livelock
Root-cause evidence, validated workaround, and reproducer pack.

**Status (July 2026):** deadlock reproduced repeatedly ✅ · cuda-gdb capture obtained ✅ · workaround validated (560+ sessions, 200K restored) ✅ · awaiting upstream investigation ⏳

**Purpose:** this repository exists to give FlashInfer, vLLM, and NVIDIA developers sufficient evidence to reproduce, diagnose, and ultimately fix the sparse-MLA mbarrier livelock, and, until then, to give affected operators a validated workaround.

**What failed:** Serving GLM-5.2 (TP=4, MTP) on GB10/sm_121, any cold-prefill-heavy request could livelock a rank GPU inside FlashInfer's sparse_mla_sm120 attention kernels, wedging the whole cluster (8/8 at >=60K-token cold prefills).

**What we captured:** cuda-gdb on a live wedge: a single resident block spinning at an mbarrier expect-tx wait (SASS receipt), 96% GPU util at ~18W with zero memory traffic. See nvidia-addendum/.

**The workaround (not a kernel fix):** route the main model's attention to the portable Triton sparse-MLA implementations (--attention-backend FLASHMLA_SPARSE + sm12x Triton drop-ins; kernels by @jasl). Zero wedges since.

**How tested:** 560+ context-ceiling sessions clean incl. 500 consecutive under the strictest config, a ~15h unattended overnight, and a cold staged climb to 200,000 tokens with a clean boundary completion, the exact workload that previously wedged 8/8. Decode ~25-27 tok/s.

**Also documented:** a separate, earlier-captured mechanism, rank-divergent MTP drafter gating at the context ceiling, plus a TP-consensus guard (safety-proven, efficacy honestly unexercised: the disease did not reproduce on the Triton route, which is consistent with, but does not prove, the kernel-nondeterminism seed hypothesis). See the RFC update.

**Where to go:** NVIDIA/kernel engineers: nvidia-addendum/. vLLM maintainers: rfc-48720-update-20260717.md and pr-drafts/. Raw receipts: evidence-pack/ (start with its README). Two separate platform bugs remain open and are NOT addressed by the workaround: the sm_12x sampler event-sync hang (vllm #41725) and the GB10 0x51 UMA memdesc leak.

One cluster, one model family; replication invited, configs of record are in the RFC update.
