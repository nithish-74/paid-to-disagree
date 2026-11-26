# "Do Councils Cave?" — NAACL 2027 SRW outline (4 pages + references)

Working outline for the fallback paper (architecture doc: Phases 1–2). Each section lists
what it must say, the facts and numbers it uses, and where they come from. The authors write
the prose. Numbers are never typed by hand: they come from `analysis/paper_tables.py`
(`tables.tex`, `summary.json`) run on the combined Kaggle outputs.

**Deadline:** 11 Jan 2027. **Format:** 4 pages + unlimited references (check the SRW call
for the template, anonymity rules and whether an appendix is allowed).

**Before submitting, check:**
- [ ] SRW call: page limit, anonymous submission, mentoring option, appendix policy.
- [ ] ACL Responsible NLP checklist and the venue's policy on AI-assisted writing and code,
      and your college's policy; disclose what they require.
- [ ] Every citation below verified (title, authors, venue, arXiv id). Paper 7 in the
      architecture doc ("Peer-Predictive Self-Training") is marked *not yet verified*.
- [ ] Every number traced to a file in `runs/` (architecture doc integrity rule).
- [ ] Negative results reported, not hidden.

---

## Title (options)
- Do Councils Cave? Sycophancy and Correlated Errors in Small-Model LLM Councils
- Do Councils Cave? Pressure-Testing Small LLM Councils and Their Aggregators

## Abstract (~150 words) — write last
Must contain, in this order:
1. Problem: LLMs change correct answers under pressure; councils of LLMs are proposed as a fix.
2. What we test: councils of three 1–1.5B models from different families (Qwen2.5-1.5B,
   Llama-3.2-1B, Gemma-3-1B) under user and peer pressure, three protocols, five aggregators.
3. Headline finding 1: caving rate under user pressure → `summary.json:cave_rate.user_wrong`.
4. Headline finding 2: do aggregators beat majority vote? → `summary.json:aggregation`.
5. Headline finding 3: error correlation → `summary.json:error_corr`.
6. Pilot: single-model BTS-GRPO changes flip-to-wrong by X with accuracy change Y (B9 run).
7. One sentence on what this implies for training councils (motivates the full project).

## 1. Introduction (~0.6 page)
- **Sycophancy:** models flip correct answers when a user pushes back. Cite the sycophancy
  literature you use for "are you sure?" and answer-suggestion tests (SycophancyEval).
- **Councils as a proposed fix:** Karpathy's LLM Council (Nov 2025) — several models answer,
  rank each other anonymously, a chairman decides.
- **Why it may not help:** members may conform to each other; *Consensus is Not
  Verification* (ICML 2026, arXiv 2603.06612) shows models share mistakes, so voting and
  aggregation gain little.
- **Research questions for this paper** (subset of the full project):
  - RQ1: How often do small-model councils cave to a user's wrong answer, and to a peer
    arguing for a wrong answer? (baseline for H1, H5)
  - RQ2: Do protocols (independent, Karpathy-style review, debate) or aggregators
    (confidence-weighted, surprisingly popular, Dawid–Skene, chairman) beat majority vote?
    (baseline for H4)
  - RQ3: How correlated are members' errors? (baseline for H3)
  - RQ4 (pilot): does a label-free BTS reward reduce caving for one model?
- **Contributions** (3 bullets): an open pressure-testing harness for councils (5 conditions ×
  3 protocols × 5 aggregators, every output logged); measurements on 4 benchmarks with
  3 seeds and 95% CIs; a single-model BTS-GRPO pilot.

## 2. Related work (~0.5 page)
Group, one short paragraph each; say what each did and how this paper differs.
- **Sycophancy and pressure tests:** SycophancyEval; SYCON-Bench; "are you sure?" studies.
- **Multi-agent councils and debate:** Karpathy LLM Council; Du et al. (multi-agent debate);
  Choi, Zhu, Li — *Debate or Vote* (NeurIPS 2025); trained debaters MAPoRL / MACA / DEAR
  (correctness or consensus rewards).
- **Aggregation beyond majority vote:** Ai, Pan, Simchi-Levi, Tambe, Xu — *Beyond Majority
  Voting* (ICML 2026, arXiv 2510.01499; OW, ISP); Prelec's surprisingly popular;
  Dawid & Skene (1979).
- **Correlated errors:** *Consensus is Not Verification* (ICML 2026).
- **Peer prediction for LLMs:** Mytsyk, Zhang, Krishnamurthy 2026 (arXiv 2608.25267;
  BTS reward in GRPO, one model scored against its own samples — closest to our pilot);
  Qiu, Carroll, Allen (ICLR 2026); Chen et al. (*Peer Elicitation Games*, DMI).

## 3. Setup (~1 page)
### 3.1 Council members
- Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, Gemma-3-1B-it: three families on purpose.
- Every member returns JSON `{reason, answer, confidence, peer_prediction}`; one retry on
  unreadable output; missing `peer_prediction` filled from option-token probabilities.
- Phase 1 check (100 ARC dev questions): invalid JSON qwen 0%, llama 1%, gemma 0%
  (`runs/kaggle_phase1/summary.json`). Dev accuracy 80 / 54 / 50%.
- Hardware note: Kaggle T4; Qwen and Llama via vLLM (fp16), Gemma via transformers (fp32,
  vLLM cannot run Gemma 3 on a T4). Temperature 0.

### 3.2 Data
- TruthfulQA-MC1, ARC-Challenge, MMLU-Pro (4-option subset), GSM8K (numeric).
  300 test questions each; MCQ reduced to 4 options with seeded distractors.
- (If time) SycophancyEval answer-suggestion / are-you-sure parts, closed-form SYCON-Bench.

### 3.3 Pressure conditions (C2)
Table or list: neutral; user_wrong ("I'm fairly sure the answer is X", X wrong);
user_right (X correct — tests over-resistance); are_you_sure (two-turn challenge);
peer_wrong (a planted council member argues for a wrong option). Wrong option seeded per
question, identical across conditions and seeds.

### 3.4 Protocols and aggregators (C4, C5)
- Independent; Karpathy-style (anonymised peer ranking → chairman = Qwen); debate (1 round).
- Majority vote, confidence-weighted, surprisingly popular, Dawid–Skene (EM), chairman.
  OW / ISP: *only if* the equations from Ai et al. are implemented in time — otherwise list as
  future work, do not report.

### 3.5 Metrics (C8)
- Accuracy; flip-to-wrong (right → wrong under pressure); flip-to-right; cave rate (moved to
  the pushed answer); peer-flip; pairwise error correlation (phi) and Cohen's kappa; ECE,
  Brier; aggregation gain over majority vote.
- Statistics: 3 seeds; mean over seeds with 95% t-interval across seeds and per-seed
  bootstrap intervals; a claim needs both (rule in `analysis/combine_seeds.py`).

## 4. Results (~1.3 pages) — fill from `analysis/paper_tables.py`
- **Table 1 (caving):** `tables.tex: tab:caving`. Points to make: which model caves most;
  user vs peer pressure; dataset differences (TruthfulQA vs ARC vs MMLU-Pro vs GSM8K).
- **Fig. 2 (harmful vs beneficial revision):** `fig_flips.pdf`. Does pressure mostly break
  right answers or fix wrong ones? Over-resistance under user_right?
- **Table 3 (protocols × aggregators):** `tables.tex: tab:aggregation`. Does any protocol or
  aggregator beat majority vote with † support? Does the chairman help or hurt?
- **Fig. 1 (error correlation):** `fig_error_corr.pdf`. Are errors correlated, and is it
  higher for some pairs or datasets?
- Report negative results plainly ("no aggregator beat majority vote on X").

## 5. Pilot: single-model BTS-GRPO (~0.4 page)
- Method: GRPO on Qwen (LoRA r=8, fp16, G=8, ≤256 tokens, lr 5e-6, no KL); reward = BTS
  information + prediction score (`configs/reward_b9.yaml`); peers = frozen copies of Qwen
  (its own cached samples), as in Mytsyk et al.; 600 steps or fewer (state the number).
- Result: dev flip-to-wrong and accuracy before vs after (`runs/b9_single_qwen/round1_qwen/
  monitor.jsonl`), test-set evaluation of the trained Qwen.
- Phase 2 criterion: flip-to-wrong down, accuracy drop ≤ 3 points. Report either way.
- Caveat to state: reward weights not yet matched to Mytsyk et al. → "single-model
  BTS-GRPO", not a replication (unless the TODO in `configs/reward_b9.yaml` is resolved).

## 6. Discussion and limitations (~0.3 page)
- Small models (1–1.5B), four benchmarks, closed-form answers only, one hardware setup.
- Gemma run in fp32 on a different backend than the other two members.
- Temperature 0: seeds vary the pushed wrong option and ranking order, not sampling.
- Self-consistency (B2) and 2-round debate (if not run) → future work.
- Next step (the full project): cross-model peer-prediction training (H1–H4).

## Ethics / broader impact (short, if required)
Sycophancy reduction must not become over-resistance to correct users (report flip-to-right
under user_right). All data public benchmarks.

## Appendix (if allowed)
`tables.md` appendix tables: members alone (accuracy, invalid rate, ECE, prediction KL),
flip rates per dataset with intervals.
