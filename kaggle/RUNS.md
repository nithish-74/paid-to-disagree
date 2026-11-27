# Running the experiments on several Kaggle accounts

Every run below is independent of the others except where "needs" says otherwise, so each
team member can take rows and run them on their own Kaggle account (30 GPU hours a week each,
at most two GPU runs at a time per account). Each account must belong to a different person;
Kaggle does not allow one person to use several accounts.

## One-time setup per account (about 10 minutes)

1. Kaggle account with a **verified phone number** (needed for GPU and Internet in notebooks).
2. Create a notebook: kaggle.com/code -> New Notebook -> File -> Import Notebook -> Link, and paste
   the raw link of the notebook from the table below, e.g.
   `https://raw.githubusercontent.com/nithish-74/paid-to-disagree/main/kaggle/c7_train.ipynb`.
3. Settings (right panel or Settings menu): Accelerator **GPU T4 x2**, Internet **on**.
4. Optional: Add-ons -> Secrets -> `HF_TOKEN` (after accepting the Llama 3.2 and Gemma 3 licences on
   Hugging Face). Without it the notebooks use Unsloth's ungated copies of the same weights.
5. Edit the settings cell as the table says, then **Save Version -> Save & Run All (Commit)**. A
   committed run keeps going after the browser closes and keeps its output.

## Training runs (`kaggle/c7_train.ipynb`, MODE = "leg")

Ablations run **1 round** (one leg per member) to halve their cost; the main runs and B10 run 2 rounds.
Compare an ablation with the main run's round-1 checkpoints, not its final ones: evaluate the main run once more with `TRAINED_ROUND = 1` in `kaggle/baseline_eval.ipynb` (it reads each member's round-1 LoRA from `state.json`).

Each training run = 2 rounds x its members, one leg per session, about 1 + 6 sessions
(the first session only caches the peers). Between sessions: on the finished version's page,
Output -> "New Dataset" from `<experiment>_runs.zip` (or download and upload it), attach that dataset
to the next session (Add Input) and set `RESUME_FROM` to its `runs` folder, e.g.
`/kaggle/input/<dataset-name>/runs`.

| Run | CONFIG | TRAIN_SEED | Notes |
| --- | --- | --- | --- |
| Main, seed 0 | `configs/train.yaml` | 0 | H1-H4 |
| Main, seed 1 | `configs/train.yaml` | 1 | H1-H4 |
| Main, seed 2 | `configs/train.yaml` | 2 | H1-H4 |
| B10 correctness reward (labels) | `configs/ablations/train_b10_correctness.yaml` | 0 | upper bound |
| Ablation 1: no PP_cross | `configs/ablations/train_abl1_no_pp_cross.yaml` | 0 | |
| Ablation 1: no BTS_info | `configs/ablations/train_abl1_no_bts_info.yaml` | 0 | |
| Ablation 1: no Pred | `configs/ablations/train_abl1_no_pred.yaml` | 0 | the doc: matters most |
| Ablation 1: no Calib | `configs/ablations/train_abl1_no_calib.yaml` | 0 | |
| Ablation 1: no Flip | `configs/ablations/train_abl1_no_flip.yaml` | 0 | the doc: matters most |
| Ablation 2: DMI | `configs/ablations/train_abl2_mech_dmi.yaml` | 0 | main run = CA |
| Ablation 2: PTS | `configs/ablations/train_abl2_mech_pts.yaml` | 0 | |
| Ablation 2: BTS only | `configs/ablations/train_abl2_mech_bts.yaml` | 0 | |
| Ablation 3: single-model group | `configs/ablations/train_abl3_single_model.yaml` | 0 | full reward |
| Ablation 4: homogeneous council | `configs/ablations/train_abl4_homogeneous.yaml` | 0 | 3 x Qwen |
| Ablation 5: G = 16 | `configs/ablations/train_abl5_g16.yaml` | 0 | slower legs |
| Ablation 6: no user_right | `configs/ablations/train_abl6_no_user_right.yaml` | 0 | |
| Ablation 7: council of 2 | `configs/ablations/train_abl7_size2.yaml` | 0 | 4 legs |
| Ablation 7: council of 4 | `configs/ablations/train_abl7_size4.yaml` | 0 | 8 legs, adds SmolLM2-1.7B |

## Other runs

| Run | Notebook | Settings | Notes |
| --- | --- | --- | --- |
| Baseline seeds 0, 1, 2 | `kaggle/baseline_eval.ipynb` | one seed per run (`SEEDS = [k]`) | about 5.5 h each |
| Combine baseline seeds | `kaggle/combine_seeds.ipynb` | Accelerator None; add the 3 seed notebooks as inputs | CPU, no GPU quota |
| B9 single-model BTS-GRPO | `kaggle/b9_train.ipynb` | defaults | one session |
| B8 anti-sycophancy SFT (Wei et al.) | `kaggle/b8_train.ipynb` | `MEMBERS = ["qwen", "llama"]`, then `["gemma"]` with `RESUME_FROM` | two sessions |
| Trained-council evaluation | `kaggle/baseline_eval.ipynb` | attach the run's zip as a dataset; `TRAINED_FROM` = its `runs` folder, `TRAINED_EXPERIMENT` = e.g. `cross_model_v1_seed0`, `SEEDS = [k]` (k = its TRAIN_SEED); ablations 4/7: `TRAINED_COUNCIL` | about 4.5 h; after training |
| Sycophancy sets (SycophancyEval, SYCON) | `kaggle/baseline_eval.ipynb` | `EVAL_CONFIG = "configs/eval_sycophancy.yaml"`, untrained and each trained run | short |
| Trained vs untrained comparison (H1, H3, H4, H5) | any CPU notebook / laptop | `python -m analysis.compare_runs <untrained_combined> <trained_combined> --out runs/compare_<name>` | CPU |

## Sharing results

Each run's output zip holds everything (`runs/<experiment>/`, including `state.json`, LoRAs,
`monitor.jsonl`, `reward_terms.jsonl`). Share a finished zip with the team as a private Kaggle
dataset (Settings -> Sharing -> add the teammates), so anyone can resume or evaluate it.
