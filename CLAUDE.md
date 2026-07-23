# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For architecture, project structure, command usage, and configuration reference, see README.md.

PLAN.md and MEMO.md are ephemeral scratch space for jotting in-progress plans/memos; durable decisions belong in CONTEXT.md, this file, or docs/adr/.

## Write a program based on the Unix philosophy

- Write programs that do one thing and do it well.
- Write programs to work together.
- Write programs to handle text streams, because that is a universal interface.

## Interaction contract

- If requirements are ambiguous or underspecified, stop and ask 1–3 targeted questions before proceeding.
- Before making any irreversible change (deletes, migrations, dependency upgrades, infra changes), ask for explicit confirmation.
- Never assume environment details (OS, shell, package manager, project conventions). Ask or infer only from repo evidence.
- Start each task by restating: Goal, Non-goals, Constraints, Success criteria (brief).
- When multiple approaches exist, present 2 options with tradeoffs, then ask which to take.

## Comment & Context Policy

Write comments generously — treat them as first-class documentation, not noise. Current AI models benefit from heavy inline context; so do human readers six months later.

In addition to comments, write function and class descriptions and intentions in docstring.

- Always comment intent, not mechanics. Explain why a block exists, what invariant it protects, or what would break without it. Don't restate what the code does — explain what it means.
- Record fix provenance inline. When code exists because of a specific bug or incident, leave a dated note: `# 2026-05-12 crash fix: bare ifconfig omits netmask → classful /8 on Class A`. This is the kind of context that git blame buries and developers lose.
- Keep context close to code. A comment explaining a constraint belongs next to the line it constrains, not in a separate design doc. If someone reads the function, they should see the warning without leaving the file.
- Don't write comments that rot. Avoid referencing ticket numbers, PR links, or caller names ("used by X") — those change. Describe the constraint the code enforces; that outlives the ticket.

## Notice

Separate functions into separate files by type, and do not recreate existing functions in the execution script. If you need to edit them, edit the existing function and check that the modifications have been made. Make functions as flexible as possible by using variables.

# Development

Cautions for changing this repository. Each rule below was codified after an actual (or narrowly avoided) incident; see docs/adr/ for the full history.

## 1. Freeze contract — seven document artifacts you must not touch

`doc/final_report/tables/{lolo,protocol_a}.tex` and `doc/final_report/figures/{cdf_lolo,cdf_lolo_tier4,cdf_protocol_a_forward_to_backward,cdf_protocol_a_backward_to_forward,segment_heatmap}.pdf` are the frozen main-body document artifacts. **Never hand-edit or casually regenerate them.** When regeneration is genuinely needed, use only the command in README "Tier ごとの評価手順" with the 15 main-body methods listed explicitly. Table TeX flows through `%.2f` and stays byte-identical to HEAD across OS; figure PDFs are visually invariant despite ULP-level numeric drift and have been versioned since commit `076bec5 add: pdf contents`.

# 2026-07-23 freeze-contract pivot: `results/{protocol_a,lolo_ledger,lolo_summary}.csv` — previously frozen — are now gitignored regeneratable derivatives. Mac(Accelerate) ↔ Linux(OpenBLAS) BLAS implementation differences drift them at ULP scale, so any cross-OS byte-identity claim breaks. Document-visible numbers survive the drift because they go through `%.2f`. See `docs/adr/0001-freeze-main-body-artifacts.md` for the pre-pivot rationale and README §凍結契約 for the current state.

## 2. Never run `run_all_methods.py` without `--methods`

# 2026-07-14 contamination incident: the registry auto-discovers `src/icsr8/methods/`, so the moment a new method carries `@register`, an unfiltered run sweeps it too. An unfiltered run leaked the seven Tier 4 methods into the frozen files. Every new method you add makes this landmine **bigger** — do not forget it.

## 3. Isolated evaluation needs all three output flags

Evaluate new/experimental methods with `run_tier4.py --methods <name> --output <dir> --tables-dir <dir> --figures-dir <dir>`, pointing **all three output channels** at a dedicated directory (e.g. `results/extra/`). If `--tables-dir`/`--figures-dir` are omitted, the run writes into the default `doc/final_report/` and overwrites the committed Appendix-A tables `tier4_*.tex`.

## 4. Always pass both gates after a change

```bash
uv run pytest                           # 374 tests (freeze guard, leak contract, published-value reproduction)
uv run python scripts/verify_report.py  # byte-level CSV↔TeX reconciliation + main.tex reference-path existence
```

Run both no matter what you touched — code, results, or LaTeX. One alone is insufficient (pytest does not check document/table consistency; verify_report does not check behavior). **On a fresh clone `results/*.csv` are absent (gitignored since 2026-07-23); run `scripts/run_all_methods.py --methods …` and `scripts/run_tier4.py` first to populate them before invoking these gates.**

## 5. Conventions for adding a method

- One module = one method (`src/icsr8/methods/<name>.py`), decorated with `@register`, declaring `name` and `uses_geometry`.
- Never pass test-side information into `fit`. Leak prevention is structural: `run_method` (`methods/__init__.py`) filters location coordinates to training locations, and the spy test `test_iter_lolo_leakage_contract_spy` pins that contract. Do not write direct calls that bypass this guarantee.
- For iterative algorithms, the contract is "iterate until converged"; set the cap with generous margin over the observed maximum and leave a dated comment justifying it (# 2026-07-22 vWCL: the paper expects 5–10 iterations, but real data required up to 53).
- Add property tests plus an end-to-end test through `run_method` in `tests/test_<name>.py`.

## 6. LaTeX contracts

- Builds use LuaLaTeX + latexmk (`doc/*/.latexmkrc`). The report is `ltjsarticle[twocolumn]`; the slides are beamer + luatexja. Do not mix in pLaTeX-family classes (ieicej etc.).
- The path strings of `\input{tables/...}` and `\includegraphics{...figures/...}` inside `doc/final_report/main.tex` are **inspected by verify_report.py**. Restructure sections freely, but do not change a single character of those path strings.
- Numbers should be `\input` from the table generator's `%.2f` output. When a number appears in prose, it must match the tables/CSVs exactly — never re-round on your own.

## 7. Public-repository placeholders

`[GROUP_NAME]` / `[AUTHOR_NAME]` are placeholders for the public repository. Never commit real names filled in (build the submission PDFs locally with the placeholders replaced).

## 8. Do not break reproducibility

seed=0, bootstrap B=1000, and the deterministic tie-break (rssi_median desc → frequency asc → ssid asc → ap_name asc; the rule that reproduces the five published tie events P19/P30/P35/P43/P49) are the foundation of result identity. Any proposal to change them means the published-value reproduction tests will break.

## 9. Data originals

`data/*.zip` are the originals. Never edit the extracted directories (`data/dataset/` etc.) directly. Regenerate test fixtures only via `scripts/extract_baseline_fixtures.py`.
