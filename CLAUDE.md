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

## 1. Freeze contract — nine document artifacts, allowlisted writers only

`doc/final_report/tables/{lolo,protocol_a,tier4_lolo,tier4_protocol_a}.tex` and `doc/final_report/figures/{cdf_lolo,cdf_lolo_tier4,cdf_protocol_a_forward_to_backward,cdf_protocol_a_backward_to_forward,segment_heatmap}.pdf` (6 main-body + 3 Appendix A = 9 files, enumerated verbatim in `src/icsr8/harness_tier4.FROZEN_OUTPUT_PATHS`) are frozen document artifacts. **Never hand-edit or casually regenerate them.**

# 2026-07-29 allowlist pivot: enforcement moved from a blocklist (any writer, reject specific paths) to an allowlist keyed on *writer identity*. `harness_tier4._guard_frozen(targets, *, writer_id=None)` raises `ValueError` unless `writer_id` is one of `_SANCTIONED_WRITERS` — currently exactly `icsr8.report.regenerate_main_body` and `icsr8.report.regenerate_appendix_a` (`src/icsr8/report.py`, both zero-argument functions). Every other caller — `scripts/run_experimental_tier4.py`, hand-edits, ad-hoc scripts — passes no writer_id and is rejected structurally, not by convention. The count grew from the pre-pivot 7 (2 TeX + 5 figure PDFs, main-body only) to 9 because Appendix A's `tier4_*.tex` and `cdf_lolo_tier4.pdf` had never been in `FROZEN_OUTPUT_PATHS` at all — they were unguarded by construction, not by oversight, since the old blocklist only ever covered the main-body path. Regenerate with `uv run python scripts/regenerate_main_body.py` / `scripts/regenerate_appendix_a.py` (both take no arguments — there is nothing to mistype). See `docs/adr/0004-deep-module-freeze-invariant.md` for the full rationale and `docs/adr/0001-freeze-main-body-artifacts.md` (Superseded) for the pre-pivot history. Table TeX flows through `%.2f` and stays byte-identical to HEAD across OS — `scripts/verify_report.py` reconstructs the 4 table fragments from the committed CSVs via `icsr8.harness`/`harness_tier4`'s own `_protocol_a_tex`/`_lolo_tex`/`_protocol_tex`/`_lolo_tex` and asserts exact byte equality, not just a line-set match. Figure PDFs are pinned by sha256 against `scripts/frozen_pdf_hashes.json` (tripwire on the committed tree, checked by `verify_report.py`; `--skip-pdf-hash` opts out on non-Mac dev hosts) — **this is not a claim that the 5 PDFs are byte-reproducible even on the same Mac**: 4 of the 5 (`cdf_lolo`, both `cdf_protocol_a_*`, `segment_heatmap`) embed a wall-clock `CreationDate` via `matplotlib`'s default `savefig`, so re-running `regenerate_main_body.py` changes their bytes even with identical numeric output (only `cdf_lolo_tier4.pdf` sets `metadata={"CreationDate": None}` and is deterministic). A legitimate regeneration must re-pin `scripts/frozen_pdf_hashes.json` as a deliberate, reviewed part of that same commit.

# 2026-07-29: `results/*.csv` are git-tracked but not part of this byte-identity contract: Mac(Accelerate) ↔ Linux(OpenBLAS) BLAS implementation differences drift them at ULP scale. Document-visible numbers survive the drift because they go through `%.2f`. See README §凍結契約 for the current state.

## 2. Isolated evaluation is structurally non-sanctioned

`scripts/run_experimental_tier4.py --methods <name> --output <dir> --tables-dir <dir> --figures-dir <dir>` evaluates new/experimental methods. **All four flags are `required=True`** — there is no default to accidentally omit, unlike the deleted `run_all_methods.py`/`run_tier4.py` that this replaced (see §1). This CLI never passes a `writer_id` to `run_tier4()`, so even if you point `--tables-dir`/`--figures-dir` at `doc/final_report/tables`, `_guard_frozen` rejects the write with `ValueError` before any file touches disk. Point real runs at a dedicated directory (e.g. `results/extra/<name>/`).

## 3. Always pass both gates after a change

```bash
uv run pytest                           # 515 tests (freeze guard, leak contract, published-value reproduction)
uv run python scripts/verify_report.py  # byte-level CSV↔TeX reconciliation + main.tex reference-path existence
```

Run both no matter what you touched — code, results, or LaTeX. One alone is insufficient (pytest does not check document/table consistency; verify_report does not check behavior).

## 4. Conventions for adding a method

- One module = one method (`src/icsr8/methods/<name>.py`), decorated with `@register`, declaring `name` and `uses_geometry`.
- Never pass test-side information into `fit`. Leak prevention is structural: `run_method` (`methods/__init__.py`) filters location coordinates to training locations, and the spy test `test_iter_lolo_leakage_contract_spy` pins that contract. Do not write direct calls that bypass this guarantee.
- For iterative algorithms, the contract is "iterate until converged"; set the cap with generous margin over the observed maximum and leave a dated comment justifying it (# 2026-07-22 vWCL: the paper expects 5–10 iterations, but real data required up to 53).
- Add property tests plus an end-to-end test through `run_method` in `tests/test_<name>.py`.

## 5. LaTeX contracts

- Builds use LuaLaTeX + latexmk (`doc/*/.latexmkrc`). The report is `ltjsarticle[twocolumn]`; the slides are beamer + luatexja. Do not mix in pLaTeX-family classes (ieicej etc.).
- The path strings of `\input{tables/...}` and `\includegraphics{...figures/...}` inside `doc/final_report/main.tex` are **inspected by verify_report.py**. Restructure sections freely, but do not change a single character of those path strings.
- Numbers should be `\input` from the table generator's `%.2f` output. When a number appears in prose, it must match the tables/CSVs exactly — never re-round on your own.

## 6. Public-repository placeholders

`[GROUP_NAME]` / `[AUTHOR_NAME]` are placeholders for the public repository. Never commit real names filled in (build the submission PDFs locally with the placeholders replaced).

## 7. Do not break reproducibility

seed=0, bootstrap B=1000, and the deterministic tie-break (rssi_median desc → frequency asc → ssid asc → ap_name asc; the rule that reproduces the five published tie events P19/P30/P35/P43/P49) are the foundation of result identity. Any proposal to change them means the published-value reproduction tests will break.

## 8. Data originals

`data/*.zip` are the originals. Never edit the extracted directories (`data/dataset/` etc.) directly. Regenerate test fixtures only via `scripts/extract_baseline_fixtures.py`.
