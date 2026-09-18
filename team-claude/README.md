# Team Claude Setup — 2 minutes, do this once

These files make Claude Code understand our project without you re-explaining it every session.

## How the two layers work

| File | Who sees it | Committed to git? |
|---|---|---|
| `CLAUDE.md` (repo root) | **everyone** — shared project rules, frozen contracts, gotchas | ✅ yes |
| `CLAUDE.local.md` (repo root) | **only you** — your role, your checklist | ❌ no, gitignored |

Claude Code auto-loads both every session and layers your personal file on top of the shared one.

## Setup

```bash
# from the repo root, once the repo exists
cp team-claude/CLAUDE.md              ./CLAUDE.md
cp team-claude/roles/<YOUR-ROLE>.md   ./CLAUDE.local.md
echo "CLAUDE.local.md" >> .gitignore
git add CLAUDE.md .gitignore && git commit -m "docs: add shared Claude project context"
```

Then just run `claude` in the repo. Ask it things like *"what's next on my checklist?"* or *"implement the next unchecked item"*.

## Role assignment — **swap these freely, agree in the week-1 meeting**

| Role | File | Owns | Suggested |
|---|---|---|---|
| P1 Data lead | [`roles/P1-data-lead.md`](roles/P1-data-lead.md) | loaders, splits, augmentation, video set, ethics | Bulagala |
| P2 Baseline & Infra lead | [`roles/P2-baseline-infra-lead.md`](roles/P2-baseline-infra-lead.md) | trainer, Hydra, CI, RGB/SSIM, FC-Siam-Diff | Ekanayake |
| P3 Modeling A | [`roles/P3-modeling-encoders.md`](roles/P3-modeling-encoders.md) | ResNet-18 baseline, encoder + fusion registries | Lelwala |
| P4 Modeling B (novelty) | [`roles/P4-modeling-novelty.md`](roles/P4-modeling-novelty.md) | bounded alignment, confidence head, pair-order loss | Pabasara |
| P5 Metrics & Integration | [`roles/P5-metrics-integration.md`](roles/P5-metrics-integration.md) | all metrics, ablation orchestration, paper integration | Rajapaksha |

**P4 is the highest-risk seat** (it owns the actual novelty). Whoever takes it, item 1 of their checklist is a prior-art check that must happen in week 1 — see the risk note in the root `CLAUDE.md`.

## Also in this folder

`CODEOWNERS` — copy to `.github/CODEOWNERS` and replace `@P1`..`@P5` with real GitHub handles. It is resolution-tested; **do not alphabetise it** (last match wins, and one override depends on line order).

## Keeping your checklist alive

Your `CLAUDE.local.md` checklist is meant to be edited. Tick boxes as you go — Claude reads the ticks and will pick up where you left off. Tell it *"tick the items I finished and show me what's blocking"* at the end of a session.

## Where the research lives

All six research briefs are in [`../research/`](../research/). The root `CLAUDE.md` points Claude at the right brief for each question so it doesn't guess.
