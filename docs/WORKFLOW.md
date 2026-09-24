# Multi-agent workflow (Orca + Claude Code + Pi + Codex)

## Roles

- **You (tech lead):** own the contracts, write task briefs, review every PR, merge, integrate.
- **Agents (implementers):** one task per worktree, one PR per task, never merge.
- **Orca:** runs the agents side by side, each in its own git worktree, tracked in one place.

## One task = one worktree = one agent

1. Pick a task from `docs/tasks/` whose dependencies are merged.
2. In Orca, create a worktree named after the branch: `lane-c/T03-intent-classifier`.
3. Choose the agent (see `docs/PLAN.md` for suggestions) and send the kickoff prompt below.
4. **Check that the agent actually started.** Orca can report success while the prompt sits
   unsubmitted in a freshly launched TUI. Glance at each pane after launch.
5. Let it work. Steer when it drifts out of scope (Orca's mobile app is handy here).
6. The agent opens a PR. You review with the checklist below.

### Kickoff prompt (same for every agent)

```
Read AGENTS.md, docs/ARCHITECTURE.md and docs/tasks/<TASK_FILE>.md.
Implement the task exactly as specified, staying inside the allowed paths.
Work in small commits. Run ruff, pyright and pytest before each commit.
When all acceptance criteria pass, open a PR using the template.
List any assumptions at the top of the PR description.
```

## Bake-off (compare agents on the same task)

For T03, fan the same prompt out to **Pi** and **Claude Code** in two worktrees
(`lane-c/T03-pi`, `lane-c/T03-cc`). Compare them on:

| Criterion | How to measure |
|---|---|
| Correctness | Acceptance criteria and tests pass |
| Quality of result | Macro-F1 on the test split |
| Code quality | Your review: readability, scope creep, test quality |
| Cost / time | Tokens or $ spent, wall-clock time |
| Steering needed | Number of interventions |

Merge the winner. Record the comparison in `docs/LOG.md`, because it makes good README and interview material.

## Pi specifics

- Pi reads `AGENTS.md` automatically and keeps plans in files (`PLAN.md` / `TODO.md`), not in a hidden mode.
- **Pi runs with no permission prompts by default. Always run it inside the devcontainer or a
  Docker container**, with only the API keys it needs (never cloud admin credentials).
- Pi is minimal (few tools, short system prompt). Give it precise task briefs, where it shines.
  Install and configure it following https://pi.dev.

## Claude Code specifics

- Reads `AGENTS.md` when there is no `CLAUDE.md` (v2.1.277+). Keep **only** `AGENTS.md`.
- If AGENTS.md is not picked up (older version, Bedrock/Vertex, feature flags unavailable),
  check "Project instructions" in `/config`. As a last resort, create a one-line `CLAUDE.md`
  containing `@AGENTS.md`.

## Review checklist (you, on every PR)

- [ ] Stays inside the allowed paths? No edits to `contracts/`?
- [ ] Implements the `Operator` protocol and fills `OperatorResult` metadata?
- [ ] Tests meaningful (not just asserting mocks were called)? External calls mocked?
- [ ] Spans + metrics added with the agreed names?
- [ ] No PII in logs/fixtures, no secrets, no inline prompts?
- [ ] Dependencies justified?
- [ ] **Do I understand every line?** If not, ask the agent to explain or simplify before merging.

## Merge and rebase

- Squash-merge. After each merge, tell the remaining agents: `git fetch && git rebase origin/main`, then re-run tests.
- Conflicts in shared files (`pyproject.toml`, `uv.lock`, `.env.example`): you resolve them, not the agent.

## Logging what you learn

Keep `docs/LOG.md` as a dated engineering journal: decisions, agent failures, surprises,
numbers. It feeds the README's "How I built this" section and your interview stories.
