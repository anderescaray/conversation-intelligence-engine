# Multi-agent workflow (Claude Code + Devin + Pi + Cursor)

## Roles

- **You (tech lead):** own the contracts, write task briefs, review every PR, merge, integrate.
- **Agents (implementers):** one task per branch, one PR per task, never merge.
  - **Claude Code** (Pro plan) and **Pi** run locally, each in its own git worktree.
  - **Devin** runs in its own cloud environment and pushes its branch and PR itself.
  - **Cursor** is for interactive review and small fixes you drive; it owns no lane.

## One task = one branch = one agent

1. Pick a task from `docs/tasks/` (and its GitHub issue) whose dependencies are merged.
2. Create the branch named after the task: `lane-c/T03-intent-classifier`.
   - **Local agents:** `git worktree add ../cie-T03 -b lane-c/T03-intent-classifier origin/main`
     and start the agent inside that worktree.
   - **Devin:** start a session on the repo and give it the branch name in the kickoff prompt.
3. Choose the agent (see `docs/PLAN.md`) and send the kickoff prompt below.
4. **Check that the agent actually started.** A freshly launched TUI can leave the prompt sitting
   unsubmitted. Glance at each session after launch.
5. Let it work. Steer when it drifts out of scope.
6. The agent opens a PR that references the issue. You review with the checklist below.

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

Merge the winner. Record the comparison in `docs/LOG.md`, because it makes good README material.

## Pi specifics

- Uses the Anthropic API key. Give it only the keys it needs (never cloud admin credentials).
- Pi reads `AGENTS.md` automatically and keeps plans in files (`PLAN.md` / `TODO.md`), not in a hidden mode.
- **Pi runs with no permission prompts by default. Always run it inside the devcontainer.**
- Pi is minimal (few tools, short system prompt). Give it precise task briefs, where it shines.
  Install and configure it following https://pi.dev.

## Claude Code specifics

- Runs on the Pro plan: usage limits are shared across sessions, so two parallel lanes can hit
  them. Stagger launches and prefer fewer, well-scoped sessions.
- Reads `AGENTS.md` when there is no `CLAUDE.md` (v2.1.277+). Keep **only** `AGENTS.md`.
- If AGENTS.md is not picked up (older version, Bedrock/Vertex, feature flags unavailable),
  check "Project instructions" in `/config`. As a last resort, create a one-line `CLAUDE.md`
  containing `@AGENTS.md`.

## Devin specifics

- Works in its own cloud machine, not in a local worktree: it clones the repo, creates the branch
  and opens the PR itself.
- Point it at `AGENTS.md` and the task brief in the kickoff prompt, and give it the exact branch name.
- It cannot reach your local docker compose stack, so integration tests that need it are run by
  you on the PR branch.
- Review its PRs exactly like any other agent's.

## Cursor specifics

- Reads `AGENTS.md`. Use it to read, review and make small edits on a PR branch you have checked out.
- It is not assigned a lane; if a fix grows beyond a few lines, hand it back to the lane's agent.

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
numbers. It feeds the README's "How I built this" section.
