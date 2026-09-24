# ADR-0002: LangGraph for pipeline orchestration

**Status:** Accepted · **Date:** 2026-09

## Context
The pipeline needs parallel operators, conditional routing (verifier), per-node error handling,
and later a tool-using Q&A agent. Options considered: plain asyncio, LangGraph, CrewAI, AutoGen.

## Decision
LangGraph. It models the pipeline as an explicit state graph: typed state, fan-out/fan-in, conditional edges,
checkpointing. It also integrates with OpenInference tracing. CrewAI and AutoGen are role/chat-centric
multi-agent frameworks, a poor fit for a mostly deterministic pipeline with a few decision points.

## Consequences
- Deterministic steps stay as plain functions wrapped as nodes. Only the verifier and the Q&A agent use an LLM to decide.
- The graph is visualised (Mermaid export) in the README.
- Adds a dependency on the LangChain ecosystem. Mitigation: operators depend only on our `Operator`
  protocol, so the orchestrator could be swapped out.
