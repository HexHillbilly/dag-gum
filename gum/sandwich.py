"""gum.sandwich — the weight-free steering sandwich orchestrator.

Three layers, in generation order:

    top bread     SCP/SSP pre-generation conditioning — retrieve a voice/style
                  line from a corpus and wrap the prompt ("weave its spirit" /
                  "write like X" / "explain in their terms").

    the meat      AVG mid-generation governance — the Active Variety Governor
                  watches the residual stream during decode and suppresses
                  repetition / early-EOS attractors.

    bottom bread  two-stage post-generation synthesis — state the grounded fact
                  plainly, then voice it.

The "gum" is the routing + binding. Tool-needing turns drop the persona during
the tool call (persona-free router → call → execute) and resume in voice; this
keeps the persona off the tool-cognition path (see scp/gate).

Dependency-injected (mirrors scp.gate) so the orchestrator stays backend-agnostic.
"""
from __future__ import annotations

from typing import Callable, Optional

from scp import gate


class Sandwich:
    """Compose pre-gen conditioning + mid-gen governance + synthesis.

    Args:
        generate: raw persona-free completion ``(prompt, max_new_tokens) -> str``
            used for the router and tool calls.
        retrieve: ``(query, k=1) -> list[str]`` corpus retrieval (index/corpus
            already bound by the caller).
        wrap: ``(user_msg, seed, name, desc) -> str`` a persona/style wrapper —
            e.g. ``scp.core.wrap_persona_hardened``, ``scp.core.wrap_explain_in_terms``,
            or ``ssp.core.wrap_style``.
        governor: optional ``AVG.governor.controller.ActiveVarietyGovernor``; when
            set, its ``.generate(...)`` replaces the raw path for the final answer.
        router_prompt: the persona-free ``[tool]/[chat]`` classifier prompt
            (``gate.ROUTER_DEFINED`` for 8B-class models, ``gate.ROUTER_FEWSHOT``
            for Phi-3-mini-class; see scp/gate for the measured rationale).
        persona_name / persona_desc: passed through to the wrapper.
    """

    def __init__(
        self,
        *,
        generate: Callable,
        retrieve: Callable,
        wrap: Callable,
        governor: Optional[object] = None,
        router_prompt: Optional[str] = None,
        persona_name: str = "Daisy",
        persona_desc: str = "a warm, lowercase, 2000-era chatbot",
    ):
        self.generate = generate
        self.retrieve = retrieve
        self.wrap = wrap
        self.governor = governor
        self.router_prompt = router_prompt
        self.persona_name = persona_name
        self.persona_desc = persona_desc

    def steer(self, prompt: str, *, max_new_tokens: int = 256, use_tools: bool = True) -> str:
        """One full sandwich turn: route → condition → govern → synthesize."""
        if use_tools:
            decision = gate.classify_tool_intent(
                prompt, self.generate, prompt=self.router_prompt or gate.ROUTER_DEFINED
            )
            if decision == "tool":
                return self._tool_turn(prompt, max_new_tokens)
        return self._govern(self._condition(prompt), max_new_tokens)

    # -- the three layers ----------------------------------------------------

    def _condition(self, prompt: str) -> str:
        """Top bread: pre-generation voice/style conditioning."""
        matched = self.retrieve(prompt, k=1) or []
        if not matched:
            return prompt
        return self.wrap(prompt, matched[0], self.persona_name, self.persona_desc)

    def _govern(self, wrapped: str, max_new_tokens: int) -> str:
        """The meat: mid-generation governance (AVG), else raw generation."""
        if self.governor is not None:
            out = self.governor.generate(
                wrapped, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.8, intervene=True,
            )
            if isinstance(out, dict):
                return (out.get("gen_only_text") or "").strip()
            return str(out).strip()
        return self.generate(wrapped, max_new_tokens)

    def _tool_turn(self, prompt: str, max_new_tokens: int) -> str:
        """Bottom bread: tool call (persona-free) → execute → two-stage resume."""
        call = gate.emit_tool_call(prompt, self.generate)
        result = gate.execute_tool(call)
        resume = gate.build_resume_prompt(prompt, result)
        matched = self.retrieve(prompt, k=1) or []  # seed retrieved on the ORIGINAL prompt
        if matched:
            resume = self.wrap(resume, matched[0], self.persona_name, self.persona_desc)
        return self._govern(resume, max_new_tokens)
