"""Part 13.1 - The LLM Wiki: a Deep Agent worth evaluating.

A script-first Deep Agents example that builds and maintains a persistent
knowledge base. An agent researches a topic and writes results into a wiki repo,
syncing each change to a (simulated) Context Hub revision. It exposes three modes:

* ``ingest`` - expand raw sources into canonical wiki pages; optionally a
  two-phase review/apply flow that *skips writes on decline* (human-in-the-loop).
* ``query``  - read ``wiki/index.md``, then ``log.md`` for recency, then expand
  the relevant pages and answer with citations, optionally filing the answer.
* ``lint``   - reconcile contradictions, stale claims, orphan pages.

Every phase appends a machine-parseable heading to ``log.md``
(``## [date] mode.phase | outcome=...``).

The point of the example is that it touches most of the Part 9 agent-eval
surface, so it reuses the Part 9 :class:`~trustworthy_evals.agent_eval.Transcript`
and graders and the Part 8 :func:`~trustworthy_evals.frameworks.faithfulness`
metric. The graders are *bespoke per case* -- "the ingest review skipped writes
on decline" and "the query answer was grounded in the cited pages" are different
assertions on different artifacts, exactly as LangChain found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .agent_eval import Grader, GraderResult, ToolCall, Transcript
from .frameworks import faithfulness


# ---------------------------------------------------------------------------
# A tiny virtual filesystem + the wiki agent
# ---------------------------------------------------------------------------


@dataclass
class WikiResult:
    """What a mode run returns: the gradeable transcript plus the answer text."""

    transcript: Transcript
    output: str = ""


class LLMWiki:
    """A simulated LLM Wiki Deep Agent over an in-memory virtual filesystem."""

    def __init__(self) -> None:
        self.fs: Dict[str, str] = {
            "wiki/index.md": "# Wiki Index\n",
            "wiki/log.md": "# Log\n",
        }
        self.revision = 0  # simulated Context Hub revision counter

    # -- vfs helpers --------------------------------------------------------

    def _read(self, path: str) -> str:
        return self.fs.get(path, "")

    def _write(self, path: str, content: str) -> None:
        self.fs[path] = content

    def _log(self, mode: str, phase: str, outcome: str) -> None:
        # The append-only, machine-parseable log heading code graders rely on.
        entry = f"## [2026-06-02] {mode}.{phase} | outcome={outcome}\n"
        self.fs["wiki/log.md"] = self._read("wiki/log.md") + entry

    def log_text(self) -> str:
        return self._read("wiki/log.md")

    def _push_revision(self) -> int:
        self.revision += 1
        return self.revision

    # -- ingest -------------------------------------------------------------

    def ingest(self, name: str, claims: Sequence[str], *, review: bool = False, confirm: bool = True) -> WikiResult:
        """Expand a raw source into a canonical wiki page.

        With ``review=True`` this runs a read-only review pass first; it writes
        only if ``confirm`` is true (the apply pass). A decline (``confirm=False``)
        must skip all writes -- a textbook human-in-the-loop checkpoint.
        """
        calls: List[ToolCall] = [ToolCall("read_file", {"path": "wiki/index.md"})]
        page = f"wiki/pages/{name}.md"

        if review:
            calls.append(ToolCall("review", {"page": page, "proposed_claims": list(claims)}))
            self._log("ingest", "review", "proposed" if confirm else "skipped")
            if not confirm:
                # Decline: no writes, no revision.
                tr = Transcript(
                    final_output=f"ingest review for {name}: declined, no writes",
                    tool_calls=calls,
                    final_state={"wrote": False, "outcome": "skipped", "log": self.log_text()},
                )
                return WikiResult(tr, tr.final_output)

        body = f"# {name}\n" + "\n".join(f"- {c}" for c in claims) + "\n"
        self._write(page, body)
        calls.append(ToolCall("write_file", {"path": page}))
        # Refresh the index and push a revision.
        index = self._read("wiki/index.md")
        if page not in index:
            self._write("wiki/index.md", index + f"- [{name}]({page})\n")
        calls.append(ToolCall("edit_file", {"path": "wiki/index.md"}))
        rev = self._push_revision()
        self._log("ingest", "apply", "written")

        tr = Transcript(
            final_output=f"ingested {name}: wrote {page}, refreshed index, pushed revision {rev}",
            tool_calls=calls,
            final_state={
                "wrote": True,
                "outcome": "written",
                "index_refreshed": True,
                "revision_pushed": True,
                "pages_written": [page],
                "log": self.log_text(),
            },
        )
        return WikiResult(tr, tr.final_output)

    # -- query --------------------------------------------------------------

    def query(self, question: str, claims: Sequence[str], cite: Sequence[str], *, file_answer: bool = True) -> WikiResult:
        """Answer from the wiki with citations; optionally file a durable answer.

        ``claims`` are the answer's assertions; ``cite`` are the page paths it
        cites. Groundedness is later graded by checking the claims against the
        concatenated cited-page text.
        """
        consulted = ["wiki/index.md", "wiki/log.md"] + list(cite)
        calls = [ToolCall("read_file", {"path": p}) for p in consulted]
        cited_context = "\n".join(self._read(p) for p in cite)
        answer = f"{question} -> " + " ".join(claims) + " [cites: " + ", ".join(cite) + "]"

        filed = False
        if file_answer:
            self._write(f"wiki/query/{abs(hash(question)) % 1000}.md", answer)
            calls.append(ToolCall("write_file", {"path": "wiki/query/"}))
            self._push_revision()
            self._log("query", "apply", "filed")
            filed = True
        else:
            self._log("query", "answer", "ephemeral")

        tr = Transcript(
            final_output=answer,
            tool_calls=calls,
            final_state={
                "answer_claims": list(claims),
                "cited_context": cited_context,
                "consulted": consulted,
                "filed": filed,
                "log": self.log_text(),
            },
        )
        return WikiResult(tr, answer)

    # -- lint ---------------------------------------------------------------

    def lint(self) -> WikiResult:
        """Find orphan pages (not linked from the index) and report gaps."""
        index = self._read("wiki/index.md")
        pages = [p for p in self.fs if p.startswith("wiki/pages/")]
        orphans = [p for p in pages if p not in index]
        self._log("lint", "report", f"orphans={len(orphans)}")
        tr = Transcript(
            final_output=f"lint: {len(orphans)} orphan page(s): {orphans}",
            tool_calls=[ToolCall("glob", {"pattern": "wiki/pages/*.md"}), ToolCall("read_file", {"path": "wiki/index.md"})],
            final_state={"orphans": orphans, "log": self.log_text()},
        )
        return WikiResult(tr, tr.final_output)


# ---------------------------------------------------------------------------
# Bespoke, per-case graders (the research-agent playbook + state + log)
# ---------------------------------------------------------------------------


@dataclass
class WikiGroundedness(Grader):
    """Are the query answer's claims supported by the cited pages? (Part 7/8.)"""

    threshold: float = 0.5
    weight: float = 1.0
    required: bool = False
    name: str = "groundedness"

    def grade(self, t: Transcript) -> GraderResult:
        claims = t.final_state.get("answer_claims", [])  # type: ignore[assignment]
        context = str(t.final_state.get("cited_context", ""))
        score = faithfulness(list(claims), context) if claims else 0.0
        return GraderResult(self.name, score, score >= self.threshold, "claims supported by cited pages")


@dataclass
class Coverage(Grader):
    """Did the agent consult the pages it was supposed to? (research coverage.)"""

    required: Sequence[str] = field(default_factory=list)
    weight: float = 1.0
    name: str = "coverage"

    def grade(self, t: Transcript) -> GraderResult:
        consulted = set(t.final_state.get("consulted", []))  # type: ignore[arg-type]
        need = list(self.required)
        present = sum(1 for p in need if p in consulted)
        score = present / len(need) if need else 1.0
        return GraderResult(self.name, score, score == 1.0, f"consulted {present}/{len(need)} required pages")


@dataclass
class LogContains(Grader):
    """Code-based grader: assert a parseable log entry appeared (grep the log)."""

    pattern: str = ""
    weight: float = 1.0
    required: bool = False
    name: str = "log_contains"

    def grade(self, t: Transcript) -> GraderResult:
        log = str(t.final_state.get("log", ""))
        ok = self.pattern in log
        return GraderResult(self.name, float(ok), ok, f"log must contain {self.pattern!r}")


__all__ = [
    "WikiResult",
    "LLMWiki",
    "WikiGroundedness",
    "Coverage",
    "LogContains",
]
