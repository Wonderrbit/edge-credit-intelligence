from __future__ import annotations

import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from mlops.agents.guardrails import check_input, check_output
from mlops.agents.llm import get_llm
from mlops.agents.prompts import OUT_OF_SCOPE_ANSWER, SYNTHESIS_PROMPT, SYSTEM_PROMPT
from mlops.agents.rag.ingest import tokenize
from mlops.agents.router import classify_intent, extract_client_id
from mlops.agents.state import AgentState
from mlops.agents.tools import AgentTools
from mlops.agents.toon import encode_toon
from mlops.agents.tracing import AgentTrace, new_run_id
from mlops.judgment.batteries import GUARD_QUESTIONS, INTENT_QUESTIONS, RAG_QUESTIONS
from mlops.judgment.client import get_jev
from mlops.judgment.compose import (
    compose_guard_assessment,
    compose_intent_decision,
    compose_rag_verification,
)
from mlops.judgment.fallback import (
    fallback_guard_assessment,
    fallback_intent_decision,
    fallback_rag_verification,
)
from mlops.serving.predictor import get_predictor

DISCLAIMER = (
    "Esta é uma saída de apoio à decisão. Decisões adversas de crédito, limite ou "
    "cobrança exigem revisão humana."
)


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value * 100:.1f}%"


def _money(value: float | None) -> str:
    if value is None:
        return "n/d"
    formatted = f"{value:,.2f}"
    return "R$ " + formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = []
    for part in parts:
        part = re.sub(r"^[-*|]\s*", "", part).strip()
        if len(part) >= 25 and not part.startswith("**Documento"):
            cleaned.append(part)
    return cleaned


def _best_sentences(question: str, retrieved: list[dict[str, Any]], max_sentences: int = 4) -> str:
    query_tokens = set(tokenize(question))
    candidates: list[tuple[float, int, str]] = []
    order = 0
    for item in retrieved[:2]:
        text = item.get("chunk", {}).get("text", "")
        for sentence in _sentences(text):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(query_tokens & sentence_tokens)
            numeric_bonus = 0.5 if re.search(r"\d", sentence) else 0.0
            candidates.append((overlap + numeric_bonus, order, sentence))
            order += 1
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = sorted(candidates[:max_sentences], key=lambda row: row[1])
    if not selected:
        return ""
    return " ".join(row[2] for row in selected)


def _format_sources(citations: list[dict[str, str]]) -> str:
    if not citations:
        return ""
    seen: set[tuple[str, str]] = set()
    lines = []
    for citation in citations:
        key = (citation["doc_id"], citation["section"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{citation['doc_id']} | {citation['section']}] {citation['citation']}")
    return "\n".join(lines)


class AgentRunner:
    def __init__(self) -> None:
        self.tools = AgentTools(predictor=get_predictor())
        self.llm = get_llm()
        self.jev = get_jev()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("guard", self._guard)
        builder.add_node("router", self._router)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("tools", self._run_tools)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("review", self._review)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "guard")
        builder.add_conditional_edges(
            "guard",
            lambda state: "finalize" if state.get("blocked") else "router",
            {"finalize": "finalize", "router": "router"},
        )
        builder.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "retrieve": "retrieve",
                "tools": "tools",
                "synthesize": "synthesize",
            },
        )
        builder.add_edge("retrieve", "synthesize")
        builder.add_edge("tools", "synthesize")
        builder.add_edge("synthesize", "review")
        builder.add_edge("review", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def _route_after_router(state: AgentState) -> str:
        intent = state.get("intent", "out_of_scope")
        if intent in {"policy"}:
            return "retrieve"
        if intent in {"client_score", "limit", "collection", "model_metrics", "simulation"}:
            return "tools"
        return "synthesize"

    def _guard(self, state: AgentState) -> dict[str, Any]:
        result = check_input(state["question"], state.get("role", "analista"))
        state["trace"].log("guard", allowed=result.allowed, reason=result.reason)
        if not result.allowed:
            return {
                "blocked": True,
                "guardrail_reason": result.reason,
                "human_review": result.human_review,
                "answer": f"Não posso atender a esta solicitação. {result.reason}",
                "citations": [],
                "intent": "blocked",
            }
        response = self.jev.ask({"mensagem": result.sanitized_question}, GUARD_QUESTIONS)
        assessment = compose_guard_assessment(response) if response else fallback_guard_assessment(
            result.sanitized_question
        )
        state["trace"].log(
            "guard_jev",
            decision=assessment.decision,
            hazards=assessment.hazards,
            source=assessment.meta.source,
        )
        if assessment.decision == "block":
            return {
                "blocked": True,
                "guardrail_reason": "A solicitação foi classificada como insegura pelo Jev.",
                "human_review": True,
                "answer": (
                    "Não posso atender a esta solicitação. "
                    "Ela foi classificada como insegura e será registrada para auditoria."
                ),
                "citations": [],
                "intent": "blocked",
            }
        return {
            "blocked": False,
            "question": result.sanitized_question,
            "human_review": assessment.decision == "review",
        }

    def _router(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        intent = classify_intent(question, self.llm)
        if intent == "out_of_scope" and self.jev.available:
            response = self.jev.ask({"pergunta": question}, INTENT_QUESTIONS)
            if response:
                jev_intent = compose_intent_decision(response)
                if jev_intent.intent != "out_of_scope":
                    intent = jev_intent.intent
                    state["trace"].log(
                        "router_jev",
                        intent=jev_intent.intent,
                        confidence=jev_intent.confidence,
                    )
        elif not self.jev.available:
            state["trace"].log("router_jev", intent=fallback_intent_decision(question).intent, source="fallback")
        client_id = state.get("client_id") or extract_client_id(question)
        state["trace"].log("router", intent=intent, client_id=client_id)
        return {"intent": intent, "client_id": client_id}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        result = self.tools.policy_search(state["question"], k=5)
        state["trace"].log(
            "retrieve",
            query=state["question"],
            retriever=result.get("retriever"),
            context_chars=result.get("context_chars"),
            citations=result["citations"],
        )
        return {
            "retrieved": result["results"],
            "citations": result["citations"],
            "tool_results": {"policy": result},
        }

    def _run_tools(self, state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", "")
        client_id = state.get("client_id")
        tool_results: dict[str, Any] = dict(state.get("tool_results") or {})
        policy_query = state["question"]
        if intent == "limit":
            policy_query = "política de limite recomendação preset risco"
        elif intent == "collection":
            policy_query = "política de cobrança priorização canal valor esperado"
        elif intent == "model_metrics":
            policy_query = "modelo métricas calibração validação governança"
        elif intent == "client_score":
            policy_query = "política de crédito score risco revisão humana"
        policy = self.tools.policy_search(policy_query, k=4)
        tool_results["policy"] = policy
        citations = policy["citations"]

        try:
            if intent == "client_score":
                if not client_id:
                    return {"answer": "Informe o client_id no formato C000001 para consultar o score."}
                tool_results["score"] = self.tools.score_client(client_id)
            elif intent == "limit":
                if not client_id:
                    return {"answer": "Informe o client_id para recomendar limite."}
                tool_results["limit"] = self.tools.limit_recommendation(client_id)
                tool_results["score"] = self.tools.score_client(client_id)
            elif intent == "collection":
                tool_results["queue"] = self.tools.collection_queue(top_n=10)
                if client_id:
                    tool_results["score"] = self.tools.score_client(client_id)
                    tool_results["simulation"] = self.tools.simulate_collection(client_id, "whatsapp")
            elif intent == "model_metrics":
                tool_results["metrics"] = self.tools.model_metrics()
            elif intent == "simulation":
                if not client_id:
                    return {"answer": "Informe o client_id para simular uma ação de cobrança."}
                tool_results["score"] = self.tools.score_client(client_id)
                tool_results["simulation"] = self.tools.simulate_collection(client_id, "whatsapp")
        except Exception as exc:
            state["trace"].log("tool_error", error=str(exc))
            return {
                "answer": (
                    "Não consegui concluir a consulta. Verifique o client_id e tente novamente. "
                    "Se o problema persistir, encaminhe para revisão humana."
                ),
                "citations": citations,
                "tool_results": tool_results,
                "tool_calls": list(self.tools.calls),
                "human_review": True,
            }
        state["trace"].log(
            "tools",
            intent=intent,
            calls=self.tools.calls[-5:],
        )
        return {
            "citations": citations,
            "tool_results": tool_results,
            "tool_calls": list(self.tools.calls),
        }

    def _synthesize_offline(self, state: AgentState) -> str:
        intent = state.get("intent", "out_of_scope")
        results = state.get("tool_results") or {}
        citations = state.get("citations") or []
        if intent == "policy":
            snippet = _best_sentences(state["question"], state.get("retrieved", []))
            if not snippet:
                return OUT_OF_SCOPE_ANSWER
            return (
                f"Com base nas políticas indexadas: {snippet}\n\n"
                f"**Fontes**\n{_format_sources(citations)}\n\n"
                f"{DISCLAIMER}"
            )
        if intent == "client_score":
            score = results.get("score")
            if not score:
                return "Não localizei o cliente informado."
            scores = score["scores"]
            drivers = score.get("top_drivers", [])
            driver_text = ", ".join(
                f"{driver.get('feature')} ({driver.get('attribution', 0):+.3f})"
                for driver in drivers[:3]
            )
            return (
                f"**Cliente {score['client_id']}** — {score.get('segmento')} / {score.get('regiao')}\n\n"
                f"- Pagamento em 30 dias: **{_pct(scores['pagou_30d'])}**\n"
                f"- Inadimplência em 90 dias: **{_pct(scores['inadimplente_90d'])}**\n"
                f"- Aceite de limite: **{_pct(scores['aceita_limite'])}**\n"
                f"- Consumo alto: **{_pct(scores['consumo_alto'])}**\n"
                f"- Decisão de risco sugerida: **{score['decision']['risco']}**\n"
                f"- Limite recomendado: **{_money(score['decision']['limite'].get('limite_recomendado'))}** "
                f"(atual {_money(score.get('limite_atual'))})\n"
                f"- Principais fatores: {driver_text or 'não disponíveis'}\n\n"
                f"**Fontes**\n{_format_sources(citations)}\n\n{DISCLAIMER}"
            )
        if intent == "limit":
            limit = results.get("limit")
            if not limit:
                return "Não localizei o cliente informado."
            data = limit["limit"]
            return (
                f"**Recomendação de limite — {limit['client_id']}** (preset {limit['preset']})\n\n"
                f"- Limite atual: {_money(data.get('limite_atual'))}\n"
                f"- Limite recomendado: **{_money(data.get('limite_recomendado'))}**\n"
                f"- Crescimento: {_pct((data.get('growth') or 1) - 1)}\n"
                f"- Probabilidade de aceite: {_pct(limit['scores']['aceita_limite'])}\n"
                f"- Probabilidade de inadimplência: {_pct(limit['scores']['inadimplente_90d'])}\n"
                f"- Lucro esperado: {_money(data.get('lucro_esperado'))}\n"
                f"- Decisão de risco: {limit['risk_decision']}\n\n"
                f"**Fontes**\n{_format_sources(citations)}\n\n{DISCLAIMER}"
            )
        if intent == "collection":
            queue = (results.get("queue") or {}).get("queue", [])
            lines = [
                f"- {row['client_id']}: {_money(row['valor_atrasado_cutoff'])}, "
                f"p_pag={_pct(row['p_pagou_30d'])}, EV={_money(row['ev_cobranca'])}, "
                f"prioridade={row['prioridade']}"
                for row in queue[:8]
            ]
            simulation = results.get("simulation")
            sim_text = ""
            if simulation:
                sim_text = (
                    f"\n\nSimulação para {simulation['client_id']} com {simulation['action']}: "
                    f"EV base {_money(simulation['ev_base'])}, "
                    f"EV com ação {_money(simulation['ev_with_action'])}, "
                    f"delta {_money(simulation['delta_ev'])}."
                )
            return (
                "**Fila de cobrança priorizada por valor esperado**\n\n"
                + ("\n".join(lines) if lines else "Sem clientes elegíveis.")
                + sim_text
                + f"\n\n**Fontes**\n{_format_sources(citations)}\n\n{DISCLAIMER}"
            )
        if intent == "model_metrics":
            metrics = (results.get("metrics") or {}).get("edge_metrics", {})
            lines = [
                f"- {target}: AUC {value.get('roc_auc', 0):.3f}, "
                f"KS {value.get('ks', 0):.3f}, Brier {value.get('brier', 0):.3f}"
                for target, value in metrics.items()
            ]
            calibration = (results.get("metrics") or {}).get("calibration", {})
            ece = ", ".join(
                f"{target} {value.get('ece_after', 0):.3f}" for target, value in calibration.items()
            )
            return (
                "**Métricas do modelo Edge**\n\n"
                + ("\n".join(lines) if lines else "Métricas indisponíveis.")
                + f"\n\nECE após calibração: {ece or 'n/d'}\n\n"
                f"**Fontes**\n{_format_sources(citations)}\n\n{DISCLAIMER}"
            )
        if intent == "simulation":
            simulation = results.get("simulation")
            if not simulation:
                return "Não foi possível simular o cenário."
            return (
                f"**Simulação de cobrança — {simulation['client_id']}**\n\n"
                f"- Ação: {simulation['action']}\n"
                f"- Dívida: {_money(simulation['debt'])}\n"
                f"- p_pagamento base: {_pct(simulation['p_pay'])}\n"
                f"- p_pagamento com ação: {_pct(simulation['p_pay_with_action'])}\n"
                f"- EV base: {_money(simulation['ev_base'])}\n"
                f"- EV com ação: {_money(simulation['ev_with_action'])}\n"
                f"- Delta: {_money(simulation['delta_ev'])}\n\n"
                f"**Fontes**\n{_format_sources(citations)}\n\n{DISCLAIMER}"
            )
        return OUT_OF_SCOPE_ANSWER

    def _synthesize(self, state: AgentState) -> dict[str, Any]:
        if state.get("answer"):
            return {}
        offline = self._synthesize_offline(state)
        if not self.llm.available or state.get("intent") in {"out_of_scope", "blocked"}:
            return {"answer": offline}
        context = (state.get("tool_results") or {}).get("policy", {}).get("context", "")
        tool_payload = {
            key: value
            for key, value in (state.get("tool_results") or {}).items()
            if key != "policy"
        }
        try:
            answer = self.llm.complete(
                SYSTEM_PROMPT,
                SYNTHESIS_PROMPT.format(
                    question=state["question"],
                    intent=state.get("intent"),
                    context=context,
                    tool_results=encode_toon(tool_payload),
                ),
                temperature=0.1,
            )
            if not answer:
                answer = offline
        except Exception:
            answer = offline
        if state.get("citations") and "Fonte" not in answer and "[" not in answer:
            answer += f"\n\n**Fontes**\n{_format_sources(state['citations'])}"
        return {"answer": answer}

    def _review(self, state: AgentState) -> dict[str, Any]:
        answer, human_review, reason = check_output(
            state.get("answer", ""),
            state.get("citations", []),
            state.get("intent", "out_of_scope"),
        )
        human_review = human_review or bool(state.get("human_review"))
        intent = state.get("intent", "out_of_scope")
        context = (state.get("tool_results") or {}).get("policy", {}).get("context", "")
        if intent in {"policy", "model_metrics"} and answer:
            response = None
            if self.jev.available:
                response = self.jev.ask(
                    {
                        "pergunta": state.get("question", ""),
                        "resposta": answer,
                        "contexto": context,
                    },
                    RAG_QUESTIONS,
                )
            verification = (
                compose_rag_verification(response)
                if response
                else fallback_rag_verification(answer, state.get("citations", []))
            )
            state["trace"].log(
                "review_jev",
                decision=verification.decision,
                supported=verification.supported,
                contradicts=verification.contradicts,
                source=verification.meta.source,
            )
            if verification.decision != "aceitar":
                human_review = True
                reason = "A verificação Jev não confirmou a ancoragem da resposta."
                if "verificação" not in answer.lower():
                    answer += "\n\n**Verificação:** resposta parcialmente ancorada; revisar fontes."
        if human_review and "revisão humana" not in answer.lower():
            answer += "\n\n**Revisão humana obrigatória antes de qualquer decisão adversa.**"
        state["trace"].log(
            "review",
            human_review=human_review,
            reason=reason,
        )
        return {"answer": answer, "human_review": human_review}

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 2)
        answer = state.get("answer", OUT_OF_SCOPE_ANSWER)
        state["trace"].log("finalize", latency_ms=latency_ms, human_review=bool(state.get("human_review")))
        state["trace"].write(answer, state.get("intent", "out_of_scope"), bool(state.get("human_review")))
        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "tool_calls": state.get("tool_calls", []),
            "citations": state.get("citations", []),
        }

    def run(
        self,
        question: str,
        thread_id: str = "demo",
        role: str = "analista",
        client_id: str | None = None,
    ) -> dict[str, Any]:
        self.tools.calls = []
        started_at = time.perf_counter()
        trace = AgentTrace(
            run_id=new_run_id(),
            question=question,
            role=role,
            thread_id=thread_id,
        )
        state: AgentState = {
            "question": question,
            "role": role,
            "client_id": client_id,
            "thread_id": thread_id,
            "run_id": trace.run_id,
            "started_at": started_at,
            "trace": trace,
            "errors": [],
        }
        result = self.graph.invoke(state)
        return {
            "run_id": trace.run_id,
            "answer": result.get("answer", OUT_OF_SCOPE_ANSWER),
            "intent": result.get("intent", "out_of_scope"),
            "citations": result.get("citations", []),
            "tool_calls": result.get("tool_calls", []),
            "human_review": bool(result.get("human_review")),
            "latency_ms": result.get("latency_ms", 0.0),
        }


def run_agent(
    question: str,
    thread_id: str = "demo",
    role: str = "analista",
    client_id: str | None = None,
) -> dict[str, Any]:
    return AgentRunner().run(question, thread_id=thread_id, role=role, client_id=client_id)
