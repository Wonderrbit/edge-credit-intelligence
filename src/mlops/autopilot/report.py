from __future__ import annotations

from typing import Any


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/d"
    return f"R$ {number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/d"


def _num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/d"


def build_markdown(
    manifest: dict[str, Any],
    portfolio: dict[str, Any],
    comparison: dict[str, Any],
    decisions: list[dict[str, Any]],
    quality: dict[str, Any],
    alerts: list[dict[str, Any]],
    jev_stats: dict[str, int],
) -> str:
    lines = [
        f"# Autopilot Akropoli — {manifest.get('run_id')}",
        "",
        f"- Início: {manifest.get('started_at')}",
        f"- Fim: {manifest.get('finished_at')}",
        f"- Modelo: {manifest.get('model_version')}",
        f"- Política: {manifest.get('policy_version')}",
        f"- Autonomia: {manifest.get('autonomy')}",
        f"- Modo Jev: {manifest.get('judgment_mode')}",
        "",
        "## Qualidade dos dados",
        "",
        f"- Status: **{'OK' if quality.get('passed') else 'ATENÇÃO'}**",
        f"- Linhas: {quality.get('rows')} | Colunas: {quality.get('columns')}",
    ]
    for check in quality.get("checks", []):
        status = "ok" if check.get("passed") else "falha"
        lines.append(f"- `{check.get('name')}`: {status} — {check.get('detail')}")

    lines += [
        "",
        "## Indicadores da carteira",
        "",
        "| Indicador | Valor |",
        "|---|---:|",
        f"| Clientes | {portfolio.get('n_clientes')} |",
        f"| Carteira total | {_money(portfolio.get('carteira_total'))} |",
        f"| Limite médio | {_money(portfolio.get('limite_medio'))} |",
        f"| Renda média | {_money(portfolio.get('renda_media'))} |",
        f"| PD média 90d | {_pct(portfolio.get('pd_media_90d'))} |",
        f"| Perda esperada | {_money(portfolio.get('perda_esperada'))} |",
        f"| Taxa de atraso | {_pct(portfolio.get('taxa_atraso'))} |",
        f"| Valor em atraso | {_money(portfolio.get('valor_em_atraso'))} |",
        f"| Utilização média | {_pct(portfolio.get('utilizacao_media'))} |",
        f"| Comprometimento médio | {_pct(portfolio.get('comprometimento_medio'))} |",
        f"| Taxa de aprovação | {_pct(portfolio.get('taxa_aprovacao'))} |",
        f"| EV de cobrança | {_money(portfolio.get('ev_cobranca'))} |",
        f"| Concentração top 10 | {_pct(portfolio.get('concentracao_top10'))} |",
        f"| PSI vs. run anterior | {_num(comparison.get('psi_score'))} |",
        "",
        "## Alertas",
        "",
    ]
    if alerts:
        for alert in alerts:
            lines.append(
                f"- **{alert.get('severity', 'info').upper()}** `{alert.get('code')}`: "
                f"{alert.get('message')}"
            )
    else:
        lines.append("- Nenhum alerta disparado.")

    risk_ranked = sorted(
        decisions,
        key=lambda item: float(item.get("risk", {}).get("composite_risk", 0.0)),
        reverse=True,
    )[:10]
    lines += [
        "",
        "## Top riscos",
        "",
        "| Cliente | Ação | PD | Composto | Confiança | Revisão |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in risk_ranked:
        risk = item.get("risk", {})
        lines.append(
            f"| {item.get('client_id')} | {risk.get('action')} | "
            f"{_pct(risk.get('model_pd'))} | {_num(risk.get('composite_risk'))} | "
            f"{_num(risk.get('confidence'), 2)} | "
            f"{'sim' if risk.get('manual_review_recommended') else 'não'} |"
        )

    collection_ranked = sorted(
        decisions,
        key=lambda item: float(item.get("collection", {}).get("ev", 0.0)),
        reverse=True,
    )[:10]
    lines += [
        "",
        "## Top cobrança",
        "",
        "| Cliente | Ação | Prioridade | EV | Confiança |",
        "|---|---|---|---:|---:|",
    ]
    for item in collection_ranked:
        collection = item.get("collection", {})
        lines.append(
            f"| {item.get('client_id')} | {collection.get('action')} | "
            f"{collection.get('priority')} | {_money(collection.get('ev'))} | "
            f"{_num(collection.get('channel_confidence'), 2)} |"
        )

    lines += [
        "",
        "## Operação autônoma",
        "",
        "- Decisões adversas são executadas e auditadas (modo `full`), sem bloqueio humano.",
        "- Cada decisão carrega `decision_id`, `inputs_hash`, versão de modelo e política.",
        "- A fila de revisão é apenas recomendação (`manual_review_recommended`), não um gate.",
        "",
    ]
    narrative = manifest.get("narrative")
    if narrative:
        lines += [
            "## Narrativa executiva (OpenAI)",
            "",
            f"**{narrative.get('headline', '')}**",
            "",
        ]
        for finding in narrative.get("key_findings", []):
            lines.append(f"- {finding}")
        if narrative.get("risks"):
            lines += ["", "**Riscos**", ""]
            lines += [f"- {risk}" for risk in narrative["risks"]]
        if narrative.get("recommended_actions"):
            lines += ["", "**Ações recomendadas**", ""]
            lines += [f"- {action}" for action in narrative["recommended_actions"]]
        lines += ["", f"_{narrative.get('disclaimer', '')}_", ""]

    lines += [
        "## Jev / custo",
        "",
        f"- Chamadas Jev: {jev_stats.get('calls', 0)}",
        f"- Fallbacks: {jev_stats.get('fallbacks', 0)}",
        f"- Custo estimado: US$ {float(manifest.get('estimated_cost_usd', 0.0)):.4f}",
        "",
        "## Arquivos",
        "",
        f"- `{manifest.get('run_dir')}/indicators.parquet`",
        f"- `{manifest.get('run_dir')}/decisions.jsonl`",
        f"- `{manifest.get('run_dir')}/review_queue.jsonl`",
        f"- `{manifest.get('run_dir')}/audit.jsonl`",
        f"- `{manifest.get('run_dir')}/portfolio.json`",
        "",
    ]
    return "\n".join(lines)
