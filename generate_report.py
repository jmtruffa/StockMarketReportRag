# generate_report.py
# Standalone script that generates a market report and writes it to a .txt file.
#
# Usage:
#   python generate_report.py --date 2026-01-27 --market US
#   python generate_report.py --date 2026-01-27 --market US --debug
#   python generate_report.py --date 2026-01-27 --market AR --out informe_ar.txt
#
# The workflow mirrors app_streamlit.py but runs headless (no Streamlit).

import argparse
import datetime
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package is required. pip install openai")
    sys.exit(1)

from config.market_config import get_market_config, MarketConfig
from core.compute_variations import compute_variations
from core.evaluator import (
    load_dataset,
    build_eval_prompt,
    call_evaluator,
    extract_date_from_prompt,
    find_reference_for_date,
    normalize_decimal,
)
from core.utils import format_variations_for_prompt, fetch_url_text
from core.debug_logger import DebugSession

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def build_system_prompt(config: MarketConfig, csv_block: str, question: str,
                        news_text: str = "", news_urls: List[str] | None = None) -> str:
    """Load the system prompt template and fill in context + question."""
    # Gather news context
    extra_context_parts: list[str] = []
    if news_text and news_text.strip():
        extra_context_parts.append(news_text.strip())
    if news_urls:
        for url in news_urls:
            txt = fetch_url_text(url)
            if txt:
                extra_context_parts.append(txt)

    merged_context = csv_block
    if extra_context_parts:
        merged_context += "\n\n" + "\n\n".join(extra_context_parts)

    path = config.system_prompt_path
    if not os.path.exists(path):
        print(f"WARNING: system prompt template not found at {path}, using fallback")
        return f"{merged_context}\n\n{question}"

    with open(path, "r", encoding="utf-8") as fh:
        template = fh.read()
    return template.format(context=merged_context, question=question)


def run_generation(
    config: MarketConfig,
    target_date: str,
    news_text: str = "",
    news_urls: Optional[List[str]] = None,
    temperature: float = 0.0,
) -> tuple[str, float, DebugSession]:
    """
    End-to-end report generation with evaluation loop.
    Returns (final_answer, final_score, debug_session).
    """
    debug = DebugSession()
    debug.start(market_id=config.market_id, target_date=target_date)

    # ── 1. Compute variations ─────────────────────────────────────
    print(f"📥 Downloading ticker data for {config.market_name} ({target_date})…")
    df_out, close_df, data_date_mode, failed_tickers = compute_variations(
        config.ticker_map, lookback=config.default_lookback, target_date=target_date
    )
    if failed_tickers:
        print(f"⚠️  Failed tickers: {', '.join(failed_tickers)}")

    # Determine effective report date
    if data_date_mode:
        report_date_obj = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
    else:
        report_date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    dia = DIAS_SEMANA[report_date_obj.weekday()]
    qdate = f"{report_date_obj.strftime('%d/%m/%Y')} ({dia})"

    requested_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    if data_date_mode and requested_date > report_date_obj:
        print(f"⚠️  Market not yet open for {target_date}. Using data from {data_date_mode}.")

    # ── 2. Build prompt ───────────────────────────────────────────
    csv_block = format_variations_for_prompt(df_out)
    question = f"Generá resumen para {qdate}"
    system_prompt = build_system_prompt(config, csv_block, question, news_text, news_urls)
    user_message = question

    # ── 3. Load evaluation dataset ────────────────────────────────
    ds = load_dataset(config.threshold_dataset_path)
    few_shot = ds[:3]
    csv_for_eval = csv_block

    # Incluir TODAS las noticias en el contexto del evaluador
    news_parts = []
    if news_text and news_text.strip():
        news_parts.append(news_text.strip())
    if news_urls:
        for url in news_urls:
            txt = fetch_url_text(url)
            if txt:
                news_parts.append(f"[Fuente: {url}]\n{txt[:3000]}")
    if news_parts:
        csv_for_eval += "\n\n=== NOTICIAS (contexto válido proporcionado al escritor) ===\n"
        csv_for_eval += "\n\n".join(news_parts)

    query_date = extract_date_from_prompt(csv_block)
    reference_response = None
    if query_date:
        ref_entry = find_reference_for_date(ds, query_date)
        if ref_entry:
            reference_response = ref_entry.get("response")
            print(f"📊 Reference found in dataset for {query_date}")

    # ── 4. Generate + evaluate loop ───────────────────────────────
    client = OpenAI()
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    PLATEAU_THRESHOLD = 0.02
    GOOD_ENOUGH_AFTER = 3
    GOOD_ENOUGH_SCORE = 0.88

    best_answer = ""
    best_score = 0.0
    accumulated_feedback: list[dict] = []
    eval_history: list[dict] = []
    consecutive_plateau = 0
    prev_score = 0.0

    for attempt in range(config.max_eval_retries):
        # Temperatura: subir muy poco (+0.03/intento, max +0.12)
        retry_temp = min(temperature + (attempt * 0.03), temperature + 0.12)
        print(f"\n🔄 Iteration {attempt + 1}/{config.max_eval_retries} (temp={retry_temp:.2f})…")

        if attempt == 0:
            send_messages = base_messages.copy()
        else:
            # Tomar SOLO el último feedback (el más reciente y relevante)
            last_fb = accumulated_feedback[-1]
            feedback_block = f"Score: {last_fb['score']:.2f}\n"
            if last_fb.get('reason'):
                feedback_block += f"Análisis: {last_fb['reason']}\n"
            if last_fb.get('datos_correctos') is False:
                feedback_block += "⚠️ HAY VALORES NUMÉRICOS INCORRECTOS. Verificá cada dato contra el CSV.\n"
            if last_fb.get('narrativa_quality'):
                feedback_block += f"Calidad narrativa: {last_fb['narrativa_quality']}\n"
            if last_fb.get('mejoras'):
                feedback_block += "MEJORAS PRIORITARIAS:\n"
                for mejora in last_fb['mejoras']:
                    feedback_block += f"  • {mejora}\n"

            retry_user_msg = f"""{user_message}

=== DATOS CSV DE REFERENCIA (verificá tus valores contra estos) ===
{csv_for_eval}

=== TU RESPUESTA ANTERIOR (intento {attempt}, score {best_score:.2f}) ===
{best_answer}

=== FEEDBACK DEL EVALUADOR ===
{feedback_block}

=== INSTRUCCIONES DE MEJORA ===
Tomá tu respuesta anterior como base y EDITALA aplicando las mejoras pedidas.
NO reescribas desde cero: mejorá lo que ya tenés sin perder lo que estaba bien.

1. Aplicá las mejoras prioritarias listadas arriba.
2. Verificá que TODOS los valores numéricos coincidan exactamente con el CSV.
3. Mejorá la narrativa: explicá POR QUÉ se movió el mercado, conectando causas y efectos.
4. Integrá noticias y datos macro como causas de los movimientos.
5. No pierdas información correcta que ya tenías en la versión anterior.

Generá la respuesta mejorada:"""
            send_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user_msg},
            ]

        resp = client.chat.completions.create(
            model=config.openai_model,
            messages=send_messages,
            temperature=min(retry_temp, 1.0),
        )
        answer = resp.choices[0].message.content

        # Evaluate
        eval_prompt = build_eval_prompt(
            few_shot, csv_for_eval, answer, reference_response,
            iteration=attempt + 1,
            previous_attempts=eval_history if eval_history else None,
            user_prompt=question,
        )
        eval_res, eval_raw = call_evaluator(eval_prompt, openai_model=config.openai_model, temperature=0.0)
        score = eval_res.get("score", 0.0)
        reason = eval_res.get("reason", "")

        # Record in debug session
        writer_user = send_messages[-1]["content"]
        debug.add_iteration(
            iteration=attempt + 1,
            writer_system=system_prompt,
            writer_user=writer_user,
            writer_response=answer,
            writer_temperature=retry_temp,
            evaluator_prompt=eval_prompt,
            evaluator_raw=eval_raw,
            eval_score=score,
            eval_ok=score >= config.min_eval_score,
            eval_reason=reason,
        )

        emoji = "✅" if score >= config.min_eval_score else "⚠️"
        print(f"  {emoji} Score: {score:.2f}  —  {reason[:120]}")

        if score > best_score:
            best_score = score
            best_answer = answer

        if score >= config.min_eval_score:
            break

        # ── Detección de plateau ──
        improvement = score - prev_score if attempt > 0 else score
        prev_score = score

        if attempt > 0 and improvement < PLATEAU_THRESHOLD:
            consecutive_plateau += 1
        else:
            consecutive_plateau = 0

        if consecutive_plateau >= 2 and best_score >= GOOD_ENOUGH_SCORE:
            print(f"  ⏹️  Plateau detected (score stable at ~{best_score:.2f}). Accepting best result.")
            break
        if attempt + 1 >= GOOD_ENOUGH_AFTER and best_score >= GOOD_ENOUGH_SCORE:
            print(f"  ⏹️  Good enough after {attempt + 1} attempts (score={best_score:.2f}). Accepting.")
            break

        # Acumular feedback estructurado para el próximo intento
        if attempt < config.max_eval_retries - 1:
            accumulated_feedback.append({
                "score": score,
                "reason": reason,
                "datos_correctos": eval_res.get("datos_correctos", True),
                "narrativa_quality": eval_res.get("narrativa_quality", ""),
                "mejoras": eval_res.get("mejoras", []),
            })
            eval_history.append({
                "iteration": attempt + 1,
                "response": answer,
                "score": score,
                "reason": reason,
                "mejoras": eval_res.get("mejoras", []),
            })
    debug.finish(final_answer=best_answer, final_score=best_score)
    return best_answer, best_score, debug


def main():
    parser = argparse.ArgumentParser(description="Generate a financial market report")
    parser.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    parser.add_argument("--market", default="US", help="Market ID: US, AR, …")
    parser.add_argument("--out", default=None, help="Output .txt path (default: informe_<market>_<date>.txt)")
    parser.add_argument("--news", default="", help="Inline news text")
    parser.add_argument("--news-urls", nargs="*", default=[], help="News URLs (space-separated)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    args = parser.parse_args()

    config = get_market_config(args.market)
    print(f"🦜 PARROT — Generating {config.market_name} report for {args.date}")

    answer, score, debug_session = run_generation(
        config=config,
        target_date=args.date,
        news_text=args.news,
        news_urls=args.news_urls or None,
        temperature=args.temperature,
    )

    # Save report
    out_path = args.out or f"informe_{config.market_id}_{args.date}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(answer)
    print(f"\n📄 Report saved to {out_path}  (score={score:.2f})")

    # Print iteration summary
    if debug_session.iterations:
        print(f"\n🔄 Iterations: {debug_session.total_iterations}")
        for it in debug_session.iterations:
            emoji = "✅" if it.eval_ok else "⚠️"
            print(f"  {emoji} #{it.iteration}  score={it.eval_score:.2f}  reason={it.eval_reason[:100]}")

    # Exit code based on score
    if score < config.min_eval_score:
        print(f"\n⚠️  Final score {score:.2f} is below threshold {config.min_eval_score}.")
        sys.exit(2)


if __name__ == "__main__":
    main()
