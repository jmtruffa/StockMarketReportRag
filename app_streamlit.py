# app_streamlit.py
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os, json, datetime
from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import pandas as pd

# langchain HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# openai optional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from core.utils import clean_text, fetch_url_text, SimpleDoc, extract_date_from_text, format_variations_for_prompt, df_to_single_doc
from core.evaluator import load_dataset, build_eval_prompt, call_evaluator, extract_date_from_prompt, find_reference_for_date
from core.compute_variations import compute_variations
from config.market_config import get_market_config
from core.debug_logger import DebugSession

# --------- Config / defaults ----------
st.set_page_config(page_title="PARROT RAG", layout="wide")
st.title("🦜 PARROT RAG — informe de rueda (CSV -> resumen)")

DEFAULT_EMBED     = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# --------- Sidebar ----------
st.sidebar.header("⚙️ Configuración")

# Opciones para informe
st.sidebar.markdown("### Opciones para informe")
generate_report_checkbox = st.sidebar.checkbox("Generar informe de la rueda", value=False)

# Market toggle (US ↔ AR) — right below the report checkbox
is_argentina = st.sidebar.toggle("🇦🇷 Mercado Argentina", value=False, help="Desactivado = 🇺🇸 EEUU · Activado = 🇦🇷 Argentina")
selected_market = "AR" if is_argentina else "US"
market_config = get_market_config(selected_market)

# Derive paths from market config
SYSTEM_PROMPT_PATH = market_config.system_prompt_path
RAG_PROMPT_PATH = market_config.rag_prompt_path
THRESHOLD_PATH = market_config.threshold_dataset_path
TICKER_MAP = market_config.ticker_map

date_picker = st.sidebar.date_input("Fecha del informe (default:hoy)", value=None)

# Debug toggle
enable_debug = st.sidebar.checkbox("🐛 Mostrar debug (conversación agentes)", value=False)
st.sidebar.markdown("---")

# news inputs
news_text = st.sidebar.text_area("Texto de noticias (opcional)", value="", height=150)
news_urls = st.sidebar.text_area("URLs de noticias (una por línea)", value="", height=120)

k_total = st.sidebar.slider("Top-k total", 1, 30, 10, 1)
openai_model = st.sidebar.text_input("Modelo OpenAI", value=DEFAULT_OAI_MODEL)
embed_model = st.sidebar.text_input("Modelo de Embeddings (HF)", value=DEFAULT_EMBED)
history_path = st.sidebar.text_input("Archivo de historial (JSONL)", value="./data/history/chat_history.jsonl")
show_sources = st.sidebar.checkbox("Mostrar fuentes", value=True)
show_scores = st.sidebar.checkbox("Mostrar score", value=False)
show_preview = st.sidebar.checkbox("Mostrar preview", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Requiere OPENAI_API_KEY si querés usar verificador OpenAI.")

# --------- caches ----------
@st.cache_resource(show_spinner=False)
def get_embedder_cached(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})

embedder = get_embedder_cached(embed_model)

# no vectordb usage: retrieval will use only `news_text`, `news_urls` and computed CSV

# session_state for variations
if "computed_variations" not in st.session_state:
    st.session_state["computed_variations"] = None
# date for which computed_variations was generated (YYYY-MM-DD or None)
if "computed_variations_date" not in st.session_state:
    st.session_state["computed_variations_date"] = None
# data date mode (la moda de las fechas de los datos)
if "data_date_mode" not in st.session_state:
    st.session_state["data_date_mode"] = None
# failed tickers
if "failed_tickers" not in st.session_state:
    st.session_state["failed_tickers"] = []
# market status warning
if "market_warning" not in st.session_state:
    st.session_state["market_warning"] = None

# Nota: el checkbox "Generar informe" solo indica usar el `systemprompt_template`.
# No se descarga/ejecuta `compute_variations` hasta que el usuario envíe la consulta.
if generate_report_checkbox:
    st.sidebar.info("Al enviar la consulta se usará el sistema de 'informe' (system prompt). El CSV se descargará solo si es necesario.")

# ----- Chat render previo (history) -----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_sources and m.get("sources"):
            with st.expander("Fuentes"):
                for s in m.get("sources", []):
                    st.text(clean_text(s))

# ----- Main interaction -----
user_input = st.chat_input("Escribí tu pregunta (ej: \"Generá resumen para 16/01/2026\")")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # build retrieval context (very similar to previous pipeline)
    candidates_all = []
    local_docs = []

    # add news_text as individual news items (do NOT chunk)
    if news_text and news_text.strip():
        # split on blank lines to separate distinct news items; fallback to whole text
        parts = [p.strip() for p in news_text.split("\n\n") if p.strip()]
        if not parts:
            parts = [news_text.strip()]
        for i, part in enumerate(parts, 1):
            local_docs.append(SimpleDoc(page_content=part, metadata={"_collection":"news_user","source":"news_text","news_id":i}))

    # add each URL as a single SimpleDoc (do NOT chunk)
    if news_urls and news_urls.strip():
        for i, url in enumerate([u.strip() for u in news_urls.splitlines() if u.strip()], 1):
            txt = fetch_url_text(url)
            if not txt:
                continue
            local_docs.append(SimpleDoc(page_content=txt, metadata={"_collection":"news_url","source":url,"news_id":i}))

    # add CSV rows (computed)
    # If the user selected 'Generar informe' (use system prompt) we compute the CSV
    df_computed = st.session_state.get("computed_variations")
    # determine date to request for compute: prefer date in user_input, otherwise date_picker
    extracted_date = None
    try:
        extracted_date = extract_date_from_text(user_input)
    except Exception:
        extracted_date = None
    compute_target_date = None
    if extracted_date:
        compute_target_date = extracted_date
    elif date_picker is not None:
        try:
            compute_target_date = date_picker.strftime("%Y-%m-%d")
        except Exception:
            compute_target_date = None
    else:
        # default to today's date if none provided
        try:
            compute_target_date = datetime.date.today().strftime("%Y-%m-%d")
        except Exception:
            compute_target_date = None

    # If user asked for system prompt / informe, compute CSV
    # or if the already-computed CSV was generated for a different date.
    existing_computed_date = st.session_state.get("computed_variations_date")
    existing_data_date_mode = st.session_state.get("data_date_mode")
    
    if generate_report_checkbox:
        need_compute = False
        if df_computed is None:
            need_compute = True
        else:
            # Comparar fechas: recalcular si la fecha solicitada cambió
            if existing_computed_date != compute_target_date:
                need_compute = True
            # TAMBIÉN recalcular si los datos existentes son de una fecha anterior a la solicitada
            # (por si el mercado ya abrió desde la última vez que se calculó)
            elif existing_data_date_mode and compute_target_date:
                data_mode_date = datetime.datetime.strptime(existing_data_date_mode, "%Y-%m-%d").date()
                requested_date = datetime.datetime.strptime(compute_target_date, "%Y-%m-%d").date()
                if data_mode_date < requested_date:
                    # Los datos son viejos, intentar recalcular por si el mercado ya abrió
                    need_compute = True
                    
        if need_compute:
            try:
                df_out, close, data_date_mode, failed_tickers = compute_variations(TICKER_MAP, lookback="30d", target_date=compute_target_date)
                st.session_state["computed_variations"] = df_out
                # store the date used for this computation (can be None)
                st.session_state["computed_variations_date"] = compute_target_date
                st.session_state["data_date_mode"] = data_date_mode
                st.session_state["failed_tickers"] = failed_tickers
                df_computed = df_out
                
                # Verificar si hay tickers que fallaron
                if failed_tickers:
                    st.warning(f"⚠️ Los siguientes tickers fallaron en la descarga después de reintentos: {', '.join(failed_tickers)}")
                
                # Verificar si el mercado abrió comparando la fecha solicitada con la moda de fechas
                if data_date_mode and compute_target_date:
                    requested_date = datetime.datetime.strptime(compute_target_date, "%Y-%m-%d").date()
                    data_mode_date = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
                    
                    if requested_date > data_mode_date:
                        # El mercado todavía no abrió para la fecha solicitada
                        warning_msg = f"⚠️ **ATENCIÓN**: El mercado de EEUU todavía no abrió para el {requested_date.strftime('%d/%m/%Y')}. Los datos disponibles son del {data_mode_date.strftime('%d/%m/%Y')}. El informe se generará con los datos del día anterior."
                        st.session_state["market_warning"] = warning_msg
                        st.warning(warning_msg)
                    else:
                        st.session_state["market_warning"] = None
                
                st.info("CSV calculado y almacenado en sesión (computed_variations).")
            except Exception as e:
                st.warning(f"No se pudo descargar el CSV: {e}")

    df_use = df_computed
    if df_use is not None:
        # crea un único SimpleDoc que contiene TODO el CSV
        single_doc = df_to_single_doc(
            df_use,
            source_name="variacion_diaria.csv",
            extra_meta={"_collection": "tickers_csv", "source": "csv"}
        )
        local_docs.append(single_doc)

    # try to embed local_docs and add to candidates_all (so they are considered for retrieval)
    try:
        embedder_local = embedder
        texts = [d.page_content for d in local_docs]
        if texts:
            # compute query embedding (prefer embed_query, fallback to embed_documents)
            query_emb = None
            try:
                if hasattr(embedder_local, "embed_query"):
                    query_emb = embedder_local.embed_query(user_input)
                else:
                    q_embs = embedder_local.embed_documents([user_input])
                    query_emb = q_embs[0] if q_embs else None
            except Exception:
                query_emb = None

            # compute embeddings for local docs
            try:
                emb_list = embedder_local.embed_documents(texts)
            except Exception:
                emb_list = []

            import numpy as _np
            def cos(a,b):
                a=_np.array(a); b=_np.array(b)
                if a.size==0 or b.size==0: return 0.0
                na=_np.linalg.norm(a); nb=_np.linalg.norm(b)
                if na==0 or nb==0: return 0.0
                return float(_np.dot(a,b)/(na*nb))

            if emb_list and query_emb is not None:
                for d, emb in zip(local_docs, emb_list):
                    sim = cos(query_emb, emb)
                    dist = 1.0 - sim
                    candidates_all.append((d, dist))
            else:
                # fallback: include local_docs with neutral distance so they appear in context
                for d in local_docs:
                    candidates_all.append((d, 1.0))
    except Exception:
        # ignore embed errors but ensure local_docs are still considered
        for d in local_docs:
            candidates_all.append((d, 1.0))

    # Build top docs (simple scoring: use dist as score proxy, no recency for brevity)
    scored = []
    for doc, dist in candidates_all:
        scored.append((doc, dist, 1.0/(1.0+dist)))
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:k_total]
    top_docs = [doc for (doc,_,_) in top]

    # Build context string for RAG
    context_parts = []
    for i, doc in enumerate(top_docs,1):
        context_parts.append(f"[{i}] ({doc.metadata.get('_collection')})\n{doc.page_content}")
    context = "\n\n".join(context_parts)


    # If generate_report_checkbox is True -> use systemprompt_template and include the CSV (if exists)
    use_system_prompt = generate_report_checkbox and os.path.exists(SYSTEM_PROMPT_PATH)
    # Ensure the date for the summary is present in the question passed to the system prompt.
    # Prefer an explicit date in the user_input (extracted), otherwise use the date_picker if provided.
    try:
        extracted_date = extract_date_from_text(user_input)
    except Exception:
        extracted_date = None
    
    # Diccionario para días de la semana en español
    DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    
    # Verificar si el mercado no abrió y debemos usar la fecha de los datos reales
    market_warning = st.session_state.get("market_warning")
    data_date_mode = st.session_state.get("data_date_mode")
    
    # Determinar la fecha real a usar para el informe
    # Si el mercado no abrió, usar la fecha de los datos (data_date_mode)
    if market_warning and data_date_mode:
        # Usar la fecha de los datos reales (día anterior)
        report_date_obj = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
        dia_semana = DIAS_SEMANA[report_date_obj.weekday()]
        qdate = f"{report_date_obj.strftime('%d/%m/%Y')} ({dia_semana})"
        market_note = f"\n\n[NOTA: El mercado de EEUU todavía no abrió para la fecha solicitada. Los datos corresponden al {qdate}. Usá esta fecha para el informe.]"
    else:
        # Usar la fecha extraída, date_picker, o la fecha de los datos si está disponible
        if extracted_date:
            report_date_obj = datetime.datetime.strptime(extracted_date, "%Y-%m-%d").date()
        elif date_picker is not None:
            report_date_obj = date_picker
        elif data_date_mode:
            # Usar la fecha real de los datos del CSV
            report_date_obj = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
        else:
            report_date_obj = datetime.date.today()
        dia_semana = DIAS_SEMANA[report_date_obj.weekday()]
        qdate = f"{report_date_obj.strftime('%d/%m/%Y')} ({dia_semana})"
        market_note = ""
    
    # Construir question_for_prompt con la fecha y día de la semana correctos
    if user_input and user_input.strip():
        question_for_prompt = f"{user_input} para {qdate}"
    else:
        question_for_prompt = f"Generá resumen para {qdate}"
    
    # Agregar nota de mercado si corresponde
    if market_note:
        question_for_prompt = question_for_prompt + market_note
    
    if use_system_prompt:
        # build CSV textual block
        if df_use is None:
            st.warning("No hay CSV disponible: subí un CSV o presioná 'Calcular CSV' en la barra lateral.")
        csv_block = format_variations_for_prompt(df_use) if df_use is not None else ""
        
        # Filtrar el CSV del context para no duplicarlo (ya está en csv_block)
        context_without_csv = "\n\n".join([
            part for part in context_parts 
            if "tickers_csv" not in part and "variacion_diaria.csv" not in part
        ])
        
        # final context: CSV + news/docs (sin duplicar el CSV)
        if context_without_csv.strip():
            merged_context = csv_block + "\n\n" + context_without_csv
        else:
            merged_context = csv_block
            
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as fh:
                system_prompt_template = fh.read()
        except Exception:
            st.error("No se pudo cargar system prompt.")
            system_prompt_template = "{context}\n\n{question}"
        system_prompt = system_prompt_template.format(context=merged_context, question=question_for_prompt)
    else:
        # normal RAG prompt
        try:
            with open(RAG_PROMPT_PATH, "r", encoding="utf-8") as fh:
                rag_prompt_template = fh.read()
        except Exception:
            rag_prompt_template = "{context}\n\n{question}"
        system_prompt = rag_prompt_template.format(context=context, question=question_for_prompt)

    # Preparar los mensajes que se enviarán al modelo
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_for_prompt}
    ]

    # Preparar evaluación si es informe
    ds = None
    few_shot = None
    reference_response = None
    query_date = None
    if use_system_prompt and df_use is not None:
        ds = load_dataset(THRESHOLD_PATH)
        few_shot = ds[:3]
        prompt_text = format_variations_for_prompt(df_use)
        query_date = extract_date_from_prompt(prompt_text)
        if query_date:
            ref_entry = find_reference_for_date(ds, query_date)
            if ref_entry:
                reference_response = ref_entry.get("response")

    # call LLM (OpenAI) to generate answer - with retry if score < threshold
    MAX_RETRIES = market_config.max_eval_retries
    MIN_SCORE = market_config.min_eval_score
    PLATEAU_THRESHOLD = 0.02   # si el score mejora menos que esto 2 veces seguidas → plateau
    GOOD_ENOUGH_AFTER = 3      # después de N intentos, aceptar si score >= GOOD_ENOUGH_SCORE
    GOOD_ENOUGH_SCORE = 0.88   # umbral secundario para cortar si no alcanza el ideal
    best_answer = None
    best_score = 0.0
    best_eval_res = None
    attempts = []

    # Initialise debug session
    debug_session = DebugSession()
    debug_session.start(market_id=market_config.market_id, target_date=compute_target_date or "")
    
    with st.spinner("Generando respuesta con LLM..."):
        if OpenAI is None:
            answer = "⚠️ OpenAI SDK no disponible en el entorno. Configura OPENAI_API_KEY o instala openai."
        else:
            client = OpenAI()
            # Datos CSV formateados para re-anclar en cada reintento
            csv_data_for_eval = ""
            if use_system_prompt and df_use is not None:
                csv_data_for_eval = format_variations_for_prompt(df_use)

                # Incluir TODAS las noticias en el contexto del evaluador
                news_eval_parts = []
                if news_text and news_text.strip():
                    news_eval_parts.append(news_text.strip())
                if news_urls and news_urls.strip():
                    for url in [u.strip() for u in news_urls.splitlines() if u.strip()]:
                        txt = fetch_url_text(url)
                        if txt:
                            news_eval_parts.append(f"[Fuente: {url}]\n{txt[:3000]}")
                if news_eval_parts:
                    csv_data_for_eval += "\n\n=== NOTICIAS (contexto válido proporcionado al escritor) ===\n"
                    csv_data_for_eval += "\n\n".join(news_eval_parts)

            accumulated_feedback: list[dict] = []  # Historial de correcciones estructuradas
            eval_history: list[dict] = []  # Historial completo para el evaluador
            consecutive_plateau = 0  # cuántas veces seguidas la mejora fue < PLATEAU_THRESHOLD
            prev_score = 0.0

            for attempt in range(MAX_RETRIES):
                try:
                    if attempt == 0:
                        # Primera iteración: prompt original
                        send_messages = llm_messages.copy()
                    else:
                        # Construir bloque de feedback holístico del evaluador
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

                        # En vez de reescribir de cero, darle SU MEJOR respuesta anterior
                        # y pedirle que la EDITE aplicando las mejoras
                        retry_user_msg = f"""{question_for_prompt}

=== DATOS CSV DE REFERENCIA (verificá tus valores contra estos) ===
{csv_data_for_eval}

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
                        model=openai_model,
                        messages=send_messages,
                    )
                    answer = resp.choices[0].message.content
                    
                    # Evaluar solo si es informe
                    if use_system_prompt and df_use is not None and few_shot:
                        eval_prompt = build_eval_prompt(
                            few_shot, csv_data_for_eval, answer, reference_response,
                            iteration=attempt + 1,
                            previous_attempts=eval_history if eval_history else None,
                            user_prompt=question_for_prompt,
                        )
                        eval_res, eval_raw = call_evaluator(eval_prompt, openai_model=openai_model)
                        score = eval_res.get("score", 0.0)
                        reason = eval_res.get("reason", "")
                        attempts.append({"attempt": attempt + 1, "score": score, "reason": reason})

                        # Record in debug session
                        writer_user_msg = send_messages[-1]["content"]
                        debug_session.add_iteration(
                            iteration=attempt + 1,
                            writer_system=system_prompt,
                            writer_user=writer_user_msg,
                            writer_response=answer,
                            evaluator_prompt=eval_prompt,
                            evaluator_raw=eval_raw,
                            eval_score=score,
                            eval_ok=score >= MIN_SCORE,
                            eval_reason=reason,
                        )
                        
                        # Guardar mejor respuesta
                        if score > best_score:
                            best_score = score
                            best_answer = answer
                            best_eval_res = eval_res
                        
                        # Si alcanzamos el umbral, salir
                        if score >= MIN_SCORE:
                            break
                        
                        # ── Detección de plateau ──
                        # Si el score dejó de mejorar significativamente, no gastar más intentos
                        improvement = score - prev_score if attempt > 0 else score
                        prev_score = score
                        
                        if attempt > 0 and improvement < PLATEAU_THRESHOLD:
                            consecutive_plateau += 1
                        else:
                            consecutive_plateau = 0
                        
                        # Cortar si: plateau 2 veces seguidas Y score "bueno suficiente",
                        # O si ya pasamos GOOD_ENOUGH_AFTER intentos con score decente
                        if consecutive_plateau >= 2 and best_score >= GOOD_ENOUGH_SCORE:
                            break
                        if attempt + 1 >= GOOD_ENOUGH_AFTER and best_score >= GOOD_ENOUGH_SCORE:
                            break
                        
                        # Acumular feedback holístico para el próximo intento
                        if attempt < MAX_RETRIES - 1:
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
                    else:
                        # No es informe, no evaluar
                        best_answer = answer
                        break
                        
                except Exception as e:
                    answer = f"⚠️ Error llamando a OpenAI: {e}"
                    best_answer = answer
                    break
            
            # Usar la mejor respuesta encontrada
            if best_answer:
                answer = best_answer

    # Finish debug session
    debug_session.finish(final_answer=answer, final_score=best_score)

    # show sources
    sources = []
    for i, (doc, dist, cscore) in enumerate(top, 1):
        tag = f"{doc.metadata.get('_collection')} - {doc.metadata.get('source','')}"
        preview = (doc.page_content[:240] + "…") if len(doc.page_content)>240 else doc.page_content
        sources.append(f"[{i}] ({tag})\n> {preview}")

    # render answer
    with st.chat_message("assistant"):
        st.markdown(answer)
        if show_sources and sources:
            with st.expander("Fuentes"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})

    # If we generated an "informe", show evaluation results (already computed during generation)
    if use_system_prompt and df_use is not None:
        # Guardar en session_state para que persista al hacer click en botones
        st.session_state["last_eval_result"] = {
            "score": best_score,
            "eval_res": best_eval_res,
            "answer": answer,
            "df_use": df_use,
            "news_text": news_text,
            "attempts": attempts,
            "query_date": query_date,
            "reference_response": reference_response
        }
        # Store debug session in session_state so it survives reruns
        st.session_state["last_debug_session"] = debug_session
        
        if query_date and reference_response:
            st.sidebar.info(f"📊 Usando referencia del dataset para {query_date}")
        
        # Mostrar intentos de generación
        if attempts:
            st.sidebar.markdown("### 🔄 Intentos de generación")
            for att in attempts:
                emoji = "✅" if att["score"] >= MIN_SCORE else "⚠️"
                st.sidebar.write(f"{emoji} Intento {att['attempt']}: Score **{att['score']:.2f}**")
            st.sidebar.write(f"Total de intentos: **{len(attempts)}**")
        
        # Usar los resultados de evaluación ya calculados
        if best_eval_res:
            score = best_eval_res.get("score", 0.0)
            ok = score >= MIN_SCORE
            reason = best_eval_res.get("reason", "")
        else:
            score = 0.0
            ok = False
            reason = "No se pudo evaluar"
        
        st.sidebar.markdown("### Evaluación del informe")
        if score >= MIN_SCORE:
            st.sidebar.success(f"✅ Score: **{score:.2f}** — Aprobado")
        else:
            st.sidebar.warning(f"⚠️ Score: **{score:.2f}** — Mejor resultado después de {len(attempts)} intentos")
        st.sidebar.write(f"Motivo: {reason}")

# ─── Debug visual: timeline de agentes ───────────────────────────
if enable_debug and "last_debug_session" in st.session_state and st.session_state["last_debug_session"]:
    dsess: DebugSession = st.session_state["last_debug_session"]

    st.markdown("---")
    # Header con resumen visual
    score_color = "🟢" if dsess.final_score >= market_config.min_eval_score else "🔴"
    st.markdown(
        f"## 🐛 Debug — Conversación entre agentes\n"
        f"**Iteraciones:** {dsess.total_iterations}  ·  "
        f"**Score final:** {score_color} {dsess.final_score:.2f}"
    )

    for it in dsess.iterations:
        iter_emoji = "✅" if it.eval_ok else "🔄"
        score_bar_pct = int(it.eval_score * 100)

        st.markdown(f"---\n### {iter_emoji} Iteración {it.iteration} de {dsess.total_iterations}")
        # Score progress bar
        st.progress(min(it.eval_score, 1.0), text=f"Score: {it.eval_score:.2f}")

        # ── MODELO ESCRITOR ──────────────────────────────────
        st.markdown("#### ✍️ Modelo Escritor")

        if it.iteration == 1:
            st.caption("📌 Primera iteración — sin correcciones previas")
        else:
            st.caption(f"📌 Iteración {it.iteration} — con correcciones del evaluador")

        # Lo que le llega al escritor (prompt)
        with st.expander("📥 Prompt que recibe el Modelo Escritor", expanded=False):
            st.markdown("**System prompt:**")
            st.text_area(
                "system_writer", it.writer_prompt_system,
                height=200, disabled=True, label_visibility="collapsed",
                key=f"dbg_writer_sys_{it.iteration}"
            )
            label_user = "User prompt (pregunta original)" if it.iteration == 1 else "User prompt (con feedback / correcciones)"
            st.markdown(f"**{label_user}:**")
            st.text_area(
                "user_writer", it.writer_prompt_user,
                height=150, disabled=True, label_visibility="collapsed",
                key=f"dbg_writer_usr_{it.iteration}"
            )

        # Lo que responde el escritor
        st.markdown("**📤 Respuesta del Modelo Escritor:**")
        st.info(it.writer_response)

        # ── MODELO EVALUADOR ─────────────────────────────────
        st.markdown("#### 🔍 Modelo Evaluador")

        # Lo que le llega al evaluador (prompt)
        with st.expander("📥 Prompt que recibe el Modelo Evaluador", expanded=False):
            st.text_area(
                "eval_prompt", it.evaluator_prompt,
                height=250, disabled=True, label_visibility="collapsed",
                key=f"dbg_eval_prompt_{it.iteration}"
            )

        # Lo que responde el evaluador
        st.markdown("**📤 Respuesta del Modelo Evaluador:**")
        # Parsear la respuesta del evaluador para mostrarla de forma legible
        try:
            eval_parsed = json.loads(it.evaluator_raw_response)
            eval_cols = st.columns([1, 1, 1])
            with eval_cols[0]:
                s = eval_parsed.get("score", 0)
                st.metric("Score", f"{s:.2f}", delta=None)
            with eval_cols[1]:
                datos_ok = eval_parsed.get("datos_correctos", None)
                st.metric("Datos correctos", "✅ Sí" if datos_ok else "❌ No")
            with eval_cols[2]:
                narrativa = eval_parsed.get("narrativa_quality", "?")
                emoji_narr = {"alta": "🟢", "media": "🟡", "baja": "🔴"}.get(narrativa, "⚪")
                st.metric("Narrativa", f"{emoji_narr} {narrativa}")

            # Análisis holístico
            reason = eval_parsed.get("reason", "")
            if reason:
                st.info(f"💬 **Análisis:** {reason}")

            # Mejoras sugeridas
            mejoras = eval_parsed.get("mejoras", [])
            if mejoras:
                st.markdown("**📋 Mejoras prioritarias:**")
                for j, mejora in enumerate(mejoras, 1):
                    st.markdown(f"{j}. {mejora}")
            else:
                st.success("Sin mejoras sugeridas — informe excelente")

        except (json.JSONDecodeError, TypeError):
            # Fallback: mostrar raw si no se puede parsear
            st.code(it.evaluator_raw_response, language="json")
            if it.eval_reason:
                st.caption(f"💬 {it.eval_reason}")

    # Footer
    st.markdown("---")
    final_emoji = "✅" if dsess.final_score >= market_config.min_eval_score else "⚠️"
    st.markdown(
        f"### {final_emoji} Resultado final: score **{dsess.final_score:.2f}** "
        f"en **{dsess.total_iterations}** iteración(es)"
    )

# Botón para agregar al dataset (fuera del bloque de generación, usa session_state)
if "last_eval_result" in st.session_state and st.session_state["last_eval_result"]:
    eval_data = st.session_state["last_eval_result"]
    if eval_data.get("score", 0) >= market_config.min_eval_score:
        if st.sidebar.button("➕ Agregar ejemplo al dataset"):
            try:
                df_use_saved = eval_data["df_use"]
                news_text_saved = eval_data.get("news_text", "")
                answer_saved = eval_data["answer"]
                score_saved = eval_data["score"]
                rec = {
                    "prompt": format_variations_for_prompt(df_use_saved) + "\n\nNoticias:\n" + (news_text_saved or ""),
                    "response": answer_saved,
                    "accuracy": score_saved
                }
                with open(THRESHOLD_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st.sidebar.success("✅ Ejemplo agregado al dataset.")
                # Limpiar para no agregar duplicados
                st.session_state["last_eval_result"] = None
            except Exception as e:
                st.sidebar.error(f"Error guardando: {e}")

    # save history
    try:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "timestamp": datetime.datetime.now().isoformat(),
                "query": user_input,
                "answer": answer,
                "sources": sources,
                "generate_report": bool(generate_report_checkbox)
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
