"""
TechTrinetra7 — Premium Pipeline Dashboard
Renders every stage of the Advanced RAG architecture for inspection:
ingestion -> chunking -> dense/sparse/llamaindex retrieval -> RRF fusion
-> reranking -> generation, with timings and token estimates.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from theme import CYAN, VIOLET, GREEN, AMBER, MUTED, TEXT, CARD_BG


def _card_open(title: str, badge_text: str = "", badge_class: str = "tt7-pill-cyan"):
    badge = f'<span class="tt7-badge {badge_class}">{badge_text}</span>' if badge_text else ""
    st.markdown(f'<div class="tt7-card"><h4 style="margin:0 0 8px 0;">{title} {badge}</h4>', unsafe_allow_html=True)


def _card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def render_pipeline_overview(trace, chunk_count: int):
    _card_open("🧭 Advanced RAG Pipeline — Architecture Trace", "Enterprise Document Search", "tt7-pill-violet")
    stages = ["Ingestion", "Chunking", "Query Rewrite", "Dense", "Sparse", "LlamaIndex", "RRF Fusion", "Rerank", "Generation"]
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages):
        with col:
            st.markdown(
                f'<div style="text-align:center;font-size:11px;color:{MUTED};">{stage}</div>'
                f'<div style="text-align:center;font-size:18px;color:{CYAN};">●</div>',
                unsafe_allow_html=True,
            )
    st.caption(f"Indexed chunks in corpus: **{chunk_count}**  •  Query variants generated: **{len(trace.query_variants)}**")
    _card_close()


def render_timings(trace):
    _card_open("⏱️ Stage Latency Breakdown")
    df = pd.DataFrame(
        [{"stage": k.replace("_", " ").title(), "ms": v} for k, v in trace.timings_ms.items() if k != "total"]
    )
    if not df.empty:
        fig = go.Figure(go.Bar(
            x=df["ms"], y=df["stage"], orientation="h",
            marker=dict(color=CYAN), text=df["ms"].round(0), textposition="outside",
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT), xaxis_title="milliseconds",
        )
        st.plotly_chart(fig, use_container_width=True)
    total = trace.timings_ms.get("total", 0)
    st.caption(f"Total round-trip: **{total:.0f} ms**  •  Context tokens (est.): **{trace.token_estimate.get('context_tokens_est','-')}**")
    _card_close()


def render_query_expansion(trace):
    _card_open("🔁 Query Rewrite (LCEL chain · gpt-4o-mini)", "Stage 3", "tt7-pill-cyan")
    for i, v in enumerate(trace.query_variants):
        tag = "original" if i == 0 else f"variant {i}"
        st.markdown(f'<div class="tt7-chunk"><b style="color:{CYAN};">{tag}</b> — {v}</div>', unsafe_allow_html=True)
    _card_close()


def _render_result_list(results, title, badge, badge_class, color, score_label="score"):
    _card_open(title, badge, badge_class)
    if not results:
        st.caption("No results.")
    for r in results[:6]:
        meta = r["metadata"]
        score_txt = f" · {score_label}: {r['score']:.4f}" if r.get("score") is not None else ""
        src = meta.get("source", "?")
        loc = meta.get("page") or meta.get("slide") or meta.get("chunk_id") or meta.get("block") or ""
        st.markdown(
            f'<div class="tt7-chunk"><b style="color:{color};">#{r.get("rank","?")}</b> '
            f'<span style="color:{MUTED};">{src} {f"(loc {loc})" if loc != "" else ""}{score_txt}</span><br>'
            f'{r["content"][:220].replace(chr(10), " ")}…</div>',
            unsafe_allow_html=True,
        )
    _card_close()


def render_hybrid_retrieval(trace):
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_result_list(trace.dense_results, "🔵 Dense Retrieval", "FAISS · OpenAI embeddings", "tt7-pill-green", GREEN)
    with col2:
        _render_result_list(trace.sparse_results, "🟠 Sparse Retrieval", "BM25 keyword", "tt7-pill-amber", AMBER)
    with col3:
        _render_result_list(trace.llamaindex_results, "🟣 LlamaIndex Retrieval", "Secondary dense path", "tt7-pill-violet", VIOLET)


def render_fusion(trace):
    _render_result_list(
        trace.fused_results, "🧩 Reciprocal Rank Fusion (RRF)",
        "merged dense + sparse + llamaindex", "tt7-pill-cyan", CYAN, score_label="rrf score"
    )


def render_reranking(trace):
    _render_result_list(
        trace.reranked_results, "🎯 Cross-Encoder Reranking",
        "local model · no API cost", "tt7-pill-violet", VIOLET, score_label="relevance"
    )


def render_generation(trace):
    _card_open("✨ Final Grounded Answer", "gpt-4o-mini", "tt7-pill-cyan")
    st.markdown(trace.answer)
    with st.expander("View final context sent to the LLM"):
        st.text(trace.final_context[:6000])
    _card_close()


def render_full_dashboard(trace, chunk_count: int):
    render_pipeline_overview(trace, chunk_count)
    render_timings(trace)
    render_query_expansion(trace)
    st.markdown("#### Hybrid Retrieval — Dense vs Sparse vs LlamaIndex")
    render_hybrid_retrieval(trace)
    render_fusion(trace)
    render_reranking(trace)
    render_generation(trace)
