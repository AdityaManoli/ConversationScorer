"""
Ahoum Conversation Evaluation — Streamlit UI
Run: streamlit run ui/app.py
"""

import streamlit as st
import pandas as pd
import json
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.scorer as _scorer_module
from src.scorer import score_turn

# ── Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ahoum — Conversation Evaluator",
    page_icon="🧠",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'facets_cleaned.csv')

@st.cache_data
def load_facets():
    rows = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

facets_all = load_facets()
categories = sorted(set(f["category"] for f in facets_all))

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🦙 Ollama Settings")
    ollama_url = st.text_input(
        "Ollama base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    ollama_model = st.selectbox(
        "Model",
        options=["llama3.1:8b", "llama3.1", "llama3", "llama3:8b", "qwen2", "qwen2:7b", "mistral", "mixtral", "phi3", "gemma2"],
        index=0,
        help="Must be pulled in Ollama first: `ollama pull llama3.1:8b`"
    )

    # Patch env so scorer picks up changes
    os.environ["OLLAMA_BASE_URL"] = ollama_url
    os.environ["OLLAMA_MODEL"] = ollama_model
    # Patch the live module variables so get_llm instantiates ChatOllama cleanly
    _scorer_module.OLLAMA_BASE_URL = ollama_url
    _scorer_module.OLLAMA_MODEL = ollama_model

    st.markdown("---")
    selected_categories = st.multiselect(
        "Facet Categories",
        options=categories,
        default=categories[:4],
        help="Filter which facet categories to evaluate"
    )
    
    observable_only = st.checkbox("Observable facets only", value=True,
        help="Exclude facets that can't be measured from text alone")
    
    batch_size = st.slider("Batch size (facets per LLM call)", 10, 300, 150,
        help="Larger = much faster but ensure you are using a model with a large context limit like Llama 3.1")
    
    st.markdown("---")
    st.markdown(f"**Total facets loaded:** {len(facets_all)}")
    
    # Filter facets based on selection
    filtered = [
        f for f in facets_all
        if f["category"] in selected_categories
        and (not observable_only or f.get("observable_in_text", "True") == "True")
    ]
    st.markdown(f"**Facets to score:** {len(filtered)}")
    
    st.markdown("---")
    st.markdown("### 📊 Facet Distribution")
    cat_counts = pd.DataFrame(
        [(f["category"], 1) for f in facets_all],
        columns=["Category", "Count"]
    ).groupby("Category").sum().reset_index()
    st.bar_chart(cat_counts.set_index("Category"))


# ── Main Panel ─────────────────────────────────────────────────────────────

st.title("🧠 Ahoum — Conversation Evaluation")
st.caption("Score conversation turns across 300+ psychological, linguistic & safety facets")

tab1, tab2, tab3 = st.tabs(["📝 Score a Turn", "💬 Score Conversation", "📂 Browse Sample Results"])

# ── Tab 1: Single Turn Scoring ─────────────────────────────────────────────

with tab1:
    st.subheader("Score a Single Conversation Turn")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        speaker = st.text_input("Speaker", value="User")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
    
    turn_text = st.text_area(
        "Conversation turn text",
        height=120,
        placeholder="e.g. I don't know what to do anymore. Everything feels pointless..."
    )
    
    if st.button("🔍 Score Turn", type="primary", disabled=not turn_text.strip()):
        if not filtered:
            st.error("No facets selected. Please pick at least one category in the sidebar.")
        else:
            facet_dicts = [
                {"facet_id": int(f["facet_id"]), "facet_name": f["facet_name"], "category": f["category"]}
                for f in filtered
            ]
            
            with st.spinner(f"Scoring {len(facet_dicts)} facets in {len(facet_dicts)//batch_size + 1} batch(es)..."):
                full_text = f"{speaker}: {turn_text}"
                scores = score_turn(full_text, facet_dicts, batch_size=batch_size, model=ollama_model)
            
            st.success(f"✅ Scored {len(scores)} facets!")
            
            df = pd.DataFrame(scores)
            df["score_bar"] = df["score"].apply(lambda s: "█" * s + "░" * (5-s))
            df["conf_pct"] = (df["confidence"] * 100).round(1).astype(str) + "%"
            
            # Top hits
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🔝 Top 10 Highest Scores")
                top = df.nlargest(10, "score")[["facet_name", "score", "conf_pct", "rationale"]]
                st.dataframe(top, width="stretch", hide_index=True)
            with col_b:
                st.markdown("#### ⬇️ 10 Lowest Scores")
                bot = df.nsmallest(10, "score")[["facet_name", "score", "conf_pct", "rationale"]]
                st.dataframe(bot, width="stretch", hide_index=True)
            
            st.markdown("#### 📋 All Scores")
            st.dataframe(
                df[["facet_id", "facet_name", "score", "confidence", "rationale"]],
                width="stretch",
                hide_index=True,
            )
            
            # Download
            csv_data = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv_data, "scores.csv", "text/csv")


# ── Tab 2: Multi-turn Conversation ─────────────────────────────────────────

with tab2:
    st.subheader("Score a Multi-turn Conversation")
    st.caption("Enter each turn as JSON or use the form below")
    
    num_turns = st.number_input("Number of turns", min_value=1, max_value=10, value=3)
    
    turns = []
    for i in range(int(num_turns)):
        col1, col2 = st.columns([1, 4])
        with col1:
            sp = st.selectbox(f"Speaker {i+1}", ["User", "Assistant", "Other"],
                key=f"sp_{i}")
        with col2:
            txt = st.text_input(f"Turn {i+1}", key=f"txt_{i}",
                placeholder=f"Turn {i+1} text...")
        if txt.strip():
            turns.append({"turn_id": i+1, "speaker": sp, "text": txt})
    
    if st.button("🔍 Score All Turns", type="primary", disabled=len(turns)==0):
        if not filtered:
            st.error("No facets selected.")
        else:
            facet_dicts = [
                {"facet_id": int(f["facet_id"]), "facet_name": f["facet_name"], "category": f["category"]}
                for f in filtered
            ]
            
            all_turn_scores = []
            progress = st.progress(0)
            
            for idx, turn in enumerate(turns):
                with st.spinner(f"Scoring turn {idx+1}/{len(turns)}..."):
                    full_text = f"{turn['speaker']}: {turn['text']}"
                    scores = score_turn(full_text, facet_dicts, batch_size=batch_size, model=ollama_model)
                    all_turn_scores.append({**turn, "scores": scores})
                progress.progress((idx+1)/len(turns))
            
            st.success(f"✅ Scored {len(turns)} turns × {len(facet_dicts)} facets each!")
            
            for ts in all_turn_scores:
                with st.expander(f"Turn {ts['turn_id']} — {ts['speaker']}: \"{ts['text'][:60]}...\""):
                    df = pd.DataFrame(ts["scores"])
                    st.dataframe(df[["facet_name","score","confidence","rationale"]],
                                 width="stretch", hide_index=True)


# ── Tab 3: Browse Sample Conversations ─────────────────────────────────────

with tab3:
    st.subheader("Browse Pre-scored Sample Conversations")
    
    conv_dir = os.path.join(os.path.dirname(__file__), '..', 'conversations')
    
    files = [f for f in os.listdir(conv_dir) if f.endswith('.json') and f != 'all_conversations_scored.json'] \
        if os.path.exists(conv_dir) else []
    
    if not files:
        st.info("No sample conversations found. Run `python src/generate_sample_conversations.py` to generate them.")
    else:
        selected_file = st.selectbox("Select conversation", sorted(files))
        fpath = os.path.join(conv_dir, selected_file)
        
        with open(fpath) as f:
            conv = json.load(f)
        
        st.markdown(f"**Theme:** `{conv.get('theme')}` — {conv.get('description', '')}")
        
        st.markdown("#### 💬 Conversation")
        for turn in conv.get("turns", []):
            role = "👤" if turn["speaker"] == "User" else "🤖"
            st.markdown(f"{role} **{turn['speaker']}:** {turn['text']}")
        
        if "scored_turn" in conv:
            st.markdown("#### 📊 Scores for Highlighted Turn")
            st.caption(f"Scored: \"{conv['scored_turn']['text'][:80]}...\"")
            
            scores_df = pd.DataFrame(conv["scored_turn"]["scores"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Score", f"{scores_df['score'].mean():.2f}/5")
            col2.metric("Avg Confidence", f"{scores_df['confidence'].mean():.0%}")
            col3.metric("Facets Scored", len(scores_df))
            
            # Score distribution
            score_dist = scores_df["score"].value_counts().sort_index()
            st.bar_chart(score_dist)
            
            st.dataframe(
                scores_df[["facet_name","score","confidence","rationale"]],
                width="stretch",
                hide_index=True,
            )
