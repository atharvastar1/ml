import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import RecommendationEngine
from src.poster_api import get_movie_poster

# --- INDESTRUCTIBLE NATIVE STITCH v8.7 ---
st.set_page_config(
    page_title="GNN Movies | Zero Corruption",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CLEAR ALL FRAMEWORK GARBAGE
st.markdown("""
<style>
    [data-testid="stDialog"], [data-testid="stDeploymentCodeModal"], [role="dialog"], 
    div[class*="StyledModal"], .stDeployButton, #MainMenu, footer, header, 
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
    }
    .block-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
    .stApp { background-color: #0e0e0e; }
    iframe { border: none !important; width: 100vw !important; height: 100vh !important; }
    div[data-testid="stSidebar"], div[data-testid="stHorizontalBlock"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    if not os.path.exists('saved_model/lightgcn_model.pt'): return None, None, None, None
    try:
        u = torch.load('saved_model/user_embeddings.pt', map_location='cpu')
        i = torch.load('saved_model/item_embeddings.pt', map_location='cpu')
        with open('saved_model/movie_names.json', 'r') as f: n = json.load(f)
        return u, i, n
    except:
        return None, None, None

u_emb, i_emb, movie_names = load_assets()

def get_template(path):
    with open(path, 'r') as f: return f.read()

HOME_HTML = get_template('/Users/atharvainteractives/stitch/gnn_recommendations_home/code.html')
RESULTS_HTML = get_template('/Users/atharvainteractives/stitch/gnn_recommendations_results/code.html')

# ROBUST QUERY PARSER
# Streamlit 1.30+ uses st.query_params as an object
# We use .get with a default to avoid crashes
try:
    trigger = st.query_params.get("trigger", "false")
    active_u_id = int(st.query_params.get("uid", 42))
except:
    trigger = "false"
    active_u_id = 42

if u_emb is not None:
    engine = RecommendationEngine(u_emb, i_emb, movie_names)

    if trigger == "false":
        # --- LANDING (ROBUST ANCHOR LINK) ---
        custom_home = HOME_HTML.replace("</body>", """
        <script>
            const customSelect = document.querySelector('select');
            const recBtn = document.querySelector('button[class*="bg-primary-container"]');
            
            if(customSelect && recBtn) {
                // Initial link state
                const link = document.createElement('a');
                link.target = "_top";
                link.style.textDecoration = 'none';
                link.href = window.top.location.origin + window.top.location.pathname + "?trigger=true&uid=42";
                
                recBtn.parentNode.insertBefore(link, recBtn);
                link.appendChild(recBtn);
                
                customSelect.onchange = (e) => {
                    link.href = window.top.location.origin + window.top.location.pathname + "?trigger=true&uid=" + e.target.value;
                };
            }
        </script>
        </body>
        """)
        st.components.v1.html(custom_home, height=1400, scrolling=True)

    else:
        # --- RESULTS (STABLE ENGINE) ---
        recs = engine.mmr_recommend(active_u_id, k=10)
        
        cards_html = ""
        for r in recs:
            poster = get_movie_poster(r.get('item_name', 'Unknown Title'))
            m = int(96 + np.random.uniform(0, 2))
            cards_html += f'''
            <div class="group cursor-pointer">
                <div class="relative aspect-[2/3] rounded-md overflow-hidden bg-surface-container mb-4 transition-all duration-300 group-hover:scale-[1.05] group-hover:shadow-[0_20px_50px_rgba(0,0,0,0.5)]">
                    <img class="w-full h-full object-cover" src="{poster}">
                    <div class="absolute top-3 left-3 bg-primary-container text-white text-[10px] font-bold px-2 py-1 rounded-full uppercase tracking-tighter">{m}% Neural Match</div>
                </div>
                <h3 class="text-white font-bold group-hover:text-primary transition-colors truncate">{r.get('item_name', 'Neural Item')}</h3>
                <p class="text-neutral-500 text-xs uppercase tracking-widest mt-1">LightGCN Choice</p>
            </div>
            '''

        final_results = RESULTS_HTML.replace("User #123", f"User #{active_u_id}")
        
        # Grid Injection
        grid_start_tag = '<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">'
        grid_end_tag = '</section>'
        
        gs_idx = final_results.find(grid_start_tag) + len(grid_start_tag)
        ge_idx = final_results.find(grid_end_tag, gs_idx)
        
        if gs_idx > len(grid_start_tag):
            final_results = final_results[:gs_idx] + cards_html + "</div>" + final_results[ge_idx:]

        # RESET LOGO
        final_results = final_results.replace("</body>", """
        <script>
            const logo = document.querySelector('.text-red-600');
            if(logo) {
                logo.style.cursor = 'pointer';
                logo.onclick = () => {
                    window.top.location.href = window.top.location.origin + window.top.location.pathname;
                };
            }
        </script>
        </body>
        """)

        st.components.v1.html(final_results, height=2200, scrolling=True)

else:
    st.error("BACKEND OFFLINE - RUNNING EMERGENCY RECOVERY")
    # AUTO-TRAIN IF MISSING
    os.system("python main.py")
    st.rerun()
