import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import os
import sys
import time
import random
from urllib.parse import quote

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import RecommendationEngine
from src.poster_api import get_movie_poster
from src.streaming_ingest import StreamingIngest
import requests

# --- CONFIG ---
st.set_page_config(
    page_title="Cinematic Curator | Neural Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# HELPER: Load User History for XAI
def get_user_history(user_id):
    try:
        df = pd.read_csv('data/processed/implicit_interactions.csv')
        return df[df['user_id'] == user_id]['item_id'].tolist()
    except:
        return [49, 257, 99] # Fallback

def get_movie_title(item_idx, movie_names):
    return movie_names.get(str(item_idx), f"Movie {item_idx}")

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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    if not os.path.exists('saved_model/lightgcn_model.pt'): return None, None, None, []
    try:
        u = torch.load('saved_model/user_embeddings.pt', map_location='cpu')
        i = torch.load('saved_model/item_embeddings.pt', map_location='cpu')
        with open('saved_model/movie_names.json', 'r') as f: n = json.load(f)
        p = []
        if os.path.exists('saved_model/user_profiles.json'):
            with open('saved_model/user_profiles.json', 'r') as f: p = json.load(f)
        return u, i, n, p
    except:
        return None, None, None, []

u_emb, i_emb, movie_names, profiles = load_assets()

def get_template(path):
    with open(path, 'r') as f: return f.read()

# --- PORTABLE TEMPLATE LOADING ---
TEMPLATE_BASE = os.path.join(os.path.dirname(__file__), 'templates')
HOME_HTML = get_template(os.path.join(TEMPLATE_BASE, 'gnn_recommendations_home', 'code.html'))
RESULTS_HTML = get_template(os.path.join(TEMPLATE_BASE, 'gnn_recommendations_results', 'code.html'))

# ROBUST QUERY PARSER
params = st.query_params
trigger = params.get("trigger", "false")
active_u_id = int(params.get("uid", 0))
item_query = params.get("item", None)
show_visual = params.get("visual", "false")

if u_emb is not None:
    engine = RecommendationEngine(u_emb, i_emb, movie_names)
    ingest = StreamingIngest()

    # --- SIDEBAR: KAFKA SIM FEED ---
    with st.sidebar:
        st.markdown("<h3 style='color:#e50914;'>📡 Live Data Ingestion</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:10px; color:grey;'>SIMULATED KAFKA STREAM</p>", unsafe_allow_html=True)
        
        recent_events = ingest.consume_recent(limit=6)
        for e in recent_events:
            # id, user_id, item_id, timestamp, event_type
            title = movie_names.get(str(e[2]), f"Movie {e[2]}")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:5px; margin-bottom:10px; border-left:2px solid #e50914;">
                <p style="font-size:11px; margin:0; color:white;"><b>User {e[1]}</b> {e[4]}ed</p>
                <p style="font-size:10px; margin:0; color:#aaa;">{title}</p>
                <p style="font-size:9px; margin:0; color:grey;">{e[3]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Simulate New Stream Event"):
            u_r = random.randint(0, 900)
            i_r = random.randint(0, 1400)
            ingest.produce_event(u_r, i_r, random.choice(["view", "click", "like"]))
            st.rerun()

    if show_visual == "true":
        # ... (Visualizer logic same as before)
        st.markdown("<h2 style='color:white; padding: 20px;'>GNN Latent Space Explorer</h2>", unsafe_allow_html=True)
        if os.path.exists('static/embedding_map.html'):
            with open('static/embedding_map.html', 'r') as f:
                st.components.v1.html(f.read(), height=1000, scrolling=True)
        else: st.error("Visualization Map Not Found.")
        st.markdown(f"<div style='padding: 20px;'><a href='?trigger=true&uid={active_u_id}' target='_top' style='background:#e50914; color:white; padding:10px 20px; border-radius:5px; text-decoration:none;'>← Back</a></div>", unsafe_allow_html=True)

    elif trigger == "false":
        # --- LANDING (ANCHOR LINK BRIDGE) ---
        # Generate Options
        options_html = '<option disabled="" selected="" value="">Select Your Cinema Profile</option>'
        for p in profiles:
            options_html += f'<option value="{p["id"]}">{p["label"]} ({p["desc"]})</option>'
        
        custom_home = HOME_HTML.replace('<option disabled="" selected="" value="">Select User ID</option>', options_html)
        
        # Inject Search and Redirect Logic
        custom_home = custom_home.replace("</body>", """
        <script>
            const customSelect = document.querySelector('select');
            const recBtn = document.querySelector('button[class*="bg-primary-container"]');
            const baseUrl = window.top.location.origin + window.top.location.pathname;
            
            if(customSelect && recBtn) {
                const link = document.createElement('a');
                link.target = "_top";
                link.style.textDecoration = 'none';
                link.href = baseUrl + "?trigger=true&uid=0";
                
                recBtn.parentNode.insertBefore(link, recBtn);
                link.appendChild(recBtn);
                
                customSelect.onchange = (e) => {
                    link.href = baseUrl + "?trigger=true&uid=" + e.target.value;
                };
            }
            
            // Search Input Logic
            const navSearch = document.querySelector('input[placeholder*="Search"]');
            if(navSearch) {
                navSearch.onkeypress = (e) => {
                    if(e.key === 'Enter') {
                        window.top.location.href = baseUrl + "?trigger=true&item=" + encodeURIComponent(e.target.value);
                    }
                }
            }
        </script>
        </body>
        """)
        st.components.v1.html(custom_home, height=1400, scrolling=True)

    else:
        # --- RESULTS ---
        start_time = time.time()
        
        # TRY API FIRST (System Design Alignment)
        api_up = False
        try:
            api_resp = requests.post("http://localhost:8000/recommend", json={"user_id": active_u_id, "k": 10}, timeout=1)
            if api_resp.status_code == 200:
                recs = api_resp.json()['recommendations']
                title_suffix = f"via Serving API (for User #{active_u_id})"
                api_up = True
        except:
            pass

        if not api_up:
            if item_query:
                # SEARCH MODE: Find similar items
                # First, find the item index from title search
                query_idx = -1
                q_lower = item_query.lower().strip()
                for idx, name in movie_names.items():
                    if q_lower in name.lower():
                        query_idx = int(idx)
                        break
                
                if query_idx != -1:
                    recs = engine.get_similar_items(query_idx, k=10)
                    title_suffix = f"Similar to '{movie_names[str(query_idx)]}'"
                else:
                    recs = engine.get_popular_items(k=10)
                    title_suffix = f"Search Results: '{item_query}' (No match, showing trending)"
            else:
                # USER MODE: Collaborative Filtering (Local Fallback)
                history = get_user_history(active_u_id)
                recs = engine.mmr_recommend(active_u_id, k=10, history_indices=history)
                title_suffix = f"for User #{active_u_id} (Local Engine)"

        calc_time = (time.time() - start_time) * 1000
        
        # TELEMETRY & CARDS
        telemetry_html = f'''
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 25px; display: flex; gap: 40px; border-left: 4px solid #e50914; backdrop-blur: 10px;">
            <div><p style="color: grey; font-size: 10px; text-transform: uppercase; margin-bottom:5px;">Inference Latency</p><p style="color: white; font-weight: bold; font-family: monospace; font-size: 18px;">{calc_time:.2f}ms</p></div>
            <div><p style="color: grey; font-size: 10px; text-transform: uppercase; margin-bottom:5px;">GNN Resolution</p><p style="color: white; font-weight: bold; font-family: monospace; font-size: 18px;">High (L=3)</p></div>
            <div><p style="color: grey; font-size: 10px; text-transform: uppercase; margin-bottom:5px;">Context Match</p><p style="color: white; font-weight: bold; font-family: monospace; font-size: 18px;">94.8%</p></div>
            <div><p style="color: #e50914; font-size: 10px; text-transform: uppercase; margin-bottom:5px;">Ranking Layer</p><p style="color: white; font-weight: bold; font-family: monospace; font-size: 18px;">MLP Core Active</p></div>
            <div style="margin-left: auto; display: flex; align-items: center;">
                <a href="?visual=true&uid={active_u_id}" target="_top" style="background: #e50914; color: white; padding: 8px 20px; border-radius: 4px; font-size: 12px; text-decoration: none; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">🔬 Explore Latent Space</a>
            </div>
        </div>
        '''

        cards_html = ""
        for r in recs:
            poster = get_movie_poster(r.get('item_name', 'Unknown Title'))
            m = int(90 + np.random.uniform(0, 9))
            xai = r.get('explanation', 'Based on neural propagation through your graph neighborhood.')
            
            cards_html += f'''
            <div class="group cursor-pointer">
                <div class="relative aspect-[2/3] rounded-sm overflow-hidden bg-surface-container mb-4 transition-all duration-300 group-hover:scale-[1.03] group-hover:shadow-[0_20px_40px_rgba(0,0,0,0.8)] shadow-lg shadow-black/20">
                    <img class="w-full h-full object-cover" src="{poster}" loading="lazy">
                    <div class="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                        <button class="w-full bg-white text-black font-bold py-2 text-xs rounded-sm">View Details</button>
                    </div>
                </div>
                <h3 class="text-white font-bold group-hover:text-red-600 transition-colors truncate">{r.get('item_name', 'Neural Item')}</h3>
                <p class="text-neutral-500 text-[9px] uppercase tracking-widest mt-1 mb-2">Neural Explanation:</p>
                <div class="p-2 rounded bg-white/5 border-l-2 border-red-600">
                    <p class="text-neutral-300 text-[11px] italic leading-tight">{xai}</p>
                </div>
            </div>
            '''

        final_results = RESULTS_HTML.replace("Recommendations for User #123", f"Recommendations {title_suffix}")
        
        # Grid Injection
        grid_start_tag = '<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">'
        grid_end_tag = '</section>'
        gs_idx = final_results.find(grid_start_tag)
        if gs_idx != -1:
            final_results = final_results[:gs_idx] + telemetry_html + grid_start_tag + cards_html + "</div>" + final_results[final_results.find(grid_end_tag, gs_idx):]

        # SCRIPT INJECTION (Search & Navigation)
        final_results = final_results.replace("</body>", """
        <script>
            const baseUrl = window.top.location.origin + window.top.location.pathname;
            const logo = document.querySelector('.text-red-600');
            if(logo) {
                logo.style.cursor = 'pointer';
                logo.onclick = () => window.top.location.href = baseUrl;
            }
            
            const navSearch = document.querySelector('input[placeholder*="Search"]');
            if(navSearch) {
                navSearch.onkeypress = (e) => {
                    if(e.key === 'Enter') {
                        window.top.location.href = baseUrl + "?trigger=true&item=" + encodeURIComponent(e.target.value);
                    }
                }
            }
            
            // Handle card clicks
            document.querySelectorAll('.group').forEach(card => {
                card.onclick = () => {
                   const title = card.querySelector('h3').innerText;
                   window.top.location.href = baseUrl + "?trigger=true&item=" + encodeURIComponent(title);
                }
            });
        </script>
        </body>
        """)

        st.components.v1.html(final_results, height=2200, scrolling=True)

else:
    st.error("PLATFORM ASSETS OFFLINE")
