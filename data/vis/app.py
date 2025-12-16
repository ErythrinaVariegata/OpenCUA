import streamlit as st
import pandas as pd
import json
import os
import re
import random
import argparse
import sys
from PIL import Image, ImageDraw

# ================= Argument Parsing =================
parser = argparse.ArgumentParser(description="AgentNet Visualizer")
parser.add_argument("--traj_file", type=str, help="Path to trajectory jsonl file")
parser.add_argument("--meta_file", type=str, help="Path to meta metadata jsonl file")
parser.add_argument("--image_dir", type=str, help="Path to the directory containing images")

try:
    args, unknown = parser.parse_known_args()
except SystemExit:
    st.stop()

# Validate arguments
if not args.traj_file or not args.meta_file or not args.image_dir:
    st.error("❌ Missing required arguments.")
    st.code("Usage: streamlit run app.py -- --traj_file <PATH> --meta_file <PATH> --image_dir <PATH>")
    st.stop()

TRAJ_FILE = args.traj_file
META_FILE = args.meta_file
IMAGE_DIR = args.image_dir

# ================= Page Settings & Style =================
st.set_page_config(layout="wide", page_title="AgentNet Visualizer Pro", page_icon="🎨")

def set_custom_style():
    st.markdown("""
        <style>
        .stApp { font-family: 'Inter', sans-serif; }
        section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
        /* Metric Styling */
        div[data-testid="stMetric"] {
            background-color: #ffffff; padding: 10px; border-radius: 8px;
            border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            margin-top: 10px;
        }
        .info-badge {
            display: inline-block; padding: 3px 8px; border-radius: 4px;
            font-size: 0.8em; font-weight: 600; margin-right: 5px; margin-bottom: 5px;
        }
        .badge-system { background-color: #e0f2fe; color: #0284c7; }
        .badge-quality { background-color: #dcfce7; color: #16a34a; }
        .badge-domain { background-color: #f3e8ff; color: #9333ea; }
        .stButton button { width: 100%; border-radius: 6px; }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# ================= Data Processing =================

@st.cache_data
def load_data(traj_path, meta_path):
    # 1. Load Trajectory
    traj_data = []
    if os.path.exists(traj_path):
        with open(traj_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: traj_data.append(json.loads(line))
                except: continue
    else:
        st.error(f"Trajectory file not found: {traj_path}")
        return pd.DataFrame()

    df_traj = pd.DataFrame(traj_data)
    df_traj['calc_step_len'] = df_traj['traj'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 2. Load Meta
    df_meta = pd.DataFrame()
    if os.path.exists(meta_path):
        meta_list = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: meta_list.append(json.loads(line))
                except: continue
        df_meta = pd.DataFrame(meta_list)
    
    # 3. Merge
    if not df_meta.empty:
        def extract_quality(feedback):
            return feedback.get('quality', 'N/A') if isinstance(feedback, dict) else 'N/A'
        df_meta['quality'] = df_meta['verify_feedback'].apply(extract_quality)
        df_traj['task_id'] = df_traj['task_id'].astype(str)
        df_meta['task_id'] = df_meta['task_id'].astype(str)
        df_merged = pd.merge(df_traj, df_meta, on='task_id', how='left', suffixes=('', '_meta'))
    else:
        df_merged = df_traj
        df_merged['quality'] = 'N/A'
        df_merged['system'] = 'N/A'
        df_merged['domains'] = None

    # 4. Clean
    df_merged['quality'] = df_merged['quality'].fillna('N/A')
    df_merged['system'] = df_merged['system'].fillna('N/A')
    
    def normalize_domains(d):
        if isinstance(d, list): return ", ".join(d)
        if pd.isna(d): return "N/A"
        return str(d)
    
    df_merged['domains_str'] = df_merged['domains'].apply(normalize_domains)
    return df_merged

def parse_coordinates(code_str):
    if not isinstance(code_str, str): return None, None
    x_match = re.search(r'x\s*=\s*(\d+(\.\d+)?)', code_str)
    y_match = re.search(r'y\s*=\s*(\d+(\.\d+)?)', code_str)
    if x_match and y_match:
        try: return float(x_match.group(1)), float(y_match.group(1))
        except: return None, None
    return None, None

def draw_circle_on_image(image_path, x, y):
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)
        final_x, final_y = (x * img_w, y * img_h) if (x <= 1.0 and y <= 1.0) else (x, y)
        r, color = 15, '#ff2b2b'
        draw.ellipse((final_x-r, final_y-r, final_x+r, final_y+r), outline=color, width=5)
        draw.ellipse((final_x-3, final_y-3, final_x+3, final_y+3), fill=color)
        return img
    except: return None

# ================= Main Application =================

def main():
    if 'current_task_id' not in st.session_state:
        st.session_state['current_task_id'] = None

    st.sidebar.title("🔍 AgentNet Pro")
    
    with st.spinner('Loading Dataset...'):
        df = load_data(TRAJ_FILE, META_FILE)

    if df.empty:
        st.stop()
    
    total_count = len(df)

    # ================= 1. Filters & Search Section =================
    st.sidebar.subheader("Filters & Search")
    
    # --- Fuzzy Search Input ---
    search_query = st.sidebar.text_input("🔍 Fuzzy Search", placeholder="Type ID or Instruction...")

    # --- Filters ---
    min_s, max_s = int(df['calc_step_len'].min()), int(df['calc_step_len'].max())
    default_min = 20 if max_s >= 20 else min_s
    step_range = st.sidebar.slider("Step Length", min_s, max_s, (default_min, max_s))
    
    all_qs = sorted(df['quality'].unique().tolist())
    def_qs = ['good'] if 'good' in all_qs else all_qs
    sel_qs = st.sidebar.multiselect("Quality", all_qs, default=def_qs)
    
    all_sys = sorted(df['system'].unique().tolist())
    sel_sys = st.sidebar.multiselect("System", all_sys, default=all_sys)
    
    all_dom = sorted(df['domains_str'].unique().tolist())
    sel_dom = st.sidebar.multiselect("Domains", all_dom, default=all_dom)

    # --- Apply Logic ---
    filtered_df = df[
        (df['calc_step_len'] >= step_range[0]) &
        (df['calc_step_len'] <= step_range[1]) &
        (df['quality'].isin(sel_qs)) &
        (df['system'].isin(sel_sys)) &
        (df['domains_str'].isin(sel_dom))
    ]

    # Apply Fuzzy Search
    if search_query:
        mask = (
            filtered_df['task_id'].str.contains(search_query, case=False, na=False) | 
            filtered_df['instruction'].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    # --- 2. Metric & Search Result Selector (Moved Here) ---
    st.sidebar.markdown("---")
    
    # Display Count
    st.sidebar.metric(label="Tasks Matched", value=len(filtered_df), delta=f"Total: {total_count}", delta_color="off")

    # [NEW] Dropdown Selection for Search Results
    # Only show this dropdown if a search query is active
    if search_query:
        if not filtered_df.empty:
            st.sidebar.markdown("### 👇 Select Result")
            search_results = filtered_df['task_id'].tolist()
            
            # Helper to sync selection
            def on_search_select():
                st.session_state['current_task_id'] = st.session_state.search_select_box
            
            # Format function to show ID + Short Instruction
            def search_format(tid):
                instr = filtered_df[filtered_df['task_id']==tid]['instruction'].values[0]
                return f"{tid} | {instr[:30]}..."

            st.sidebar.selectbox(
                "Matches:",
                options=search_results,
                format_func=search_format,
                key="search_select_box",
                on_change=on_search_select
            )
        else:
            st.sidebar.warning("No matches found.")

    if filtered_df.empty:
        st.warning("⚠️ No tasks match the current filters.")
        st.stop()

    # ================= Navigation =================
    
    task_ids = filtered_df['task_id'].tolist()
    
    # Sync Session State
    if st.session_state['current_task_id'] not in task_ids:
        # If searching, the selectbox above handles assignment, but if not searching, reset to 0
        if not search_query:
             st.session_state['current_task_id'] = task_ids[0]
        else:
             # If searching but current ID is invalid (e.g. typing new query), default to first match
             if not filtered_df.empty:
                 st.session_state['current_task_id'] = task_ids[0]

    # Safe get index
    try:
        current_idx = task_ids.index(st.session_state['current_task_id'])
    except ValueError:
        current_idx = 0
        st.session_state['current_task_id'] = task_ids[0]

    # Top Navigation Bar
    c_prev, c_rand, c_next, c_jump = st.columns([1, 1, 1, 2])
    
    with c_prev:
        if st.button("⬅️ Prev", disabled=(current_idx == 0)):
            st.session_state['current_task_id'] = task_ids[current_idx - 1]
            st.rerun()
            
    with c_rand:
        if st.button("Random 🎲"):
            st.session_state['current_task_id'] = random.choice(task_ids)
            st.rerun()

    with c_next:
        if st.button("Next ➡️", disabled=(current_idx == len(task_ids) - 1)):
            st.session_state['current_task_id'] = task_ids[current_idx + 1]
            st.rerun()

    with c_jump:
        def jump_to_id():
            target = st.session_state.jump_input.strip()
            if target in df['task_id'].values:
                st.session_state['current_task_id'] = target
                if target not in task_ids:
                    st.toast(f"ID {target} loaded (hidden by current filters)", icon="⚠️")
            else:
                st.toast("ID not found", icon="❌")
        st.text_input("Jump to ID", key="jump_input", placeholder="Paste Task ID...", on_change=jump_to_id)

    # Main List Dropdown (Syncs with Sidebar)
    def on_main_select():
        st.session_state['current_task_id'] = st.session_state.main_nav_select

    # Dynamic label
    list_label = "Current Filtered List" if not search_query else "Search Results List"

    selected_tid = st.selectbox(
        list_label, 
        options=task_ids, 
        index=current_idx,
        key="main_nav_select",
        on_change=on_main_select,
        format_func=lambda x: f"[{filtered_df[filtered_df['task_id']==x]['quality'].values[0].upper()}] {x} (Steps: {filtered_df[filtered_df['task_id']==x]['calc_step_len'].values[0]})"
    )

    # ================= Details View =================
    
    if selected_tid in filtered_df['task_id'].values:
        row = filtered_df[filtered_df['task_id'] == selected_tid].iloc[0]
    elif selected_tid in df['task_id'].values:
        row = df[df['task_id'] == selected_tid].iloc[0]
        st.warning(f"Note: Task {selected_tid} is hidden by filters.")
    else:
        st.error("Task not found.")
        st.stop()

    with st.container():
        st.markdown(f"## 🆔 {row['task_id']}")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(f"**Instruction:** {row['instruction']}", icon="🎯")
            if 'natural_language_task' in row and row['natural_language_task']:
                st.markdown(f"**Natural Language:** {row['natural_language_task']}")
        with c2:
            st.markdown(f"""
                <span class="info-badge badge-quality">{row['quality']}</span>
                <span class="info-badge badge-system">{row['system']}</span>
                <span class="info-badge badge-domain">{row['domains_str']}</span>
                <span class="info-badge" style="background-color:#f1f5f9;color:#475569">Steps: {row['calc_step_len']}</span>
            """, unsafe_allow_html=True)
            with st.expander("Meta JSON"):
                meta_dict = row.to_dict()
                if 'traj' in meta_dict: del meta_dict['traj']
                st.json(meta_dict)

    st.markdown("---")

    st.subheader(f"🎞️ Trajectory ({row['calc_step_len']} steps)")
    traj_list = row['traj']
    
    for i, step in enumerate(traj_list):
        step_value = step.get('value', {})
        action_code = step_value.get('code', '')
        img_filename = step.get('image')

        col_img, col_text = st.columns([1.6, 1])

        with col_img:
            if img_filename:
                full_img_path = os.path.join(IMAGE_DIR, img_filename)
                if os.path.exists(full_img_path):
                    cx, cy = parse_coordinates(action_code)
                    display_img = draw_circle_on_image(full_img_path, cx, cy) if cx else Image.open(full_img_path)
                    caption = f"Coords: ({cx}, {cy})" if cx else "No coords"
                    st.image(display_img, caption=f"Step {i+1} | {caption}", use_container_width=True)
                else:
                    st.error(f"Image missing: {img_filename}")
            else:
                st.warning("No Image")

        with col_text:
            with st.container():
                st.markdown(f"#### Step {i+1}")
                st.markdown("**Action:**")
                st.success(step_value.get('action', 'N/A'))
                if action_code:
                    st.code(action_code, language='python')
                
                thought = step_value.get('thought', '')
                obs = step_value.get('observation', '')
                if thought or obs:
                    with st.expander("🧠 Logic"):
                        if thought: st.markdown(f"**Thought:**\n{thought}")
                        if obs: 
                            st.divider()
                            st.markdown(f"**Observation:**\n{obs}")
        st.divider()

if __name__ == "__main__":
    main()