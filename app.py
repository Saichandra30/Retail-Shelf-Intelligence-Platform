import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import streamlit as st
import pandas as pd
from pipeline import run_pipeline, get_pipeline_components

st.set_page_config(page_title="Retail Shelf Intelligence", layout="wide", page_icon="🛒")

st.markdown("""
<style>
.metric-value { font-size: 2rem; font-weight: bold; color: #10B981; }
.metric-label { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-container { background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }
.welcome-card { background-color: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 30px; margin-top: 20px; line-height: 1.8; }
.ocr-tag { display: inline-block; background-color: #1E293B; border: 1px solid #10B981; border-radius: 6px; padding: 4px 10px; margin: 4px; font-family: monospace; color: #10B981; }
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state["results"] = {}

st.sidebar.title("🛒 Control Center")
st.sidebar.markdown("### Upload Shelf Images")
uploaded_files = st.sidebar.file_uploader("Accepted formats: .jpg  .jpeg  .png", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

selected_file_name = None
if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.sidebar.selectbox("Select image to inspect", file_names)

st.sidebar.markdown("### Actions")
run_btn = st.sidebar.button("▶  Run Inference Pipeline", type="primary", use_container_width=True)  # noqa: keep for sidebar compat

with st.sidebar.expander("⚙️ Model Metadata", expanded=False):
    st.markdown("- **Detection:** YOLOv8n (SKU-110K weights)")
    st.markdown("- **OCR:** EasyOCR (CRAFT + CRNN, CPU Mode)")
    st.markdown("- **Classification:** Hierarchical Zero-Shot (SigLIP, CPU Mode)")

st.title("🛍️ Retail Shelf Intelligence Platform")
st.markdown("Automated **on-shelf availability**, **share of shelf**, and **planogram auditing** powered by YOLOv8 + Hybrid OCR-SigLIP + EasyOCR.")

if not uploaded_files:
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Welcome to the Retail Shelf Intelligence Dashboard</h2>
        <p>To begin auditing your retail shelf assets, follow these steps:</p>
        <ol>
            <li><b>Upload Images</b> — Use the sidebar file uploader (.jpg / .jpeg / .png)</li>
            <li><b>Select Image</b> — Pick one from the dropdown to inspect</li>
            <li><b>Run Pipeline</b> — Click <b>"Run Inference Pipeline"</b> to execute processes</li>
        </ol>
        <p><b>Supported shelf types:</b> Beverages · Snacks · Dairy · General FMCG</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if run_btn:
        os.makedirs("images", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        for file in uploaded_files:
            # Save a unique dynamic copy to explicitly bust frozen system state registers
            import time
            unique_id = int(time.time())
            img_path = os.path.join("images", f"{unique_id}_{file.name}")
            
            with open(img_path, "wb") as fh:
                fh.write(file.getbuffer())
            
            # FIX: Unified into a single, clean spinner block to eliminate the indentation crash
            with st.spinner(f"Running live pipeline evaluation on {file.name} ..."):
                try:
                    # Force run fresh, live code compilation pass
                    result = run_pipeline(img_path, "outputs")
                    clean = {
                        "image_name": result.get("image_name", ""),
                        "total_products": result.get("total_products", 0),
                        "brands": result.get("brands", {}),
                        "ocr_labels": result.get("ocr_labels", [])
                    }
                    st.session_state["results"][file.name] = result
                    st.success(f"✅ {file.name} processed successfully.")
                except Exception as e:
                    # FIX: Cleaned up duplicate calls to output a single, professional error box
                    st.error(f"❌ Pipeline runtime failure for {file.name}: {e}")

    if selected_file_name and selected_file_name in st.session_state["results"]:
        res = st.session_state["results"][selected_file_name]
        
        # Expand the JSON payload to fulfill the final assignment requirements
        clean = {
            "image_name": res.get("image_name", ""),
            "total_products": res.get("total_products", 0),
            "brands": res.get("brands", {}),
            "ocr_labels": res.get("ocr_labels", [])
        }
        
        st.download_button(
            label="⬇️  Download Metrics JSON",
            data=json.dumps(clean, indent=4),
            file_name=f"metrics_{os.path.splitext(selected_file_name)[0]}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_metrics_top"
        )
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🖼️ Annotated Image", "📊 Business Metrics", "🏷️ OCR Labels", "⚙️ Raw JSON Output", "📥 Download Report"])

        with tab1:
            st.subheader("Annotated Shelf Composite Layout")
            ann_path = res.get("annotated_image") or ""
            if ann_path and os.path.exists(ann_path):
                st.image(ann_path, use_container_width=True, caption=f"Pipeline output — {selected_file_name}")

        with tab2:
            st.subheader("Business Metrics & Planogram Integrity")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-container"><div class="metric-value">{res.get("total_products") or 0}</div><div class="metric-label">Total Products</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-container"><div class="metric-value">{len(res.get("brands") or {})}</div><div class="metric-label">Unique Brands</div></div>', unsafe_allow_html=True)
            
            osa_status = res.get("on_shelf_availability") or "UNKNOWN"
            osa_c = "#10B981" if osa_status == "IN_STOCK" else "#EF4444"
            c3.markdown(f'<div class="metric-container"><div class="metric-value" style="color:{osa_c}">{osa_status}</div><div class="metric-label">OSA Status</div></div>', unsafe_allow_html=True)
            
            plano = res.get("planogram") or {"score": 0.0, "observation": "N/A"}
            sc = plano.get("score") or 0.0
            sc_c = "#10B981" if sc >= 0.7 else "#F59E0B" if sc >= 0.5 else "#EF4444"
            c4.markdown(f'<div class="metric-container"><div class="metric-value" style="color:{sc_c}">{sc:.2f}</div><div class="metric-label">Planogram Health</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"📋 **Planogram Observation:** {plano.get('observation') or 'N/A'}")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("#### Category Distribution")
                if res.get("categories"):
                    st.bar_chart(pd.DataFrame(list(res["categories"].items()), columns=["Category", "Count"]).set_index("Category"))
            with col_b:
                st.markdown("#### Brand Distribution")
                if res.get("brands"):
                    st.bar_chart(pd.DataFrame(list(res["brands"].items()), columns=["Brand", "Count"]).set_index("Brand"))
            with col_c:
                st.markdown("#### Share of Shelf Space (%)")
                if res.get("shelf_space_percent"):
                    st.bar_chart(pd.DataFrame(list(res["shelf_space_percent"].items()), columns=["Brand", "Share (%)"]).set_index("Brand"))

        with tab3:
            st.subheader("Extracted Price & Promotional Labels")
            labels = res.get("ocr_labels", [])
            if labels:
                st.markdown("".join(f'<span class="ocr-tag">🏷️ {tag}</span>' for tag in labels), unsafe_allow_html=True)
            else:
                st.info("No pricing tags matching criteria were isolated.")

        with tab4:
            st.subheader("Raw API Response Payload")
            st.json(clean)

        with tab5:
            st.subheader("Export Audit Reports")
            d1, d2 = st.columns(2)
            with d1:
                ann_path = res.get("annotated_image") or ""
                if ann_path and os.path.exists(ann_path):
                    with open(ann_path, "rb") as fh:
                        st.download_button(label="⬇️  Download Annotated Image", data=fh.read(), file_name=f"annotated_{selected_file_name}", mime="image/jpeg", use_container_width=True, key="dl_annotated_tab")
            with d2:
                st.download_button(label="⬇️  Download Metrics JSON", data=json.dumps(clean, indent=4), file_name=f"metrics_{os.path.splitext(selected_file_name)[0]}.json", mime="application/json", use_container_width=True, key="dl_metrics_tab")
