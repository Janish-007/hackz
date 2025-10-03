import streamlit as st
import requests
import io
from concurrent.futures import ThreadPoolExecutor

# API endpoints
AI_TAMPERED_URL = "https://hifi-app-47f8c5.ambitioussand-4fd0f12d.eastus.azurecontainerapps.io/detect"
AI_GENERATED_URL = "https://ai-detector-api.politeflower-7bb2893a.eastus.azurecontainerapps.io/predict"

# Set page configuration
st.set_page_config(page_title="AI Image Classifiers", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357abd;
        transform: translateY(-2px);
    }
    .stFileUploader {
        border: 2px dashed #4a90e2;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        max-width: 300px;
    }
    .result-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .auto-result-card {
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #4a90e2;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
    }
    .stSubheader {
        color: #34495e;
        font-weight: 600;
    }
    .stCaption {
        color: #7f8c8d;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    .verdict {
        font-size: 2.5em;
        font-weight: bold;
        margin: 20px 0;
    }
    .real {
        color: #27ae60;
    }
    .ai {
        color: #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)

# Main title and description
st.title("AI Image Classifier")
st.markdown("Upload an image to analyze whether it is **AI-tampered** or **AI-generated**.", unsafe_allow_html=True)

# Mode selection
st.markdown("### Analysis Mode")
analysis_mode = st.radio(
    "Choose analysis mode:",
    options=["Auto", "Manual"],
    horizontal=True,
    help="Auto: Get consolidated results. Manual: Select specific models to test."
)

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

# Clear results when a new file is uploaded
if uploaded_file:
    # Check if the uploaded file is different from the previous one
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        # Clear previous results
        if "results" in st.session_state:
            del st.session_state.results

if uploaded_file:
    # Layout: image on left, results on right (after classification)
    col_image, col_results = st.columns([1, 2], gap="medium")

    with col_image:
        st.image(uploaded_file, caption="Uploaded Image Preview", use_column_width=True)
        
        # Manual mode: model selection
        if analysis_mode == "Manual":
            st.markdown("#### Select Models")
            selected_models = st.radio(
                "Choose which model to use:",
                options=["AI-Tampered Detection", "AI-Generated Detection", "Both"],
                help="Select which analysis to perform"
            )
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                file_bytes = uploaded_file.read()

                def call_api(url, file_name, file_data, file_type):
                    try:
                        files = {"file": (file_name, io.BytesIO(file_data), file_type)}
                        response = requests.post(url, files=files, timeout=20)
                        response.raise_for_status()
                        return response.json()
                    except Exception as e:
                        return {"error": str(e)}

                # Determine which APIs to call based on mode
                if analysis_mode == "Auto" or (analysis_mode == "Manual" and selected_models in ["Both", "AI-Tampered Detection", "AI-Generated Detection"]):
                    # Run API calls in parallel
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        if analysis_mode == "Auto" or selected_models in ["Both", "AI-Tampered Detection"]:
                            future_tampered = executor.submit(call_api, AI_TAMPERED_URL, uploaded_file.name, file_bytes, uploaded_file.type)
                        else:
                            future_tampered = None
                            
                        if analysis_mode == "Auto" or selected_models in ["Both", "AI-Generated Detection"]:
                            future_generated = executor.submit(call_api, AI_GENERATED_URL, uploaded_file.name, file_bytes, uploaded_file.type)
                        else:
                            future_generated = None
                        
                        result_tampered = future_tampered.result() if future_tampered else None
                        result_generated = future_generated.result() if future_generated else None

                    # Store results in session state
                    st.session_state.results = {
                        "tampered": result_tampered, 
                        "generated": result_generated,
                        "mode": analysis_mode,
                        "selected_models": selected_models if analysis_mode == "Manual" else None
                    }

    with col_results:
        # Only show results if they exist in session state
        if "results" in st.session_state:
            if st.session_state.results["mode"] == "Auto":
                # AUTO MODE: Consolidated results
                st.markdown("### Consolidated Analysis")
                
                result_tampered = st.session_state.results["tampered"]
                result_generated = st.session_state.results["generated"]
                
                # Calculate consolidated verdict
                is_tampered = False
                is_generated = False
                confidence = 0.0
                reasons = []
                
                if result_tampered and "error" not in result_tampered:
                    if result_tampered.get("is_forged"):
                        is_tampered = True
                        confidence = max(confidence, result_tampered.get("probability", 0))
                        reasons.append("Image shows signs of tampering")
                
                if result_generated and "error" not in result_generated:
                    prediction = result_generated.get("final_prediction", "").upper()
                    
                    # Check multiple possible values for AI detection
                    if prediction in ["AI", "SYNTHETIC", "FAKE", "GENERATED"]:
                        is_generated = True
                        confidence = max(confidence, result_generated.get("p_synth", 0))
                        reasons.append("Image appears to be AI-generated")
                    
                    # Also check probability threshold as backup
                    p_synth = result_generated.get("p_synth", 0)
                    if p_synth > 0.5:  # If confidence is over 50%, consider it AI
                        is_generated = True
                        confidence = max(confidence, p_synth)
                        if "Image appears to be AI-generated" not in reasons:
                            reasons.append("Image appears to be AI-generated")
                
                # Display consolidated result
                with st.container():
                    st.markdown('<div class="auto-result-card">', unsafe_allow_html=True)
                    st.markdown("## Result:")
                    
                    if is_tampered or is_generated:
                        st.markdown('<div class="verdict ai">AI-TAMPERED / AI-GENERATED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict real">REAL IMAGE</div>', unsafe_allow_html=True)
                    
                    st.progress(min(max(confidence, 0.0), 1.0))
                    
                    
                    if reasons:
                        st.markdown("**Reasons:**")
                        for reason in reasons:
                            st.markdown(f"• {reason}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed breakdown in expander
                with st.expander("View Detailed Breakdown"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### AI-Tampered Detection")
                        if result_tampered and "error" not in result_tampered:
                            forged = result_tampered.get("is_forged")
                            prob = result_tampered.get("probability")
                            if forged is not None:
                                status = "Yes" if forged else "No"
                                st.markdown(f"**Forged:** {status}")
                            if prob is not None:
                                st.progress(min(max(prob, 0.0), 1.0))
                                
                            st.caption(f"Model: {result_tampered.get('model', 'N/A')}")
                        else:
                            st.error("Error in tampered detection")
                    
                    with col2:
                        st.markdown("#### AI-Generated Detection")
                        if result_generated and "error" not in result_generated:
                            prediction = result_generated.get("final_prediction", "N/A")
                            synth_prob = result_generated.get("p_synth")
                            st.markdown(f"**Prediction:** `{prediction}`")
                            if synth_prob is not None:
                                st.progress(min(max(synth_prob, 0.0), 1.0))
                            
                        else:
                            st.error("Error in generated detection")
            
            else:
                # MANUAL MODE: Individual results
                st.markdown("### Analysis Results")
                selected = st.session_state.results["selected_models"]
                
                if selected == "Both":
                    col1, col2 = st.columns(2, gap="small")
                    
                    # AI-Tampered Result
                    with col1:
                        st.subheader("AI-Tampered Detection")
                        with st.container():
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            result_tampered = st.session_state.results["tampered"]
                            if "error" in result_tampered:
                                st.error(f"Error: {result_tampered['error']}")
                            else:
                                forged = result_tampered.get("is_forged")
                                prob = result_tampered.get("probability")
                                if forged is not None:
                                    status = "Yes" if forged else "No"
                                    st.markdown(f"**Forged:** {status}", unsafe_allow_html=True)
                                if prob is not None:
                                    st.progress(min(max(prob, 0.0), 1.0))
                            
                                st.caption(f"Model: {result_tampered.get('model', 'N/A')}")
                                with st.expander("View Raw JSON"):
                                    st.json(result_tampered)
                            st.markdown('</div>', unsafe_allow_html=True)

                    # AI-Generated Result
                    with col2:
                        st.subheader("AI-Generated Detection")
                        with st.container():
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            result_generated = st.session_state.results["generated"]
                            if "error" in result_generated:
                                st.error(f"Error: {result_generated['error']}")
                            else:
                                prediction = result_generated.get("final_prediction", "N/A")
                                synth_prob = result_generated.get("p_synth")
                                st.markdown(f"**Prediction:** `{prediction}`", unsafe_allow_html=True)
                                if synth_prob is not None:
                                    st.progress(min(max(synth_prob, 0.0), 1.0))
                                
                                st.caption(f"AI by Model: {result_generated.get('ai_by_model', 'N/A')}")
                                st.caption(f"AI by EXIF: {result_generated.get('ai_by_exif', 'N/A')}")
                                st.caption(f"AI by C2PA: {result_generated.get('ai_by_c2pa', 'N/A')}")
                                with st.expander("View Raw JSON"):
                                    st.json(result_generated)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                elif selected == "AI-Tampered Detection":
                    st.subheader("AI-Tampered Detection")
                    with st.container():
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        result_tampered = st.session_state.results["tampered"]
                        if "error" in result_tampered:
                            st.error(f"Error: {result_tampered['error']}")
                        else:
                            forged = result_tampered.get("is_forged")
                            prob = result_tampered.get("probability")
                            if forged is not None:
                                status = "Yes" if forged else "No"
                                st.markdown(f"**Forged:** {status}", unsafe_allow_html=True)
                            if prob is not None:
                                st.progress(min(max(prob, 0.0), 1.0))
                            
                            st.caption(f"Model: {result_tampered.get('model', 'N/A')}")
                            with st.expander("View Raw JSON"):
                                st.json(result_tampered)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                elif selected == "AI-Generated Detection":
                    st.subheader("AI-Generated Detection")
                    with st.container():
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        result_generated = st.session_state.results["generated"]
                        if "error" in result_generated:
                            st.error(f"Error: {result_generated['error']}")
                        else:
                            prediction = result_generated.get("final_prediction", "N/A")
                            synth_prob = result_generated.get("p_synth")
                            st.markdown(f"**Prediction:** `{prediction}`", unsafe_allow_html=True)
                            if synth_prob is not None:
                                st.progress(min(max(synth_prob, 0.0), 1.0))
                            st.caption(f"AI by Model: {result_generated.get('ai_by_model', 'N/A')}")
                            st.caption(f"AI by EXIF: {result_generated.get('ai_by_exif', 'N/A')}")
                            st.caption(f"AI by C2PA: {result_generated.get('ai_by_c2pa', 'N/A')}")
                            with st.expander("View Raw JSON"):
                                st.json(result_generated)
                        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
