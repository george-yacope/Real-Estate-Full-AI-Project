import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
import re
import os
import fitz  # PyMuPDF
import plotly.express as px

# --- 1. Page Configuration & Custom Styling ---
st.set_page_config(page_title="Real Estate Machine", page_icon="🏠", layout="wide")

# Function to detect Arabic for RTL support
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

# Professional Dark Mode Styling for WhatsApp-like Chat & RTL
st.markdown("""
<style>
    .chat-wrapper {
        background-color: #0b141a; 
        padding: 15px;
        border-radius: 10px;
        height: 550px;
        overflow-y: auto;
        border: 1px solid #3b4a54;
        display: flex;
        flex-direction: column;
    }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        line-height: 1.5;
        font-size: 15px;
        color: #e9edef;
    }
    .user-bubble {
        background-color: #005c4b; 
        align-self: flex-end;
        border-bottom-right-radius: 2px;
    }
    .assistant-bubble {
        background-color: #202c33; 
        align-self: flex-start;
        border-bottom-left-radius: 2px;
    }
    .rtl { direction: rtl; text-align: right; }
    .ltr { direction: ltr; text-align: left; }
    
    /* Remove streamlit default gaps */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Data & Model Loading Logic ---
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('models/robust_scaler.pkl')
        classifier = joblib.load('models/classifier_market_segment.pkl')
        reg_models = {
            0: joblib.load('models/regressor_cluster_0_premium.pkl'),
            1: joblib.load('models/regressor_cluster_1_economic.pkl'),
            2: joblib.load('models/regressor_cluster_2_family.pkl')
        }
        features = joblib.load('models/features_list.pkl')
        with open('models/cluster_mapping.json', 'r') as f:
            cluster_map = json.load(f)
        
        # Load Reference Data for plotting
        ref_data = pd.read_csv('data\house_data_classification.csv')
        return scaler, classifier, reg_models, features, cluster_map, ref_data
    except Exception as e:
        st.error(f"Error loading system assets: {e}")
        return None, None, None, None, None, None

scaler, classifier, reg_models, features, cluster_map, ref_data = load_assets()

# --- 3. PDF Context Engine (From your latest code) ---
def read_pdf_text(pdf_path):
    if not os.path.exists(pdf_path):
        return None
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_chunks(text, max_words=200):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

@st.cache_data
def get_pdf_knowledge():
    pdf_text = read_pdf_text("data/smart_real_estate_report.pdf")
    if pdf_text:
        chunks = split_into_chunks(pdf_text, max_words=200)
        return "\n".join(chunks[:15]) # Inject first 15 chunks as primary context
    return "No local market report available."

internal_pdf_context = get_pdf_knowledge()

# --- 4. Sidebar & Navigation ---
st.title("🏠 Real Estate Machine - Washington State")
tab1, tab2 = st.tabs(["💎 Valuation Dashboard", "💬 AI Market Consultant"])

# --- TAB 1: Valuation & Strategy ---
with tab1:
    st.header("Property Inputs")
    with st.form("valuation_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1.0, step=0.5, value=2.0)
            sqft_living = st.number_input("Living Space (sqft)", min_value=300, value=2000)
            sqft_lot = st.number_input("Lot Size (sqft)", min_value=300, value=5000)
        with col2:
            floors = st.selectbox("Floors", [1, 1.5, 2, 2.5, 3, 3.5])
            waterfront = st.selectbox("Waterfront?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            view = st.slider("View Quality", 0, 4, 0)
            condition = st.slider("Condition", 1, 5, 3)
        with col3:
            sqft_above = st.number_input("Above Ground (sqft)", value=1500)
            sqft_basement = st.number_input("Basement (sqft)", value=500)
            house_age = st.number_input("House Age", min_value=0, value=20)
            city_freq = st.slider("City Demand Index", 0.0, 1.0, 0.1)
        
        submitted = st.form_submit_button("Generate Valuation Analysis")

    if submitted and classifier:
        input_df = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, 
                                waterfront, view, condition, sqft_above, 
                                sqft_basement, house_age, city_freq]], columns=features)
        
        # Logic Sequence
        scaled_x = scaler.transform(input_df)
        cluster_id = classifier.predict(scaled_x)[0]
        segment_name = cluster_map.get(str(cluster_id), "General Market")
        
        # Price Calculation
        pred_log = reg_models[cluster_id].predict(input_df)[0]
        final_price = np.expm1(pred_log)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.subheader("Identification")
            st.metric("Target Segment", segment_name)
            st.metric("Estimated Price", f"${final_price:,.2f}")
            st.success(f"Categorized under Cluster {cluster_id}")

        with res_col2:
            st.subheader("Market Position Analysis")
            # Show where this property sits compared to others in the same cluster
            cluster_data = ref_data[ref_data['cluster'] == cluster_id]['price']
            fig = px.histogram(cluster_data, nbins=50, title=f"Price Distribution for {segment_name}",
                               color_discrete_sequence=['#00a884'], labels={'value': 'Price ($)'})
            fig.add_vline(x=final_price, line_width=4, line_dash="dash", line_color="#ea5e24", 
                         annotation_text="Your Property", annotation_position="top right")
            fig.update_layout(showlegend=False, height=350, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: AI Consultant (Groq + PDF Context) ---
with tab2:
    st.header("WhatsApp-Style Smart Consultant")
    
    # Strictly applying your prompt logic
    system_prompt = f"""
    You are a Washington real estate sales consultant.
    KNOWLEDGE BASE (FROM SMART PDF):
    {internal_pdf_context}

        YOUR CORE MISSION:
    - Persuade customers by highlighting key assets: Location, Living Space, and Finishing Quality.
    - Demonstrate "Value for Money" using ACTUAL NUMBERS and COMPARISONS from the KNOWLEDGE BASE. Do not just say "value"; use specific prices and metrics.
    - If a property doesn't fit their budget, suggest specific alternatives from the data.
    - Be concise, direct, and avoid any repetitive phrases.

    STRICT RULES:
    1. Base all facts strictly on the KNOWLEDGE BASE provided. Use real prices and property IDs in your comparisons.
    2. LANGUAGE: Match the user's language perfectly. 
       - If they speak Arabic (Egyptian, Standard, or any dialect), respond in a warm, polite, and welcoming Egyptian Arabic (عامية مصرية).
       - If they speak English, respond in professional and inviting English.
    3. TONE: Always maintain a pleasant and helpful attitude. Never be rude or overly technical.
    4. TRADE-OFFS: Every piece of advice MUST include a clear comparison (Trade-off). Specifically suggest ways to save money, e.g., "We can reduce the budget by $X if we look at properties with smaller areas or in different neighborhoods from our data."
    5. SALES STRATEGY: Focus on "The Deal". Use phrases like "This is a catch because the price per sqft is only $X..." or "You are paying for the location premium here".
    6. DO NOT mention technical system errors like "No report available".
    7. FORMATTING & READABILITY: 
       - Ensure clear spacing between numbers, currencies ($), and text. 
       - NEVER let numbers or currency symbols stick to Arabic words (e.g., use '130,000 $ و' instead of '130,000$و'). 
       - Use proper punctuation to separate sentences.
    8. Length: Keep it short, high-impact, and professional.
    9. Ending: Always end with a specific choice or an inviting next step based on a numerical trade-off.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    # Rendering chat as a single HTML block to prevent "empty space" issues
    chat_html = '<div class="chat-wrapper">'
    for msg in st.session_state.messages:
        if msg["role"] == "system": continue
        
        b_type = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        direction = "rtl" if is_arabic(msg["content"]) else "ltr"
        chat_html += f'<div class="chat-bubble {b_type} {direction}">{msg["content"]}</div>'
    chat_html += '</div>'
    
    st.markdown(chat_html, unsafe_allow_html=True)

    if user_query := st.chat_input("Ask about market trade-offs..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        GROQ_API_KEY = "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" 
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": st.session_state.messages,
            "temperature": 0.5
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.error("AI Service connection error.")
        except Exception as e:

            st.error(f"Error: {e}")


