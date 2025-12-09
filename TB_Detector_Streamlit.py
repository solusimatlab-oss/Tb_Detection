import streamlit as st
import sys
import subprocess
import os
import time
import streamlit.components.v1 as components

# ==========================================
# AUTO-INSTALLER
# ==========================================
def check_and_install_libs():
    required = {'seaborn': 'seaborn', 'sklearn': 'scikit-learn', 'pandas': 'pandas'}
    missing = []
    for lib_import, lib_install in required.items():
        try:
            __import__(lib_import)
        except ImportError:
            missing.append(lib_install)
    
    if missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Gagal install: {e}")
            st.stop()

try:
    import seaborn
    import sklearn
except ImportError:
    check_and_install_libs()

# ==========================================
# IMPORT LIBRARY
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datetime import datetime
from supabase import create_client, Client 

# ==========================================
# KONFIGURASI DATABASE (HYBRID: CLOUD + LOCAL)
# ==========================================
SUPABASE_URL = "https://knisufytmlpdlgajrlqm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtuaXN1Znl0bWxwZGxnYWpybHFtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDMxNTUwNywiZXhwIjoyMDc5ODkxNTA3fQ.tR4PL_vpbQMJUcKrCHKhnfnGfQGyjFMQ7CaUoXRDQVQ"

DB_MODE = "LOCAL" 
supabase_client = None
LOCAL_DB_FILE = "medical_records_local.csv"

try:
    if "MASUKKAN" not in SUPABASE_URL and len(SUPABASE_URL) > 5:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        DB_MODE = "CLOUD"
except Exception as e:
    print(f"Supabase init skipped: {e}")

def save_record(data_dict):
    if DB_MODE == "CLOUD" and supabase_client:
        try:
            supabase_client.table("medical_records").insert(data_dict).execute()
            return True, "Data successfully uploaded to Cloud Database."
        except Exception as e:
            return False, f"Cloud Error: {str(e)}"
    else:
        try:
            df_new = pd.DataFrame([data_dict])
            if 'created_at' not in df_new.columns:
                df_new['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if os.path.exists(LOCAL_DB_FILE):
                df_new.to_csv(LOCAL_DB_FILE, mode='a', header=False, index=False)
            else:
                df_new.to_csv(LOCAL_DB_FILE, mode='w', header=True, index=False)
            return True, "Data saved locally (Offline Mode)."
        except Exception as e:
            return False, f"Local Disk Error: {str(e)}"

def fetch_records():
    if DB_MODE == "CLOUD" and supabase_client:
        try:
            response = supabase_client.table("medical_records").select("*").order("created_at", desc=True).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            st.error(f"Failed to fetch cloud data: {e}")
            return pd.DataFrame()
    else:
        if os.path.exists(LOCAL_DB_FILE):
            try:
                df = pd.read_csv(LOCAL_DB_FILE)
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    df = df.sort_values(by='created_at', ascending=False)
                return df
            except Exception as e:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

# ==========================================
# KONFIGURASI MEDIS
# ==========================================
OPTIMAL_THRESHOLD = 0.10 

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="MedSys Pro V.X",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS "VIEW MAHAL" (PREMIUM UI)
# ==========================================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #f1f5f9; }
[data-testid="stSidebar"] { background-color: #0f172a; color: #f8fafc; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #f8fafc !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] div[data-baseweb="select"] > div { background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important; }
[data-testid="stRadio"] label { color: #f8fafc !important; font-weight: 500; }
.hero-header { background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); padding: 2rem; border-radius: 16px; color: white; box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4); margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center; }
.glass-card { background: white; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: transform 0.2s ease, box-shadow 0.2s ease; height: 100%; margin-bottom: 20px; }
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
.metric-box { text-align: center; padding: 1rem; border-radius: 12px; background: #f8fafc; border: 1px solid #e2e8f0; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; }
.metric-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2563eb; color: white; border: none; transition: all 0.2s; }
.stButton>button:hover { background-color: #1d4ed8; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# JAVASCRIPT COMPONENTS
# ==========================================
def js_sidebar_widget():
    components.html(
        """
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; color: #94a3b8; padding: 10px 0;">
            <div style="font-size: 12px; font-weight: 600; letter-spacing: 1px;">SYSTEM STATUS</div>
            <div style="display:flex; align-items:center; gap:10px; margin-top:5px;">
                <div style="width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow: 0 0 8px #22c55e;"></div>
                <div style="font-size: 14px; color: #e2e8f0;">ONLINE</div>
            </div>
            <div id="clock" style="font-size: 20px; font-weight: 700; color: white; margin-top: 10px;"></div>
            <div id="date" style="font-size: 11px; color: #64748b;"></div>
        </div>
        <script>
            function updateTime() {
                const now = new Date();
                document.getElementById('clock').innerHTML = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
                document.getElementById('date').innerHTML = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
            }
            setInterval(updateTime, 1000);
            updateTime();
        </script>
        """,
        height=120
    )

# ==========================================
# BACKEND LOGIC
# ==========================================
@st.cache_resource
def load_tb_model():
    model_path = 'tb_detection_model.h5'
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except:
            return None
    return None

def get_gradcam(model, img_array, layer_name=None):
    if layer_name is None:
        target_layer = None
        for layer in reversed(model.layers):
            if len(layer.output.shape) == 4 and 'global' not in layer.name.lower():
                target_layer = layer.name
                break
    else:
        target_layer = layer_name

    grad_model = Model([model.inputs], [model.get_layer(target_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0: max_val = 1e-10
    heatmap /= max_val
    return heatmap.numpy()

class StreamlitPlotCallback(Callback):
    def __init__(self, plot_placeholder):
        super().__init__()
        self.plot_placeholder = plot_placeholder
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    def on_epoch_end(self, epoch, logs=None):
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        df = pd.DataFrame(self.history)
        with self.plot_placeholder.container():
            st.markdown("<div class='glass-card'><h5>📈 Live Training Metrics</h5>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.line_chart(df[['accuracy', 'val_accuracy']], color=["#3b82f6", "#10b981"])
            c2.line_chart(df[['loss', 'val_loss']], color=["#ef4444", "#f59e0b"])
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# LOGIN PAGE
# ==========================================
def login_page():
    # Pusatkan form login dengan kolom
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div style='text-align: center; margin-top: 50px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        # Ikon Medis
        st.markdown("<i class='fa-solid fa-user-doctor fa-4x' style='color:#2563eb;'></i>", unsafe_allow_html=True)
        st.markdown("<h2 style='color:#0f172a;'>MedSys Pro Security</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;'>Authorized Personnel Only</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Access System", type="primary")
            
            if submitted:
                # CREDENTIALS SEDERHANA (HARDCODED)
                # Username: admin, Password: admin123
                if username == "admin" and password == "admin123":
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("Access Granted. Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")

def logout():
    st.session_state['logged_in'] = False
    st.rerun()

# ==========================================
# MAIN UI CONTROLLER
# ==========================================
def main():
    # Inisialisasi Session State
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Logika Tampilan: Login vs Dashboard
    if not st.session_state['logged_in']:
        login_page()
    else:
        dashboard()

# ==========================================
# DASHBOARD UTAMA (LOGIC LAMA DIPINDAHKAN KESINI)
# ==========================================
def dashboard():
    # --- SIDEBAR (NAVIGASI PRO) ---
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <i class="fa-solid fa-hospital-user fa-3x" style="color: #3b82f6;"></i>
                <h3 style="color: white; margin-top: 10px;">MedSys Pro</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.write(f"User: **{st.session_state.get('username', 'Doctor')}**")
        if st.button("Log Out", type="secondary"):
            logout()
            
        st.markdown("---")
        
        js_sidebar_widget()
        st.markdown("<br><p style='color:#64748b; font-size:12px; font-weight:700;'>MAIN MODULES</p>", unsafe_allow_html=True)
        app_mode = st.radio("Select Operation:", ["Radiology Diagnosis", "Patient Database", "Batch Evaluation", "Model Training"], label_visibility="collapsed")
        
        st.markdown("<br><p style='color:#64748b; font-size:12px; font-weight:700;'>PATIENT CONTEXT</p>", unsafe_allow_html=True)
        with st.form("patient_form"):
            p_id = st.text_input("Medical Record ID (MRN)", value="MRN-2023-8821")
            p_name = st.text_input("Patient Name")
            c1, c2 = st.columns(2)
            with c1: st.number_input("Age", 0, 100, 45)
            with c2: st.selectbox("Sex", ["M", "F"])
            st.form_submit_button("Update Context", type="primary")
        
        st.markdown("---")
        # Status Koneksi Database (Auto-Detect)
        if DB_MODE == "CLOUD":
            st.markdown('<div style="color:#4ade80; font-size:12px;"><i class="fa-solid fa-cloud"></i> CLOUD DATABASE CONNECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#f59e0b; font-size:12px;"><i class="fa-solid fa-database"></i> LOCAL DATABASE ACTIVE</div>', unsafe_allow_html=True)

    st.markdown(f"""
        <div class="hero-header">
            <div>
                <h1 style="margin:0; font-size: 2rem; font-weight: 700;">AI Tuberculosis Analyzer</h1>
                <p style="margin:0; opacity: 0.9; font-weight: 300;">Computer Aided Diagnosis (CADx) System • {DB_MODE} Mode</p>
            </div>
            <div style="text-align: right;">
                <i class="fa-solid fa-dna fa-3x" style="opacity: 0.3;"></i>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- DIAGNOSIS ---
    if app_mode == "Radiology Diagnosis":
        model = load_tb_model()
        if model is None:
            st.warning("⚠️ Neural Network Model not found. Please train model first.")
            st.stop()

        c_left, c_right = st.columns([1, 2], gap="medium")

        with c_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### <i class='fa-solid fa-upload'></i> Image Acquisition", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if uploaded_file:
                st.markdown("---")
                st.image(uploaded_file, caption="Input Preview", use_column_width=True)
                with st.expander("⚙️ Advanced Filters"):
                    alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4)
                    threshold = st.slider("Visual Threshold", 0.0, 1.0, OPTIMAL_THRESHOLD)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### <i class='fa-solid fa-microscope'></i> Diagnostic Analysis", unsafe_allow_html=True)
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_array = np.expand_dims(img_resized, axis=0) / 255.0
                
                with st.spinner("Analyzing..."):
                    prediction = model.predict(img_array)[0][0]
                    heatmap = get_gradcam(model, img_array)
                    time.sleep(0.5)

                is_tb = prediction > OPTIMAL_THRESHOLD
                confidence = prediction if is_tb else 1 - prediction
                
                result_str = "POSITIVE TB" if is_tb else "NORMAL"
                
                if is_tb:
                    banner = f"""<div style="background:#fee2e2; border-left:6px solid #991b1b; padding:20px; border-radius:8px;">
                        <h3 style="color:#991b1b; margin:0;">POSITIVE: TUBERCULOSIS DETECTED</h3>
                        <p style="margin:0; color:#991b1b;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                    </div>"""
                else:
                    banner = f"""<div style="background:#dcfce7; border-left:6px solid #166534; padding:20px; border-radius:8px;">
                        <h3 style="color:#166534; margin:0;">NEGATIVE: NORMAL LUNG</h3>
                        <p style="margin:0; color:#166534;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                    </div>"""
                st.markdown(banner, unsafe_allow_html=True)
                st.write("")

                heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
                heatmap_resized[heatmap_resized < threshold] = 0
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                superimposed = cv2.addWeighted(img_rgb, 1.0, heatmap_colored, alpha, 0)
                st.image(superimposed, caption="AI Heatmap Analysis", use_column_width=True)
                
                # FITUR SAVE TO DATABASE (Hybrid)
                st.markdown("---")
                if st.button("💾 Save Record to Database", type="secondary"):
                    with st.spinner("Saving data..."):
                        data = {
                            "patient_id": p_id,
                            "patient_name": p_name,
                            "diagnosis": result_str,
                            "confidence": float(confidence)
                        }
                        success, msg = save_record(data)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)

            else:
                st.info("Waiting for image input...")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- DATABASE VIEWER ---
    elif app_mode == "Patient Database":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"#### <i class='fa-solid fa-server'></i> Medical Records ({DB_MODE} Storage)", unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        df = fetch_records()
        
        if not df.empty:
            st.dataframe(
                df, 
                column_config={
                    "created_at": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, HH:mm"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")
                },
                use_container_width=True
            )
            
            # Statistik Cepat
            c1, c2 = st.columns(2)
            c1.metric("Total Records", len(df))
            if 'diagnosis' in df.columns:
                pos_count = len(df[df['diagnosis'] == 'POSITIVE TB'])
                c2.metric("Positive Cases", pos_count)
        else:
            st.info("Database is empty or file not found.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- BATCH EVALUATION ---
    elif app_mode == "Batch Evaluation":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### <i class='fa-solid fa-chart-line'></i> Performance Evaluation", unsafe_allow_html=True)
        test_path = st.text_input("Test Dataset Path", placeholder=r"C:\Medical_Data\Test_Set")

        if st.button("Run Evaluation", type="primary"):
            model = load_tb_model()
            if model and os.path.exists(test_path):
                normal_p = os.path.join(test_path, 'Normal')
                if not os.path.exists(normal_p): normal_p = os.path.join(test_path, 'normal')
                tb_p = os.path.join(test_path, 'TB')
                if not os.path.exists(tb_p): tb_p = os.path.join(test_path, 'tb')

                if os.path.exists(normal_p) and os.path.exists(tb_p):
                    try:
                        test_datagen = ImageDataGenerator(rescale=1./255)
                        test_gen = test_datagen.flow_from_directory(test_path, target_size=(224,224), batch_size=32, class_mode='binary', shuffle=False)
                        
                        y_pred_prob = model.predict(test_gen)
                        y_pred = (y_pred_prob > OPTIMAL_THRESHOLD).astype(int).reshape(-1)
                        y_true = test_gen.classes
                        
                        cm = confusion_matrix(y_true, y_pred)
                        tn, fp, fn, tp = cm.ravel()
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        sensitivity = tp / (tp + fn) if (tp + fn)>0 else 0
                        specificity = tn / (tn + fp) if (tn + fp)>0 else 0
                        precision = tp / (tp + fp) if (tp + fp)>0 else 0
                        f1 = 2*(precision*sensitivity)/(precision+sensitivity) if (precision+sensitivity)>0 else 0
                        
                        st.markdown(f"### 📊 Metrics (Threshold: {OPTIMAL_THRESHOLD})")
                        c1,c2,c3,c4,c5 = st.columns(5)
                        c1.markdown(f"<div class='metric-box'><div class='metric-value'>{accuracy:.1%}</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-box'><div class='metric-value'>{sensitivity:.1%}</div><div class='metric-label'>Sensitivity</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-box'><div class='metric-value'>{specificity:.1%}</div><div class='metric-label'>Specificity</div></div>", unsafe_allow_html=True)
                        c4.markdown(f"<div class='metric-box'><div class='metric-value'>{precision:.1%}</div><div class='metric-label'>Precision</div></div>", unsafe_allow_html=True)
                        c5.markdown(f"<div class='metric-box'><div class='metric-value'>{f1:.1%}</div><div class='metric-label'>F1-Score</div></div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        c_plot, c_detail = st.columns(2)
                        with c_plot:
                            fig, ax = plt.subplots(figsize=(6,4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'TB'], yticklabels=['Normal', 'TB'])
                            plt.ylabel('Actual'); plt.xlabel('Predicted')
                            st.pyplot(fig)
                        with c_detail:
                            st.write(pd.DataFrame(classification_report(y_true, y_pred, target_names=['Normal', 'TB'], output_dict=True)).transpose())
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Invalid folder structure (Normal/TB not found).")
            else:
                st.error("Check Path / Model.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TRAINING ---
    elif app_mode == "Model Training":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### <i class='fa-solid fa-brain'></i> Neural Network Training", unsafe_allow_html=True)
        data_path = st.text_input("Dataset Source Path", placeholder=r"C:\Medical_Data\Chest_Xray")
        
        c1, c2 = st.columns(2)
        epochs = c1.number_input("Epochs", 1, 100, 10)
        batch_size = c2.selectbox("Batch Size", [16, 32, 64], index=1)
        fine_tune = st.checkbox("Enable Advanced Fine-Tuning", value=True)
        
        if st.button("Start Training Protocol", type="primary"):
            if os.path.exists(data_path):
                plot_placeholder = st.empty()
                train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, horizontal_flip=True, validation_split=0.2)
                try:
                    train_gen = train_datagen.flow_from_directory(data_path, target_size=(224,224), batch_size=batch_size, class_mode='binary', subset='training')
                    val_gen = train_datagen.flow_from_directory(data_path, target_size=(224,224), batch_size=batch_size, class_mode='binary', subset='validation', shuffle=False)
                    
                    if train_gen.samples > 0:
                        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                        base_model.trainable = False 
                        x = base_model.output
                        x = GlobalAveragePooling2D()(x)
                        x = Dense(128, activation='relu')(x)
                        x = Dropout(0.4)(x)
                        predictions = Dense(1, activation='sigmoid')(x)
                        
                        new_model = Model(inputs=base_model.input, outputs=predictions)
                        new_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
                        
                        streamlit_cb = StreamlitPlotCallback(plot_placeholder)
                        
                        with st.spinner("Phase 1: Training Classification Head..."):
                            new_model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[streamlit_cb])
                        
                        if fine_tune:
                            with st.spinner("Phase 2: Fine-Tuning Deep Layers..."):
                                base_model.trainable = True
                                for layer in base_model.layers[:100]: layer.trainable = False
                                new_model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
                                ft_epochs = max(3, int(epochs/2))
                                new_model.fit(train_gen, epochs=ft_epochs, validation_data=val_gen, callbacks=[streamlit_cb])

                        new_model.save('tb_detection_model.h5')
                        st.success("Training Completed Successfully.")
                    else:
                        st.error("No images found.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Invalid path.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()