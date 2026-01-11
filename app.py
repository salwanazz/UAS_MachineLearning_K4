import streamlit as st
import joblib
import random

# ===============================
# LOAD MODEL
# ===============================
models = joblib.load("models/models.pkl")
tfidf = joblib.load("models/tfidf.pkl")
model_lr = models["logistic"]

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Sistem Deteksi Spam Email",
    page_icon="📧",
    layout="wide"
)

# ===============================
# SESSION STATE
# ===============================
if "show_effect" not in st.session_state:
    st.session_state.show_effect = False

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background:
        linear-gradient(180deg, #020617 0%, #020617 40%, #020617 100%),
        radial-gradient(circle at 20% 30%, rgba(99,102,241,0.15), transparent 40%),
        radial-gradient(circle at 80% 20%, rgba(56,189,248,0.12), transparent 45%),
        radial-gradient(circle at 50% 80%, rgba(167,139,250,0.1), transparent 50%);
    background-blend-mode: screen;
    overflow: hidden;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ===== HEADER ===== */
header {
    background: linear-gradient(
        180deg,
        rgba(251,207,232,0.25),
        rgba(221,214,254,0.15)
    ) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.15);
}

header * {
    color: #fdf4ff !important;
}


/* ===== TEXT ===== */
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: white;
    line-height: 1.2;
}

.main-desc {
    margin-top: 18px;
    font-size: 16px;
    color: #e5e7eb;
    max-width: 520px;
}

.input-label {
    font-size: 16px;
    font-weight: 600;
    color: white;
    margin-bottom: 10px;
}

/* ===== TEXTAREA ===== */
textarea {
    background: rgba(255,255,255,0.07) !important;
    color: white !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    padding: 16px !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(135deg, #020617, #1e293b);
    color: #e0e7ff;
    border-radius: 16px;
    padding: 0.8rem;
    font-weight: 600;
    width: 100%;
    border: 1px solid rgba(255,255,255,0.25);
    animation: neonPulse 2s infinite alternate;
}

@keyframes neonPulse {
    from { box-shadow: 0 0 12px rgba(99,102,241,0.3); }
    to { box-shadow: 0 0 30px rgba(99,102,241,0.9); }
}

/* ===== RESULT ===== */
.result-box {
    margin-top: 22px;
    padding: 1.3rem;
    border-radius: 18px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    animation: fadeUp 0.5s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-spam {
    background: rgba(239,68,68,0.2);
    color: #fee2e2;
    border: 1px solid rgba(239,68,68,0.7);
    animation: glowRed 1.3s infinite alternate;
}

@keyframes glowRed {
    from { box-shadow: 0 0 14px rgba(239,68,68,0.4); }
    to { box-shadow: 0 0 40px rgba(239,68,68,1); }
}

.result-ham {
    background: rgba(34,197,94,0.18);
    color: #dcfce7;
    border: 1px solid rgba(34,197,94,0.6);
    animation: glowGreen 1.5s infinite alternate;
}

@keyframes glowGreen {
    from {
        box-shadow:
            0 0 10px rgba(34,197,94,0.4),
            0 0 20px rgba(56,189,248,0.25);
    }
    to {
        box-shadow:
            0 0 26px rgba(34,197,94,0.9),
            0 0 45px rgba(56,189,248,0.6);
    }
}


/* ===== METEOR ===== */
.meteor {
    position: fixed;
    top: -200px;
    left: var(--x);
    width: 4px;
    height: 280px;
    background: linear-gradient(
        180deg,
        rgba(255,255,255,1),
        rgba(96,165,250,0.9),
        transparent
    );
    filter: blur(0.5px);
    animation: meteorFall 1.8s ease-in forwards;
    opacity: 0.95;
    z-index: 0;
}

@keyframes meteorFall {
    0% {
        transform: translate(0,0) rotate(45deg);
        opacity: 1;
    }
    100% {
        transform: translate(-600px,1200px) rotate(45deg);
        opacity: 0;
    }
}

/* ===== FOOTER ===== */
.footer {
    margin-top: 80px;
    text-align: center;
    font-size: 14px;
    color: #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# LAYOUT
# ===============================
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown('<div class="main-title">Kemana emailmu akan pergi?<br>Deteksi spam secara cerdas!</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="main-desc">
    Sistem ini digunakan untuk mendeteksi email <b>Spam</b> dan 
    <b>Non-Spam (Ham)</b> menggunakan pendekatan <b>Machine Learning</b>
    dengan dukungan teknik <i>Natural Language Processing </i>
    pada tahap pra-pemrosesan teks. Model telah melalui proses pelatihan
    dan diimplementasikan pada tahap <i>deployment</i> untuk melakukan
    prediksi secara otomatis.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-label">Masukkan isi pesan email 👇</div>', unsafe_allow_html=True)

    email_text = st.text_area(
        "Pesan Email",
        placeholder="Contoh: Anda memenangkan hadiah besar! Klik di sini..."
    )

    if email_text != st.session_state.last_text:
        st.session_state.show_effect = False
        st.session_state.last_text = email_text

    if st.button("Deteksi Sekarang ➜"):
        if not email_text.strip():
            st.session_state.show_effect = False
            st.warning("⚠️ Silakan masukkan isi pesan email terlebih dahulu.")
        else:
            st.session_state.show_effect = True
            vector = tfidf.transform([email_text])
            proba = model_lr.predict_proba(vector)[0]
            pred = model_lr.predict(vector)[0]

            spam_prob = proba[1] * 100
            ham_prob = proba[0] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="result-box result-spam">
                🚨 <b>HASIL: SPAM</b><br><br>
                Spam: {spam_prob:.2f}%<br>
                Non-Spam: {ham_prob:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-ham">
                ✅ <b>HASIL: NON-SPAM (HAM)</b><br><br>
                Spam: {spam_prob:.2f}%<br>
                Non-Spam: {ham_prob:.2f}%
                </div>
                """, unsafe_allow_html=True)


# ===============================
# METEOR LOOP
# ===============================
if st.session_state.show_effect:
    for _ in range(8):  # diperbanyak
        x = random.randint(5, 95)
        st.markdown(
            f'<div class="meteor" style="--x:{x}%"></div>',
            unsafe_allow_html=True
        )

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
<b>Kelompok 4 – Machine Learning</b><br>
Abdul Fatah (10222182) •
Salwa Nurazizah (10222154) •
Anisa (10222134) •
Tiara Kurniawati (10222155) •
Wina Apriliani Rahayu (10222111)<br><br>
© 2026 | Sistem Deteksi Spam Email
</div>
""", unsafe_allow_html=True)
