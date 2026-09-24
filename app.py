
import os
import random
import json

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Spotify Mood Finder",
    page_icon="🎵",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

CREATOR_NAME = "Jihan Timmy Nisrina"
DATASET_LINK = "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"


# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    cluster_profile = pd.read_csv(
        os.path.join(MODEL_DIR, "cluster_profile.csv"),
        index_col="cluster",
    )
    cluster_songs = pd.read_csv(
        os.path.join(MODEL_DIR, "cluster_songs.csv")
    )

    return scaler, model, feature_cols, cluster_profile, cluster_songs


scaler, model, feature_cols, cluster_profile, cluster_songs = load_artifacts()


# ============================================================
# MOOD DATA
# ============================================================
CLUSTER_INFO = {
    0: {
        "name": "Upbeat & Happy",
        "emoji": "💃",
        "color": "#FF6B6B",
        "desc": "Ceria, ritmis, dan gampang bikin badan ikut gerak.",
    },
    1: {
        "name": "Calm & Ambient",
        "emoji": "🌙",
        "color": "#4D96FF",
        "desc": "Tenang dan minim distraksi. Cocok buat santai atau tidur.",
    },
    2: {
        "name": "Acoustic & Mellow",
        "emoji": "🎻",
        "color": "#B983FF",
        "desc": "Hangat, lembut, dan terasa natural.",
    },
    3: {
        "name": "Electronic & Energetic",
        "emoji": "🎛️",
        "color": "#1DB954",
        "desc": "Nuansa elektronik dengan energi yang tetap terasa.",
    },
    4: {
        "name": "High-Energy & Intense",
        "emoji": "🔥",
        "color": "#FF4757",
        "desc": "Kuat, keras, dan intens. Cocok buat olahraga atau aktivitas berat.",
    },
}

PRESETS = [
    ("💃", "Joget / Pesta", 0, "Ceria dan ritmis"),
    ("🌙", "Tidur / Santai", 1, "Tenang dan minim distraksi"),
    ("🎻", "Akustik / Mellow", 2, "Hangat dan lembut"),
    ("🎛️", "Fokus Kerja", 3, "Elektronik dan tetap berenergi"),
    ("🔥", "Olahraga", 4, "Keras dan intens"),
    ("🎲", "Kejutkan Aku", "random", "Biar model yang memilih"),
]

DEFAULT_VALUES = {
    "danceability": 0.5,
    "energy": 0.5,
    "acousticness": 0.5,
    "valence": 0.5,
    "loudness": -10.0,
    "instrumentalness": 0.1,
}

SLIDER_META = {
    "danceability": {
        "label": "🕺 Danceability",
        "help": "Seberapa enak lagu ini diikuti untuk joget. Kiri = kurang danceable, kanan = lebih danceable.",
    },
    "energy": {
        "label": "⚡ Energy",
        "help": "Seberapa kuat dan aktif karakter lagunya. Kiri = lebih lembut, kanan = lebih energik.",
    },
    "valence": {
        "label": "😊 Mood",
        "help": "Nuansa emosional lagu. Kiri = lebih mellow, kanan = lebih ceria.",
    },
    "acousticness": {
        "label": "🎸 Acousticness",
        "help": "Seberapa kuat karakter akustiknya. Kiri = lebih elektronik, kanan = lebih akustik.",
    },
    "loudness": {
        "label": "🔊 Loudness",
        "help": "Tingkat keras suara rata-rata. Nilai mendekati 0 dB berarti secara umum lebih keras.",
    },
    "instrumentalness": {
        "label": "🎹 Instrumentalness",
        "help": "Seberapa mungkin lagu minim vokal. Kiri = dominan vokal, kanan = lebih instrumental.",
    },
}


# ============================================================
# SESSION STATE
# ============================================================
for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "predicted_cluster" not in st.session_state:
    st.session_state.predicted_cluster = None

if "song_seed" not in st.session_state:
    st.session_state.song_seed = 0

if "show_custom" not in st.session_state:
    st.session_state.show_custom = False


# ============================================================
# CALLBACKS
# ============================================================
def run_prediction():
    values = [[st.session_state[feature] for feature in feature_cols]]
    input_df = pd.DataFrame(values, columns=feature_cols)

    input_scaled = scaler.transform(input_df)
    st.session_state.predicted_cluster = int(model.predict(input_scaled)[0])

    st.session_state.song_seed = 0
    st.session_state.show_result = True


def reset_form():
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value

    st.session_state.show_result = False
    st.session_state.predicted_cluster = None
    st.session_state.song_seed = 0


def apply_preset(cluster_id):
    if cluster_id == "random":
        st.session_state["danceability"] = round(random.uniform(0.1, 0.95), 2)
        st.session_state["energy"] = round(random.uniform(0.1, 0.95), 2)
        st.session_state["acousticness"] = round(random.uniform(0.0, 0.95), 2)
        st.session_state["valence"] = round(random.uniform(0.1, 0.95), 2)
        st.session_state["loudness"] = round(random.uniform(-25.0, -3.0), 1)
        st.session_state["instrumentalness"] = round(random.uniform(0.0, 0.9), 2)
    else:
        for feature in feature_cols:
            st.session_state[feature] = float(
                cluster_profile.loc[cluster_id, feature]
            )

    run_prediction()


def shuffle_songs():
    st.session_state.song_seed += 1


def open_custom_mode():
    st.session_state.show_custom = True


def close_custom_mode():
    st.session_state.show_custom = False


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&display=swap');

    h1, h2, h3, h4 {
        font-family: 'Sora', sans-serif;
        letter-spacing: -0.02em;
    }

    .hero-kicker {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 16px;
        line-height: 1.55;
        opacity: 0.72;
        max-width: 720px;
        margin-bottom: 8px;
    }

    .section-kicker {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.5;
        margin-bottom: 2px;
    }

    .preset-card {
        padding: 2px 2px 8px 2px;
        min-height: 0;
    }

    .preset-icon {
        font-size: 25px;
        line-height: 1;
        margin-bottom: 7px;
    }

    .preset-title {
        font-family: 'Sora', sans-serif;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .preset-desc {
        font-size: 11.5px;
        line-height: 1.4;
        opacity: 0.62;
        min-height: 32px;
    }

    .custom-note {
        font-size: 13px;
        opacity: 0.65;
        margin: 2px 0 10px 0;
    }

    .sidebar-brand {
        font-family: 'Sora', sans-serif;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }

    .sidebar-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.5;
        margin-bottom: 5px;
    }

    .sidebar-card {
        padding: 2px 0;
    }

    .sidebar-divider {
        margin: 14px 0;
    }

    .tech-chip {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px 3px 2px 0;
        border-radius: 999px;
        background: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.12);
        font-size: 11px;
    }

    .mood-card {
        padding: 24px;
        border-radius: 18px;
        margin: 10px 0 22px 0;
    }

    .mood-emoji {
        font-size: 40px;
        line-height: 1;
        margin-bottom: 10px;
    }

    .mood-title {
        font-family: 'Sora', sans-serif;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .mood-desc {
        font-size: 15px;
        line-height: 1.55;
        opacity: 0.80;
    }

    .song-meta {
        opacity: 0.60;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .sidebar-small {
        font-size: 12px;
        line-height: 1.55;
        opacity: 0.70;
    }

    div[data-testid="stButton"] > button {
        border-radius: 11px;
        min-height: 40px;
        font-weight: 600;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🎵 Spotify Mood Finder</div>',
        unsafe_allow_html=True
    )
    st.caption("Temukan vibe lagu dari karakter audio.")

    st.markdown(
        '<div class="sidebar-label">Project</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '''
        <div class="sidebar-card sidebar-small">
        Aplikasi rekomendasi mood lagu berbasis <b>K-Means clustering</b>.
        Pengguna cukup memilih vibe atau mengatur karakter lagu secara manual.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown("")
    
    st.markdown(
        '<div class="sidebar-label">Model</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '''
        <div class="sidebar-card sidebar-small">
        <b>Unsupervised Learning</b><br>
        6 audio features → 5 mood clusters<br>
        Silhouette Score ≈ 0.27
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown("")
    
    st.markdown(
        '''
        <span class="tech-chip">Python</span>
        <span class="tech-chip">Streamlit</span>
        <span class="tech-chip">K-Means</span>
        <span class="tech-chip">Spotify Audio Features</span>
        ''',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**Dataset**")
    st.caption("Spotify Tracks Dataset")
    st.caption("Source: Maharshi Pandya, Kaggle (2022)")
    st.markdown(f"[↗ Lihat sumber dataset]({DATASET_LINK})")

    st.divider()

    st.caption(f"Dibuat oleh {CREATOR_NAME}")

# ============================================================
# HERO
# ============================================================
st.markdown(
    '<div class="hero-kicker">Spotify audio clustering</div>',
    unsafe_allow_html=True,
)
st.title("🎵 Spotify Mood Finder")
st.markdown(
    """
    <div class="hero-subtitle">
    Pilih suasana yang kamu cari. Model akan mencocokkan karakter audio lagu dengan mood yang paling dekat.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# MAIN INPUT
# ============================================================
if not st.session_state.show_custom:
    st.markdown('<div class="section-kicker">Pilih cepat</div>', unsafe_allow_html=True)
    st.markdown("### ✨ Lagi cari vibe yang mana?")
    st.caption("Pilih mood yang paling dekat dengan suasana yang kamu mau.")

    for start in range(0, len(PRESETS), 3):
        row = PRESETS[start:start + 3]
        cols = st.columns(3, gap="medium")

        for col, (icon, title, cluster_id, description) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'''
                        <div class="preset-card">
                            <div class="preset-icon">{icon}</div>
                            <div class="preset-title">{title}</div>
                            <div class="preset-desc">{description}</div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

                    st.button(
                        "Pilih vibe",
                        key=f"preset_{cluster_id}",
                        use_container_width=True,
                        on_click=apply_preset,
                        args=(cluster_id,),
                    )

    st.markdown("")
    st.button(
        "🎛️ Atur vibe sendiri",
        use_container_width=True,
        on_click=open_custom_mode,
    )

else:
    st.markdown("### 🎛️ Racik mood sendiri")
    st.markdown(
        '<div class="custom-note">Mau mengatur musik sesuai keinginanmu? Sesuaikan karakter audio melalui slider di bawah, lalu model akan menentukan mood yang paling sesuai.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.slider(
            SLIDER_META["danceability"]["label"],
            0.0,
            1.0,
            key="danceability",
            step=0.01,
            help=SLIDER_META["danceability"]["help"],
        )

        st.slider(
            SLIDER_META["energy"]["label"],
            0.0,
            1.0,
            key="energy",
            step=0.01,
            help=SLIDER_META["energy"]["help"],
        )

        st.slider(
            SLIDER_META["valence"]["label"],
            0.0,
            1.0,
            key="valence",
            step=0.01,
            help=SLIDER_META["valence"]["help"],
        )

        st.slider(
            SLIDER_META["acousticness"]["label"],
            0.0,
            1.0,
            key="acousticness",
            step=0.01,
            help=SLIDER_META["acousticness"]["help"],
        )

        st.slider(
            SLIDER_META["loudness"]["label"],
            -60.0,
            0.0,
            key="loudness",
            step=0.5,
            help=SLIDER_META["loudness"]["help"],
        )

        st.slider(
            SLIDER_META["instrumentalness"]["label"],
            0.0,
            1.0,
            key="instrumentalness",
            step=0.01,
            help=SLIDER_META["instrumentalness"]["help"],
        )

        action1, action2 = st.columns([2, 1])

        with action1:
            st.button(
                "🔍 Temukan Mood",
                type="primary",
                use_container_width=True,
                on_click=run_prediction,
            )

        with action2:
            st.button(
                "↺ Reset",
                use_container_width=True,
                on_click=reset_form,
            )

        st.button(
            "← Kembali ke pilihan vibe",
            use_container_width=True,
            on_click=close_custom_mode,
        )

    with right:
        with st.container(border=True):
            st.markdown("#### Preview")

            preview_specs = [
                ("danceability", "🕺 Dance", 0.0, 1.0, "#1DB954"),
                ("energy", "⚡ Energy", 0.0, 1.0, "#FF6B6B"),
                ("valence", "😊 Mood", 0.0, 1.0, "#FFC048"),
                ("acousticness", "🎸 Acoustic", 0.0, 1.0, "#B983FF"),
                ("loudness", "🔊 Loudness", -60.0, 0.0, "#4D96FF"),
                ("instrumentalness", "🎹 Instrumental", 0.0, 1.0, "#1DD3B0"),
            ]

            for key, label, lo, hi, color in preview_specs:
                value = st.session_state[key]
                percentage = (value - lo) / (hi - lo) * 100

                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:9px;margin:11px 0;">
                        <div style="width:105px;font-size:12px;opacity:0.72;">{label}</div>
                        <div style="flex:1;height:8px;border-radius:20px;background:rgba(128,128,128,.14);overflow:hidden;">
                            <div style="width:{percentage}%;height:100%;border-radius:20px;background:{color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption("Preview karakter lagu. Hasil cluster muncul setelah kamu menekan tombol di kiri.")

        with st.expander("Apa arti slider ini?"):
            st.markdown(
                """
                **Danceability** → seberapa cocok buat joget  
                **Energy** → seberapa lembut atau intens  
                **Mood** → seberapa mellow atau ceria  
                **Acousticness** → elektronik atau akustik  
                **Loudness** → seberapa keras  
                **Instrumentalness** → dominan vokal atau instrumental
                """
            )


# ============================================================
# RESULT
# ============================================================
st.divider()

if not st.session_state.show_result:
    st.caption("Pilih salah satu vibe atau racik mood sendiri untuk melihat hasilnya.")

else:
    predicted_cluster = st.session_state.predicted_cluster

    info = CLUSTER_INFO.get(
        predicted_cluster,
        {
            "name": f"Cluster {predicted_cluster}",
            "emoji": "🎵",
            "color": "#888888",
            "desc": "",
        },
    )

    st.markdown("### 🎯 Mood kamu")

    st.markdown(
        f"""
        <div class="mood-card"
             style="background-color:{info['color']}18;
                    border:1px solid {info['color']}40;
                    border-left:5px solid {info['color']};">
            <div class="mood-emoji">{info['emoji']}</div>
            <div class="mood-title">{info['name']}</div>
            <div class="mood-desc">{info['desc']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    title_col, action_col = st.columns([4, 1])

    with title_col:
        st.markdown("### 🎧 Contoh lagu")
        st.caption("Beberapa lagu yang berada di cluster mood yang sama.")

    with action_col:
        st.button(
            "🔄 Lagu lain",
            use_container_width=True,
            on_click=shuffle_songs,
        )

    all_songs_in_cluster = cluster_songs[
        cluster_songs["cluster"] == predicted_cluster
    ]

    if all_songs_in_cluster.empty:
        st.info("Belum ada contoh lagu untuk mood ini.")
    else:
        n_show = min(6, len(all_songs_in_cluster))

        songs = all_songs_in_cluster.sample(
            n=n_show,
            random_state=(
                st.session_state.song_seed
                + predicted_cluster * 1000
            ),
        )

        song_cols = st.columns(2, gap="medium")

        for i, (_, row) in enumerate(songs.iterrows()):
            with song_cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{row['track_name']}**")
                    st.caption(f"oleh {row['artists']}")

                    st.markdown(
                        f"""
                        <div class="song-meta">
                        {row['track_genre']} · Popularity {row['popularity']}/100
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    components.iframe(
                        f"https://open.spotify.com/embed/track/{row['track_id']}",
                        height=80,
                    )

        st.caption(
            f"Menampilkan {n_show} dari {len(all_songs_in_cluster)} lagu untuk mood ini."
        )

    with st.expander("📊 Detail teknis"):
        st.write(
            "Rata-rata nilai audio feature untuk cluster ini "
            "(skala asli, sebelum normalisasi):"
        )

        technical_df = cluster_profile.loc[
            [predicted_cluster], feature_cols
        ]

        st.dataframe(
            technical_df,
            use_container_width=True,
        )
