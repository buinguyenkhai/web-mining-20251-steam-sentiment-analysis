import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models import (
    load_fasttext_model, load_baseline_model, load_bigru_model, 
    load_textcnn_model, get_device
)
from app.data_utils import (
    load_test_data, load_train_data, create_and_fit_encoders
)

st.set_page_config(
    page_title="Steam Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Game card styling */
    .game-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #0f3460;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Positive/Negative labels */
    .label-positive {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .label-negative {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Genre badge */
    .genre-badge {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #333;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Stats display */
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00cec9;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00cec9 0%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading FastText embeddings... (this may take a minute)")
def load_fasttext():
    """Load FastText model (cached at app startup)."""
    model_path = PROJECT_ROOT / 'models' / 'steam_fasttext.bin'
    if not model_path.exists():
        st.error(f"FastText model not found at {model_path}")
        return None
    return load_fasttext_model(model_path)


@st.cache_resource(show_spinner="Loading baseline model...")
def load_baseline():
    """Load TF-IDF + Naive Bayes baseline (cached)."""
    model_path = PROJECT_ROOT / 'models' / 'baseline.joblib'
    if not model_path.exists():
        st.error(f"Baseline model not found at {model_path}")
        return None
    return load_baseline_model(model_path)


@st.cache_resource(show_spinner="Loading TextCNN model...")
def load_textcnn():
    """Load TextCNN + Metadata model (cached)."""
    model_path = PROJECT_ROOT / 'models' / 'textcnn_metadata.pt'
    config_path = PROJECT_ROOT / 'results' / 'textcnn_metadata_optuna_best_params.json'
    if not model_path.exists():
        st.warning(f"TextCNN model not found at {model_path}")
        return None
    device = get_device()
    return load_textcnn_model(model_path, device, config_path)


@st.cache_resource(show_spinner="Loading BiGRU model...")
def load_bigru():
    """Load BiGRU + Attention + Metadata model (cached)."""
    model_path = PROJECT_ROOT / 'models' / 'bigru_attention_metadata.pt'
    config_path = PROJECT_ROOT / 'results' / 'bigru_metadata_optuna_best_params.json'
    if not model_path.exists():
        st.warning(f"BiGRU model not found at {model_path}")
        return None
    device = get_device()
    return load_bigru_model(model_path, device, config_path)


@st.cache_data(show_spinner="Loading test data...")
def load_data():
    """Load test data and create encoders (cached)."""
    data_dir = PROJECT_ROOT / 'processed_data'
    test_df = load_test_data(data_dir)
    train_df = load_train_data(data_dir)
    encoders = create_and_fit_encoders(train_df)
    return test_df, encoders

def init_session_state():
    """Initialize session state variables."""
    if 'selected_game' not in st.session_state:
        st.session_state.selected_game = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

def go_to_game(appid: int, name: str):
    """Navigate to game detail page."""
    st.session_state.selected_game = {'appid': appid, 'name': name}
    st.session_state.current_page = 'game_detail'

def go_home():
    """Navigate back to home page."""
    st.session_state.selected_game = None
    st.session_state.current_page = 'home'

def main():
    """Main application entry point."""
    init_session_state()
    
    # Load models and data
    ft_model = load_fasttext()
    baseline_model = load_baseline()
    textcnn_model = load_textcnn()
    bigru_model = load_bigru()
    test_df, encoders = load_data()
    
    # Store in session state for access in pages
    st.session_state.ft_model = ft_model
    st.session_state.baseline_model = baseline_model
    st.session_state.textcnn_model = textcnn_model
    st.session_state.bigru_model = bigru_model
    st.session_state.test_df = test_df
    st.session_state.encoders = encoders
    st.session_state.device = get_device()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        
        if st.button(" Home", use_container_width=True):
            go_home()
        
        st.divider()
        
        st.markdown("### Model Status")
        
        # Model availability status
        models_status = {
            "TF-IDF + NB": baseline_model is not None,
            "FT-TextCNN-Meta": textcnn_model is not None,
            "FT-BiGRU-Att-Meta": bigru_model is not None,
            "FastText Embeddings": ft_model is not None
        }
        
        for name, available in models_status.items():
            status = "✅" if available else "❌"
            st.markdown(f"{status} {name}")
        
        st.divider()
        
        st.markdown("### Dataset Info")
        if test_df is not None:
            st.markdown(f"**Reviews:** {len(test_df):,}")
            st.markdown(f"**Games:** {test_df['appid'].nunique()}")
            st.markdown(f"**Genres:** {test_df['primary_genre'].nunique()}")

    if st.session_state.current_page == 'home':
        from app.views.home import render_home_page
        render_home_page()
    elif st.session_state.current_page == 'game_detail':
        from app.views.game_detail import render_game_detail_page
        render_game_detail_page()


if __name__ == "__main__":
    main()
