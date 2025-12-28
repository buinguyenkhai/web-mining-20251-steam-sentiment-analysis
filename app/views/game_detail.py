"""
Game Detail Page - Reviews and Classification

Displays game details, reviews, and allows model-based classification.
Uses table view with expandable reviews for quick comparison.
"""

import streamlit as st
import pandas as pd

from app.data_utils import (
    get_game_reviews, preprocess_for_fasttext, truncate_text
)
from app.models import (
    predict_baseline, predict_deep_model, get_device
)


def infer_length_tier(text: str) -> str:
    """Infer length tier based on word count."""
    if not text:
        return 'short'
    word_count = len(text.split())
    if word_count < 20:
        return 'short'
    elif word_count < 100:
        return 'medium'
    else:
        return 'long'


def render_game_detail_page():
    """Render the game detail page with reviews and classification."""
    
    # Get selected game
    selected_game = st.session_state.get('selected_game')
    
    if selected_game is None:
        st.warning("No game selected. Please go back to home.")
        if st.button("Go Home"):
            st.session_state.current_page = 'home'
            st.rerun()
        return
    
    appid = selected_game['appid']
    game_name = selected_game['name']
    
    # Get data
    test_df = st.session_state.get('test_df')
    encoders = st.session_state.get('encoders')
    
    if test_df is None or encoders is None:
        st.error("Data not loaded properly.")
        return
    
    # Get reviews for this game
    reviews_df = get_game_reviews(test_df, appid)
    
    if len(reviews_df) == 0:
        st.error(f"No reviews found for game: {game_name}")
        return
    
    # Back button
    if st.button("← Back to Games"):
        st.session_state.current_page = 'home'
        st.session_state.selected_game = None
        st.rerun()
    
    # Game Header
    render_game_header(game_name, reviews_df)
    
    st.divider()
    
    # Tabs for Classification and Custom Input
    tab1, tab2 = st.tabs(["Classify Reviews", "Custom Review"])
    
    with tab1:
        render_classification_section(reviews_df, encoders)
    
    with tab2:
        render_custom_review_section(encoders, reviews_df)


def render_game_header(game_name: str, reviews_df: pd.DataFrame):
    """Render game header with metadata."""
    
    total_reviews = len(reviews_df)
    positive_count = int(reviews_df['voted_up'].sum())
    negative_count = total_reviews - positive_count
    positive_ratio = positive_count / total_reviews if total_reviews > 0 else 0
    genre = reviews_df['primary_genre'].iloc[0]
    
    st.title(f"{game_name}")
    st.caption(f"Genre: **{genre}**")
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col2:
        st.metric("Positive", f"{positive_count:,}", delta=f"{positive_ratio*100:.1f}%")
    with col3:
        st.metric("Negative", f"{negative_count:,}")
    with col4:
        st.metric("Positive Ratio", f"{positive_ratio*100:.1f}%")


def render_classification_section(reviews_df: pd.DataFrame, encoders):
    """Render model selection and classification controls."""
    
    st.markdown("### Sentiment Classification")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            options=[
                "TF-IDF + Naive Bayes",
                "FT-TextCNN-Meta",
                "FT-BiGRU-Att-Meta",
                "Compare All Models"
            ],
            index=3,  # Default to compare all
            help="Choose a model to classify reviews"
        )
    
    with col2:
        num_reviews = st.slider(
            "Reviews to display",
            min_value=5,
            max_value=min(100, len(reviews_df)),
            value=min(20, len(reviews_df)),
            step=5
        )
    
    # Get subset of reviews
    subset_df = reviews_df.head(num_reviews).copy()
    
    # Classification button
    if st.button("Classify Reviews", type="primary", use_container_width=True):
        predictions = classify_reviews(subset_df, model_choice, encoders)
        if predictions is not None:
            display_results_with_expanders(subset_df, predictions, model_choice)
    else:
        # Show reviews without classification
        st.info("Click 'Classify Reviews' to run sentiment prediction on the reviews below.")
        display_reviews_simple(subset_df)


def classify_reviews(reviews_df: pd.DataFrame, model_choice: str, encoders) -> dict:
    """Run classification and return results."""
    
    # Get models from session state
    baseline_model = st.session_state.get('baseline_model')
    textcnn_model = st.session_state.get('textcnn_model')
    bigru_model = st.session_state.get('bigru_model')
    ft_model = st.session_state.get('ft_model')
    device = st.session_state.get('device', get_device())
    
    # Prepare data
    texts = reviews_df['processed_text'].fillna('').tolist()
    genre_ids = encoders.transform_genre(reviews_df['primary_genre'].values).tolist()
    playtime_ids = encoders.transform_playtime_tier(reviews_df['playtime_tier'].values).tolist()
    length_ids = encoders.transform_length_tier(reviews_df['length_tier'].values).tolist()
    
    predictions = {}
    
    with st.spinner("Classifying reviews..."):
        try:
            if model_choice in ["TF-IDF + Naive Bayes", "Compare All Models"]:
                if baseline_model is not None:
                    predictions['TF-IDF + NB'] = predict_baseline(texts, baseline_model)
                else:
                    st.warning("Baseline model not loaded.")
            
            if model_choice in ["FT-TextCNN-Meta", "Compare All Models"]:
                if textcnn_model is not None and ft_model is not None:
                    predictions['TextCNN'] = predict_deep_model(
                        texts, genre_ids, playtime_ids, length_ids,
                        textcnn_model, ft_model, device
                    )
                else:
                    st.warning("TextCNN model not loaded.")
            
            if model_choice in ["FT-BiGRU-Att-Meta", "Compare All Models"]:
                if bigru_model is not None and ft_model is not None:
                    predictions['BiGRU+Att'] = predict_deep_model(
                        texts, genre_ids, playtime_ids, length_ids,
                        bigru_model, ft_model, device
                    )
                else:
                    st.warning("BiGRU model not loaded.")
                    
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return None
    
    if not predictions:
        st.error("No models available for classification.")
        return None
    
    return predictions


def display_reviews_simple(reviews_df: pd.DataFrame):
    """Display reviews without predictions."""
    for i, (_, row) in enumerate(reviews_df.iterrows()):
        actual = "👍" if row['voted_up'] else "👎"
        with st.expander(f"{actual} {truncate_text(str(row['review_text']), 80)}"):
            st.write(row['review_text'])


def display_results_with_expanders(reviews_df: pd.DataFrame, predictions: dict, model_choice: str):
    """Display results with expandable reviews showing predictions at a glance."""
    
    st.markdown("### Classification Results")
    
    # Calculate accuracy for each model
    model_accuracies = {}
    for model_name, preds in predictions.items():
        correct = sum(1 for i, pred in enumerate(preds) 
                     if pred['prediction'] == bool(reviews_df.iloc[i]['voted_up']))
        model_accuracies[model_name] = correct / len(preds) * 100
    
    # Show accuracy metrics
    acc_cols = st.columns(len(predictions))
    for col, (model_name, acc) in zip(acc_cols, model_accuracies.items()):
        with col:
            st.metric(f"{model_name} Accuracy", f"{acc:.1f}%")
    
    st.divider()
    st.caption("👍=Positive, 👎=Negative, ✅=Correct, ❌=Wrong | Click to expand full review")
    
    # Display each review as an expander with summary in header
    for i, (_, row) in enumerate(reviews_df.iterrows()):
        actual_label = bool(row['voted_up'])
        actual_emoji = "👍" if actual_label else "👎"
        
        # Build prediction summary for header
        pred_summary = []
        for model_name, preds in predictions.items():
            if i < len(preds):
                pred = preds[i]
                pred_label = pred['prediction']
                confidence = pred['confidence']
                pred_emoji = "👍" if pred_label else "👎"
                correct_emoji = "✅" if pred_label == actual_label else "❌"
                # Short model name
                short_name = model_name.replace("TF-IDF + ", "").replace("+Att", "")
                pred_summary.append(f"{short_name}:{pred_emoji}{correct_emoji}{confidence:.0f}%")
        
        header = f"Actual:{actual_emoji} | {' | '.join(pred_summary)} | {truncate_text(str(row['review_text']), 50)}"
        
        with st.expander(header):
            # Full review text
            st.markdown("**Review:**")
            st.write(row['review_text'])
            
            st.divider()
            
            # Detailed predictions
            cols = st.columns(len(predictions))
            for col, (model_name, preds) in zip(cols, predictions.items()):
                with col:
                    if i < len(preds):
                        pred = preds[i]
                        pred_label = pred['prediction']
                        confidence = pred['confidence']
                        correct = pred_label == actual_label
                        
                        st.markdown(f"**{model_name}**")
                        st.markdown(f"{'👍 Positive' if pred_label else '👎 Negative'}")
                        st.markdown(f"{'✅ Correct' if correct else '❌ Wrong'}")
                        st.progress(confidence / 100, text=f"{confidence:.1f}%")


def render_custom_review_section(encoders, reviews_df: pd.DataFrame):
    """Render custom review classification input."""
    
    st.markdown("### Classify Your Own Review")
    
    # Get game's genre
    game_genre = reviews_df['primary_genre'].iloc[0] if len(reviews_df) > 0 else encoders.genre_classes[0]
    st.info(f"Genre: **{game_genre}** (based on current game)")
    
    # Text input
    custom_text = st.text_area(
        "Enter your review text",
        placeholder="Type or paste a review here to classify it...",
        height=100
    )
    
    # Playtime tier (user might want to specify this)
    playtime_options = encoders.playtime_tier_classes
    selected_playtime = st.selectbox(
        "Playtime Tier",
        options=playtime_options,
        index=1 if 'medium' in playtime_options else 0,
        help="How long have you played the game?"
    )
    
    # Length tier is auto-inferred (shown as info)
    if custom_text.strip():
        inferred_length = infer_length_tier(custom_text)
        st.caption(f"Review length tier: **{inferred_length}** (auto-detected based on word count)")
    else:
        inferred_length = 'medium'
    
    # Classify button
    if st.button("Classify Custom Review", type="primary"):
        if not custom_text.strip():
            st.warning("Please enter some review text.")
            return
        
        classify_custom_review(
            custom_text,
            game_genre,
            selected_playtime,
            inferred_length,
            encoders
        )


def classify_custom_review(
    text: str,
    genre: str,
    playtime_tier: str,
    length_tier: str,
    encoders
):
    """Classify a custom review input with all available models."""
    
    # Preprocess text
    processed_text = preprocess_for_fasttext(text)
    
    # Get models
    baseline_model = st.session_state.get('baseline_model')
    textcnn_model = st.session_state.get('textcnn_model')
    bigru_model = st.session_state.get('bigru_model')
    ft_model = st.session_state.get('ft_model')
    device = st.session_state.get('device', get_device())
    
    # Encode metadata
    genre_id = encoders.transform_genre([genre])[0]
    playtime_id = encoders.transform_playtime_tier([playtime_tier])[0]
    length_id = encoders.transform_length_tier([length_tier])[0]
    
    results = {}
    
    with st.spinner("Classifying..."):
        try:
            if baseline_model is not None:
                results["TF-IDF + NB"] = predict_baseline([processed_text], baseline_model)[0]
            
            if textcnn_model is not None and ft_model is not None:
                results["TextCNN"] = predict_deep_model(
                    [processed_text], [genre_id], [playtime_id], [length_id],
                    textcnn_model, ft_model, device
                )[0]
            
            if bigru_model is not None and ft_model is not None:
                results["BiGRU+Att"] = predict_deep_model(
                    [processed_text], [genre_id], [playtime_id], [length_id],
                    bigru_model, ft_model, device
                )[0]
                    
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return
    
    # Display results
    if not results:
        st.warning("No models available for classification.")
        return
    
    st.markdown("### Classification Results")
    
    cols = st.columns(len(results))
    
    for col, (model_name, pred) in zip(cols, results.items()):
        with col:
            pred_label = pred['prediction']
            confidence = pred['confidence']
            emoji = "👍" if pred_label else "👎"
            label_text = "Positive" if pred_label else "Negative"
            
            st.markdown(f"**{model_name}**")
            st.markdown(f"### {emoji} {label_text}")
            st.progress(confidence / 100, text=f"{confidence:.1f}% confidence")
