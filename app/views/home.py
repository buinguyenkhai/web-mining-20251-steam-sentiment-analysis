import streamlit as st
from app.data_utils import get_games_from_test_set, format_game_card_data


def render_home_page():
    """Render the home page with game list and filters."""
    
    # Header
    st.title("Steam Review Sentiment Analysis")
    st.markdown("""
    Analyze Steam game reviews using ML models: **TF-IDF + Naive Bayes**, **TextCNN**, and **BiGRU with Attention**
    """)
    
    # Get data
    test_df = st.session_state.get('test_df')
    
    if test_df is None:
        st.error("Test data not loaded. Please check the data path.")
        return
    
    # Get aggregated game stats
    games_df = get_games_from_test_set(test_df)
    
    # Filters Section
    st.markdown("### Filter Games")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Game search/select
        game_names = ["All Games"] + sorted(games_df['name'].tolist())
        selected_game_name = st.selectbox(
            "Select Game",
            options=game_names,
            index=0,
            help="Select a specific game to view its details"
        )
    
    with col2:
        # Genre filter
        genres = ["All Genres"] + sorted(games_df['primary_genre'].unique().tolist())
        selected_genre = st.selectbox(
            "Filter by Genre",
            options=genres,
            index=0,
            help="Filter games by genre"
        )
    
    with col3:
        # Sort option
        sort_option = st.selectbox(
            "Sort By",
            options=["Name", "Reviews ↓", "Reviews ↑", "Positive Ratio ↓"],
            index=0
        )
    
    # Apply filters
    filtered_df = games_df.copy()
    
    if selected_game_name != "All Games":
        # Navigate directly to game detail
        game_data = games_df[games_df['name'] == selected_game_name].iloc[0]
        st.session_state.selected_game = {
            'appid': int(game_data['appid']),
            'name': game_data['name']
        }
        st.session_state.current_page = 'game_detail'
        st.rerun()
    
    if selected_genre != "All Genres":
        filtered_df = filtered_df[filtered_df['primary_genre'] == selected_genre]
    
    # Apply sorting
    if sort_option == "Name":
        filtered_df = filtered_df.sort_values('name')
    elif sort_option == "Reviews ↓":
        filtered_df = filtered_df.sort_values('total_reviews', ascending=False)
    elif sort_option == "Reviews ↑":
        filtered_df = filtered_df.sort_values('total_reviews', ascending=True)
    elif sort_option == "Positive Ratio ↓":
        filtered_df = filtered_df.sort_values('positive_ratio', ascending=False)
    
    # Summary stats
    st.divider()
    
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total Games", len(filtered_df))
    with stat_cols[1]:
        st.metric("Total Reviews", f"{filtered_df['total_reviews'].sum():,}")
    with stat_cols[2]:
        avg_positive = filtered_df['positive_ratio'].mean() * 100
        st.metric("Avg Positive Rate", f"{avg_positive:.1f}%")
    with stat_cols[3]:
        unique_genres = filtered_df['primary_genre'].nunique()
        st.metric("Genres", unique_genres)
    
    st.divider()
    
    # Game Grid
    st.markdown(f"### Games ({len(filtered_df)})")
    st.caption("Click on any game card to view its reviews")
    
    # Display games in grid layout (3 per row)
    cols_per_row = 3
    
    for i in range(0, len(filtered_df), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered_df):
                break
            
            game_row = filtered_df.iloc[idx]
            game_data = format_game_card_data(game_row)
            
            with col:
                render_game_card(game_data)


def render_game_card(game_data: dict):
    """Render a single clickable game card - entire card is the button."""
    
    # Determine sentiment based on positive ratio
    ratio = game_data['positive_ratio']
    if ratio >= 0.7:
        sentiment_emoji = "🟢"
        sentiment_label = "Very Positive"
    elif ratio >= 0.5:
        sentiment_emoji = "🔵"
        sentiment_label = "Mostly Positive"
    elif ratio >= 0.3:
        sentiment_emoji = "🟡"
        sentiment_label = "Mixed"
    else:
        sentiment_emoji = "🔴"
        sentiment_label = "Mostly Negative"
    
    # Create button with all info displayed as label
    # This makes the entire card clickable
    button_label = f"""**{game_data['name'][:30]}{'...' if len(game_data['name']) > 30 else ''}**  
 {game_data['genre']}  
 {game_data['total_reviews']:,} reviews  
{sentiment_emoji} {game_data['positive_percent']} ({sentiment_label})"""
    
    if st.button(
        button_label,
        key=f"game_{game_data['appid']}",
        use_container_width=True,
        type="secondary"
    ):
        st.session_state.selected_game = {
            'appid': game_data['appid'],
            'name': game_data['name']
        }
        st.session_state.current_page = 'game_detail'
        st.rerun()
