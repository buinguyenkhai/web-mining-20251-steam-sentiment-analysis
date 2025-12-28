import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import LabelEncoder

def preprocess_for_fasttext(text: str) -> str:
    """
    Official FastText preprocessing adapted for Steam reviews.
    """
    if not isinstance(text, str):
        return ''
    
    # Steam-specific: Remove URLs first
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Normalize smart quotes to regular apostrophe
    text = re.sub(r"['′''`]", "'", text)
    
    # Add space around apostrophes
    text = re.sub(r"'", " ' ", text)
    
    # Remove double quotes including smart quotes
    text = re.sub(r'["""]', '', text)
    
    # Add space around periods
    text = re.sub(r'\.', ' . ', text)
    
    # Remove <br /> tags
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    # Add space around commas
    text = re.sub(r',', ' , ', text)
    
    # Add space around parentheses
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\)', ' ) ', text)
    
    # Add space around exclamation marks
    text = re.sub(r'!', ' ! ', text)
    
    # Add space around question marks
    text = re.sub(r'\?', ' ? ', text)
    
    # Remove semicolons and replace with space
    text = re.sub(r';', ' ', text)
    
    # Remove colons and replace with space
    text = re.sub(r':', ' ', text)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_test_data(data_dir: Path) -> pd.DataFrame:
    """Load test data with reviews."""
    test_df = pd.read_csv(data_dir / 'test.csv')
    return test_df


def load_selected_games(data_dir: Path) -> pd.DataFrame:
    """Load selected games metadata."""
    games_df = pd.read_csv(data_dir / 'selected_games.csv')
    return games_df


def load_train_data(data_dir: Path) -> pd.DataFrame:
    """Load training data (for fitting encoders)."""
    train_df = pd.read_csv(data_dir / 'train.csv')
    return train_df


def get_games_from_test_set(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique games from test set with aggregated stats.
    """
    # Aggregate by appid
    game_stats = test_df.groupby(['appid', 'name', 'primary_genre']).agg({
        'voted_up': ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    game_stats.columns = ['appid', 'name', 'primary_genre', 'total_reviews', 'positive_count']
    game_stats['negative_count'] = game_stats['total_reviews'] - game_stats['positive_count']
    game_stats['positive_ratio'] = game_stats['positive_count'] / game_stats['total_reviews']
    
    # Sort by name
    game_stats = game_stats.sort_values('name').reset_index(drop=True)
    
    return game_stats


def get_game_reviews(test_df: pd.DataFrame, appid: int) -> pd.DataFrame:
    """Get all reviews for a specific game."""
    reviews = test_df[test_df['appid'] == appid].copy()
    return reviews


def get_unique_genres(test_df: pd.DataFrame) -> List[str]:
    """Get list of unique genres from test set."""
    return sorted(test_df['primary_genre'].unique().tolist())


def get_unique_playtime_tiers(test_df: pd.DataFrame) -> List[str]:
    """Get list of unique playtime tiers."""
    return sorted(test_df['playtime_tier'].unique().tolist())


def get_unique_length_tiers(test_df: pd.DataFrame) -> List[str]:
    """Get list of unique length tiers."""
    return sorted(test_df['length_tier'].unique().tolist())

class MetadataEncoders:
    """Container for metadata label encoders."""
    
    def __init__(self):
        self.genre_encoder = LabelEncoder()
        self.playtime_tier_encoder = LabelEncoder()
        self.length_tier_encoder = LabelEncoder()
        self._fitted = False
    
    def fit(self, train_df: pd.DataFrame):
        """Fit encoders on training data."""
        self.genre_encoder.fit(train_df['primary_genre'])
        self.playtime_tier_encoder.fit(train_df['playtime_tier'])
        self.length_tier_encoder.fit(train_df['length_tier'])
        self._fitted = True
    
    def transform_genre(self, genres: List[str]) -> np.ndarray:
        """Transform genre labels to IDs."""
        return self.genre_encoder.transform(genres)
    
    def transform_playtime_tier(self, tiers: List[str]) -> np.ndarray:
        """Transform playtime tier labels to IDs."""
        return self.playtime_tier_encoder.transform(tiers)
    
    def transform_length_tier(self, tiers: List[str]) -> np.ndarray:
        """Transform length tier labels to IDs."""
        return self.length_tier_encoder.transform(tiers)
    
    def transform_all(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform all metadata columns."""
        genre_ids = self.transform_genre(df['primary_genre'].values)
        playtime_tier_ids = self.transform_playtime_tier(df['playtime_tier'].values)
        length_tier_ids = self.transform_length_tier(df['length_tier'].values)
        return genre_ids, playtime_tier_ids, length_tier_ids
    
    @property
    def genre_classes(self) -> List[str]:
        """Get genre class names."""
        return list(self.genre_encoder.classes_)
    
    @property
    def playtime_tier_classes(self) -> List[str]:
        """Get playtime tier class names."""
        return list(self.playtime_tier_encoder.classes_)
    
    @property
    def length_tier_classes(self) -> List[str]:
        """Get length tier class names."""
        return list(self.length_tier_encoder.classes_)


def create_and_fit_encoders(train_df: pd.DataFrame) -> MetadataEncoders:
    """Create and fit metadata encoders."""
    encoders = MetadataEncoders()
    encoders.fit(train_df)
    return encoders

def format_game_card_data(game_row: pd.Series) -> Dict[str, Any]:
    """Format game data for display in UI cards."""
    return {
        'appid': int(game_row['appid']),
        'name': game_row['name'],
        'genre': game_row['primary_genre'],
        'total_reviews': int(game_row['total_reviews']),
        'positive_count': int(game_row['positive_count']),
        'negative_count': int(game_row['negative_count']),
        'positive_ratio': float(game_row['positive_ratio']),
        'positive_percent': f"{game_row['positive_ratio'] * 100:.1f}%"
    }


def format_review_for_display(review_row: pd.Series, max_chars: int = 200) -> Dict[str, Any]:
    """Format review data for display in UI."""
    review_text = review_row['review_text'] if pd.notna(review_row['review_text']) else ''
    truncated = review_text[:max_chars] + '...' if len(review_text) > max_chars else review_text
    
    return {
        'recommendation_id': review_row['recommendationid'],
        'text': review_text,
        'truncated_text': truncated,
        'processed_text': review_row['processed_text'] if pd.notna(review_row['processed_text']) else '',
        'original_label': bool(review_row['voted_up']),
        'original_label_text': 'Positive' if review_row['voted_up'] else 'Negative',
        'playtime_tier': review_row['playtime_tier'],
        'length_tier': review_row['length_tier'],
        'genre': review_row['primary_genre']
    }


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) > max_chars:
        return text[:max_chars] + '...'
    return text
