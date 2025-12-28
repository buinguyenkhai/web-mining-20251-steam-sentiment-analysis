import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import fasttext
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

MAX_LENGTH = 256
EMBEDDING_DIM = 200
NUM_GENRES = 12
NUM_PLAYTIME_TIERS = 4
NUM_LENGTH_TIERS = 3


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load hyperparameters from JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('best_params', config)


def get_bigru_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get BiGRU config from JSON or use defaults."""
    defaults = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.4,
        'metadata_embed_dim': 16
    }
    if config_path and config_path.exists():
        params = load_config(config_path)
        return {
            'hidden_dim': params.get('hidden_dim', defaults['hidden_dim']),
            'num_layers': params.get('num_layers', defaults['num_layers']),
            'dropout': params.get('dropout', defaults['dropout']),
            'metadata_embed_dim': params.get('metadata_embed_dim', defaults['metadata_embed_dim'])
        }
    return defaults


def get_textcnn_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get TextCNN config from JSON or use defaults."""
    defaults = {
        'n_filters': 128,
        'filter_sizes': [3, 4, 5],
        'hidden_dim': 128,
        'dropout': 0.3,
        'metadata_embed_dim': 16
    }
    if config_path and config_path.exists():
        params = load_config(config_path)
        filter_sizes = params.get('filter_sizes', '3,4,5')
        if isinstance(filter_sizes, str):
            filter_sizes = [int(x) for x in filter_sizes.split(',')]
        return {
            'n_filters': params.get('n_filters', defaults['n_filters']),
            'filter_sizes': filter_sizes,
            'hidden_dim': params.get('hidden_dim', defaults['hidden_dim']),
            'dropout': params.get('dropout', defaults['dropout']),
            'metadata_embed_dim': params.get('metadata_embed_dim', defaults['metadata_embed_dim'])
        }
    return defaults


class Attention(nn.Module):
    """Attention mechanism for BiGRU."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, gru_output: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(gru_output).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e4)
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), gru_output).squeeze(1)
        return context, attention_weights


class BiGRUAttentionWithMetadata(nn.Module):
    """FastText + BiGRU + Attention + Metadata + FNN for sentiment classification."""
    
    def __init__(self, embedding_dim: int = 200, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.4, num_genres: int = 12, num_playtime_tiers: int = 4,
                 num_length_tiers: int = 3, metadata_embed_dim: int = 16):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim * 2)
        
        # Metadata embeddings
        self.genre_embed = nn.Embedding(num_genres, metadata_embed_dim)
        self.playtime_tier_embed = nn.Embedding(num_playtime_tiers, metadata_embed_dim)
        self.length_tier_embed = nn.Embedding(num_length_tiers, metadata_embed_dim)
        
        combined_dim = hidden_dim * 2 + metadata_embed_dim * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, 
                genre_ids: torch.Tensor, playtime_tier_ids: torch.Tensor, 
                length_tier_ids: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        gru_output, _ = self.gru(embeddings)
        text_context, _ = self.attention(gru_output, mask)
        
        genre_emb = self.genre_embed(genre_ids)
        playtime_emb = self.playtime_tier_embed(playtime_tier_ids)
        length_emb = self.length_tier_embed(length_tier_ids)
        
        combined = torch.cat([text_context, genre_emb, playtime_emb, length_emb], dim=-1)
        return self.classifier(combined)


class TextCNNWithMetadata(nn.Module):
    """TextCNN + Metadata for sentiment classification."""
    
    def __init__(self, embedding_dim: int = 200, n_filters: int = 128,
                 filter_sizes: List[int] = [3, 4, 5], hidden_dim: int = 128,
                 dropout: float = 0.3, num_genres: int = 12, num_playtime_tiers: int = 4,
                 num_length_tiers: int = 3, metadata_embed_dim: int = 16):
        super().__init__()
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        # Metadata embeddings
        self.genre_embed = nn.Embedding(num_genres, metadata_embed_dim)
        self.playtime_tier_embed = nn.Embedding(num_playtime_tiers, metadata_embed_dim)
        self.length_tier_embed = nn.Embedding(num_length_tiers, metadata_embed_dim)
        
        cnn_output_dim = n_filters * len(filter_sizes)
        combined_dim = cnn_output_dim + metadata_embed_dim * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor,
                genre_ids: torch.Tensor, playtime_tier_ids: torch.Tensor,
                length_tier_ids: torch.Tensor) -> torch.Tensor:
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        x = masked_embeddings.permute(0, 2, 1)
        
        conved = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        text_features = torch.cat(pooled, dim=1)
        
        genre_emb = self.genre_embed(genre_ids)
        playtime_emb = self.playtime_tier_embed(playtime_tier_ids)
        length_emb = self.length_tier_embed(length_tier_ids)
        
        combined = torch.cat([text_features, genre_emb, playtime_emb, length_emb], dim=-1)
        return self.classifier(combined)

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_fasttext_model(model_path: Path) -> Any:
    """Load FastText embeddings model."""
    return fasttext.load_model(str(model_path))


def load_baseline_model(model_path: Path) -> Dict[str, Any]:
    """Load TF-IDF + Naive Bayes baseline model."""
    return joblib.load(model_path)


def load_metadata_encoders(encoders_path: Path) -> Dict[str, Any]:
    """Load label encoders for metadata."""
    return joblib.load(encoders_path)


def load_bigru_model(model_path: Path, device: torch.device, 
                     config_path: Optional[Path] = None) -> BiGRUAttentionWithMetadata:
    """Load BiGRU + Attention + Metadata model with config from JSON."""
    config = get_bigru_config(config_path)
    
    model = BiGRUAttentionWithMetadata(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_genres=NUM_GENRES,
        num_playtime_tiers=NUM_PLAYTIME_TIERS,
        num_length_tiers=NUM_LENGTH_TIERS,
        metadata_embed_dim=config['metadata_embed_dim']
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def load_textcnn_model(model_path: Path, device: torch.device,
                       config_path: Optional[Path] = None) -> TextCNNWithMetadata:
    """Load TextCNN + Metadata model with config from JSON."""
    config = get_textcnn_config(config_path)
    
    model = TextCNNWithMetadata(
        embedding_dim=EMBEDDING_DIM,
        n_filters=config['n_filters'],
        filter_sizes=config['filter_sizes'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        num_genres=NUM_GENRES,
        num_playtime_tiers=NUM_PLAYTIME_TIERS,
        num_length_tiers=NUM_LENGTH_TIERS,
        metadata_embed_dim=config['metadata_embed_dim']
    )
    
    # Load checkpoint - handle both dict format and direct state_dict
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def get_embeddings_for_text(text: str, ft_model: Any, max_length: int = MAX_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """Convert text to FastText embeddings with mask."""
    tokens = text.split()[:max_length]
    embedding_dim = ft_model.get_dimension()
    
    embeddings = np.zeros((max_length, embedding_dim), dtype=np.float32)
    for i, token in enumerate(tokens):
        embeddings[i] = ft_model.get_word_vector(token)
    
    mask = np.zeros(max_length, dtype=np.float32)
    mask[:len(tokens)] = 1.0
    
    return embeddings, mask


def predict_baseline(texts: List[str], pipeline) -> List[Dict[str, Any]]:
    """
    Predict using TF-IDF + Naive Bayes baseline.
    Returns list of dicts with 'prediction', 'confidence', 'probabilities'.
    """
    results = []
    
    # Get probabilities
    probas = pipeline.predict_proba(texts)
    predictions = pipeline.predict(texts)
    
    for i, (pred, proba) in enumerate(zip(predictions, probas)):
        confidence = max(proba) * 100
        results.append({
            'prediction': bool(pred),
            'confidence': confidence,
            'prob_negative': proba[0] * 100,
            'prob_positive': proba[1] * 100
        })
    
    return results


def predict_deep_model(
    texts: List[str],
    genre_ids: List[int],
    playtime_tier_ids: List[int],
    length_tier_ids: List[int],
    model: nn.Module,
    ft_model: Any,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Predict using a deep learning model (BiGRU or TextCNN).
    Returns list of dicts with 'prediction', 'confidence', 'probability'.
    """
    results = []
    model.eval()
    
    with torch.no_grad():
        # Process in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_genre_ids = genre_ids[i:i+batch_size]
            batch_playtime_ids = playtime_tier_ids[i:i+batch_size]
            batch_length_ids = length_tier_ids[i:i+batch_size]
            
            # Create embeddings
            embeddings_list = []
            masks_list = []
            for text in batch_texts:
                emb, mask = get_embeddings_for_text(text, ft_model)
                embeddings_list.append(emb)
                masks_list.append(mask)
            
            # Convert to tensors
            embeddings = torch.tensor(np.array(embeddings_list), dtype=torch.float32).to(device)
            masks = torch.tensor(np.array(masks_list), dtype=torch.float32).to(device)
            genre_tensor = torch.tensor(batch_genre_ids, dtype=torch.long).to(device)
            playtime_tensor = torch.tensor(batch_playtime_ids, dtype=torch.long).to(device)
            length_tensor = torch.tensor(batch_length_ids, dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(embeddings, masks, genre_tensor, playtime_tensor, length_tensor).squeeze(-1)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities > 0.5).astype(bool)
            
            for pred, prob in zip(predictions, probabilities):
                confidence = prob * 100 if pred else (1 - prob) * 100
                results.append({
                    'prediction': bool(pred),
                    'confidence': float(confidence),
                    'prob_positive': float(prob * 100),
                    'prob_negative': float((1 - prob) * 100)
                })
    
    return results


def predict_all_models(
    texts: List[str],
    genre_ids: List[int],
    playtime_tier_ids: List[int],
    length_tier_ids: List[int],
    baseline_pipeline,
    textcnn_model: nn.Module,
    bigru_model: nn.Module,
    ft_model: Any,
    device: torch.device
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Predict using all three models for comparison.
    Returns list of dicts with predictions from each model.
    """
    baseline_results = predict_baseline(texts, baseline_pipeline)
    textcnn_results = predict_deep_model(
        texts, genre_ids, playtime_tier_ids, length_tier_ids,
        textcnn_model, ft_model, device
    )
    bigru_results = predict_deep_model(
        texts, genre_ids, playtime_tier_ids, length_tier_ids,
        bigru_model, ft_model, device
    )
    
    combined_results = []
    for i in range(len(texts)):
        combined_results.append({
            'TF-IDF + NB': baseline_results[i],
            'FT-TextCNN-Meta': textcnn_results[i],
            'FT-BiGRU-Att-Meta': bigru_results[i]
        })
    
    return combined_results
