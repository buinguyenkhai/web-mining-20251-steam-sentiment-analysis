import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main():
    # Paths
    project_root = Path(__file__).parent
    data_dir = project_root / 'processed_data'
    models_dir = project_root / 'models'
    
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(data_dir / 'train.csv')
    
    # Create and fit encoders
    print("Fitting encoders...")
    genre_encoder = LabelEncoder()
    playtime_tier_encoder = LabelEncoder()
    length_tier_encoder = LabelEncoder()
    
    genre_encoder.fit(train_df['primary_genre'])
    playtime_tier_encoder.fit(train_df['playtime_tier'])
    length_tier_encoder.fit(train_df['length_tier'])
    
    # Print info
    print(f"Genre categories ({len(genre_encoder.classes_)}): {list(genre_encoder.classes_)}")
    print(f"Playtime tier categories ({len(playtime_tier_encoder.classes_)}): {list(playtime_tier_encoder.classes_)}")
    print(f"Length tier categories ({len(length_tier_encoder.classes_)}): {list(length_tier_encoder.classes_)}")
    
    # Save encoders
    encoders = {
        'genre_encoder': genre_encoder,
        'playtime_tier_encoder': playtime_tier_encoder,
        'length_tier_encoder': length_tier_encoder
    }
    
    output_path = models_dir / 'metadata_encoders.joblib'
    joblib.dump(encoders, output_path)
    print(f"\nEncoders saved to: {output_path}")


if __name__ == '__main__':
    main()
