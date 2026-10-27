import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
# --- FIX: Use FeatureExtractor specifically, not AutoProcessor ---
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/wavlm-base-plus" 
SAMPLE_RATE = 16000  
MAX_DURATION = 5     
SEED = 42

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# ==========================================
# 2. LOAD PRETRAINED MODEL & EXTRACTOR
# ==========================================
print(f"📥 Loading Model: {MODEL_NAME}...")

try:
    # FIX: WavLM uses Wav2Vec2's feature extraction logic
    # This bypasses the need for a text tokenizer (which caused your error)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

model.eval() # Freeze the model

# ==========================================
# 3. EXTRACT EMBEDDINGS
# ==========================================
# Ensure 'df' exists from Step 1.
if 'df' not in locals():
    # Fallback if df is missing (e.g. restart) - minimal reload for testing
    print("⚠️ DataFrame 'df' missing. Please re-run Step 1 to load data.")
else:
    print("\n" + "="*80)
    print("🌊 Starting Feature Extraction (WavLM)...")
    print("="*80)

    X_features = []
    y_labels = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        try:
            # 1. Load Audio
            y, sr = librosa.load(row['path'], sr=SAMPLE_RATE)
            
            # 2. Fix Length
            max_len = SAMPLE_RATE * MAX_DURATION
            if len(y) < max_len:
                y = np.pad(y, (0, max_len - len(y)), 'constant')
            else:
                y = y[:max_len]
                
            # 3. Preprocess (Use feature_extractor instead of processor)
            inputs = feature_extractor(
                y, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            ).to(device)
            
            # 4. Inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 5. Mean Pooling
            # Extract last_hidden_state (Batch, Time, Features)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]
                
            # Average over time dimension -> (1, 768)
            embedding = torch.mean(hidden_states, dim=1).cpu().numpy().squeeze()
            
            X_features.append(embedding)
            y_labels.append(row['label'])
            
        except Exception as e:
            # print(f"Skipping {row['filename']}: {e}")
            continue

    X = np.array(X_features)
    y = np.array(y_labels)

    print(f"\n✅ Extraction Complete!")
    print(f"   X Shape: {X.shape}") 
    print(f"   y Shape: {y.shape}")

    # ==========================================
    # 4. TRAIN CLASSIFIER
    # ==========================================
    if len(X) > 0:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

        print("\n🚀 Training Keras Classifier...")

        keras_model = Sequential([
            Input(shape=(X.shape[1],)), # 768 dimensions
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        keras_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        history = keras_model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=20, 
            batch_size=32
        )

        # Plot
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.title('Deepfake Detection Accuracy')
        plt.legend()
        plt.show()

        keras_model.save("wavlm_classifier.keras")
        print("✅ Model Saved!")
    else:
        print("❌ No features extracted. Check your DataFrame or Audio paths.")