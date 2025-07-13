import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data import load_metadata, filter_quality, batch_extract_features, split_data
from models import build_cnn, encode_labels
from config import Config

def main(config_path=None):
    cfg = Config(config_path)
    # Load and preprocess data
    df = load_metadata(cfg['data']['metadata_path'])
    df = filter_quality(df, cfg['data'].get('quality_keep', ['A', 'B', 'C']))
    X, y = batch_extract_features(df, cfg['data']['audio_dir'], sr=cfg['features']['sr'], n_mfcc=cfg['features']['n_mfcc'])
    y, label_encoder = encode_labels(y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=cfg['split']['test_size'], val_size=cfg['split']['val_size'])
    # Build model
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(np.unique(y)))
    # Prepare datasets
    batch_size = cfg['train']['batch_size']
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=cfg['train']['patience'], restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(cfg['train']['checkpoint_path'], save_best_only=True, monitor='val_loss', verbose=1)
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg['train']['epochs'],
        callbacks=[early_stop, checkpoint],
        verbose=2
    )
    # Save final model
    model.save(cfg['train']['final_model_path'])
    # Evaluate
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.config)
