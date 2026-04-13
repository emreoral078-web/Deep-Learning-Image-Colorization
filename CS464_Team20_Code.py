# ==============================================================================
# UNIFIED COLORIZATION PROJECT: BASELINE vs. U-NET vs. FUSION MODEL
# ==============================================================================

# ==============================================================================
# PART 1: SHARED SETUP & DATA LOADING (Run Once for All Models)
# ==============================================================================

import kagglehub
import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, BatchNormalization, Concatenate, Activation, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files

# 1.1 Optimize for A100 GPU
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("GPU and Mixed Precision Policy set.")

# 1.2 Download Dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
TRAIN_PATH = os.path.join(path, 'seg_train', 'seg_train')

# 1.3 Global Settings
IMG_SIZE = 160
BATCH_SIZE = 64

# 1.4 Data Loading Function
def load_data_with_labels(folder_path):
    images = []
    labels = []
    if not os.path.exists(folder_path): return np.array([]), np.array([])
    
    print("Loading images from disk...")
    # Iterate through sorted folders
    for label_name in sorted(os.listdir(folder_path)):
        sub_folder_path = os.path.join(folder_path, label_name)
        if os.path.isdir(sub_folder_path):
            # Limit to 1500 images per class for RAM safety
            file_names = os.listdir(sub_folder_path)[:1500] 
            for file_name in file_names: 
                try:
                    img_path = os.path.join(sub_folder_path, file_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label_name) 
                except: pass
    return np.array(images), np.array(labels)

# 1.5 Load & Process
full_data, full_labels = load_data_with_labels(TRAIN_PATH)
full_data = full_data.astype('float32') / 255.0 # Normalize to 0-1
print(f"Total Images Loaded: {len(full_data)}")

# Create Grayscale Input (X) and RGB Target (y)
print("Converting inputs to Grayscale...")
with tf.device('/GPU:0'):
    X = tf.image.rgb_to_grayscale(tf.convert_to_tensor(full_data)).numpy()
    y = full_data

# Stratified Split (Used by all models)
X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
    X, y, full_labels, test_size=0.2, random_state=42, stratify=full_labels
)
print(f"Train Set: {len(X_train)} | Validation Set: {len(X_val)}")


# ==============================================================================
# PART 2: BASELINE CNN (SIMPLE AUTOENCODER)
# ==============================================================================
print("\n" + "="*50)
print("PART 2: TRAINING BASELINE CNN")
print("="*50)

def build_simple_cnn():
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    x = Conv2D(64, (3, 3), padding='same')(input_img); x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(256, (3, 3), padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x); x = UpSampling2D((2, 2))(x)
    
    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mse') # MSE for Baseline
    return model

model_cnn = build_simple_cnn()


history_cnn = model_cnn.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=20, validation_data=(X_val, y_val), verbose=1)

# Visualization Function
def plot_results(model, X_test, y_test, title_prefix, num_samples=3):
    predicted = model.predict(X_test[:num_samples])
    plt.figure(figsize=(15, num_samples * 3))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3 + 1); plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray'); plt.title("Input"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 2); plt.imshow(predicted[i]); plt.title(f"{title_prefix} Prediction"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 3); plt.imshow(y_test[i]); plt.title("Ground Truth"); plt.axis('off')
    plt.show()

print("Visualizing Baseline CNN Results...")
plot_results(model_cnn, X_val, y_val, "Baseline CNN")


# ==============================================================================
# PART 3: STANDARD U-NET (WITH SKIP CONNECTIONS)
# ==============================================================================
print("\n" + "="*50)
print("PART 3: TRAINING STANDARD U-NET")
print("="*50)

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor); x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x)
    return x

def build_unet_model():
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))
    # Encoder
    c1 = conv_block(inputs, 64); p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 128); p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 256); p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 512) # Bottleneck
    # Decoder
    u1 = UpSampling2D((2, 2))(c4); u1 = Concatenate()([u1, c3]); c5 = conv_block(u1, 256)
    u2 = UpSampling2D((2, 2))(c5); u2 = Concatenate()([u2, c2]); c6 = conv_block(u2, 128)
    u3 = UpSampling2D((2, 2))(c6); u3 = Concatenate()([u3, c1]); c7 = conv_block(u3, 64)
    
    outputs = Conv2D(3, (1, 1), activation='sigmoid', dtype='float32')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mae') # MAE for sharper colors
    return model

model_unet = build_unet_model()

history_unet = model_unet.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=30, validation_data=(X_val, y_val), verbose=1)

print("Visualizing Standard U-Net Results...")
plot_results(model_unet, X_val, y_val, "Standard U-Net")


# ==============================================================================
# PART 4: FUSION U-NET (HISTOGRAM INJECTION + CROSS COLORIZATION)
# ==============================================================================
print("\n" + "="*50)
print("PART 4: TRAINING FUSION U-NET (THE FINAL MODEL)")
print("="*50)

# 4.1 Histogram Helpers
def compute_single_histogram(img, bins=32):
    img_uint8 = (img * 255).astype('uint8')
    h = [cv2.calcHist([img_uint8], [i], None, [bins], [0, 256]) for i in range(3)]
    for channel in h: cv2.normalize(channel, channel)
    return np.concatenate(h).flatten()

def compute_batch_histograms(images):
    return np.array([compute_single_histogram(img) for img in images])

print("Calculating Histograms...")
train_histograms = compute_batch_histograms(y_train)
val_histograms = compute_batch_histograms(y_val)

# 4.2 Fusion Architecture
def build_fusion_model():
    input_img = Input((IMG_SIZE, IMG_SIZE, 1))
    input_hist = Input((96,))
    
    # Encoder
    c1 = conv_block(input_img, 64); p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 128); p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 256); p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 512)
    
    # Fusion (Histogram Injection)
    h = Dense(256, activation='relu')(input_hist)
    h = Dense(256, activation='relu')(h)
    h_reshaped = Reshape((1, 1, 256))(h)
    h_tiled = UpSampling2D(size=(20, 20))(h_reshaped)
    fusion = Concatenate(axis=-1)([c4, h_tiled])
    fusion = Conv2D(512, (1, 1), activation='relu')(fusion)
    
    # Decoder
    u1 = UpSampling2D((2, 2))(fusion); u1 = Concatenate()([u1, c3]); c5 = conv_block(u1, 256)
    u2 = UpSampling2D((2, 2))(c5); u2 = Concatenate()([u2, c2]); c6 = conv_block(u2, 128)
    u3 = UpSampling2D((2, 2))(c6); u3 = Concatenate()([u3, c1]); c7 = conv_block(u3, 64)
    
    outputs = Conv2D(3, (1, 1), activation='sigmoid', dtype='float32')(c7)
    model = Model(inputs=[input_img, input_hist], outputs=outputs)
    model.compile(optimizer='adam', loss='mae')
    return model

model_fusion = build_fusion_model()

# 4.3 Training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
print("Starting Fusion Model Training...")
history_fusion = model_fusion.fit(
    [X_train, train_histograms], y_train,
    batch_size=BATCH_SIZE, epochs=100, # Higher epoch with Early Stopping
    validation_data=([X_val, val_histograms], y_val),
    callbacks=[early_stopping], verbose=1
)

# 4.4 Reference Selection & Saturation Boost
print("Selecting Reference Styles...")
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
REFERENCE_HISTS = {}
for cls in CLASSES:
    indices = np.where(labels_val == cls)[0]
    if len(indices) > 0:
        ref_idx = indices[min(10, len(indices)-1)]
        ref_img = y_val[ref_idx]
        REFERENCE_HISTS[cls] = compute_single_histogram(ref_img)

def increase_saturation(img_rgb, factor=1.4):
    img_uint8 = (img_rgb * 255).astype('uint8')
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype('float32')
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB).astype('float32') / 255.0

# 4.5 Generating Labeled Results & Zip
output_dir = "Cross_Colorization_Results"
if os.path.exists(output_dir): 
    import shutil
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print("\nGenerating Labeled Cross-Colorization Images...")
NUM_TO_PROCESS = 200
subset_indices = range(min(len(X_val), NUM_TO_PROCESS))
sep = np.ones((IMG_SIZE, 5, 3))
TITLES = ["Input"] + [c.capitalize() for c in CLASSES] + ["Real"]

for i in subset_indices:
    # Prepare Input
    bw_img = X_val[i].reshape(IMG_SIZE, IMG_SIZE)
    input_rgb = np.stack((bw_img,)*3, axis=-1)
    row_images = [input_rgb]
    input_tensor = bw_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    # Apply 6 Styles
    for cls in CLASSES:
        if cls in REFERENCE_HISTS:
            hist_input = REFERENCE_HISTS[cls].reshape(1, 96)
            pred = model_fusion.predict([input_tensor, hist_input], verbose=0)[0]
            pred_boosted = increase_saturation(pred, factor=1.4)
            row_images.append(sep); row_images.append(pred_boosted)
            
    # Add Real
    row_images.append(sep); row_images.append(y_val[i])
    
    # Stitch
    full_strip = np.hstack(row_images)
    
    # Add Header Labels
    full_strip_uint8 = (full_strip * 255).astype(np.uint8)
    header = np.ones((30, full_strip_uint8.shape[1], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    current_x = 0
    step_x = IMG_SIZE + 5
    
    for title in TITLES:
        (text_w, text_h), _ = cv2.getTextSize(title, font, 0.4, 1)
        text_x = current_x + (IMG_SIZE - text_w) // 2
        cv2.putText(header, title, (text_x, 20), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        current_x += step_x
        
    final_image = np.vstack((header, full_strip_uint8))
    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{labels_val[i]}_{i}.png"), final_image_bgr)
    
    if i % 50 == 0 and i > 0: print(f"Processed {i} images...")

# 4.6 Zip and Download
print("Creating Zip Archive...")
zip_name = "Unified_Project_Results.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files_in_dir in os.walk(output_dir):
        for file in files_in_dir:
            zipf.write(os.path.join(root, file), arcname=file)

print("Download Starting!")
files.download(zip_name)