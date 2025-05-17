import os
import re
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Workaround SSL certificate issues when downloading pretrained weights
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# === Path configuration ===
# Replace with the right folder path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/datasets/top50/content/EVA_together'
    VOTES_CSV = '/content/drive/MyDrive/datasets/eva-dataset-master/data/votes_filtered.csv'
    CHECKPOINT_DIR = '/content/drive/MyDrive/models'
    print("Mounted Google Drive and set dataset paths.")
except ImportError:
    # Local paths (not running in Colab)
    BASE_DIR = 'datasets/top50/content/EVA_together'
    VOTES_CSV = 'datasets/eva-dataset-master/data/votes_filtered.csv'
    CHECKPOINT_DIR = 'checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Load and preprocess votes ===
cols = ['image_id','user_id','score','difficulty','visual','composition','quality','semantic','vote_time','1','2','3','4']
votes = pd.read_csv(VOTES_CSV, sep='=', header=0, names=cols, engine='python')
means = (
    votes
    .groupby('image_id')[['score', 'difficulty', 'visual', 'composition', 'quality', 'semantic']]
    .mean()
    .reset_index()
)
means['image_id'] = means['image_id'].astype(str)
means.rename(columns={'score':'total'}, inplace=True)

# === Create labels dataframe ===
filepaths, totals, diffs, viss, comps, quals, sems = [], [], [], [], [], [], []
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    image_id = os.path.splitext(folder)[0]
    row = means[means['image_id'] == image_id]
    if row.empty:
        continue
    t, d, v, c, q, s = row[['total','difficulty','visual','composition','quality','semantic']].iloc[0]
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            fp = os.path.join(folder_path, fname)
            filepaths.append(fp)
            totals.append(t)
            diffs.append(d)
            viss.append(v)
            comps.append(c)
            quals.append(q)
            sems.append(s)

labels_df = pd.DataFrame({
    'filepath': filepaths,
    'total': totals,
    'difficulty': diffs,
    'visual': viss,
    'composition': comps,
    'quality': quals,
    'semantic': sems
})

# === Split dataset: train/val/test ===
df_train_val, df_test = train_test_split(labels_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(df_train_val, test_size=0.25, random_state=42)  # 0.25*0.8 = 0.2

# === Data generators ===
target_size = (299,299)
batch_size = 32
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

y_cols = ['total','difficulty','visual','composition','quality','semantic']
train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col=y_cols,
    target_size=target_size, batch_size=batch_size,
    class_mode='raw', shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col=y_cols,
    target_size=target_size, batch_size=batch_size,
    class_mode='raw', shuffle=False
)
test_gen = test_datagen.flow_from_dataframe(
    df_test, x_col='filepath', y_col=y_cols,
    target_size=target_size, batch_size=batch_size,
    class_mode='raw', shuffle=False
)

# === Build multi-output InceptionV3 model ===
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(*target_size, 3))
# Freeze the base model layers
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
# outputs per category + total
output = Dense(6, activation='linear', name='all_scores')(x)
model = Model(inputs=base_model.input, outputs=output)

# === Compile model ===
model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])

# === Stage 1: Train head only ===
pattern1 = os.path.join(CHECKPOINT_DIR, 'stage1_epoch{epoch:02d}.weights.h5')
chkpt1 = ModelCheckpoint(pattern1, save_weights_only=True, save_freq='epoch', monitor='val_loss')

ckpts1 = glob.glob(os.path.join(CHECKPOINT_DIR, 'stage1_epoch*.weights.h5'))
if ckpts1:
    epochs_done1 = [int(re.search(r'stage1_epoch(\d+)\.weights\.h5', f).group(1)) for f in ckpts1]
    init_epoch1 = max(epochs_done1)
    last_ckpt1 = os.path.join(CHECKPOINT_DIR, f'inception_stage1_{init_epoch1:02d}.keras')
    model.load_weights(last_ckpt1)
    print(f"Resuming Stage 1 from epoch {init_epoch1}")
else:
    init_epoch1 = 0

model.fit(train_gen, validation_data=val_gen,
          epochs=20, initial_epoch=init_epoch1,
          callbacks=[chkpt1,
                     EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')])

# === Stage 2: Fine-tune top layers ===
# Unfreeze last 50 layers of base
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='mse', metrics=['mae'])

pattern2 = os.path.join(CHECKPOINT_DIR, 'stage2_epoch{epoch:02d}.weights.h5')
chkpt2 = ModelCheckpoint(pattern2, save_weights_only=True, save_freq='epoch', monitor='val_loss')

ckpts2 = glob.glob(os.path.join(CHECKPOINT_DIR, 'stage2_epoch*.weights.h5'))
if ckpts2:
    epochs_done2 = [int(re.search(r'stage2_epoch(\d+)\.weights\.h5', f).group(1)) for f in ckpts2]
    init_epoch2 = max(epochs_done2)
    last_ckpt2 = f'stage2_epoch{init_epoch2:02d}.keras'
    model.load_weights(os.path.join(CHECKPOINT_DIR, last_ckpt2))
    print(f'Resuming Stage 2 from epoch {init_epoch2}')
else:
    init_epoch2 = 0

model.fit(train_gen, validation_data=val_gen,
          epochs=10, initial_epoch=init_epoch2,
          callbacks=[chkpt2,
                     EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')])

print('Validation results:', model.evaluate(val_gen))
print('Test results:', model.evaluate(test_gen))
model.save(os.path.join(CHECKPOINT_DIR, 'inception_multiout_final.keras'))

# To load the model later:
# from tensorflow.keras.models import load_model
# model = load_model('inception_final.keras')