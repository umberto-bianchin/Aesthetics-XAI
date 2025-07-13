import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PATCHES = True

try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive'
    VOTES_CSV  = os.path.join(BASE, 'datasets/eva-dataset-master/data/votes_filtered.csv')
    IMAGES_DIR = os.path.join(BASE, 'datasets/eva-dataset-master/images/EVA_together')
    MODEL_PATH = os.path.join(BASE, 'models/inception_multiout_final.keras')
    PATCH_DIR = os.path.join(BASE, 'datasets/top50')
    print("Mounted Google Drive and set dataset paths.")
except ImportError:
    BASE = '.'
    VOTES_CSV  = os.path.join(BASE, 'datasets/eva-dataset-master/data/votes_filtered.csv')
    IMAGES_DIR = os.path.join(BASE, 'datasets/eva-dataset-master/images/EVA_together')
    MODEL_PATH = os.path.join(BASE, 'model/inception_multiout_final.keras')
    PATCH_DIR = os.path.join(BASE, 'datasets/top50/content/EVA_together')

TARGET_SIZE = (299, 299)

cols = ['image_id','user_id','score','difficulty','visual','composition','quality','semantic','vote_time','1','2','3','4']
votes = pd.read_csv(VOTES_CSV, sep='=', names=cols, header=0, engine='python')
means = (
    votes
    .groupby('image_id')[['score','difficulty','visual','composition','quality','semantic']]
    .mean()
    .reset_index()
)
means['image_id'] = means['image_id'].astype(str)
means = means.rename(columns={'score':'total'})

filepaths, image_ids, y_true = [], [], []  # each entry of y true: [total, difficulty, visual, composition, quality, semantic]
names = ['total','difficulty','visual','composition','quality','semantic']

if PATCHES:
    for img_id in sorted(os.listdir(PATCH_DIR)):
        folder = os.path.join(PATCH_DIR, img_id)
        if not os.path.isdir(folder): continue
        row = means[means['image_id']==img_id]
        if row.empty: continue
        true_vals = row[names].iloc[0].values
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
            fp = os.path.join(folder, f)
            filepaths.append(fp)
            image_ids.append(img_id)
            y_true.append(true_vals)
else:
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
        img_id = os.path.splitext(fname)[0]
        row = means[means['image_id']==img_id]
        if row.empty: continue
        fp = os.path.join(IMAGES_DIR, fname)
        filepaths.append(fp)
        image_ids.append(img_id)
        y_true.append(row[names].iloc[0].values)


y_true = np.vstack(y_true)  # shape (n_samples,6)

model = load_model(MODEL_PATH)
preds = []
for fp in filepaths:
    img = load_img(fp, target_size=TARGET_SIZE)
    x   = img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess_input(x)
    p   = model.predict(x, verbose=0)[0]  # array shape (6,)
    preds.append(p)
preds = np.vstack(preds)  # shape (n_samples,6)

if PATCHES:
    df = pd.DataFrame(preds, columns=names)
    df['image_id'] = image_ids
    df_agg = df.groupby('image_id', as_index=False)[names].mean()

    # Ground truth per image
    df_truth = means[['image_id'] + names]
    df_eval  = pd.merge(df_agg, df_truth, on='image_id', suffixes=('_pred',''))

    y_true = df_eval[names].values
    preds  = df_eval[[n+'_pred' for n in names]].values

metrics = []
for i, name in enumerate(names):
    yt   = y_true[:, i]
    yp   = preds[:, i]
    mse  = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    metrics.append((name, mse, rmse, mae, r2))

df_metrics = pd.DataFrame(metrics, columns=['score','MSE','RMSE','MAE','R2'])
print(df_metrics.round(4))
