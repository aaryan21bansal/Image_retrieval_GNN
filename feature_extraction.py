import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

dataset_dir = "data/caltech-101"   
features_dir = "features"
os.makedirs(features_dir, exist_ok=True)

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

all_features = []
all_paths = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        img_path = os.path.join(root, file)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        features = features.flatten()
        all_features.append(features)
        all_paths.append(img_path)


all_features = np.array(all_features)
all_paths = np.array(all_paths)

np.save(os.path.join(features_dir, "caltech_features.npy"), all_features)
np.save(os.path.join(features_dir, "image_paths.npy"), all_paths)

print("Feature extraction complete")


print (all_paths[0]) 