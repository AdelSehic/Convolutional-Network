import numpy as np
import itertools
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from tqdm import tqdm
from .load import Annotation


def pad_image(img: Image.Image) -> Image.Image:
    old_size = img.size
    desired_size = (64, 64)
    max_dim = max(old_size)

    padded = Image.new("RGB", (max_dim, max_dim))
    x = (max_dim - old_size[0]) // 2
    y = (max_dim - old_size[1]) // 2

    padded.paste(img, (x, y))
    out_img = padded.resize(desired_size)

    return out_img


def extract_objects(data: tuple[Image.Image, list[Annotation]]) -> list[tuple[Image.Image, str]]:
    objects = []

    image = data[0]

    for annot in data[1]:
        cropped = image.crop(annot.bbox.crop_values())
        objects.append((cropped, annot.name))

    return objects


def encode_labels(labels):
    flattened_labels = list(itertools.chain(*labels))
    flattened_labels_np = np.array(labels)
    encoder = OneHotEncoder(sparse_output=False)
    encoded_labels = encoder.fit_transform(flattened_labels_np.reshape(-1, 1)
)

    return encoded_labels, flattened_labels_np


def normalize_images(images):
    normalized = []

    for img in tqdm(images, desc="Normalizing images", unit="img"):
        normalized.append(np.array(img)/255.0)

    return np.array(normalized)


def preprocess_data(data: list[tuple[Image.Image, list[Annotation]]]):
    images = []
    labels = []

    print("🚀 Starting the data preprocessing...")

    for t in tqdm(data, desc="Extracting objects", unit="img"):
        for image, lb in extract_objects(t):
            images.append(pad_image(image))
            labels.append(lb)

    normalized_images = normalize_images(images)
    print("🔠 Encoding labels...")
    encoded_lables, flat_labels_np = encode_labels(labels)

    print(f"✅ Preprocessing complete! Processed {len(images)} objects 🎉")
    return normalized_images, encoded_lables, flat_labels_np
