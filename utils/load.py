import os
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import sys


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __repr__(self):
        return f"BoundingBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"

    def crop_values(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class Annotation:
    def __init__(self, name, bbox: BoundingBox):
        self.name = name
        self.bbox = bbox

    def __repr__(self):
        return f"Annotation(name={self.name}, bbox={self.bbox})"


def load_train_data(datadir: str, dataset: str) -> list[tuple[Image.Image, list[Annotation]]]:
    image_dir = os.path.join(datadir, 'JPEGImages')
    annotations_dir = os.path.join(datadir, 'Annotations/Horizontal Bounding Boxes')
    list_file = os.path.join(datadir, 'ImageSets/Main', dataset + '.txt')

    rvalues = []
    total_size_bytes = 0

    with open(list_file, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    print("Loading training dataset...")

    for base_name in tqdm(lines, desc="Loading", unit="file"):
        img_path = os.path.join(image_dir, base_name + '.jpg')
        xml_path = os.path.join(annotations_dir, base_name + '.xml')

        image = Image.open(img_path).convert('RGB')
        total_size_bytes += os.path.getsize(img_path)
        total_size_bytes += os.path.getsize(xml_path)

        rvalues.append((image, load_annotation(xml_path)))

    size_mb = total_size_bytes / (1024 * 1024)
    print(f"✅ Loaded `{dataset}` dataset: {len(rvalues)} images with annotations")
    print(f"📦 Total size in memory (approx file size): {size_mb:.2f} MB")

    return rvalues


def load_annotation(filepath: str):
    tree = ET.parse(filepath)
    root = tree.getroot()

    annotations = []

    for obj in root.findall('object'):
        name = obj.find('name').text

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        bounding_box = BoundingBox(xmin, ymin, xmax, ymax)
        annotation = Annotation(name, bounding_box)
        annotations.append(annotation)

    return annotations
