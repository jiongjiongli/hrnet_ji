from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime


def filter_data(data_file_paths, num_classes):
    results = []

    for data_file_path in data_file_paths:
        label_data_file_path = data_file_path.with_suffix('.png')

        if not label_data_file_path.exists():
            continue

        with Image.open(data_file_path.as_posix()) as input_data_img:
            input_data_size = input_data_img.size

        with Image.open(label_data_file_path.as_posix()) as label_img:
            label_size = label_img.size

            if input_data_size != label_size:
                continue

            if label_img.mode not in ['L', 'I']:
                continue

            label_data = np.array(label_img)

            if label_data.ndim != 2:
                continue

            unique_labels = np.unique(label_data)
            is_label_valid1 = np.any(unique_labels == 0)
            is_label_valid2 = np.all((0 <= unique_labels) & (unique_labels < num_classes))

            if not (is_label_valid1 and is_label_valid2):
                continue

        results.append(data_file_path)

    return results
