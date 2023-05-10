from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime

from utils.dataloader import SegmentationDataset


data_dir_path = r'/home/data'
data_dir_path = Path(data_dir_path)


def print_file_paths(file_paths, comment):
    if file_paths:
        print('=' * 80)
        print(comment)
        print('-' * 80)
        for file_path in file_paths:
            print(file_path.as_posix())
        print('=' * 80)
        print()


def check_data_files():
    data_file_paths = list(data_dir_path.rglob('*.jpg'))

    valid_file_paths = []
    not_existed_label_file_paths = []
    not_matched_size_file_paths = []
    invalid_mode_file_paths = []
    invalid_ndim_file_paths = []
    invalid_label_file_paths = []

    for data_file_path in data_file_paths:
        data_issues_count = 0

        if data_file_path.suffix not in ['.jpg']:
            continue

        label_data_file_path = data_file_path.with_suffix('.png')

        if not label_data_file_path.exists():
            print('Label file not exist!', label_data_file_path)
            not_existed_label_file_paths.append(label_data_file_path)
            data_issues_count += 1
            continue

        with Image.open(data_file_path.as_posix()) as input_data_img:
            input_data_size = input_data_img.size

        with Image.open(label_data_file_path.as_posix()) as label_img:
            label_size = label_img.size

            if input_data_size != label_size:
                print('input image and label size not equal!',
                      label_data_file_path,
                      input_data_size,
                      label_size)
                not_matched_size_file_paths.append(label_data_file_path)
                data_issues_count += 1

            if label_img.mode not in ['L', 'I']:
                print('Label file mode wrong!',
                      label_data_file_path,
                      label_img.mode)
                invalid_mode_file_paths.append(label_data_file_path)
                data_issues_count += 1

            label_data = np.array(label_img)

            if label_data.ndim != 2:
                print('Label ndim wrong!', label_data.shape)
                invalid_ndim_file_paths.append(label_data_file_path)
                data_issues_count += 1

            unique_labels = np.unique(label_data)
            is_label_valid1 = np.array_equal(unique_labels, np.array([0]))
            is_label_valid2 = np.array_equal(unique_labels, np.array([0, 1]))

            if not (is_label_valid1 or is_label_valid2):
                print('Label not valid!', unique_labels)
                invalid_label_file_paths.append(label_data_file_path)
                data_issues_count += 1

        if data_issues_count == 0:
            valid_file_paths.append(label_data_file_path)


    print('             data_file count:', len(data_file_paths))
    print('            valid_file count:', len(valid_file_paths))
    print('not_existed_label_file count:', len(not_existed_label_file_paths))
    print(' not_matched_size_file count:', len(not_matched_size_file_paths))
    print('     invalid_mode_file count:', len(invalid_mode_file_paths))
    print('     invalid_ndim_file count:', len(invalid_ndim_file_paths))
    print('    invalid_label_file count:', len(invalid_label_file_paths))

    print_file_paths(not_existed_label_file_paths, 'not_existed_label_file_paths')
    print_file_paths( not_matched_size_file_paths, 'not_matched_size_file_paths')
    print_file_paths(     invalid_mode_file_paths, 'invalid_mode_file_paths')
    print_file_paths(     invalid_ndim_file_paths, 'invalid_ndim_file_paths')
    print_file_paths(    invalid_label_file_paths, 'invalid_label_file_paths')


def check_data_enhance():
    print('=' * 80)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          'Start check_data_enhance.')
    print('-' * 80)

    data_file_paths = list(data_dir_path.rglob('*.jpg'))

    root = None
    img_paths = data_file_paths
    input_shape = [512, 512]
    num_classes = 2
    dataset = SegmentationDataset(img_paths, input_shape, num_classes, True, root)
    input_data_dtype_set = set()
    label_data_dtype_set = set()

    for data_file_path in data_file_paths:
        label_data_file_path = data_file_path.with_suffix('.png')

        if not label_data_file_path.exists():
            print('label_data_file_path not exist!', label_data_file_path)
            continue

        with Image.open(data_file_path.as_posix()) as origin_img, \
            Image.open(label_data_file_path.as_posix()) as origin_label:
            img, label           = dataset.get_random_data(origin_img, origin_label, input_shape, random=True)
            label_data           = np.array(label)

            if img.shape[:2] != label_data.shape:
                print('Invalid shape!', img.shape, label_data.shape)

            input_data_dtype_set.add(img.dtype)
            label_data_dtype_set.add(label_data.dtype)

            unique_labels        = np.unique(label_data)

            is_label_valid       = np.all((0 <= unique_labels) & (unique_labels <= 1))

            if not is_label_valid:
                print('Label not valid!', unique_labels, label_data_file_path)

    print('input_data_dtypes:', input_data_dtype_set)
    print('label_data_dtypes:', label_data_dtype_set)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          'End check_data_enhance.')
    print('=' * 80)

def main():
    check_data_files()
    check_data_enhance()
    check_data_enhance()


if __name__ == '__main__':
    main()
