import numpy as np
import os
import glob
from scipy.io import wavfile
from typing import List


def exist_directory(directory_path: str):
    return os.path.isdir(directory_path)


def exist_file(file_path: str):
    return os.path.isfile(file_path)


def get_directory_contents(directory_path: str, pattern: str):
    if exist_directory(directory_path):
        # Get the current working directory
        cwd = os.getcwd()
        # Change the directory into the target directory path
        os.chdir(directory_path)
        # Get the list of directory contents
        directory_contents = glob.glob(pattern)
        # Change back to the original process working directory
        os.chdir(cwd)
        # Return back the directory contents
        return directory_contents
    else:
        return []


def get_subdirectory_names(directory_path: str):
    return list(map(lambda subdirectory_name: subdirectory_name.replace('\\', '').replace('//', ''),
                    get_directory_contents(directory_path, '*//')))


def get_file_names(directory_path: str, file_extension: str):
    return get_directory_contents(directory_path, f'*{file_extension}')


def construct_path(directory_path: str, directory_content_name: str):
    return os.path.join(directory_path, directory_content_name)


def read_wav_file(file_path: str):
    if exist_file(file_path):
        return wavfile.read(file_path)
    else:
        return 44100, np.array([])


def get_sound_signals(wav_file_data):
    return wav_file_data[-1]


def shift_sound_signals(sound_signals: np.array):
    return np.int32(sound_signals + 2**15)


def normalize_sound_signals(sound_signals: np.array):
    return np.float32(sound_signals / (2. ** 15))


def get_left_channel_data(sound_signals: np.array):
    return sound_signals[:, 0]


def get_right_channel_data(sound_signals: np.array):
    return sound_signals[:, -1]


def get_sound_feature_vectors_from_file(file_path: str, normalize: bool = True, shift: bool = False):
    if normalize:
        sound_signals = normalize_sound_signals(get_sound_signals(read_wav_file(file_path)))
    else:
        sound_signals = get_sound_signals(read_wav_file(file_path))
        if shift:
            sound_signals = shift_sound_signals(sound_signals)
    left_channel_feature_vector, right_channel_feature_vector = \
        get_left_channel_data(sound_signals), get_right_channel_data(sound_signals)
    return left_channel_feature_vector, right_channel_feature_vector


def stack_data(feature_matrix: np.array, left_channel_features: np.array, right_channel_features: np.array):
    if len(feature_matrix) == 0:
        return np.stack((left_channel_features, right_channel_features))
    else:
        return np.vstack((feature_matrix, left_channel_features, right_channel_features))


def get_class_data(parent_directory_path: str, class_label: str, number_of_examples: int = 0,
                   normalize: bool = True, shift: bool = False):
    # Initializing the class feature matrix and target vector
    class_feature_matrix = []
    class_target_vector = []
    # Construct the path
    class_directory_path = construct_path(parent_directory_path, class_label)
    # Make sure the given path is correct
    if not exist_directory(class_directory_path):
        return np.array(class_feature_matrix), np.array(class_target_vector)

    # Get the names of the wav file belonging to the current class
    wav_file_names = set(get_file_names(class_directory_path, '.wav'))
    # Get the subset of the classes, if want only limited number of training examples
    if number_of_examples:
        wav_file_names = wav_file_names[:np.abs(number_of_examples)]
    print(f'Processing: {len(wav_file_names)} files')
    # Iterate through each wav file
    for file_name in wav_file_names:
        file_path = construct_path(class_directory_path, file_name)
        left_channel_features, right_channel_features = \
            get_sound_feature_vectors_from_file(file_path, normalize, shift)
        class_feature_matrix += [left_channel_features, right_channel_features]
        class_target_vector += [class_label]*2

    return np.array(class_feature_matrix), np.array(class_target_vector)


def load_irmas_data(parent_directory_path: str, class_labels_to_process: List[str],
                    number_of_training_examples_per_class: int = 0, normalize: bool = True,
                    shift: bool = False):
    if not exist_directory(parent_directory_path):
        print(f'Invalid directory: {parent_directory_path}')

    class_labels = class_labels_to_process if class_labels_to_process else get_subdirectory_names(parent_directory_path)

    feature_matrix = np.array([])
    target_vector = np.array([])

    for class_label in class_labels:
        print(f'Getting Data from {class_label}')
        class_feature_matrix, class_target_vector = get_class_data(parent_directory_path, class_label,
                                                                   number_of_training_examples_per_class, normalize,
                                                                   shift)
        print(f'Loaded all the data from the class')
        if feature_matrix.size:
            feature_matrix = np.vstack((feature_matrix, class_feature_matrix))
            target_vector = np.hstack((target_vector, class_target_vector))
        else:
            feature_matrix = class_feature_matrix
            target_vector = class_target_vector

    return feature_matrix, target_vector

