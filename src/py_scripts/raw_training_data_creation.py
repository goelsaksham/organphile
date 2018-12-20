from .wav_file_funcs import *
from typing import List
from keras.preprocessing.sequence import pad_sequences


def slide_window(data_array: np.array, window_details: tuple):
    if not len(data_array.shape) == 1:
        raise ValueError(f'Invalid shape for the given data: {data_array.shape}')

    window_size, window_advance = window_details
    if (len(data_array) - window_size) % window_advance:
        raise ValueError(f'Invalid window Specifications for the given data: {window_details}')

    num_sliding_windows = ((len(data_array) - window_size) // window_advance) + 1

    return data_array[np.arange(window_size)[None, :] + window_advance*np.arange(num_sliding_windows)[:, None]]


def get_sound_feature_vectors_from_file(file_path: str, sound_signal_window: tuple, sequence_length: int = 132300,
                                        normalize: bool = True, normalization_constant: int = 2**15,
                                        shift: bool = False, shift_constant: int = 0):
    """
    :param file_path:
    :param sound_signal_window:
    :param sequence_length:
    :param normalize:
    :param normalization_constant:
    :param shift:
    :param shift_constant:
    :return:
    """
    sound_signals = get_sound_signals(read_wav_file(file_path))

    if normalize:
        sound_signals = normalize_sound_signals(sound_signals, normalization_constant)
    if shift:
        sound_signals = shift_sound_signals(sound_signals, shift_constant)

    left_channel_feature_vector, right_channel_feature_vector = \
        get_left_channel_data(sound_signals), get_right_channel_data(sound_signals)
    left_channel_feature_vector, right_channel_feature_vector = \
        pad_sequences([left_channel_feature_vector, right_channel_feature_vector],
                      sequence_length, padding='post', truncating='post', dtype=np.float32)

    return list(slide_window(left_channel_feature_vector, sound_signal_window)) + \
           list(slide_window(right_channel_feature_vector, sound_signal_window))


def stack_data(feature_matrix: np.array, left_channel_features: np.array, right_channel_features: np.array):
    if len(feature_matrix) == 0:
        return np.stack((left_channel_features, right_channel_features))
    else:
        return np.vstack((feature_matrix, left_channel_features, right_channel_features))


def get_class_data(parent_directory_path: str, class_label: str, number_of_examples: int = 1,
                   sound_signal_window: tuple = (88200, 44100), sequence_length: int = 132300,
                   normalize: bool = True, normalization_constant: int = 2 ** 15,
                   shift: bool = False, shift_constant: int = 0):
    # Initializing the class feature matrix and target vector
    class_feature_matrix = []
    class_target_vector = []
    # Construct the path
    class_directory_path = construct_path(parent_directory_path, class_label)
    # Make sure the given path is correct
    if not exist_directory(class_directory_path):
        raise ValueError(f'Invalid directory path. Directory name: {class_label}')

    # Get the names of the wav file belonging to the current class
    wav_file_names = get_file_names(class_directory_path, '.wav')
    # Get the subset of the classes, if want only limited number of training examples
    if number_of_examples:
        wav_file_names = wav_file_names[:np.abs(number_of_examples)]

    print(f'Processing: {len(wav_file_names)} files')
    # Iterate through each wav file
    for file_name in wav_file_names:
        file_path = construct_path(class_directory_path, file_name)
        file_sound_features = \
            get_sound_feature_vectors_from_file(file_path, sound_signal_window, sequence_length, normalize,
                                                normalization_constant, shift, shift_constant)
        class_feature_matrix += file_sound_features
        class_target_vector += [class_label]*len(file_sound_features)

    return class_feature_matrix, class_target_vector


def load_irmas_data(parent_directory_path: str, class_labels_to_process: List[str], rnn_window: tuple,
                    number_of_training_examples_per_class: int = 1, sound_signal_window: tuple = (88200, 44100),
                    sequence_length: int = 132300, normalize: bool = True, normalization_constant: int = 2 ** 15,
                    shift: bool = False, shift_constant: int = 0):
    if not exist_directory(parent_directory_path):
        raise ValueError(f'Invalid directory: {parent_directory_path}')

    class_labels = class_labels_to_process if class_labels_to_process else get_subdirectory_names(parent_directory_path)

    feature_matrix = []
    target_vector = []

    for class_label in class_labels:
        print(f'Getting Data from {class_label}')
        class_feature_matrix, class_target_vector = \
            get_class_data(parent_directory_path, class_label, number_of_training_examples_per_class,
                           sound_signal_window, sequence_length, normalize, normalization_constant, shift,
                           shift_constant)
        print(f'Loaded all the data from the class')
        # Adding the already loaded data
        feature_matrix += class_feature_matrix
        target_vector += class_target_vector

    return np.array(list(map(lambda feature: slide_window(feature, rnn_window), feature_matrix))), \
           np.array(target_vector)


def main():
    X, y = load_irmas_data('../../data/whole_dataset/training/', ['cel'], (700, 700))
    print(X.shape)


if __name__ == '__main__':
    main()
