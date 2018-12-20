from wav_file_funcs import *
from directory_funcs import *
from typing import Tuple, List
from keras.preprocessing.sequence import pad_sequences
from PIL.Image import fromarray
import tensorflow as tf


def save_mrp(audio_arr, dimension, image_size, mrp_image_path):
    # Doing 1-D Max Pooling
    N = image_size-5
    window, stride = 2**(N-3), 2**(N-3)
    audio_tensor_max_pooling = tf.nn.max_pool(audio_arr.reshape(1, 1, 1, dimension),
                                              (1, 1, 1, window), (1, 1, 1, stride),
                                              padding='VALID')
    audio_tensor_max_pooling = audio_tensor_max_pooling[0, 0, 0, :]
    audio_tensor_dimension = audio_tensor_max_pooling.shape[0].value
    image_tensor = tf.stack([audio_tensor_max_pooling]*audio_tensor_dimension)
    recurrence_plot = tf.abs(tf.transpose(image_tensor) - image_tensor)
    recurrence_plot_max_pooling = recurrence_plot
    """recurrence_plot_max_pooling = tf.nn.max_pool(tf.reshape(recurrence_plot, (1, audio_tensor_dimension,
                                                                              audio_tensor_dimension, 1)),
                                                 (1, 4, 4, 1), (1, 4, 4, 1), padding='VALID')"""

    image_dimension = recurrence_plot_max_pooling.shape[1].value

    sess = tf.Session()
    with sess.as_default():
        recurrence_plot_image = tf.reshape(recurrence_plot_max_pooling, (image_dimension, image_dimension)).eval()

    print(np.max(recurrence_plot_image))
    recurrence_plot_image_array = np.zeros((image_dimension, image_dimension, 3), dtype=np.uint8)
    recurrence_plot_image_array[:, :, 0] = recurrence_plot_image
    recurrence_plot_image_array[:, :, 1] = recurrence_plot_image
    recurrence_plot_image_array[:, :, 2] = recurrence_plot_image
    image = fromarray(recurrence_plot_image_array)
    image.save(mrp_image_path)


def channel_data_to_mrp(channel_feature_vector, mrp_image_name, image_output_directory_path: str,
                        output_image_extension: str = '.bmp',
                        image_sizes: Tuple[int] = (12, 13, 14, 15, 16, 17)):
    for image_size in image_sizes:
        dimension = 2**image_size
        [audio_data] = pad_sequences([channel_feature_vector], dimension, padding='post', truncating='post')
        save_mrp(audio_data, dimension, image_size,
                 construct_path(image_output_directory_path,
                                f'{mrp_image_name}-{image_size}{output_image_extension}'))


def create_audio_file_mrp(audio_file_path: str, image_file_nm: str, image_output_directory_path: str,
                          output_image_extension: str = '.bmp', image_sizes: Tuple[int] = (12, 13, 14, 15, 16, 17),
                          normalize: bool = True, normalization_constant: int = 2 ** 16, shift: bool = True,
                          shift_constant: int = 2**15):
    if not exist_file(audio_file_path):
        raise ValueError(f'Invalid Audio File Path: {audio_file_path}')

    sound_signals = get_sound_signals(read_wav_file(audio_file_path))
    if normalize:
        sound_signals = normalize_sound_signals(sound_signals, normalization_constant)
    if shift:
        sound_signals = shift_sound_signals(sound_signals, shift_constant)

    sound_signals = sound_signals * 255

    left_channel_feature_vector, right_channel_feature_vector = \
        get_left_channel_data(sound_signals), get_right_channel_data(sound_signals)
    #left_channel_feature_vector, right_channel_feature_vector = \
    #    pad_sequences([left_channel_feature_vector, right_channel_feature_vector],
    #                  image_size**2, padding='post', truncating='post', dtype=np.uint8)
    left_channel_image_name, right_channel_image_name = f'{image_file_nm}-left', f'{image_file_nm}-right'

    channel_data_to_mrp(left_channel_feature_vector, left_channel_image_name, image_output_directory_path,
                        output_image_extension, image_sizes)
    channel_data_to_mrp(right_channel_feature_vector, right_channel_image_name, image_output_directory_path,
                        output_image_extension, image_sizes)


def create_mrp_for_class_data(audio_files_class_subdirectory_path: str, image_files_class_subdirectory_path: str,
                              output_image_extension: str = '.bmp', image_sizes: Tuple[int] = (12, 13, 14, 15, 16, 17),
                              number_of_training_examples_per_class: int = 1, normalize: bool = True,
                              normalization_constant: int = 2 ** 16, shift: bool = True,
                              shift_constant: int = 2**15):
    if not exist_directory(audio_files_class_subdirectory_path):
        raise ValueError(f'Invalid Audio File subdirectory: {audio_files_class_subdirectory_path}')

    if not check_output_directory(image_files_class_subdirectory_path):
        raise ValueError(f'Invalid Image File Output subdirectory: {image_files_class_subdirectory_path}')

    class_files = get_file_names(audio_files_class_subdirectory_path, '.wav')
    class_files = class_files[:np.abs(number_of_training_examples_per_class)] if \
        np.abs(number_of_training_examples_per_class) > 0 else class_files

    print(f'\tProcessing {len(class_files)} files')
    for file_name in class_files:
        audio_file_path = construct_path(audio_files_class_subdirectory_path, file_name)
        create_audio_file_mrp(audio_file_path, file_name.replace('.wav', ''), image_files_class_subdirectory_path,
                              output_image_extension, image_sizes, normalize, normalization_constant, shift,
                              shift_constant)


def create_mrp_for_irmas_data(parent_audio_files_directory_path: str, class_labels_to_process: List[str],
                              parent_image_output_directory_path: str, image_sizes: Tuple[int] = (14, 15, 16, 17),
                              output_image_extension: str = '.bmp',
                              number_of_training_examples_per_class: int = 1, normalize: bool = True,
                              normalization_constant: int = 2 ** 16, shift: bool = True,
                              shift_constant: int = 2**15):
    if not exist_directory(parent_audio_files_directory_path):
        raise ValueError(f'Invalid directory: {parent_audio_files_directory_path}')

    if not check_output_directory(parent_image_output_directory_path):
        raise ValueError(f'Invalid image output directory path: {parent_image_output_directory_path}')

    class_labels = class_labels_to_process if class_labels_to_process else get_subdirectory_names(parent_audio_files_directory_path)

    for class_label in class_labels:
        print(f'Processing Data from {class_label}')
        audio_files_class_subdirectory_path = construct_path(parent_audio_files_directory_path, class_label)
        image_files_class_subdirectory_path = construct_path(parent_image_output_directory_path, class_label)
        create_mrp_for_class_data(audio_files_class_subdirectory_path, image_files_class_subdirectory_path,
                                  output_image_extension, image_sizes, number_of_training_examples_per_class,
                                  normalize, normalization_constant, shift, shift_constant)
        print(f'Processed all the data from the class {class_label}')


def main():
    create_mrp_for_irmas_data('../../data/whole_dataset/training', ['pia', 'sax', 'vio', 'voi'],
                              '../../data/mrp_jpg', image_sizes=(16, 17), normalize=True,
                              normalization_constant=2**15,
                              shift=False,
                              number_of_training_examples_per_class=100, output_image_extension='.jpg')


if __name__ == '__main__':
    main()