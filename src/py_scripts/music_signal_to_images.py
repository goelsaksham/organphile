from wav_file_funcs import *
from directory_funcs import *
from keras.preprocessing.sequence import pad_sequences
from typing import List
from PIL.Image import fromarray


def channel_data_to_image(channel_feature_vector, channel_image_path, image_size):
    sound_channel_image_channel_data = channel_feature_vector.reshape(image_size, image_size)
    image_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image_array[:, :, 0] = sound_channel_image_channel_data
    image_array[:, :, 1] = sound_channel_image_channel_data
    image_array[:, :, 2] = sound_channel_image_channel_data
    image = fromarray(image_array)
    image.save(channel_image_path)


def create_audio_file_image(audio_file_path: str, image_file_nm: str, image_output_directory_path: str,
                            output_image_extension: str = '.bmp', image_size: int = 367, normalize: bool = True,
                            normalization_constant: int = 2 ** 16, shift: bool = True, shift_constant: int = 2**15):
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
    left_channel_feature_vector, right_channel_feature_vector = \
        pad_sequences([left_channel_feature_vector, right_channel_feature_vector],
                      image_size**2, padding='post', truncating='post', dtype=np.uint8)
    left_channel_image_path, right_channel_image_path = \
        construct_path(image_output_directory_path, f'{image_file_nm}-left{output_image_extension}'), \
        construct_path(image_output_directory_path, f'{image_file_nm}-right{output_image_extension}')

    channel_data_to_image(left_channel_feature_vector, left_channel_image_path, image_size)
    channel_data_to_image(right_channel_feature_vector, right_channel_image_path, image_size)


def create_images_for_class_data(audio_files_class_subdirectory_path: str, image_files_class_subdirectory_path: str,
                                 output_image_extension: str = '.bmp',  image_size: int = 367,
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
        create_audio_file_image(audio_file_path, file_name.replace('.wav', ''), image_files_class_subdirectory_path,
                                output_image_extension, image_size, normalize, normalization_constant, shift,
                                shift_constant)


def create_images_for_irmas_data(parent_audio_files_directory_path: str, class_labels_to_process: List[str],
                                 parent_image_output_directory_path: str, image_size: int = 367,
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
        create_images_for_class_data(audio_files_class_subdirectory_path, image_files_class_subdirectory_path,
                                     output_image_extension, image_size, number_of_training_examples_per_class,
                                     normalize, normalization_constant, shift, shift_constant)
        print(f'Processed all the data from the class {class_label}')


def main():
    create_images_for_irmas_data('../../data/whole_dataset/training', ['gel', 'pia', 'sax', 'vio', 'voi'],
                                 '../../data/images_jpg',
                                 number_of_training_examples_per_class=0, output_image_extension='.jpg')


if __name__ == '__main__':
    main()
