import numpy as np
from scipy.io import wavfile
from .directory_funcs import *


def read_wav_file(file_path: str):
    """
    Returns the contents of the wav file after reading it. The contents include the data about both the channels of
    the sound from the file along with the frequency at which the sound data was sampled.

    :param file_path: The path to the wavfile which should be read

    :return: The contents of the wavfile including sampling frequency and the data from both of the channels
    """
    if exist_file(file_path):
        return wavfile.read(file_path)
    else:
        raise ValueError(f'Invalid File Path given for loading data: {file_path}')


def get_sound_signals(wav_file_data: tuple):
    """
    Returns the data only about the sound signals and forget about the sampling frequency

    :param wav_file_data: The data loaded from the sound file. This data includes data about the sampling frequency
    and the sound data.

    :return: Returns only the data about the sound signals for both the channels which has been loaded from the file
    """
    if len(wav_file_data) == 2:
        return wav_file_data[-1]
    else:
        raise ValueError(f'Invalid size of wave file data. Size: {len(wav_file_data)}')


def shift_sound_signals(sound_signal: np.array, shift_constant: int = 0):
    """
    This function shifts the sound signals so that all of the values are positive. This is used so that the data can
    be prepared for the Embedding Layer, which requires the input data to be greater than 0.

    :param sound_signal: The data containing the information about the sound signal.
    :param shift_constant: The constant which should be added to the sound signal. If left to 0, then the function
    just adds the minimum value

    :return: The new array containing the values after they have been shifted.
    """
    if shift_constant:
        return sound_signal + shift_constant
    else:
        return sound_signal + np.min(sound_signal)


def normalize_sound_signals(sound_signals: np.array, normalization_constant: int = (2.**15)):
    """
    This function normalizes the value of the sound signals so that the value of the signal is present in a
    particular range. If the normalization constant is set to 0, then the function would find the maximum absolute
    value among the sound signal and divide by that value.

    :param sound_signals: The array containing data about the sound signal.
    :param normalization_constant: The constant from which we should divide the dataset.

    :return: Returns the normalized data.
    """
    if normalization_constant:
        return sound_signals / normalization_constant
    else:
        return sound_signals / np.max(np.abs(sound_signals))


def get_left_channel_data(sound_signals: np.array):
    """
    Return the data about the left channel from the sound signal

    :param sound_signals: The data loaded from the file containing data about the sound signal

    :return: Returns the left channel data corresponding to the sound signal
    """
    if sound_signals.shape[-1] == 2:
        return sound_signals[:, 0]
    else:
        raise ValueError(f'Invalid Dimensions for the sound signal data: {sound_signals.shape}')


def get_right_channel_data(sound_signals: np.array):
    """
    Return the data about the right channel from the sound signal

    :param sound_signals: The data loaded from the file containing data about the sound signal

    :return: Returns the right channel data corresponding to the sound signal
    """
    if sound_signals.shape[-1] == 2:
        return sound_signals[:, -1]
    else:
        raise ValueError(f'Invalid Dimensions for the sound signal data: {sound_signals.shape}')
