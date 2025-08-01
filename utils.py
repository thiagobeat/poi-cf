from datetime import datetime
import pandas as pd


def save_pickle_file(file, path_name, file_name):
    """
    Saves the pickle file to disk
    :param file: must be a pandas pickable file
    :param path_name: path to save
    :param file_name: file name to be concatenated with the current timestamp
    :return: nothing
    """
    file.to_pickle(path_name + file_name + datetime.now().strftime("%Y%m%d-%H%M%S") + '.pkl')
