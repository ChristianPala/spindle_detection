# libraries:
import os
from typing import Any, List, Tuple
import mne
import pandas as pd
from scipy import signal
from config import EXCERPTS


# Classes:
class DataLoader:
    """
    This class loads the dataset from the data folder and provides the required information about the dataset for
    the other classes.
    """
    def __init__(self) -> None:
        """
        The constructor of the Dataloader class.
        """
        self.__data_path = EXCERPTS
        self.__data_files = os.listdir(self.__data_path)
        self.__files, self.__channels, self.__frequencies = self.load_edf()
        # insert the txt files 1 and 3 at position 0 and 2 respectively
        txt_files = self.load_txt()
        # manually insert the txt files at the correct position for the edf files causing issues
        self.__files.insert(0, txt_files[0])
        self.__files.insert(2, txt_files[1])
        self.excerpts = []
    def load_txt(self) -> List[pd.DataFrame]:
        """
        The method to load the txt files from the data folder, creates pandas dataframes from them and
        returns the list of the dataframes, the list of the channels and the list of the sampling frequencies.
        :return: the list of txt files
        """
        file_names = ['excerpt1.txt', 'excerpt3.txt']
        original_frequencies = [100, 50]
        files = []
        for i, file in enumerate(file_names):
            df = pd.read_csv(os.path.join(self.__data_path, file))

            # Convert DataFrame column to numpy array and resample to 200 Hz
            data = df.iloc[:, 0].to_numpy()
            new_length = int(data.shape[0] * 200 / original_frequencies[i])
            resampled_data = signal.resample(data, new_length)

            # Replace column with resampled data
            df = pd.DataFrame(resampled_data, columns=df.columns)

            # Apply FIR filter between 0.3 and 35 Hz
            sos = signal.butter(10, [0.3, 35], 'bp', fs=200, output='sos')
            df[df.columns[0]] = signal.sosfilt(sos, df[df.columns[0]])

            files.append(df)

        return files

    def load_edf(self) -> Tuple[Any, List[str], List[int]]:
        """
        The method to load the edf files from the data folder, creates pandas dataframes from them and
        returns the list of the dataframes, the list of the channels and the list of the sampling frequencies.
        :return: the list of edf files
        """
        file_names = [file for file in self.__data_files if file.endswith('.edf')]
        files = [mne.io.read_raw_edf(os.path.join(self.__data_path, file), preload=True) for file in file_names if
                 '1' not in file and '3' not in file]
        channels = ['[C3-A1]', 'CZ-A1', '[C3-A1]', 'CZ-A1', 'CZ-A1', 'CZ-A1', 'CZ-A1', 'CZ-A1']
        files = [file.resample(200) for file in files]
        files = [file.filter(0.3, 35) for file in files]
        frequencies = [200] * 8
        return files, channels, frequencies

    # getters
    def get_files(self) -> List[pd.DataFrame] or List[Any]:
        """
        The getter of the files.
        """
        return self.__files

    def get_channels(self) -> List[str]:
        """
        The getter of the channels.
        """
        return self.__channels

    def get_frequencies(self) -> List[int]:
        """
        The getter of the sampling frequencies.
        """
        return self.__frequencies


if __name__ == '__main__':
    dl = DataLoader()
    print(dl.get_files()[0].head())
    assert len(dl.get_files()) == 8
    assert len(dl.get_channels()) == 8
    assert len(dl.get_frequencies()) == 8
    print('Test passed.')
    for file in dl.get_files():
        print("file name: ", file)
        if isinstance(file, pd.DataFrame):
            assert file.shape[0] == 360000
        else:
            # assert length, print error message if not
            assert file.n_times == 360000