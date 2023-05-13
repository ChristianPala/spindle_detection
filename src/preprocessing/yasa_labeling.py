# Libraries:
import os
import pandas as pd
import yasa
import numpy as np
from config import DATA
from src.preprocessing.load_dataset import DataLoader


# Classes:
class YasaLabeler:
    """
    This class generates the labels for the DREAMS dataset using the YASA library.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        """
        The constructor of the YasaLabeler class.
        """
        self.__data_loader = data_loader
        self.__files = self.__data_loader.get_files()
        self.__file_names = self.__data_loader.get_files()
        self.__sf_list = self.__data_loader.get_frequencies()
        self.__labels = []

    __window_size = 0.5
    # 0.5 seconds window to match with the 0.5 seconds window used in the paper and
    # feature extraction

    def generate_labels(self):
        """
        The method to generate the labels for the DREAMS dataset using the YASA library.
        """
        for idx, file in enumerate(self.__files):
            if idx == 0 or idx == 2:
                file = file.iloc[:, 0].to_numpy()
                channels = ['C3-A1']
                shape_0 = file.shape[0]
            else:
                channels = ['CZ-A1']
                shape_0 = len(file)

            # Detect spindles
            sp = yasa.spindles_detect(file, self.__sf_list[idx], ch_names=channels, remove_outliers=True)

            binary_spindles = np.zeros(shape_0)

            if sp is None:
                # no spindles detected
                pass
            else:
                sp_df = sp.summary()

                starts = (sp_df['Start'] * self.__sf_list[idx]).astype(int)
                ends = (sp_df['End'] * self.__sf_list[idx]).astype(int)
                indices = np.arange(len(file))

                # Loop over the starts and ends and fill in the array for the positive labels, we already
                # do outlier removal in the spindles_detect function
                for start, end in zip(starts, ends):
                    binary_spindles[(indices >= start) & (indices < end)] = 1

            # Create a new dataframe with the same index as the original data
            df_spindles = pd.DataFrame(index=pd.RangeIndex(shape_0))

            # Add a new column with the binary array
            df_spindles['patient_id'] = idx+1
            df_spindles['spindle'] = binary_spindles

            # Down-sample to match the window size
            window_size = int(self.__sf_list[idx] * self.__window_size)
            # if at least 1 spindle is detected, label the window as positive
            df_spindles = df_spindles.groupby(np.arange(len(df_spindles)) // window_size).max()
            # we use max to label the window as positive if at least 1 spindle is detected, taking the
            # mean would be more robust, but we would not be able to compare with the paper.

            self.__labels.append(df_spindles)

        df_labels = pd.concat(self.__labels)

        df_labels.to_csv(os.path.join(DATA, 'yasa_labels.csv'), index=False)


# Driver code
if __name__ == '__main__':
    dl = DataLoader()
    yasl = YasaLabeler(dl)
    yasl.generate_labels()
