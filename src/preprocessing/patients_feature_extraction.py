# Libraries
import glob
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert, welch, spectrogram
from mne.filter import filter_data
import nolds
import mne
from src.preprocessing.load_dataset import DataLoader
from config import DATA, HYPNOGRAMS


# Classes
class NoTargetChannelException(Exception):
    """
    This class is an exception class for the case when the target channel is not found in the raw data.
    """
    def __init__(self):
        super(self)


class Excerpt:
    """
    This class represents an excerpt from the dataset.
    """
    annotation_desc_2_event_id = {'Sleep stage W': 0,
                                  'Sleep stage 1': 1,
                                  'Sleep stage 2': 2,
                                  'Sleep stage 3': 3,
                                  'Sleep stage 4': 3,
                                  'Sleep stage R': 4,
                                  'Movement time': 5}

    CHANNEL_TARGETS = ['CZ-A1', 'C3-A1']  # order also determines priority

    def _set_channel_name(self) -> Exception or int:
        """
        The method to set the channel name of the excerpt.
        """
        for ch in self.CHANNEL_TARGETS:
            if ch in self.__channel_raw.ch_names:
                return self.__channel_raw.ch_names.index(ch)
        raise NoTargetChannelException

    def __init__(self, raw=None, hypno_notations=None, window=0.5) -> None:
        """
        The constructor of the Excerpt class.
        """
        self.__channel_raw = raw
        self.__hypnogram_notations = hypno_notations
        self.__windowed_data = []
        if raw is not None:
            self.__channel_name = self._set_channel_name()
            self.__sampling_freq = raw.info['sfreq']
            self.__channel_data = raw[self.__channel_name][0][0]
            self.__windowed_data = np.array(
                [self.__channel_data[int(window_idx * window * self.__sampling_freq):
                                     int((window_idx + 1) * window * self.__sampling_freq)]
                 for window_idx in range(len(self.__channel_data) // int(self.__sampling_freq * window))])

    def set_data(self, dataloader, idx, window=0.5) -> None:
        """
        The method to set the data of the excerpt.
        @param dataloader: The dataloader object.
        @param idx: The index of the excerpt stored in the dataloader.
        @param window: The window size of the excerpt.
        :return: None
        """
        self.__channel_raw = None
        self.__channel_name = dataloader.get_channels()[idx]
        self.__sampling_freq = dataloader.get_frequencies()[idx]
        if isinstance(dataloader.get_files()[idx], pd.DataFrame):
            self.__channel_data = dataloader.get_files()[idx][self.__channel_name].values
        elif isinstance(dataloader.get_files()[idx], mne.io.edf.edf.RawEDF):
            self.__channel_data = dataloader.get_files()[idx].get_data(picks=self.__channel_name)[0]
        else:
            raise TypeError
        self.__windowed_data=np.array(
            [self.__channel_data[int( window_idx * window * self.__sampling_freq ):
                                 int( (window_idx + 1) * window * self.__sampling_freq )]
             for window_idx in range( len( self.__channel_data ) // int( self.__sampling_freq * window ))] )

    def set_annotation(self):
        pass

    def get_events_from_annotation(self, annotation=None) -> np.ndarray:
        """
        The method to get the events from the annotation.
        @param annotation: The annotation to be used.
        """
        assert self.__hypnogram_notations is not None
        if annotation is None:
            annotation = self.annotation_desc_2_event_id
        events, _ = mne.events_from_annotations(self.__channel_raw, event_id=annotation, chunk_duration=30.)
        return events

    # Phase-amplitude coupling
    def _pac(self, window=4.0) -> List[float]:  # add window parameter for this feature following
        """
        The method to calculate the phase-amplitude coupling of the excerpt in the given time window.
        @param window: The time window to be used.
        :return: The phase-amplitude coupling of the excerpt in the given time window. Note that the length of the
        returned list is 8 times the length of the excerpt since the window size is 0.5 seconds for the other features.
        A length check is required when using this feature with the other features.
        """

        # Recalculate windowed data with new window size
        windowed_data = np.array(
            [self.__channel_data[int(window_idx * window * self.__sampling_freq):
                                 int((window_idx + 1) * window * self.__sampling_freq)]
             for window_idx in range(len(self.__channel_data) // int(self.__sampling_freq * window))])

        pac = []
        for w_data in windowed_data:  # use new windowed_data
            f, t, Sxx = spectrogram(w_data, fs=self.__sampling_freq)
            v = np.sum(Sxx[11:17, :]) / np.sum(Sxx)
            # add the same value 8 times to conform to the other features with 0.5 sec window
            pac.extend([v] * 8)

        return pac

    # Energy ratio
    def _energy_ratio(self, spindle_freq=None) -> List[float]:
        """
        The method to calculate the energy ratio of the excerpt in the given time window.
        @param spindle_freq: The frequency range to be used.
        :return: The energy ratio of the excerpt in the given time window.
        """
        if spindle_freq is None:
            spindle_freq = [11, 16]
        energy_ratio = []
        for w_data in self.__windowed_data:
            total_energy = np.sum(w_data ** 2)
            filtered_data = filter_data(w_data, self.__sampling_freq, l_freq=spindle_freq[0], h_freq=spindle_freq[1])
            spindle_energy = np.sum(filtered_data ** 2)
            energy_ratio.append(spindle_energy / total_energy)
        return energy_ratio

    # Peak power
    def _peak_power(self, spindle_freq: List[int] = None) -> List[float]:
        """
        The method to calculate the peak power of the excerpt in the given time window.
        @param spindle_freq: The frequency range to be used.
        :return: The peak power of the excerpt in the given time window.
        """
        if spindle_freq is None:
            spindle_freq = [11, 14]  # 14 instead of 16 as in the paper, but this is a bit strange.
        power_peak = []
        for w_data in self.__windowed_data:
            filtered_data = filter_data(w_data, self.__sampling_freq, l_freq=spindle_freq[0], h_freq=spindle_freq[1])
            frequencies, power = welch(filtered_data, self.__sampling_freq)
            freq_indices = np.where((frequencies >= spindle_freq[0]) & (frequencies <= spindle_freq[1]))
            power_peak.append(np.max(power[freq_indices]))
        return power_peak

    # Power ratio
    def _power_ratio(self, spindle_freq: List[int] = None, low_freq: List[float] = None) -> List[float]:
        """
        The method to calculate the power ratio of the excerpt in the given time window.
        @param spindle_freq: The frequency range to be used.
        @param low_freq: The frequency range to compare the spindle frequency to.
        :return: The power ratio of the excerpt in the given time window.
        """
        if spindle_freq is None:
            spindle_freq = [11, 14]
        if low_freq is None:
            low_freq = [0.3, 8.]
        power_ratio = []
        for w_data in self.__windowed_data:
            frequencies, power = welch(w_data, self.__sampling_freq)
            spindle_freq_indices = np.where((frequencies >= spindle_freq[0]) & (frequencies <= spindle_freq[1]))
            low_freq_indices = np.where((frequencies >= low_freq[0]) & (frequencies <= low_freq[1]))
            total_power_spindle = np.sum(power[spindle_freq_indices])
            total_power_low = np.sum(power[low_freq_indices])
            power_ratio.append(total_power_spindle / total_power_low)
        return power_ratio

    # Instantaneous frequency
    # Note after discussing how to handle this feature with Giuliana Monachino and Beatrice Zanchi, we
    # select the mean value of the instantaneous frequency as the feature value.
    def _mean_frequency(self, spindle_freq: List[int] = None) -> List[np.array]:
        """
        The method to calculate the instantaneous frequency of the excerpt in the given time window.
        @param spindle_freq: The frequency range to be used.
        :return: The instantaneous frequency of the excerpt in the given time window. Note that this is a list of
        numpy arrays, since the instantaneous frequency is calculated for each point in time.
        """
        if spindle_freq is None:
            spindle_freq = [11, 16]
        mean_freq = []
        for w_data in self.__windowed_data:
            filtered_data = filter_data(w_data, self.__sampling_freq, l_freq=spindle_freq[0], h_freq=spindle_freq[1])
            analytic_signal = hilbert(filtered_data)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi)) * self.__sampling_freq
            # modified from the paper. We take the mean of the instantaneous frequency as the feature value.
            mean_inst_freq = np.mean(instantaneous_frequency)
            mean_freq.append(mean_inst_freq)
        return mean_freq

    # Zero crossing rate
    def _zero_crossing_rate(self) -> List[int]:
        """
        The method to calculate the zero crossing rate of the excerpt in the given time window.
        :return: The zero crossing rate of the excerpt in the given time window.
        """
        zcr = []
        for w_data in self.__windowed_data:
            zero_crossings = np.where(np.diff(np.sign(w_data)))[0]
            zcr.append(len(zero_crossings))
        return zcr

    import numpy as np

    def _hjorth(self, window=0.5) -> List[Tuple[float, float]]:
            """
            The method to calculate the Hjorth parameters
            :return: A list of tuples where each tuple represents the Hjorth parameters (mobility, complexity)
            of the excerpt in the given time window.
            """

            hjorth_params = []
            for w_data in self.__windowed_data:
                # Calculate the first derivative of the data
                first_derivative = np.diff(w_data)

                # Calculate the second derivative of the data
                second_derivative = np.diff(w_data, 2)

                # Calculate the variances
                var_zero = np.mean(w_data ** 2)
                var_d1 = np.mean(first_derivative ** 2)
                var_d2 = np.mean(second_derivative ** 2)

                # Compute the Hjorth parameters
                mobility = np.sqrt(var_d1 / var_zero)
                complexity = np.sqrt(var_d2 / var_d1) / mobility

                hjorth_params.append((mobility, complexity))

            return hjorth_params

    def extract_features(self):
        """
        The method to extract the features from the excerpt.
        :return: A dictionary containing the features.
        """
        val_dict = {}
        # Sample Entropy
        samp_entropy = [nolds.sampen(w_data) for w_data in self.__windowed_data]
        val_dict['sample_entropy'] = samp_entropy
        # Maximum
        max_value = [np.max(w_data) for w_data in self.__windowed_data]
        val_dict['max_value'] = max_value
        # Minimum
        min_value = [np.min(w_data) for w_data in self.__windowed_data]
        val_dict['min_value'] = min_value
        # Variance
        variance = [np.var(w_data) for w_data in self.__windowed_data]
        val_dict['variance'] = variance
        # Standard deviation
        std_deviation = [np.std(w_data) for w_data in self.__windowed_data]
        val_dict['std_deviation'] = std_deviation
        # Phase-amplitude coupling
        phase_amp_coupling = self._pac()
        # Check the length of phase_amp_coupling is the same as the other features, otherwise pad with the last
        # value or truncate to the same length
        if len(phase_amp_coupling) < len(std_deviation):
            phase_amp_coupling.extend([phase_amp_coupling[-1]] * (len(std_deviation) - len(phase_amp_coupling)))
        elif len(phase_amp_coupling) > len(std_deviation):
            phase_amp_coupling = phase_amp_coupling[:len(std_deviation)]
        val_dict['phase_amp_coupling'] = phase_amp_coupling
        # Instantaneous frequency, changed to mean instantaneous frequency in the window
        inst_freq = self._mean_frequency()
        val_dict['mean_freq'] = inst_freq  # This is a list of arrays
        # Energy ratio
        energy_ratio = self._energy_ratio()
        val_dict['energy_ratio'] = energy_ratio
        # Kurtosis
        kurtosis_val = [stats.kurtosis(w_data) for w_data in self.__windowed_data]
        val_dict['kurtosis'] = kurtosis_val
        # Skewness
        skewness_val = [stats.skew(w_data) for w_data in self.__windowed_data]
        val_dict['skewness'] = skewness_val
        # Peak power
        power_peak = self._peak_power()
        val_dict['power_peak'] = power_peak
        # Power ratio
        power_ratio = self._power_ratio()
        val_dict['power_ratio'] = power_ratio
        # Interquartile range
        iqr = [stats.iqr(w_data) for w_data in self.__windowed_data]
        val_dict['iqr'] = iqr
        # zero crossing rate
        zcr = self._zero_crossing_rate()
        val_dict['zrc'] = zcr
        # Hjorth parameters, we tried to add the Hjorth parameters to the feature set since YASA uses them
        # and we have seen it is SOTA in the sleep staging task.
        hjorth_params = self._hjorth()
        mobility = [param[0] for param in hjorth_params]
        complexity = [param[1] for param in hjorth_params]
        val_dict['mobility'] = mobility
        val_dict['complexity'] = complexity

        return pd.DataFrame(val_dict)
    ####################################


class DataProcessor:
    """
    The class to process the data, extract the features and save the features and labels to csv files.
    """
    def __init__(self, data_loader, data_path) -> None:
        """
        The constructor of the class.
        """
        self.data_loader = data_loader
        self.data_path = data_path
        self.X = []
        self.y = []
        self.dfs = []

    # class constants
    __number_of_patients = 8
    __window_size = 0.5  # in seconds

    def process_data(self) -> None:
        """
        The method to process the data, extract the features and save the features and labels to csv files.
        """
        self._extract_features()
        self._load_and_process_labels()
        self._combine_data()
        self._save_data()

    def _extract_features(self, annotation=True) -> None:
        """
        Auxiliary method to extract the features from the data, uses composition with the Excerpt class.
        """
        for idx in range(DataProcessor.__number_of_patients):
            patient_id = idx + 1
            p = Excerpt()
            p.set_data(self.data_loader, idx)
            df = p.extract_features()
            df['patient_id'] = patient_id
            if annotation:
                annotation_file = os.path.join(HYPNOGRAMS,f'hypnogram_excerpt{patient_id}.txt')
                if os.path.exists(annotation_file):
                    hypnogram=pd.read_csv( annotation_file ).iloc[:, 0].to_numpy()
                    df['hypnogram']=np.array([[val]*(df.shape[0]//len(hypnogram)) for val in hypnogram]).flatten()
            self.dfs.append(df)

    def _load_and_process_labels(self):
        """
        Auxiliary method to load and process the labels.
        """
        # load the labels for the patients
        for i in range(DataProcessor.__number_of_patients):
            automatic_excerpt_path = os.path.join(self.data_path, f'Automatic_detection_excerpt{i + 1}.txt')
            visual_scoring1_path = os.path.join(self.data_path, f'Visual_scoring1_excerpt{i + 1}.txt')
            visual_scoring2_path = os.path.join(self.data_path, f'Visual_scoring2_excerpt{i + 1}.txt')

            # automatic detection
            ad = np.loadtxt(automatic_excerpt_path, skiprows=1)
            # visual scoring expert 1
            vs1 = np.loadtxt(visual_scoring1_path, skiprows=1)

            # visual scoring expert 2 for the first 6 patients
            if i < 6:
                vs2 = np.loadtxt(visual_scoring2_path, skiprows=1)
                concatenated = np.vstack((ad, vs1, vs2))
            else:
                concatenated = np.vstack((ad, vs1))

            sorted_indices = np.argsort(concatenated[:, 0])
            data = concatenated[sorted_indices]

            # create the labels for the patients
            y_patient = np.zeros(360000)

            start_indices = (self.data_loader.get_frequencies()[i] * data[:, 0]).astype(int)
            duration_indices = (self.data_loader.get_frequencies()[i] * data[:, 1]).astype(int)

            # set the labels to 1 for the indices that are in the range of the start and duration
            for start, duration in zip(start_indices, duration_indices):
                y_patient[(start <= np.arange(y_patient.shape[0])) &
                          (np.arange(y_patient.shape[0]) < start + duration)] = 1

            # Down-sample to match the window size
            window_size = int(self.data_loader.get_frequencies()[i] * DataProcessor.__window_size)
            # Here we can choose to downsample by taking the mean for a more robust approach or
            # by taking the max, due to data scarcity with the positive cases, we chose the max.
            y_patient_downsampled = y_patient.reshape(-1, window_size).max(axis=1)

            self.y.append(y_patient_downsampled)

    def _combine_data(self):
        self.X = np.concatenate(self.dfs)
        self.y = np.concatenate(self.y)

    def _save_data(self):
        features_file_name = os.path.join(self.data_path, 'features.csv')
        target_file_name = os.path.join(self.data_path, 'target.csv')
        features = pd.concat(self.dfs)
        features.to_csv(features_file_name, index=False)
        labels = pd.DataFrame(self.y, columns=["spindle"])
        labels['patient_id'] = features.reset_index()['patient_id']
        labels.to_csv(target_file_name, index=False)

    def get_labels(self):
        return self.y


# Driver code
if __name__ == '__main__':
    dl = DataLoader()
    dp = DataProcessor(dl, DATA)
    dp.process_data()
