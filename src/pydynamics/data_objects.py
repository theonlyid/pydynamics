"""
This submodule contains the class definitions and methods for loading and working with NIRS experiments and events.

Author: Ali Zaidi

Date: 20-June-2024
"""

import pandas as pd
import numpy as np
from scipy import signal, fft, stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import re


class Experiment:
    """
    A class for loading and manipulating timeseries data from a CSV file. Has methods to allow resampling, filtering, etc.
    """

    def __init__(self, source, sampling_rate, auto_resample=True, extract_ne=True, fs=100):
        """
        Initialize the Experiment object with data from a source.

        Arguments
        ---------
            source: str or pd.DataFrame
                The source of the data, either a file path or a pandas DataFrame.
            auto_resample: bool (default=True)
                Whether to automatically resample the data.
            extract_ne: bool (default=True)
                Whether to extract NE dose information.
            fs: int (default=100)
                The sampling frequency of the data.
        """
        self._log = []
        self.__log("Instantiating Experiment object")
        self.fs = fs
        self.filter_params = None
        if isinstance(source, pd.DataFrame):
            self.data, self.comments = self.from_dataframe(source)
            self.filename = f"{source}"
        else:
            self.filename = source
            self.original_filepath = source
            self.data, self.comments = self.load_file(source)
        self._data = self.data.copy()
        if extract_ne:
            self.extract_NE()
        if auto_resample:
            self.resample(sampling_rate)

    def __log(self, statement):
        """
        Append a log statement to the internal log and print it.

        Arguments
        ---------
            statement: str
                The log statement to be appended and printed.
        """
        self._log.append(statement)
        print(statement)

    def get_log(self):
        """
        Return the log of the experiment object.

        Returns
        -------
            log: list
                A list of log statements.
        """
        return self._log

    def load_file(self, filename: str, replace_data=True) -> list[pd.DataFrame]:
        """
        Load a CSV file and convert it to a pandas DataFrame.

        Arguments
        ---------
            filename: str
                The path to the CSV file to be loaded.
            replace_data: bool (default=True)
                Whether to replace the existing data with the loaded data.

        Returns
        -------
            data: pd.DataFrame
                The loaded data as a pandas DataFrame.
            comments: pd.Series
                The comments extracted from the data.
        """
        self.__log(f"Loading {filename}")
        data = pd.read_csv(filename, low_memory=False)
        
        comments_col = next((col for col in data.columns if col.lower() == 'comments'),None)
        if comments_col is None:
            raise KeyError("No 'comments' column found.")

        comments = data.pop(comments_col)

        print(data.columns)

        if "datetime" in data.columns:
            data.pop('datetime')
        if "Timestamps" in data.columns:
            data.pop('Timestamps')

        #comments = comments[~comments.isna()].copy()
        comments = comments.copy()
        return data, comments
    
    def from_dataframe(self, data: pd.DataFrame, fs: float = 1.0, replace_data=True) -> list[pd.DataFrame]:
        """
        Create an Experiment object from a pandas DataFrame.

        Arguments
        ---------
            data: pd.DataFrame
                The data to be used for creating the Experiment object.
            fs: float (default=1.0)
                The sampling frequency of the data.
            replace_data: bool (default=True)
                Whether to replace the existing data with the provided data.

        Returns
        -------
            data: pd.DataFrame
                The processed data as a pandas DataFrame.
            comments: pd.Series
                The comments extracted from the data.
        """

        # Run some basic checks on the data
        if 'comments' not in data.columns:
            raise ValueError("Dataframe must contain a 'comments' column.")
        
        self.__log(f"Generating from dataframe: {data}")
        self.data = data
        comments_col = next((col for col in data.columns if col.lower() == 'comments'),None)
        if comments_col is None:
            raise KeyError("No 'comments' column.")

        comments = data.pop(comments_col)
        comments = comments[~comments.isna()].copy()

        if "datetime" in data.columns:
            data.pop('datetime')

        if "Timestamps" in data.columns:
            data.pop('Timestamps')
        
        if "ts" not in data.columns:
            data["ts"] = np.arange(0, 1/fs, data.shape[0]/fs)
            
        return data, comments
    
    def add_comment(self, index, new_comment):
        """
        Update or add a comment at the first occurrence of the specified index
        in self.comments Series (even if there are duplicates).

        Parameters
        ----------
        index : int
            The index where the comment should be added or updated.
        new_comment : str
            The new comment string to set.
        """
        if index not in self.comments.index:
            raise IndexError(
                f"Index {index} is out of bounds. Valid indices are from {self.comments.index.min()} to {self.comments.index.max()}."
            )

        matches = (self.comments.index == index).nonzero()[0]
        if len(matches) == 0:
            raise IndexError(f"Index {index} exists but couldn't find its position (unexpected).")

        first_pos = matches[0]

        old_comment = self.comments.iloc[first_pos]
        self.comments.iloc[first_pos] = new_comment
        self.__log(f"Updated first occurrence of index {index}: '{old_comment}' → '{new_comment}'")



    def delete_comment(self, index):
        """
        Delete comments at all occurrences of the specified index by setting them to empty strings
        in the self.comments Series.

        Parameters
        ----------
        index : int
            The index where comments should be deleted.

        Raises
        ------
        IndexError
            If the index is outside the range of the comments Series.
        """
        if index not in self.comments.index:
            raise IndexError(
                f"Index {index} is out of bounds. Valid indices are from {self.comments.index.min()} to {self.comments.index.max()}."
            )

        matches = (self.comments.index == index).nonzero()[0]
        if len(matches) == 0:
            self.__log(f"No comments found at index {index}; nothing to delete.")
            return

        for pos in matches:
            old_comment = self.comments.iloc[pos]
            self.comments.iloc[pos] = ""



    def save_comments(self, filepath=None):
        """
        Replace the existing comments column (case-insensitive) in the original file with updated comments,
        preserving all other columns.
        """
        if filepath is None:
            if not hasattr(self, 'original_filepath'):
                raise ValueError("No filepath specified and no original file path stored in the object.")
            filepath = self.original_filepath

        original_df = pd.read_csv(self.original_filepath, low_memory=False, index_col=False)

        comments_col = next(
            (col for col in original_df.columns if col.lower() == 'comments'),
            None
        )
        if comments_col is None:
            raise KeyError("No 'comments' column found in the original file (case-insensitive search failed).")

        if len(original_df) != len(self.comments):
            raise ValueError(
                f"Length mismatch: original file has {len(original_df)} rows, "
                f"but comments Series has {len(self.comments)} rows."
            )

        original_df[comments_col] = self.comments.reset_index(drop=True)

        original_df.to_csv(filepath, index=False)
        self.__log(f"Updated comments saved to {filepath}")



    def extract_NE(self):
        """
        Crawl through the comments to isolate NE-dose and generate its timeseries, appending 'NE dose' to self.data.
        """
        self.__log("Extracting NE")
        tmp = self.comments.str.extract(r'(?P<event>NE.+)(?P<dose>\d\.\d{2}|Off)')
        tmp = tmp[~tmp.iloc[:, 0].isna()]

        if tmp.empty:
            self.__log("No NE events found in comments; skipping NE extraction.")
            return
        
        tmp['dose'] = pd.to_numeric(tmp['dose'], errors='coerce').fillna(0.0)

        tmp.iloc[-1, 0] = 'NE Off'
        tmp.iloc[-1, -1] = 0
        ne_dose = np.zeros((self.data.shape[0],))
        for i in range(len(tmp)):
            pos = tmp.index[i]
            ne_dose[pos:] = tmp['dose'].iloc[i]
        self.data['NE dose'] = ne_dose

    def get_challenges(self):
        """
        Crawl through comments to identify various events in the data (Hypoxia, Baseline, MAP, etc.) and return indices.

        Returns
        -------
            challenges: pd.DataFrame
                A DataFrame containing the identified challenges with their start and end indices and durations.
        """
        self.__log("Extracting physiological challenges")
        challenges = []
        #tmp = self.comments.str.extract(r" (?P<challenge>hypoxia.\d*|NE|BSL|SNP).*(?P<type>start|end|off|stop)", flags=re.IGNORECASE, expand=False)
        tmp = self.comments.str.extract(r"(?P<challenge>hypoxia(?:\s*\d+)?|(?-i:NE)|BSL|SNP|MAP)[\s\S]*?\b(?P<type>start|end|off|stop)\b", flags=re.IGNORECASE, expand=False)
        tmp = tmp[~tmp['challenge'].isna()]
        #for challenge in tmp.challenge.unique():
        #    if len(tmp[tmp['challenge'] == challenge].index):
        #        start, end = tmp[tmp['challenge'] == challenge].index
        #    challenges.append([challenge, start, end, end - start + 1])
        for challenge in tmp.challenge.dropna().unique():
            events = tmp[tmp['challenge'].str.lower() == challenge.lower()]

            start_indices = events[events['type'].str.lower() == 'start'].index
            end_indices = events[events['type'].str.lower().isin(['end', 'off', 'stop'])].index

            if len(start_indices) == 0 or len(end_indices) == 0:
                raise ValueError(
                    f"Challenge '{challenge}' has incomplete events: "
                    f"found {len(start_indices)} start(s) and {len(end_indices)} end(s). "
                    f"Check your comments for missing or malformed annotations."
                )

            start = start_indices[0]
            end = end_indices[-1]

            challenges.append([challenge, start, end, end - start + 1])


        challenges = pd.DataFrame(challenges, columns=['challenge', 'start', 'end', 'duration'])
        self.challenges = challenges
        return self.challenges
    
    def challenges_to_events(self, pre=60, post=60):
        """
        Convert identified challenges to Event objects.

        Arguments
        ---------
            pre: int (default=60)
                The number of seconds to include before the start of each challenge.
            post: int (default=60)
                The number of seconds to include after the end of each challenge.

        Returns
        -------
            events: dict
                A dictionary of Event objects for each identified challenge.
        """
        events = {}
        for i in range(len(self.challenges.challenge)):
            tmp = self.challenges.iloc[i, :]
            tmp_challenge = Event(self, tmp.start, tmp.end, pre=pre, post=post)
            tmp_challenge.calc_features()
            events[tmp.challenge] = tmp_challenge
        self.events = events
        return events
        
    def resample(self, fs_new):
        """
        Resample data to the given sampling rate.

        Arguments
        ---------
            fs_new: float
                The new sampling rate to resample the data to.
        """
        self.__log(f"Resampling from {self.fs:0.2f} to {fs_new:0.2f} Hz")
        nsamples = np.int16(self.data.shape[0] * fs_new // self.fs)
        d_np = self.data.to_numpy()
        d_r = signal.resample(d_np, nsamples, axis=0)
        df_r = pd.DataFrame(d_r, columns=self.data.columns)
        ts = np.arange(0, d_r.shape[0] / fs_new, 1 / fs_new)
        df_r['ts'] = ts
        self.data = df_r
        self._comments = self.comments.copy()

        _comments = self.comments.copy()
        _comments.index = _comments.index.to_numpy() * fs_new // self.fs
        self.comments = _comments
        self.fs = fs_new

    def get_fft(self, n=2048):
        """
        Generate the Fast Fourier Transform (FFT) of the data.

        Arguments
        ---------
            n: int (default=2048)
                The number of points to use for the FFT.
        """
        self.__log(f"Generating FFT with fs={self.fs}")
        fx = fft.fft(self.data.iloc[:, 1:].to_numpy(), n, axis=0)
        xf = fft.fftfreq(n, 1 / self.fs)

        fft_df = pd.DataFrame(fx[:len(xf) // 2, :], columns=self.data.columns[1:])
        fft_df['f'] = xf[:len(xf) // 2]
        self.fft = fft_df

    def butter(self, order=2, wn=1/10, fs=1, btype='lowpass'):
        """
        Generate a Butterworth filter.

        Arguments
        ---------
            order: int (default=2)
                The order of the filter.
            wn: float (default=1/10)
                The critical frequency.
            fs: float (default=1)
                The sampling frequency.
            btype: str (default='lowpass')
                The type of filter to generate ('lowpass', 'highpass', 'bandpass', 'bandstop').
        """
        b, a = signal.butter(order, wn, fs=fs, btype=btype)
        self.filter_params = {'type': 'butter', 'order': order, 'Wn': wn, 'fs': fs, 'btype': btype, 'b': b, 'a': a}
        self.__log(f"Generated Butterworth filter. Params:{self.filter_params}")

    def filter(self):
        """
        Apply the generated filter to the data.
        """
        if self.filter_params is None:
            print("[ERROR] First call Experiment.butter() to generate a filter before filtering data.")
            return

        self.__log(f"Filtering data with {self.filter_params['btype']} {self.filter_params['Wn']} Hz")

        data = self.data.to_numpy()
        ts = data[:, 0]

        if self.filter_params['btype'] != 'lowpass':
            data = data - data.mean(axis=0)

        b = self.filter_params['b']
        a = self.filter_params['a']

        filt_data = signal.filtfilt(b, a, data, axis=0)
        filt_df = pd.DataFrame(filt_data, columns=self.data.columns)
        filt_df['ts'] = ts
        self.data = filt_df
        self.extract_NE()

    def reset(self):
        """
        Reset data to the original dataframe generated when the object was instantiated.
        """
        self.__log("Resetting data to original values loaded from file.")
        self.data = self._data.copy()
        self.comments = self._comments.copy()
        self.filter_params = None
        self.fs = 100

    def save(self, filename):
        """
        Save the object as a pickled file.

        Arguments
        ---------
            filename: str
                The path to save the pickled object.
        """
        self.__log(f"Saving object as {filename}")
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self):
        return f"Dataset loaded from {self.filename} with {self.data.shape[0]} samples and fs={self.fs}"


class Event:
    """
    A class for working with instances of physiological challenges such as hypoxias, MAP manipulations, etc.
    Has methods that enable the estimation of features such as the area under the curve, cumulative sums, and sigmoid fits.
    """
    def __init__(self, dataset: Experiment = None, start=0, stop=-1, pre=0, post=0, detrend=False):
        """
        Initialize the Event object with data from an Experiment object.

        Arguments
        ---------
            dataset: Experiment
                The Experiment object containing the data.
            start: int (default=0)
                The start index of the event.
            stop: int (default=-1)
                The stop index of the event.
            pre: int (default=0)
                The number of seconds to include before the start of the event.
            post: int (default=0)
                The number of seconds to include after the end of the event.
            detrend: bool (default=False)
                Whether to detrend the data.
        """
        print("Instantiating Event object")
        self.fs = dataset.fs
        data = dataset.data.iloc[start - pre * self.fs:stop + post * self.fs, :].copy()
        data = data - data.iloc[:pre, :].mean(axis=0)
        print(data)
        self.init_params = {'pre': pre, 'start': start, 'stop': stop, 'post': post, 'detrend': detrend, 'Experiment': dataset.__repr__()}
        data['ts'] = data['ts'] - data['ts'].iloc[pre]
        data.set_index('ts', inplace=True, drop=False)
        self.data = data
        self.duration = (stop - start + 1) / self.fs
        self.calc_features()
        if detrend:
            self.detrend()

    def detrend(self):
        """
        Detrend the data using a linear detrend.
        """
        _d = signal.detrend(self.data.to_numpy(), axis=0, type='linear')
        self.data.iloc[:, 1:] = _d[:, 1:]

    def get_fft(self, n=2048):
        """
        Generate the Fast Fourier Transform (FFT) of the event data.

        Arguments
        ---------
            n: int (default=2048)
                The number of points to use for the FFT.
        """
        self.log(f"Generating FFT with fs={self.fs}")
        fx = fft.fft(self.data.iloc[:, 1:].to_numpy(), n, axis=0)
        xf = fft.fftfreq(n, 1 / self.fs)

        fft_df = pd.DataFrame(fx[:len(xf) // 2, :], columns=self.data.columns[2:])
        fft_df['f'] = xf[:len(xf) // 2]
        self.fft = fft_df

    def calc_features(self):
        """
        Calculate features such as the sum and cumulative sum of the event data.
        """
        z_data = (self.data - self.data.iloc[:self.init_params['pre'], :].mean(axis=0)).copy()
        f_sum = z_data.sum(axis=0).copy() / self.fs
        f_cumsum = z_data.cumsum(axis=0).copy() / self.fs
        self.features = {'data': z_data, 'sum': f_sum, 'cumsum': f_cumsum}

    def save(self, object_name):
        """
        Save the Event object as a pickled file.

        Arguments
        ---------
            object_name: str
                The path to save the pickled Event object.
        """
        with open(object_name, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self):
        return "Event object"


