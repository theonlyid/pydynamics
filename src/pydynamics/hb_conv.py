"""
Module housing methods for calculating hb-concentration changes and TOI from raw NIRS data.

Author: Ali Zaidi

Date: 05-OCT-2024
"""

import numpy as np
import pandas as pd
# from nirs_types import *


class NIRSData:
    """
    Object for storing ambient corrected PD reads, hb-concentration changes and TOI for a single NIRS channel.

    Attributes
    ----------      
        raw_abs: np.ndarray (shape=(12, nobs))
            The ambient corrected raw absorbance data.

        hbc_near: np.ndarray (shape=(4, nobs))
            The converted hb-concentration changes for the near channel.

        hbc_far: np.ndarray (shape=(4, nobs))
            The converted hb-concentration changes for the far channel.
        
        toi: np.ndarray
            The TOI calculated from the raw absorbance data (after ambient subtraction).

    """
    def __init__(self, raw_abs: np.ndarray = None, hbc_near: np.ndarray = None, hbc_far: np.ndarray = None, toi: np.ndarray = None):
        self.raw_abs = raw_abs
        self.hbc_near = hbc_near
        self.hbc_far = hbc_far
        self.toi = toi

    def __repr__(self) -> str:
        return f"NIRSData(shape={self.raw_abs.shape})"


class SVDCleaned:
    """
    Object containing the results of the SVD analysis and cleaned absorbance data.
    
    Attributes
    ----------
    uvar: np.ndarray
        The variance of each column of u. The first value being less than 0.1 is a good indicator of reliable cleaning.

    noise: np.ndarray
        The noise estimated from the SVD analysis.

    clean_abs: np.ndarray

    """
    def __init__(self, uvar: np.ndarray = None, noise: np.ndarray = None, clean_abs: np.ndarray = None,
                 comp2: np.ndarray = None, comp3: np.ndarray = None):
        self.uvar = uvar
        self.noise = noise
        self.comp2 = comp2
        self.comp3 = comp3
        self.clean_abs = clean_abs

    def __repr__(self) -> str:
        return f"SVDCleaned(uvar={self.uvar[0]:0.03f})"


class SVDDiagnostics:
    """
    Object to store the SVD cleaning results and diagnostics for both channels.
    
    Attributes
    ----------
    near: SVDCleaned
        The SVD cleaning results for the near channel.
    far: SVDCleaned
        The SVD cleaning results for the far channel.
    ods: SVDCleaned
        The SVD cleaning results for the optical density gradients.

    """
    def __init__(self, near: SVDCleaned = None, far: SVDCleaned = None, ods: SVDCleaned = None):
        self.near = near
        self.far = far
        self.ods = ods

    def __repr__(self) -> str:
        return f"SVDDiagnostics(near={self.near}, far={self.far}, ods={self.ods})"


class NIRSDataCleaned(NIRSData):
    """
    Object for storing ambient corrected PD reads, hb-concentration changes and TOI for a single NIRS channel.

    Attributes
    ----------      
        raw_abs: np.ndarray (shape=(12, nobs))
            The ambient corrected raw absorbance data.

        hbc_near: np.ndarray (shape=(4, nobs))
            The converted hb-concentration changes for the near channel.

        hbc_far: np.ndarray (shape=(4, nobs))
            The converted hb-concentration changes for the far channel.

        toi_abs: np.ndarray
            The TOI calculated from the raw absorbance data (after ambient subtraction).

        toi_ods: np.ndarray
            The TOI calculated after cleaning od_gradient data.

        svd_diagnostics: SVDDiagnostics
            The SVD cleaning results and diagnostics for both channels and optical density gradients.

    """
    def __init__(self, raw_abs: np.ndarray = None, hbc_near: np.ndarray = None, hbc_far: np.ndarray = None, toi: np.ndarray = None, toi_abs: np.ndarray = None, toi_ods: np.ndarray = None, svd_diagnostics: SVDDiagnostics = None):    
        super().__init__(raw_abs, hbc_near, hbc_far, toi)
        self.toi_abs = toi_abs
        self.toi_ods = toi_ods
        self.svd_diagnostics = svd_diagnostics


class HbConvert:
    """
    Class that handles conversion of raw PD reads to converted (and cleaned) Hb-concentration changes and TOI.

    This module contains functions and wrappers for applying various transformations for data decomposition and cleaning.
    The HbConv object takes in a data array with raw absorbances (shape: 12-channels x n-observations) and performs ambient correction,
    and calculates hb-concentration changes and TOI, along with an SVD-based cleaning of the data.
    For Hb-concentration changes, the SVD is applied to the near and far channel data.
    For TOI, the SVD is applied to both the ambient corrected absorbance data or the optical density gradients. 

    Attributes
    ----------
        observed: NIRSData
            An object with the raw and converted data as recorded. 

        cleaned: NIRSData
            An object with the raw and converted data as recorded. 

        svd_diagnostics: SVDDiagnostics
            An object with the SVD cleaning results and diagnostics for both channels.
    """
    
    def __init__(self, data: np.ndarray = None, svd_clean: bool = True):
        """
        Instantiate Object with absorbance data and obtain hb-concentration and TOI values for both observed and clean data.

        Arguments
        ---------
            data: np.ndarray (12 x nobs)
                The raw data as a 12-channel matrix with each channel as a row vector. 
            svd_clean: bool (default=True)
                Whether or not to apply the SVD cleaning on the data.
        
        Returns
        -------
            HbConv: Object
                An instance of the class object.
        """
        # The coefficients for conversion of optical density to Hb-concentrations.
        self.__params = self.__gen_params()
        
        # If data was passed during instantiation
        if data is not None:
            self.svd_diagnostics = SVDDiagnostics()
            # Convert to hb-concs and toi
            abs_data = data
            obs = self.hbc_from_abs(abs_data)
            self.observed = obs

        if svd_clean:
            # Clean the data
            self.cleaned = self.clean_hbc_toi(data)

    def __gen_params(self) -> dict:
        """Generate a dictionary of params used in calculating Hb-concs and TOI."""
        e_coef = np.array([[0.3194, 2.5713],
                           [0.4383, 1.3029],
                           [0.9291, 0.7987],
                           [1.1596, 0.7861],
                           [1.3514, 0.8968]])
        
        d = np.array([1, 1.6], dtype=np.float16)  # Source detector distance per channel
        wavelengths = np.array([680, 730, 810, 850, 910], dtype=np.float16)  # Wavelengths
        dpf = 6  # Differential path length factor
    
        # Generate matrices for conversion to Hb-concentration changes
        A_near = e_coef * d[0] * dpf
        A_near_inv = np.linalg.pinv(A_near)

        A_far = e_coef * d[1] * dpf
        A_far_inv = np.linalg.pinv(A_far)

        params = {"e_coef": e_coef,
                  "wavelengths": wavelengths,
                  "dpf": dpf,
                  "sd_distance": d,
                  "A_near_inv": A_near_inv,
                  "A_far_inv": A_far_inv,
                  "abs_offsets": np.array([0.62517, 0.81141, 0.82149, 0.81788, 0.67964])}  # calibration coefficients
        
        return params
    
    def get_params(self) -> dict:
        """
        Return the dictionary of parameters used for Hb-concentration and TOI calculation.

        Returns
        -------
            params: dict
                A dictionary containing the parameters used for Hb-concentration and TOI calculation.
        """
        return self.__params

    def hbc_from_abs(self, abs_data: np.ndarray) -> NIRSData:
        """
        Convert raw absorbance data to Hb-concentration changes and TOI.

        Arguments
        ---------
            abs_data: np.ndarray
                The raw absorbance data (before ambient subtraction). Shape is nchan x nobs.

        Returns
        -------
            hb_data: NIRSData
                Object with the raw absorbance and converted values for near and far channel.

        """
        near, far = self.subtract_ambient(abs_data)
        near_hbc, far_hbc = self.calc_hb_concs(near, far)
        toi = self.calc_toi(near, far)
        
        hb_data = NIRSData(raw_abs=abs_data, hbc_near=near_hbc, hbc_far=far_hbc, toi=toi)
        
        return hb_data

    def subtract_ambient(self, pd_reads: np.ndarray) -> np.ndarray:
        """
        Subtract ambient data from the raw absorbance data.

        Arguments
        ---------
            pd_reads: np.ndarray
                The raw data as a 12-channel matrix with each channel as a row vector.

        Returns
        -------
            near_vals: np.ndarray
                The ambient-subtracted near channel data.

            far_vals: np.ndarray
                The ambient-subtracted far channel data.
        """
        near_vals = pd_reads[:5, :] - pd_reads[5, :]
        far_vals = pd_reads[6:11, :] - pd_reads[11, :]
        return near_vals, far_vals

    def calc_hb_concs(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray) -> list[np.ndarray, np.ndarray]:
        """
        Convert raw PD reads to near and far Hb-concentration changes.

        Arguments
        ---------
            near_pd_reads: np.ndarray (5 x nobs)
                The ambient-subtracted near channel data.
            far_pd_reads: np.ndarray (5 x nobs)
                The ambient-subtracted far channel data.

        Returns
        -------
            conc_near: np.ndarray (2 x nobs)
                Array of near Hb-concentration changes with rows being O2Hb and HHb respectively. 
                
            conc_far: np.ndarray (2 x nobs)
                Array of far Hb-concentration changes with rows being O2Hb and HHb respectively. 
        """

        hb_near = np.zeros((4, near_pd_reads.shape[1]))
        hb_far = np.zeros((4, near_pd_reads.shape[1]))

        A_near_inv = self.__params["A_near_inv"]
        A_far_inv = self.__params["A_far_inv"]

        # Calculate the relative changes in OD
        near_delta_ODs = -np.log10(near_pd_reads.T / near_pd_reads[:, 0])  # Normalize PD reads to first sample
        far_delta_ODs = -np.log10(far_pd_reads.T / far_pd_reads[:, 0])

        #epsilon = 1e-12
        #safe_near = np.clip(near_pd_reads, epsilon, None)
        #safe_far = np.clip(far_pd_reads, epsilon, None)

        #near_delta_ODs = -np.log10(safe_near.T / safe_near[:, 0])
        #far_delta_ODs = -np.log10(safe_far.T / safe_far[:, 0])


        # Calculate hb concentrations
        concs_near = 1000 * (A_near_inv) @ near_delta_ODs.T
        concs_far = 1000 * (A_far_inv) @ far_delta_ODs.T

        hb_near[:2, :] = concs_near
        hb_near[2, :] = concs_near.sum(0)
        hb_near[3, :] = concs_near[0, :] - concs_near[1, :]

        hb_far[:2, :] = concs_far
        hb_far[2, :] = concs_far.sum(0)
        hb_far[3, :] = concs_far[0, :] - concs_far[1, :]

        return hb_near, hb_far
    
    def calc_gradients(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray) -> np.ndarray:
        """
        Calculate OD-gradients from ambient corrected near and far absorbance data.

        Arguments
        ---------
            near_pd_reads: np.ndarray (5 x nobs)
                The ambient-subtracted near channel data.
            far_pd_reads: np.ndarray (5 x nobs)
                The ambient-subtracted far channel data.

        Returns
        -------
            od_grads: np.ndarray (5 x nobs)
                Array of log-transformed, calibration-corrected OD-gradients. 
        """
        d = self.__params["sd_distance"]
        abs_off = self.__params["abs_offsets"]
        #OD_grads = np.log10(near_pd_reads / far_pd_reads) / (d[1] - d[0])
        epsilon = 1e-10  # avoid log10(0)
        safe_near = np.clip(near_pd_reads, epsilon, None)
        safe_far = np.clip(far_pd_reads, epsilon, None)
        OD_grads = np.log10(safe_near / safe_far) / (d[1] - d[0])

        OD_grads = OD_grads.T + abs_off / (d[1] - d[0])
        return OD_grads

    def calc_toi(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray, svd_clean=False) -> np.ndarray:
        """
        Calculate TOI from near and far absorbance data.

        Arguments
        ---------
            near_pd_reads: np.ndarray (shape=(5, nobs))
                The near channel absorbance data.
            far_pd_reads: np.ndarray (shape=(5, nobs))
                The far channel absorbance data.
            svd_clean: bool (default=False)
                Whether to apply SVD cleaning to the absorbance data. If True, the SVD is applied simultaneously
                on the near and far data by stacking them vertically.

        Returns
        -------
            toi: np.ndarray (shape=(nobs,))
                The calculated TOI values.
        """
        OD_grads = self.calc_gradients(near_pd_reads, far_pd_reads)
        
        if svd_clean:
            OD_grads_svd = self.svd_clean(OD_grads)
            self.svd_diagnostics.ods = OD_grads_svd
            OD_grads = OD_grads_svd.clean_abs.T

        h = 4.6E-4  # Wavelength dependence of scattering (1/nm)
        usp = 1 - h * self.__params["wavelengths"]

        # Step 3: Estimate absorbance coefficient
        ua = 1 / (3 * usp) * (np.log(10) * OD_grads - 2 / np.mean(self.__params["sd_distance"])) ** 2

        # Step 4: Estimate hb concentrations
        hb_concs = np.linalg.pinv(self.__params["e_coef"]) @ ua.T
        hb_concs = np.transpose(hb_concs)

        # Step 5: Calculate TOI
        toi = hb_concs[:, 0] / np.sum(hb_concs, 1) * 100

        return toi

    @staticmethod
    def get_pd_reads(df: pd.DataFrame) -> np.ndarray:
        """
        Get PD reads from Neurokey dataframe returned by NKY-utils (for online use).
        
        Arguments
        ---------
            df: pd.DataFrame
                A dataframe with the NKY-derived absorbance values.
        
        Returns
        -------
            data: np.ndarray (shape=(12, nobs))
                A numpy array with the absorbance data extracted from the dataframe.
        """
        d = np.empty((len(df), 12))
        d[:, :5] = df.iloc[:, 1:6].to_numpy()
        d[:, 5] = df.iloc[:, 11].to_numpy()
        d[:, 6:11] = df.iloc[:, 6:11].to_numpy()
        d[:, 11] = df.iloc[:, 12].to_numpy()
        return d
  
    def svd_clean(self, data: np.ndarray) -> SVDCleaned:
        """
        Clean absorbance data with SVD and return cleaned data (and noise, if return_noise is True).

        Arguments
        ---------
            data: np.ndarray (shape=(nchannels, nobs))
                The data to be decomposed with the SVD. Ambient corrected absorbance data.
        
        Returns
        -------
            clean_data: np.ndarray (shape=(nchannels, nobs))
                The cleaned data with the dominant component removed.

            noise: np.ndarray (shape=(nobs,))
                If return_noise is True, will also return the component removed during cleaning.
        """
        
        if data.shape[0] < data.shape[1]:
            data = data.T

        raw_data = data
        raw_data_means = np.mean(raw_data, axis=0)
        raw_data_mc = raw_data - raw_data_means

        u, s, v = np.linalg.svd(raw_data_mc.T, full_matrices=False)
        uvar = np.std(u, axis=0)
        
        noise = np.mean(u[:, 0]) * v[0, :]
        comp2 = np.mean(u[:, 1]) * v[1, :]
        comp3 = np.mean(u[:, 2]) * v[2, :]
        rec = u[:, 1:-1] @ np.diag(s[1:-1]) @ v[1:-1, :]
        rec = rec.T
        clean_data = (rec - rec[0, :] + raw_data[0, :]).T

        svd_cleaned = SVDCleaned(uvar=uvar, noise=noise, clean_abs=clean_data, comp2=comp2, comp3=comp3)

        return svd_cleaned

    def clean_hbc_toi(self, data: np.ndarray):
        """
        Ambient subtract, clean and convert raw absorbance data.

        Arguments
        ---------
            data: np.ndarray (shape=(12, nobs))
                The raw data as a 12-channel matrix with each channel as a row vector. 
        
        Returns
        -------
            near_hb_clean: np.ndarray (shape=(2, nobs))
                The SVD-cleaned near hb-concentration changes.
            
            far_hb_clean: np.ndarray (shape=(2, nobs))
                The SVD-cleaned far hb-concentration changes.
            
            toi_clean: np.ndarray (shape=(nobs,))
                The TOI calculated from the SVD-cleaned near and far absorbances.
        """
        clean_abs = data.copy()
        near, far = self.subtract_ambient(data)
        near_clean = self.svd_clean(near)
        far_clean = self.svd_clean(far)

        self.svd_diagnostics.near = near_clean
        self.svd_diagnostics.far = far_clean

        clean_abs[:5] = near_clean.clean_abs
        clean_abs[6:11] = far_clean.clean_abs

        near_hb_clean, far_hb_clean = self.calc_hb_concs(near_clean.clean_abs, far_clean.clean_abs)
        toi_abs = self.calc_toi(near_clean.clean_abs, far_clean.clean_abs, svd_clean=False)
        toi_ods = self.calc_toi(near, far, svd_clean=True)

        cleaned = NIRSDataCleaned(raw_abs=clean_abs, hbc_near=near_hb_clean, hbc_far=far_hb_clean, toi_abs=toi_abs, toi_ods=toi_ods)
        self.svd_diagnostics.near = near_clean
        self.svd_diagnostics.far = far_clean

        return cleaned
    
    def __repr__(self) -> str:
        return f"HbConvert(observed={self.observed}, cleaned={self.cleaned}, svd_diagnostics={self.svd_diagnostics})"


if __name__ == "__main__":
    # Example use of functionality

    # Imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load some data
    pd_reads = pd.read_csv('./src/pynirs/data/test_data.csv').to_numpy().T  
    hb = HbConvert(pd_reads)

    # Plot the data    
    plt.subplots(3, 1, sharex=True, sharey=True)
    plt.subplot(311)
    plt.plot(hb.observed.toi)
    plt.title("TOI Observed")
    plt.grid()
    plt.subplot(312)
    plt.plot(hb.cleaned.toi_abs)
    plt.title("TOI Cleaned (Abs)")
    plt.grid()
    plt.subplot(313)
    plt.plot(hb.cleaned.toi_ods)
    plt.title("TOI Cleaned (Ods)")
    plt.grid()
    plt.show()
    print("done")

