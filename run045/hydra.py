import numpy as np
import matplotlib.pyplot as plt
import h5py

from libs import readhdf5, feature, splicing, pha

class Hydra(readhdf5, feature):
    def __init__(self, rawdata_path):
        self.rawdata_path = rawdata_path

    def open_raw(self):
        readhdf5.ReadHDF5(self.rawdata_path)
        readhdf5.OpenHDF5()
        pulse = readhdf5.pulse
        noise = readhdf5.noise
        vres  = readhdf5.vres
        hres  = readhdf5.hres
        time  = readhdf5.time
        return pulse, noise, vres, hres, time

    def close_raw(self):
        readhdf5.CloseHDF5()

    def calc_features(self, pulse, noise, time):
        raw_rise, raw_fall, raw_ph, raw_bsl = feature.calc_tau(
            pulse,
            noise,
            time,
            low_lvl=0.2,
            high_lvl=0.8
        )
        return raw_rise, raw_fall, raw_ph, raw_bsl

    def import_features(self, features_path):
        with h5py.File(features_path, 'r') as h5:
            raw_rise = h5['Rise'][()]
            raw_fall = h5['Fall'][()]
            raw_ph   = h5['PulseHeight'][()]
            raw_bsl  = h5['Baseline'][()]
        
        return raw_rise, raw_fall, raw_ph, raw_bsl

        # self.pix_define = np.array([
        #     ((self.rise<10) & (self.fall<200)),
        #     ((self.rise>10) & (self.rise<30) & (self.fall>300) & (self.fall<500)),
        #     ((self.rise>40) & (self.rise<80) & (self.fall>500) & (self.fall<600)),
        #     ((self.rise>100) & (self.rise<140) & (self.fall>550) & (self.fall<800))
        # ])


    def add_new_feature(self, features_array, new_feature):
        """
        add a new feature to the existed feature array

        Parameters:
        - features: array, existed feature(s) array
        - new_feature: 1d-array, [xxx, xxx, ...]

        Returns:
        - result: 2d-array,
        """
        result = np.vstack([features_array.T, new_feature]).T
        return result
    

    def generate_mask(self, pix_define, nan=True, large=True, small=True):
        self.mask = self.splc.generate_mask(pix_define, nan, large, small)
        self.nan = self.splc.nan
        self.large = self.splc.large_outlier
        self.small = self.splc.small_outlier


    def clustering(self, features, mask, mode="kmeans"):
        self.pred, self.pixmask = self.splc.clustering(features=features[mask], n_clusters=4, mode=mode)

    
    def calc_pha(self, downsample=True, output_path='../data/energy.hdf5'):
        rawdata = (self.time, self.pulse, self.noise)
        p = pha.Calc_pha(rawdata_path=self.rawdata_path, pixmask=self.pixmask, *rawdata)
        p.calc_pha(downsample=downsample, output_path=output_path)
        