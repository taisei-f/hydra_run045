import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import libs.feature as feat
import libs.splicing
import libs.pha as pha
import libs.readhdf5

class Hydra:
    def __init__(self):
        self.rawdata_path = '../data/run045pn_b64-1.hdf5'
        self.feature_path = '../data/features.hdf5'

        self.read = libs.readhdf5.ReadHDF5()
        self.read.ReadHDF5(self.rawdata_path)
        self.read.OpenHDF5()
        
        # self.splc = libs.splicing.Splicing(self.feature_path)


    def features(self):
        ft = feat.Feature()
        # whether features.hdf5 exists or not
        isFile = os.path.isfile(self.feature_path)
        if ~isFile:
            # calculate rise time, fall time, pulse height
            self.raw_rise, self.raw_fall, self.raw_ph, self.raw_bsl = ft.calc_tau(self.read, 0.2, 0.8)
            ft.save_to_file(self.feature_path, 'Rise', self.raw_rise)
            ft.save_to_file(self.feature_path, 'Fall', self.raw_fall)
            ft.save_to_file(self.feature_path, 'PulseHeight', self.raw_ph)
            ft.save_to_file(self.feature_path, 'Baseline', self.raw_bsl)
            print('***features calculated***')


        else:
            with h5py.File(self.feature_path, 'r') as h5:
                self.raw_rise = h5['Rise'][()]
                self.raw_fall = h5['Fall'][()]
                self.raw_ph = h5['PulseHeight'][()]
                self.raw_bsl = h5['Baseline'][()]

            print('***features imported***')

        # self.pix_define = np.array([
        #     ((self.rise<10) & (self.fall<200)),
        #     ((self.rise>10) & (self.rise<30) & (self.fall>300) & (self.fall<500)),
        #     ((self.rise>40) & (self.rise<80) & (self.fall>500) & (self.fall<600)),
        #     ((self.rise>100) & (self.rise<140) & (self.fall>550) & (self.fall<800))
        # ])


    def raw(self):
        self.time = self.read.time * 1e3
        self.pulse = self.read.pulse * self.read.vres * 1e3
        self.noise = self.read.noise * self.read.vres * 1e3


    def close_raw(self):
        self.read.CloseHDF5()


    def add_new_feature(self, features, new_feature):
        """
        add a new feature to the existed feature array

        Parameters:
        - features: array, existed feature(s) array
        - new_feature: 1d-array, [xxx, xxx, ...]

        Returns:
        - result: 2d-array,
        """
        result = np.vstack([features.T, new_feature]).T
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
        