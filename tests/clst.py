import numpy as np
import h5py
from sklearn.cluster import KMeans, SpectralClustering

class Clst_3d:
    def __init__(self, feature_path):
        with h5py.File(feature_path, 'a') as h5:
            self.raw_tau = h5["Tau/Raw"][()]*10**6
            self.raw_ph = h5["PulseHeight/Raw"][()]/10**4

        self.raw_rise = self.raw_tau[:,0]
        self.raw_fall = self.raw_tau[:,1]


        print("########################################")
        print("# import data  and cleaning")
        print("########################################")

        all_counts = len(self.raw_tau)

        # nan -> True
        self.nan = np.isnan(self.raw_tau).any(axis=1)
        nantau_eventnum = np.arange(all_counts)[self.nan]
        print("cannot obtain tau : " + str(nantau_eventnum.shape) + "events")
        print("event number : ")
        print(nantau_eventnum)
        print("")

        # large outlier -> True
        large_outlier = (self.raw_rise > 200) & (self.raw_fall > 2000)
        large_outlier_eventnum = np.arange(all_counts)[large_outlier]
        print("large outlier : " + str(large_outlier_eventnum.shape) + "events")
        print("event number : ")
        print(large_outlier_eventnum)
        print("")

        # normal value -> True
        self.pixmask = []
        self.pixmask.append((self.raw_rise<10) & (self.raw_fall<200))
        self.pixmask.append((self.raw_rise>10) & (self.raw_rise<30) & (self.raw_fall>300) & (self.raw_fall<500))
        self.pixmask.append((self.raw_rise>40) & (self.raw_rise<80) & (self.raw_fall>500) & (self.raw_fall<600))
        self.pixmask.append((self.raw_rise>100) & (self.raw_rise<140) & (self.raw_fall>550) & (self.raw_fall<800))
        self.normal_mask = self.pixmask[0] | self.pixmask[1] | self.pixmask[2] | self.pixmask[3]


        small_outlier = ~(self.nan | large_outlier | self.normal_mask)
        small_outlier_eventnum = np.arange(all_counts)[small_outlier]
        print("small outlier : " + str(small_outlier_eventnum.shape) + "events")
        print("event number : ")
        print(small_outlier_eventnum)
        print("")


        normal_eventnum = np.arange(all_counts)[self.normal_mask]
        print("normal events : " + str(normal_eventnum.shape) + "events")
        print("event number : ")
        print(normal_eventnum)
        print("")


        tau = self.raw_tau[self.normal_mask, :]
        self.cleaned_rise = tau[:,0]
        self.cleaned_fall = tau[:,1]
        self.cleaned_ph = self.raw_ph[self.normal_mask]

        self.cleaned_features = np.insert(tau, 2, self.cleaned_ph, axis=1)

    def KM_clst(self, n_clusters):
        print("########################################")
        print("# k-means clustering")
        print("########################################")
        clustering = KMeans(n_clusters=n_clusters).fit(self.cleaned_features)
        return clustering

    def SP_clst(self, n_clusters):
        clustering = SpectralClustering(
                n_clusters=n_clusters,).fit(self.cleaned_features)
        return clustering
