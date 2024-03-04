import numpy as np
import h5py
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn import preprocessing

class Splicing:
    def __init__(self, feature_path):
        """
        serve you a mask to extract "exon"

        Parameters:
        - feature_path: string,
        """
        self.all_counts = len(self.raw_tau)


    def generate_mask(self, pix_define, zerotau, large, small):
        """
        generate mask to remove some dirty events

        Parameters:
        - zerotau: bool, whether remove zero-tau events or not
        - large: bool, whether remove large outlier tau events or not
        - small: bool, whether remove small outlier tau events or not

        Returns:
        - mask: array, [True, True, True, False, True, ...] (e.g., nan,large,small->False)
        """

        # all false array (len=all_counts)
        self.zerotau = self.large_outlier = self.small_outlier = np.zeros(self.all_counts, dtype=bool)
        self.eventnum = np.arange(self.all_counts)

        # define normal events
        # mask: normal value -> True
        normal_mask = pix_define[0] | pix_define[1] | pix_define[2] | pix_define[3]

        print(f"normal events : {self.eventnum[normal_mask].shape[0]} events")
        print("event number : ")
        print(self.eventnum[normal_mask])
        print("")

        if(zerotau):
            # mask: zerotau -> True
            self.zerotau = np.isnan(self.raw_tau).any(axis=1)

            print(f"cannot obtain tau : {self.eventnum[self.zerotau].shape[0]} events")
            print("event number : ")
            print(self.eventnum[self.zerotau])
            print("")

        if(large):
            # mask: large outlier -> True
            self.large_outlier = (self.raw_rise > 200) | (self.raw_fall > 2000)

            print(f"large outlier : {self.eventnum[self.large_outlier].shape[0]} events")
            print("event number : ")
            print(self.eventnum[self.large_outlier])
            print("")

        if(small):
            # mask: small outlier -> True
            self.small_outlier = ~(self.zerotau | self.large_outlier | normal_mask)

            print(f"small outlier : {self.eventnum[self.small_outlier].shape[0]} events")
            print("event number : ")
            print(self.eventnum[self.small_outlier])
            print("")

        mask = ~(self.zerotau | self.large_outlier | self.small_outlier)

        return mask


    def clustering(self, features, n_clusters, mode):
        """
        calculate clusteirng.

        Parameters:
        - features: 2d-array, [[risetime, falltime, ...], ]
        - n_clusters: int,
        - mode: string, default="kmeans"

        Returns:
        - pred: 1d-array, [pix0, pix2, pix1, -1, ...]
        - pixmask: 1d-array, [[False, False, True, ...], ]
        """

        import time
        start_time = time.time()

        # normalize data
        ss = preprocessing.StandardScaler()
        ss.fit(features)
        features_norm = ss.transform(features)
        
        if(mode == "kmeans"):
            print("k-means clustering")
            
            label = KMeans(n_clusters=n_clusters).fit_predict(features_norm)

        if(mode == "spectral"):
            print("spectral clustering")
            
            sc = SpectralClustering(
                n_clusters=n_clusters,
                gamma=10.0,
                ).fit(features_norm)
            label = sc.labels_
        
        if(mode == "dbscan"):
            print("DBSCAN clustering")

            label = DBSCAN(eps=1, min_samples=5, metric='euclidean').fit_predict(features_norm)

            # noise points are assigned -1
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Number of Noise Points: ",sum(label==-1)," (",len(label),")",sep='')



        # sort cluster numbers with risetime
        average_risetime = []
        for i in range(4):
            onelabel = [val == i for val in label]
            average_risetime.append(np.mean(features[onelabel, 0])) # 0 must be risetime
        
        labelnum = np.argsort(average_risetime) # labelnum[0] has a cluster label corresponds to pix0
        sort_dict = {labelnum[0]:0, labelnum[1]:1, labelnum[2]:2, labelnum[3]:3, -1:-1} # {labelnum:pixnum}
        
        # insert label "-1" where an event is anomalous
        anom_eventnum = np.sort(
            np.hstack([
                self.eventnum[self.nan],
                self.eventnum[self.large_outlier],
                self.eventnum[self.small_outlier]]
            )
        )
        label_allevents = label
        for evnum in anom_eventnum:
            label_allevents = np.insert(label_allevents, evnum, -1)

        pred = [sort_dict[l] for l in label_allevents]
        pixmask = []
        for i in range(4):
            onepix = [val == labelnum[i] for val in label_allevents]
            pixmask.append(onepix)

        return np.array(pred), np.array(pixmask)
    
