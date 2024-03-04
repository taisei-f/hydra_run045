import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
import itertools

class Calibration:
    def __init__(self):
        self.mask = []
        self.bsl = []
        self.pha = []
        self.energy = []

        # with h5py.File("../data/feature_run045.hdf5", 'r') as h5:
        #     self.tau_raw = h5["Tau/Raw"][()]
        #     self.normal_mask = h5["Tau/Nan/Mask"][()]
        #     self.mask.append(h5["Tau/Normal/Pix0/Mask"][()])
        #     self.mask.append(h5["Tau/Normal/Pix1/Mask"][()])
        #     self.mask.append(h5["Tau/Normal/Pix2/Mask"][()])
        #     self.mask.append(h5["Tau/Normal/Pix3/Mask"][()])

        #     self.ph_raw = h5["PulseHeight/Raw"][()]
        
        with h5py.File("../data/pha_run045_spectral.hdf5", 'r') as h5:
            self.pha.append(h5["PHA"]["Pix.0"][()])
            self.pha.append(h5["PHA"]["Pix.1"][()])
            self.pha.append(h5["PHA"]["Pix.2"][()])
            self.pha.append(h5["PHA"]["Pix.3"][()])


    def Open(self, feature_file, key):
        self.data = []
        with h5py.File(feature_file, 'r') as h5:
            self.data.append(h5["Pix0"][key][()])
            self.data.append(h5["Pix1"][key][()])
            self.data.append(h5["Pix2"][key][()])
            self.data.append(h5["Pix3"][key][()])


    def Cal(self, cor=True):
        for i in range(4):
            print("Energy Calibration for Pix" + str(i))

            # make hist
            counts, bins = np.histogram(self.pha[i], bins=1500)

            # search Ka, Kb
            Ka_index = np.argmax(counts)
            Kb_index = Ka_index+30 + np.argmax(counts[Ka_index+30:])
            pha_a = bins[Ka_index]
            pha_b = bins[Kb_index]

            # show Ka, Kb lines
            side = (pha_b - pha_a)*0.25
            plt.hist(self.pha[i], bins=500, range=(pha_a-side, pha_b+side))

            # calibration function
            def response(ka, kb):
                d1 = (5900*kb - 6500*ka)/(5900*6500*(6500-5900))
                d2 = (6500*6500*ka - 5900*5900*kb)/(5900*6500*(6500-5900))
                return d1, d2
            
            # calibration parameters
            d1, d2 = response(pha_a, pha_b)
            print("Calib. param : d1 = %.2e, d2 = %.2e" % (d1, d2))

            # obtain energy
            en = (np.sqrt(4*d1*self.pha[i]+d2*d2)-d2)/2/d1
            self.energy.append(en)


            print("")

        pha_all = list(itertools.chain.from_iterable(self.pha))
        energy_all = list(itertools.chain.from_iterable(self.energy))

        plt.hist2d(
            energy_all, pha_all,
            bins=100,
            cmap=matplotlib.cm.jet,
            norm=matplotlib.colors.LogNorm()
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("PHA")
        plt.grid(linestyle="--")
        plt.colorbar()