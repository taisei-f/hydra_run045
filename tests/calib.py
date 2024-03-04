import numpy as np
import h5py
import itertools
from scipy.optimize import curve_fit

feature_file = "../data/energy_run045.hdf5"

pha = []
with h5py.File(feature_file, 'r') as h5:
    pha.append(h5["Pix0"]["PHA"][()])
    pha.append(h5["Pix1"]["PHA"][()])
    pha.append(h5["Pix2"]["PHA"][()])
    pha.append(h5["Pix3"]["PHA"][()])

##############################
# CALIBRATION FUNCTION
##############################
def response(ka, kb):
    d1 = (5900*kb - 6500*ka)/(5900*6500*(6500-5900))
    d2 = (6500*6500*ka - 5900*5900*kb)/(5900*6500*(6500-5900))
    return d1, d2

##############################
# FIND K_ALPHA & K_BETA
##############################
rng = np.array(
    [[3.8, 4.1],
     [30, 36],
     [14, 20],
     [13, 16]]
)

Delta_PHA = np.array([7, 3.5, 2, 1.5])

binnum = np.array(
    [int((rng[0,1]-rng[0,0])/(Delta_PHA[0]/60)),
     int((rng[1,1]-rng[1,0])/(Delta_PHA[1]/60)),
     int((rng[2,1]-rng[2,0])/(Delta_PHA[2]/60)),
     int((rng[3,1]-rng[3,0])/(Delta_PHA[3]/60))]
)

energy = []
with h5py.File(feature_file, 'a') as h5:
    # del h5["Calibration"]
    # del h5["Calibration/BaselineCorrection"]
    h5.create_group("Calibration/BaselineCorrection")

    for pixnum in range(4):
        h5.create_group("Calibration/BaselineCorrection/Pix"+str(pixnum))
        print("Energy Calibration for Pix" + str(pixnum))
        counts, bins = np.histogram(pha[pixnum], bins=binnum[pixnum], range=(rng[pixnum,0],rng[pixnum,1]))
        index_KA = np.argmax(counts)
        index_KB = index_KA+30 + np.argmax(counts[index_KA+30:])

        count_KA = counts[index_KA]
        count_KB = counts[index_KB]
        print("Counts @Ka = %d, @Kb = %d" % (count_KA, count_KB))

        pha_KA = (bins[index_KA]+bins[index_KA+1])/2
        pha_KB = (bins[index_KB]+bins[index_KB+1])/2
        print("PHA @Ka = %.2e, @Kb = %.2e" % (pha_KA, pha_KB))

        d1, d2 = response(pha_KA, pha_KB)
        print("Calib. param : d1 = %.2e, d2 = %.2e" % (d1, d2))

        en = (np.sqrt(4*d1*pha[pixnum]+d2*d2)-d2)/2/d1
        energy.append(en)
        h5.create_dataset(name="Calibration/BaselineCorrection/Pix"+str(pixnum)+"/Energy", data=en)
        print("written in : ~/Calibration/BaselineCorrection/Pix"+str(pixnum)+"/Energy")
        print('')

    h5.create_group("Calibration/BaselineCorrection/All")
    pha_all = list(itertools.chain.from_iterable(pha))
    energy_all = list(itertools.chain.from_iterable(energy))
    h5.create_dataset(name='Calibration/BaselineCorrection/All/PHA', data=pha_all)
    h5.create_dataset(name='Calibration/BaselineCorrection/All/Energy', data=energy_all)
    print("written in : ~/Calibration/BaselineCorrection/All/PHA")
    print("written in : ~/Calibration/BaselineCorrection/All/Energy")
