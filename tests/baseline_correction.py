import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

feature = "../data/feature.hdf5"
arb = 10**8

# get data
pha = []
pixmask = []
with h5py.File(feature, 'a') as h5:
    pha.append(h5["Saiteki"]["Pix0"]["PHA"][()]/arb)
    pha.append(h5["Saiteki"]["Pix1"]["PHA"][()]/arb)
    pha.append(h5["Saiteki"]["Pix2"]["PHA"][()]/arb)
    pha.append(h5["Saiteki"]["Pix3"]["PHA"][()]/arb)

    bsl = h5["baseline"][()]
    # obtain mask
    pixmask.append(h5["Normal"]["pix0"]["Mask"][()])
    pixmask.append(h5["Normal"]["pix1"]["Mask"][()])
    pixmask.append(h5["Normal"]["pix2"]["Mask"][()])
    pixmask.append(h5["Normal"]["pix3"]["Mask"][()])


    def line(x, a, b):
        return a*x + b


    def fit(func, x, y, param_init):
        popt, pcov = curve_fit(func, x, y, p0=param_init)
        perr = np.sqrt(np.diag(pcov))
        result_y = func(x, *popt)
        return result_y, popt, perr

    param = [[0, 100],[0,40],[0,22],[0,13.5]]
    rng = np.array([[108.5, 109.5],
                    [39.25, 39.75],
                    [21.8, 22.2],
                    [13.4, 13.7]])

    print("/// Baseline Correction ///")
    print("")
    # del h5["Saiteki/BaselineCorrection"]
    h5.create_group("Saiteki/BaselineCorrection")
    for pixnum in range(4):
        pix_pha = pha[pixnum]
        pix_bsl = bsl[pixmask[pixnum]]

        mask = (rng[pixnum, 0] < pix_pha) & (pix_pha < rng[pixnum, 1])

        result = fit(line, pix_bsl[mask], pix_pha[mask], param[pixnum])


        print("Pix"+str(pixnum)+" :")
        print("a = %.2e +- %.2e" % (result[1][0], result[2][0]))
        print("b = %.2e +- %.2e" % (result[1][1], result[2][1]))

        
        theta = np.arctan(result[1][0])
        rot_pix_bsl = pix_bsl*np.cos(theta) + pix_pha*np.sin(theta)
        rot_pix_pha = -pix_bsl*np.sin(theta) + pix_pha*np.cos(theta)
        h5.create_dataset(name="Saiteki/BaselineCorrection/Pix"+str(pixnum)+"/PHA", data=rot_pix_pha)
        print("written in : ~/Saiteki/BaselineCorrection/Pix"+str(pixnum)+"/PHA")
        print("")
