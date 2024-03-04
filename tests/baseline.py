import numpy as np
import h5py
import readhdf5

rawfile = "../data/run045pn_b64-1.hdf5"
feature = "../data/feature_run045.hdf5"

###############################
# obtain normal event
###############################
raw = readhdf5.ReadHDF5(rawfile)
raw.OpenHDF5()
p = raw.pulse

with h5py.File(feature, "a") as h5:
    baseline = np.average(p[:, :20833], axis=1)
    # del h5["baseline"]
    h5.create_group("Baseline")
    h5.create_dataset(name="Baseline/Raw", data=baseline)
    print("written in : ~/Baseline/Raw")

raw.CloseHDF5()