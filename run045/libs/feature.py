import numpy as np
import h5py

class Feature:
    def calc_tau(self, readhdf5, low_lvl, high_lvl):
        tau = []
        pulseheight = []
        baseline = []

        all_eventnum = readhdf5.pulse.shape[0]
        for eventnum in range(all_eventnum):
            noise = readhdf5.noise[eventnum] * readhdf5.vres * 1e3
            pulse = readhdf5.pulse[eventnum] * readhdf5.vres * 1e3
            time = readhdf5.time * 1e3


            bsl = np.mean(noise)
            pmax = np.min(pulse)
            pmax_index = np.argmin(pulse)
            ph = bsl - pmax
            

            low_line = bsl - low_lvl * ph
            high_line = bsl - high_lvl * ph

            one_tau = []
            for i in range(2):
                if i == 0:
                    half_time = time[:pmax_index]
                    half_pulse = pulse[:pmax_index]
                else:
                    half_time = time[pmax_index:]
                    half_pulse = pulse[pmax_index:]

                sliced_time = half_time[(half_pulse<low_line) & (half_pulse>high_line)]

                if sliced_time.size < 2:
                    one_tau.append(time[1]-time[0])
                else:
                    one_tau.append(sliced_time[-1]-sliced_time[0])
                
            
            tau.append(one_tau)
            pulseheight.append(ph)
            baseline.append(bsl)

        tau_array = np.array(tau).reshape([-1,2])
        rise_array = tau_array[:,0]
        fall_array = tau_array[:,1]
        pulseheight_array = np.array(pulseheight)
        baseline_array = np.array(baseline)

        return rise_array, fall_array, pulseheight_array, baseline_array


    def save_to_file(self, path, name, data):
        with h5py.File(path, 'w') as h5:
            h5.create_dataset(name, data=data)
            print('***'+name+' written***')
