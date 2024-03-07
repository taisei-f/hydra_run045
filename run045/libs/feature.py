import numpy as np
import h5py

class Feature:
    def calc_baseline(self, noises):
        bsls = []
        for i, noise in enumerate(noises):
            bsl = np.mean(noise)
            bsls.append(bsl)
        return np.array(bsls)

    def calc_pulseheight(self, pulses, noises):
        phs = []
        bsls = self.calc_baseline(noises)
        for i, pulse in enumerate(pulses):
            pmax = np.min(pulse)
            ph = bsls[i] - pmax
            phs.append(ph)
        return np.array(phs)

    def calc_tau(self, pulses, noises, time, low_lvl, high_lvl):
        rises = []
        falls = []
        bsls = self.calc_baseline(noises)
        phs  = self.calc_pulseheight(pulses, noises)
        n_events = pulses.shape[0]
        sampling_rate = time[1] - time[0]

        for i in range(n_events):
            pulse = pulses[i]
            pmax_index = np.argmin(pulse)

            low_line  = bsls[i] - low_lvl  * phs[i]
            high_line = bsls[i] - high_lvl * phs[i]

            half_time1   = time[:pmax_index]
            half_pulse1  = pulse[:pmax_index]
            sliced_time1 = half_time1[(half_pulse1<low_line) & (half_pulse1>high_line)]

            half_time2   = time[pmax_index:]
            half_pulse2  = pulse[pmax_index:]
            sliced_time2 = half_time2[(half_pulse2<low_line) & (half_pulse2>high_line)]

            if not sliced_time1:
                rise = sampling_rate
                if not sliced_time2:
                    fall = sampling_rate
                else:
                    fall = sliced_time2[-1] - sliced_time2[0]
            else:
                rise = sliced_time1[-1] - sliced_time1[0]
                if not sliced_time2:
                    fall = sampling_rate
                else:
                    fall = sliced_time2[-1] - sliced_time2[0]
            
            rises.append(rise)
            falls.append(fall)

        return np.array(rises), np.array(falls)


    def save_to_file(self, path, name, data):
        with h5py.File(path, 'w') as h5:
            h5.create_dataset(name, data=data)
            print('***'+name+' written***')
