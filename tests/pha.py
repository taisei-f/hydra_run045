import numpy as np
import h5py
import readhdf5

class Calc_pha:
    def __init__(self, rawdata_path, pixmask):
        """
        Parameters:
        - rawdata_path: string,
        - output_path: string,
        - pixmask: 2d-array,

        Attributes:
        - 
        """
        self.rawdata_path = rawdata_path
        self.pixmask = pixmask

    def get_raw(self):
        """
        Parameters:
        - rawdata_path: path of a raw data file
        - downsample: downsampled data or not
        
        Returns:
        time, pulse, noise
        """
        
        print("########################################")
        print("# get raw data")
        print("########################################")

        self.raw = readhdf5.ReadHDF5()
        self.raw.ReadHDF5(self.rawdata_path)
        self.raw.OpenHDF5()
        self.time = self.raw.time
        self.pulse = self.raw.pulse
        self.noise = self.raw.noise

    def close_raw(self):
        self.raw.CloseHDF5()


    def noise_pow(self, pix_n):
        """
        average of power spectrum of noise
        """
        fourier = []
        for oneevent in pix_n:
            fourier.append(np.fft.fft(oneevent))

        fourier = np.array(fourier)
        one_pow = np.abs(fourier)**2
        return np.mean(one_pow, axis=0)


    def pulse_pow(self, ave_p):
        """
        power spectrum of average pulse
        """
        fourier= np.fft.fft(ave_p)
        return np.abs(fourier)**2


    def calc_pha(self, downsample, output_path):
        if(downsample):
            sampling_rate = 2.5e-8 # sec
        elif(~downsample):
            sampling_rate = 2.5e-8 # sec
        else:
            raise ValueError("variable downsample should be specified by True or False")

        # get raw data
        self.get_raw(self.rawdata_path)
        N_sample = len(self.time)
        self.freq = np.fft.fftfreq(N_sample, sampling_rate)

        with h5py.File(output_path, 'a') as file:
            keywords = ['AveragePulse', 'SN', 'Template', 'PHA']
            for key in keywords:
                if key not in file:
                    file.create_group(key)

            # calc template & pha for each pixel
            for i in range(4):
                pixnum = 'Pix.'+str(i)
                print('calculating PHA of '+pixnum)
                # obtain raw data of one pixel
                pix_pulse = self.pulse[self.pixmask[i]]
                pix_noise = self.noise[self.pixmask[i]]

                # average pulse
                ave_pulse = np.mean(pix_pulse, axis=0)

                # S/N ratio
                sn = np.sqrt(self.pulse_pow(ave_pulse)/self.noise_pow(pix_noise))

                # template
                tem = np.fft.ifft(np.fft.fft(ave_pulse)/self.noise_pow(pix_noise)).real
                norm = (np.max(ave_pulse)-np.min(ave_pulse))/(np.sum(tem*ave_pulse)/len(ave_pulse))
                template = tem*norm

                # obtain PHA for each event
                pha = []
                for oneevent in pix_pulse:
                    pha.append(np.sum(oneevent*template)/10**8)

                file.create_dataset(name='AveragePulse/'+pixnum, data=ave_pulse)
                file.create_dataset(name='SN/'+pixnum, data=sn)
                file.create_dataset(name='Template/'+pixnum, data=template)
                file.create_dataset(name='PHA/'+pixnum, data=pha)

                print('written in : AveragePulse/'+pixnum+' in '+output_path)
                print('written in : SN/'+pixnum+' in '+output_path)
                print('written in : Template/'+pixnum+' in '+output_path)
                print('written in : PHA/'+pixnum+' in '+output_path)
                print("")
