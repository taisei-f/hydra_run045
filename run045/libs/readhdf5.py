#!/usr/bin/env python

import time, glob
import numpy as np
import h5py

#memo
#2023.04.07 ver1.0
#2023.06.20 ver2.0

__author__ =  'Tasuku Hayashi'
__version__=  '1.0' #2023.04.07
__version__=  '2.0' #2023.06.20
__version__=  '2.1' #2023.07.18
__version__=  '2.2' #2023.08.10
__version__=  '2.4' #2023.10.13

print('========================================================')
print(f' readhdf5 version {__version__}')
print('========================================================')

class ReadHDF5:
    def __init__(self):
        self.hdf5List = glob.glob('*.hdf5')
        self.hdf5List.sort()
        for e, hdf5name in enumerate(self.hdf5List):
            print(f'{e}: {hdf5name}')

    def ReadHDF5(self, dathdf5):
        self.dathdf5 = dathdf5
        print('Checking mat2hdf version...')
        with h5py.File(self.dathdf5, 'a') as f:
            ver = f['header']['mat2hdf_Ver'][()].decode('utf-8')

        if ver == __version__:
            print('OK.')
            self.GetHeader()
            #self.OpenHDF5()
        else:
            print('mat2hdf version not match.')
            self.GetHeader()
            #self.OpenHDF5()

    def OpenHDF5(self):
        self.f = h5py.File(self.dathdf5, 'r')
        self.pulse = self.f['waveform']['pulse']
        self.noise = self.f['waveform']['noise']
        self.vres  = self.f['waveform']['vres'][()]
        self.hres  = self.f['waveform']['hres'][()]
        self.time  = np.arange(self.pulse.shape[-1]) * self.hres

    def CloseHDF5(self):
        self.f.close()

    def ReadPhiVHDF5(self, dathdf5):
        self.dathdf5 = dathdf5
        x = {}
        y = {}
        with h5py.File(self.dathdf5, 'a') as f:
            for i in f['waveform'].keys():
                x[i]  = f['waveform'][i]['x'][()]
                y[i]  = f['waveform'][i]['y'][()]
                hres  = f['waveform'][i]['hres'][()]
                hresb = f['waveform'][i]['hresb'][()]
                xt = np.arange(x[i].shape[-1]) * hres
                xt = xt - (xt[-1]/2)
                yt = np.arange(x[i].shape[-1]) * hres
                yt = yt - (yt[-1]/2)

        return xt, x, yt, y

    def GetHeader(self):
        with h5py.File(self.dathdf5, 'a') as f:
            self._MeasurementsDate = f['header']['MeasurementsDate'][()].decode('utf-8')
            self._mat2hdf_Ver      = f['header']['mat2hdf_Ver'][()].decode('utf-8')
            self._MakeDate         = f['header']['MakeDate'][()].decode('utf-8')
            self._NumOfPulseData   = f['header']['NumOfPData'][()]
            self._NumOfFileCreate  = f['header']['SpDataNum'][()]
            if float(self._mat2hdf_Ver) > 2.1:
                self._rebin            = f['header']['rebin'][()]
            self._LOTNo            = f['header']['LOT-No'][()].decode('utf-8')
            self._ChipID           = f['header']['ChipID'][()].decode('utf-8')
            if type(f['header']['Pix.ID'][()]) == str:
                self._PixID            = f['header']['Pix.ID'][()].decode('utf-8')
            else:
                self._PixID            = f['header']['Pix.ID'][()]
            self._DataType         = f['header']['DataType'][()].decode('utf-8')
            self._MagniconCh       = int(f['header']['Magnicon_Ch'][()])
            self._SQUID_Bias       = f['header']['SQUID_Bias'][()]
            self._SQUID_Vb         = f['header']['SQUID_Vb'][()]
            self._SQUID_Phib       = f['header']['SQUID_Phib'][()]
            self._TES_Bias         = f['header']['TES_Bias'][()]
            self._Rfb              = f['header']['Rfb'][()]
            self._GainBW           = f['header']['GainBW'][()]
            self._V_div            = f['header']['V_div'][()]
            self._time_div         = f['header']['time_div'][()]
            self._sample_kS        = f['header']['sample_kS'][()]
            self._bit              = f['header']['bit'][()]
            self._trigger_V        = f['header']['trigger_V'][()]
            self._edge_point       = f['header']['edge_point_percent'][()]
            self._Edge             = f['header']['Edge'][()].decode('utf-8')
            self._UseDirectory     = f['header']['UseDirectory'][()]
            self._AllDirectory     = f['header']['AllDirectory'][()]
            self._dircounts        = len(self._UseDirectory)

    def ShowHeader(self,showDir=False):
        print( ' ==============================================')
        print( ' HDF5 header Information ')
        print(f' mat2hdf version {self._mat2hdf_Ver} ')
        print( ' ==============================================')
        print(f' File name            : {self.dathdf5}')
        print(f' Measurements Date    : {self._MeasurementsDate}')
        print(f' Make Date            : {self._MakeDate}')
        print(f' Num of File Create   : {self._NumOfFileCreate}')
        print(f' Number of Pulse Data : {self._NumOfPulseData}')
        if float(self._mat2hdf_Ver) > 2.1:
            print(f' Down sample bin      : {self._rebin }')
        print( ' --------------------------------------------')
        print( ' TES information')
        print(f' LOT No.              : {self._LOTNo}')
        print(f' Chip ID              : {self._ChipID}')
        print(f' Pixel ID             : {self._PixID}')
        print(f' Data Type            : {self._DataType}')
        print( ' --------------------------------------------')
        print( ' Magnicon Setting')
        print(f' Magnicon Channel     : {self._MagniconCh}')
        print(f' SQUID Bias           : {self._SQUID_Bias} (uA)')
        print(f' SQUID Vb             : {self._SQUID_Vb} (uV)')
        print(f' SQUID Phib           : {self._SQUID_Phib} (uA)')
        print(f' TES_Bias             : {self._TES_Bias} (uA)')
        print(f' RFB                  : {self._Rfb} (kOhm)')
        print(f' Gain BW              : {self._GainBW} (GHz)')
        print( ' --------------------------------------------')
        print( ' PicoScope Setting')
        print(f' V div                : {self._V_div} (V/div)')
        print(f' time div             : {self._time_div} (ms/div)')
        print(f' sample               : {self._sample_kS} (kS)')
        print(f' bit                  : {self._bit} (bit)')
        print(f' trigger              : {self._trigger_V}')
        print(f' edge point           : {self._edge_point} (%)')
        print(f' Edge type            : {self._Edge}')
        print( ' --------------------------------------------')
        print(f' Number of used directory : {self._dircounts}')
        if showDir:
            print( ' Used Directory : ')
            print(f' {self._UseDirectory}')
            print( ' ALL Directory : ')
            print(f' {self._AllDirectory}')
        print( ' ==============================================')
