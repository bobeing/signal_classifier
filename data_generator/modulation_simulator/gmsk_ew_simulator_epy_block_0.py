"""

"""

import numpy as np
import h5py
from gnuradio import gr


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """ """

    def __init__(self, filename ='file', signaltype='comm',modulation='qpsk',samprate=1000000,datarate=0,snr=0,freqshift=0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='H5 Writer',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=None
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.filename = filename
        self.signaltype = signaltype
        self.modulation = modulation
        self.samprate = samprate
        self.datarate = datarate
        self.snr = snr
        self.freqshift = freqshift

        hf = h5py.File(self.filename, 'w')
        hf.attrs['signal_type'] =signaltype
        hf.attrs['modulation'] =modulation
        hf.attrs['samprate'] =samprate
        hf.attrs['datarate'] =datarate
        hf.attrs['snr'] =snr
        hf.attrs['freqshift'] =freqshift
        self.hf = hf
        self.init = 0

    def work(self, input_items, output_items):
        hf = h5py.File(self.filename, 'a')
        if self.init == 0:
            hf.create_dataset('signal',data=input_items[0],maxshape=(None,))
            self.init = 1
        elif self.init == 1:
            self.init = 1
            self.hf['signal'].resize((self.hf['signal'].shape[0] + input_items[0].shape[0]),axis=0)
            self.hf['signal'][-input_items[0].shape[0]:] = input_items[0]
        else:
            pass
            #self.hf['signal'].resize((self.hf['signal'].shape[0] + input_items[0].shape[0]),axis=0)
            #self.hf['signal'][-input_items[0].shape[0]:] = input_items[0]
        return len(input_items[0])
            
