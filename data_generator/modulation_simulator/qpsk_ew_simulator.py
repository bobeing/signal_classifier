#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: QPSK EW Single Burst Simulator
# Author: Hongrae Kim
# Copyright: Soletop. Co., Ltd
# GNU Radio version: 3.10.3.0

from gnuradio import blocks
import numpy
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import h5py
import qpsk_ew_simulator_epy_block_0 as epy_block_0  # embedded python block




class qpsk_ew_simulator(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "QPSK EW Single Burst Simulator", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.baud_rate = baud_rate = 10000
        self.speedoflight = speedoflight = 299792
        self.sat1_rangerate = sat1_rangerate = 0.12
        self.sat1_range = sat1_range = 517
        self.samp_rate = samp_rate = baud_rate*1000
        self.excess_bw = excess_bw = 0.35
        self.center_frequency = center_frequency = 430000000
        self.var_constellation = var_constellation = digital.constellation_qpsk().base()
        self.sensor_samp = sensor_samp = samp_rate/100
        self.sat1_snr = sat1_snr = 15
        self.sat1_doppler = sat1_doppler = -center_frequency*(2*sat1_rangerate/speedoflight)
        self.sat1_delay = sat1_delay = sat1_range/speedoflight
        self.rrc_taps = rrc_taps =  firdes.root_raised_cosine (1.0, samp_rate, samp_rate/1000, excess_bw, 11*1000)
        self.initial_time_offset = initial_time_offset = 0.1

    def initialize(self, save_path, samp_rate, sat1_snr, baud_rate):
        self.samp_rate = samp_rate
        self.sensor_samp = samp_rate/100
        self.sat1_snr = sat1_snr 
        self.baud_rate = baud_rate 
        ##################################################
        # Blocks
        ##################################################
        self.mmse_resampler_xx_0_0_1 = filter.mmse_resampler_cc((((self.initial_time_offset + self.sat1_delay) * self.sensor_samp-int((self.initial_time_offset + self.sat1_delay) * self.sensor_samp))), (int(self.samp_rate/self.sensor_samp)))
        self.epy_block_0 = epy_block_0.blk(filename=save_path, signaltype='comm', modulation='qpsk', samprate=self.samp_rate, datarate=self.baud_rate, snr=self.sat1_snr, freqshift=self.sat1_doppler)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=self.var_constellation,
            differential=True,
            samples_per_symbol=(int(self.samp_rate/self.baud_rate)),
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False,
            truncate=False)
        self.channels_impairments_0 = channels.impairments(0, 0, 0, 0, 0, 0, 0, 0)
        self.channels_channel_model_0_0 = channels.channel_model(
            noise_voltage=(numpy.power(10,-self.sat1_snr/10)),
            frequency_offset=(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate),
            epsilon=1.0,
            taps=[1],
            noise_seed=0,
            block_tags=False)
        # self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 4, 120))), False)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 4, 1200))), False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.channels_impairments_0, 0))
        self.connect((self.channels_impairments_0, 0), (self.mmse_resampler_xx_0_0_1, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.channels_channel_model_0_0, 0))
        self.connect((self.mmse_resampler_xx_0_0_1, 0), (self.epy_block_0, 0))


    def get_baud_rate(self):
        return self.baud_rate

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
        self.set_samp_rate(self.baud_rate*1000)
        self.epy_block_0.datarate = self.baud_rate

    def get_speedoflight(self):
        return self.speedoflight

    def set_speedoflight(self, speedoflight):
        self.speedoflight = speedoflight
        self.set_sat1_delay(self.sat1_range/self.speedoflight)
        self.set_sat1_doppler(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight))
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))

    def get_sat1_rangerate(self):
        return self.sat1_rangerate

    def set_sat1_rangerate(self, sat1_rangerate):
        self.sat1_rangerate = sat1_rangerate
        self.set_sat1_doppler(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight))
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))

    def get_sat1_range(self):
        return self.sat1_range

    def set_sat1_range(self, sat1_range):
        self.sat1_range = sat1_range
        self.set_sat1_delay(self.sat1_range/self.speedoflight)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rrc_taps( firdes.root_raised_cosine (1.0, self.samp_rate, self.samp_rate/1000, self.excess_bw, 11*1000))
        self.set_sensor_samp(self.samp_rate/100)
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))
        self.epy_block_0.samprate = self.samp_rate
        self.mmse_resampler_xx_0_0_1.set_resamp_ratio((int(self.samp_rate/self.sensor_samp)))

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.set_rrc_taps( firdes.root_raised_cosine (1.0, self.samp_rate, self.samp_rate/1000, self.excess_bw, 11*1000))

    def get_center_frequency(self):
        return self.center_frequency

    def set_center_frequency(self, center_frequency):
        self.center_frequency = center_frequency
        self.set_sat1_doppler(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight))
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))

    def get_var_constellation(self):
        return self.var_constellation

    def set_var_constellation(self, var_constellation):
        self.var_constellation = var_constellation

    def get_sensor_samp(self):
        return self.sensor_samp

    def set_sensor_samp(self, sensor_samp):
        self.sensor_samp = sensor_samp
        self.mmse_resampler_xx_0_0_1.set_resamp_ratio((int(self.samp_rate/self.sensor_samp)))

    def get_sat1_snr(self):
        return self.sat1_snr

    def set_sat1_snr(self, sat1_snr):
        self.sat1_snr = sat1_snr
        self.channels_channel_model_0_0.set_noise_voltage((numpy.power(10,-self.sat1_snr/10)))
        self.epy_block_0.snr = self.sat1_snr

    def get_sat1_doppler(self):
        return self.sat1_doppler

    def set_sat1_doppler(self, sat1_doppler):
        self.sat1_doppler = sat1_doppler
        self.epy_block_0.freqshift = self.sat1_doppler

    def get_sat1_delay(self):
        return self.sat1_delay

    def set_sat1_delay(self, sat1_delay):
        self.sat1_delay = sat1_delay

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_initial_time_offset(self):
        return self.initial_time_offset

    def set_initial_time_offset(self, initial_time_offset):
        self.initial_time_offset = initial_time_offset




def main(top_block_cls=qpsk_ew_simulator, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
