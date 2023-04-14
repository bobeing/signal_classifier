#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM QPSK EW Single Burst Simulator
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
import ofdm_qpsk_ew_simulator_epy_block_0 as epy_block_0  # embedded python block




class ofdm_qpsk_ew_simulator(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "OFDM QPSK EW Single Burst Simulator", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.speedoflight = speedoflight = 299792
        self.space = space = 1200
        self.sat1_rangerate = sat1_rangerate = 0.12
        self.sat1_range = sat1_range = 517
        self.samp_rate = samp_rate = 9600*1000
        self.mark = mark = 5200
        self.center_frequency = center_frequency = 430000000
        self.sensor_samp = sensor_samp = samp_rate/100
        self.sat1_snr = sat1_snr = 15
        self.sat1_doppler = sat1_doppler = -center_frequency*(2*sat1_rangerate/speedoflight)
        self.sat1_delay = sat1_delay = sat1_range/speedoflight
        self.len_tag_key = len_tag_key = "packet_len"
        self.initial_time_offset = initial_time_offset = 0.1
        self.fft_len = fft_len = 64
        self.deviation = deviation = mark-space
        self.baud_rate = baud_rate = 9600
        self.audio_samp = audio_samp = 32000

    def initialize(self, save_path, samp_rate, sat1_snr, baud_rate):
        self.samp_rate = samp_rate
        self.sensor_samp = samp_rate/100
        self.sat1_snr = sat1_snr 
        self.baud_rate = baud_rate 
        ##################################################
        # Blocks
        ##################################################
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=(int(self.samp_rate/self.sensor_samp)),
                decimation=1,
                taps=[],
                fractional_bw=0)
        self.mmse_resampler_xx_0_0_1 = filter.mmse_resampler_cc((((self.initial_time_offset + self.sat1_delay) * self.sensor_samp-int((self.initial_time_offset + self.sat1_delay) * self.sensor_samp))), (int(self.samp_rate/self.sensor_samp)))
        self.epy_block_0 = epy_block_0.blk(filename=save_path, signaltype='comm', modulation='ofdm', samprate=self.samp_rate, datarate=self.baud_rate, snr=self.sat1_snr, freqshift=self.sat1_doppler)
        self.digital_ofdm_tx_0 = digital.ofdm_tx(
            fft_len=self.fft_len,
            cp_len=(self.fft_len//4),
            packet_length_tag_key=self.len_tag_key,
            occupied_carriers=((-4,-3,-2,-1,1,2,3,4),),
            pilot_carriers=((-6,-5,5,6),),
            pilot_symbols=((-1,1,-1,1),),
            sync_word1=None,
            sync_word2=None,
            bps_header=2,
            bps_payload=2,
            rolloff=0,
            debug_log=False,
            scramble_bits=False)
        self.channels_impairments_0 = channels.impairments(0, 0, 0, 0, 0, 0, 0, 0)
        self.channels_channel_model_0_0 = channels.channel_model(
            noise_voltage=(numpy.power(10,-self.sat1_snr/10)),
            frequency_offset=(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate),
            epsilon=1.0,
            taps=[1],
            noise_seed=0,
            block_tags=False)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 50, self.len_tag_key)
        # self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 1000))), False)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 10000))), False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.digital_ofdm_tx_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.channels_impairments_0, 0))
        self.connect((self.channels_impairments_0, 0), (self.mmse_resampler_xx_0_0_1, 0))
        self.connect((self.digital_ofdm_tx_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.mmse_resampler_xx_0_0_1, 0), (self.epy_block_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.channels_channel_model_0_0, 0))


    def get_speedoflight(self):
        return self.speedoflight

    def set_speedoflight(self, speedoflight):
        self.speedoflight = speedoflight
        self.set_sat1_delay(self.sat1_range/self.speedoflight)
        self.set_sat1_doppler(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight))
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))

    def get_space(self):
        return self.space

    def set_space(self, space):
        self.space = space
        self.set_deviation(self.mark-self.space)

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
        self.set_sensor_samp(self.samp_rate/100)
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))
        self.epy_block_0.samprate = self.samp_rate
        self.mmse_resampler_xx_0_0_1.set_resamp_ratio((int(self.samp_rate/self.sensor_samp)))

    def get_mark(self):
        return self.mark

    def set_mark(self, mark):
        self.mark = mark
        self.set_deviation(self.mark-self.space)

    def get_center_frequency(self):
        return self.center_frequency

    def set_center_frequency(self, center_frequency):
        self.center_frequency = center_frequency
        self.set_sat1_doppler(-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight))
        self.channels_channel_model_0_0.set_frequency_offset((-self.center_frequency*(2*self.sat1_rangerate/self.speedoflight)/self.samp_rate))

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

    def get_len_tag_key(self):
        return self.len_tag_key

    def set_len_tag_key(self, len_tag_key):
        self.len_tag_key = len_tag_key

    def get_initial_time_offset(self):
        return self.initial_time_offset

    def set_initial_time_offset(self, initial_time_offset):
        self.initial_time_offset = initial_time_offset

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len

    def get_deviation(self):
        return self.deviation

    def set_deviation(self, deviation):
        self.deviation = deviation

    def get_baud_rate(self):
        return self.baud_rate

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
        self.epy_block_0.datarate = self.baud_rate

    def get_audio_samp(self):
        return self.audio_samp

    def set_audio_samp(self, audio_samp):
        self.audio_samp = audio_samp




def main(top_block_cls=ofdm_qpsk_ew_simulator, options=None):
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
