import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import random
import yaml
from tqdm import tqdm
from utils import utils
import signal



def sig_start(tb, options=None):
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    tb.start()
    tb.wait()

if __name__ == "__main__":

    config_path = './data_generator/simulator_config.yaml'
    save_dir = "./dataset"
    N_train_val_test = {"train":3000, "val":200, "test":200}
    modulation_type_list = {
        0: "AFSK", 
        1: "AM",
        2: "BPSK",
        3: "CP",
        4: "CPFSK",
        5: "FM",
        6: "GFSK",
        7: "GMSK",
        8: "OFDM",
        9: "QPSK",
        # 10: "QAM",
        # 11: "FSK",
        # 12: "4FSK",
        # 13: "APSK",
    }
    # save_path = 
    samp_rate = 9600*1000
    sat1_snr = 15
    baud_rate = 9600

    conf = utils.open_yaml(config_path)
    os.makedirs(save_dir, exist_ok=True)

    for task, n_task in N_train_val_test.items():
        os.makedirs(os.path.join(save_dir, task), exist_ok=True)
        f = open(os.path.join(save_dir, task+".csv"), "w")

        for i in tqdm(range(n_task), desc= f"making {task} data ..."):
            save_path = os.path.join(save_dir, task, f"{i:0>7}.h5")
            mod_type = random.randint(0, 9)
            modulation_type = modulation_type_list[mod_type]
            
            class_ = utils.initialize_module(conf[modulation_type]["path"])
            class_.initialize(save_path, samp_rate, sat1_snr, baud_rate)
            sig_start(class_)

            f.write(f'{save_path},{mod_type},{samp_rate},{baud_rate},{sat1_snr}\n')
        
        f.close()






