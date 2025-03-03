import torch
import time

import numpy as nn

from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientpfedmoe import clientpfedmoe
from flcore.trainmodel.pFMcnn import *

from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict



class pFedMoE(Server):
    def __init__(self, args, times):
        super().__init__()

        self.set_slow_clients()
        self.set_clients(clientpfedmoe)

        modellist = {"CNN_1":pFMCNN_1(),"CNN_2":pFMCNN_2(), "CNN_3":pFMCNN_3(), "CNN_4":pFMCNN_4(), "CNN_5":pFMCNN_5()}
        self.global_expert_model = modellist[args.global_expert_model]
        print(f"self.global_expert_model is :{self.global_expert_model}")
        for name, param in self.global_expert_model.named_parameters():
            print(f"Parameter: {name} | Mean: {param.data.mean()} | Std: {param.data.std()}")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

    def train(self):

        self.send_share_expert(self.global_expert_model)

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()




