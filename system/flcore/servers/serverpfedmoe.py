import torch
import time

import numpy as nn

from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientpfedmoe import clientpfedmoe
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict


class pFedMoE(Server):
    def __init__(self, args, times):
        super().__init__()

        self.set_slow_clients()
        self.set_clients(clientpfedmoe)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

    def train(self):


