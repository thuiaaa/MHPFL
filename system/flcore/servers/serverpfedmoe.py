import torch
import time

import numpy as nn

from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientpfedmoe import clientpfedmoe
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
