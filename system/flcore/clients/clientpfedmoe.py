import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from flcore.trainmodel.pFMcnn import *
from sklearn import metrics
from sklearn.preprocessing import label_binarize


class clientpfedmoe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        self.global_expert_model = pFMCNN_5().to(self.device)
        self.local_expert_model = eval(args.models[self.id % len(args.models)]).to(self.device)
        self.gate = pfedmoe_gate(m=128).to(self.device)
        self.head = pfedmoe_head().to(self.device)

        all_params = list(self.global_expert_model.parameters()) + \
                    list(self.local_expert_model.parameters()) + \
                    list(self.gate.parameters()) + \
                    list(self.head.parameters())
        self.optimizer = torch.optim.SGD(all_params, lr=self.learning_rate)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        [model.eval() for model in [self.global_expert_model, self.local_expert_model, self.gate, self.head]]

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播
                global_out = self.global_expert_model.forward_nohead(x)
                local_out = self.local_expert_model.forward_nohead(x)
                gate_weights = self.gate(x)
                fused_output = gate_weights[:, 0].unsqueeze(1)*global_out + gate_weights[:, 1].unsqueeze(1)*local_out
                output = self.head(fused_output)

                # 计算指标
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        # 处理预测结果
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # 计算AUC（修复版本）
        if self.num_classes == 2:
            auc = metrics.roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true = label_binarize(y_true, classes=np.arange(self.num_classes))
            auc = metrics.roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

        return test_acc, test_num, auc

    def train(self):
        # print(self.global_expert_model,self.local_expert_model)

        trainloader = self.load_train_data()

        self.global_expert_model.train()
        self.local_expert_model.train()
        self.gate.train()
        self.head.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                #实现pFedMoE的本地训练
                global_out = self.global_expert_model.forward_nohead(x)
                local_out = self.local_expert_model.forward_nohead(x)
                gate_weights = self.gate(x)

                fused_output = (gate_weights[:, 0].unsqueeze(1) * global_out + gate_weights[:, 1].unsqueeze(1) * local_out)

                output = self.head(fused_output)

                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time