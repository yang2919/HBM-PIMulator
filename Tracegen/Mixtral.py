from function import System
import torch.nn.functional as F
import torch

class ModelMixtral(System):
    def __init__(self, dic_model, args):
        super().__init__(args)
        self.dim = args.dim
        self.dim_expert = args.dim_expert
        self.n_expert = args.n_expert
        self.top_k = args.top_k

        self.x1 = dic_model["x1"]
        self.w1 = []
        self.w2 = []
        for i in range(self.n_expert):  
            exp = "expert" + str(i)
            self.w1.append(dic_model["w1"][exp])
            self.w2.append(dic_model["w2"][exp])

    def get_bankop(self):
        return self.row_idx < self.DRAM_row

    def set_mapping(self):
        self.row_idx = 0
        hbm = [0]
        channel = range(self.num_channels)

        self.w1_bo = []
        for w1 in self.w1:
            self.w1_bo.append(self.create_BO(len(w1), hbm, channel, [self.row_idx, 0], True))
            self.row_idx += self.w1_bo[-1].size // self.DRAM_column
        
        self.w2_bo = []
        for w2 in self.w2:
            self.w2_bo.append(self.create_BO(len(w2), hbm, channel, [self.row_idx, 0], True))
            self.row_idx += self.w2_bo[-1].size // self.DRAM_column
        
        self.x1_bo = self.create_BO(len(self.x1), hbm, channel, [self.row_idx, 0], False)
        self.row_idx += self.x1_bo.size // self.DRAM_column

        self.o1_bo = []
        for _ in range(self.top_k):
            self.o1_bo.append(self.create_BO(self.dim_expert * 16, hbm, channel, [self.row_idx, 0], True))
            self.row_idx += (self.o1_bo[-1].size // self.DRAM_column) + 1

        self.x2_bo = []
        for _ in range(self.top_k):
            self.x2_bo.append(self.create_BO(self.dim_expert, hbm, channel, [self.row_idx, 0], True))
            self.row_idx += (self.x2_bo[-1].size // self.DRAM_column) + 1

        self.o2_bo = []
        for _ in range(self.top_k):
            self.o2_bo.append(self.create_BO(self.dim * 16, hbm, channel, [self.row_idx, 0], False))
            self.row_idx += (self.o2_bo[-1].size // self.DRAM_column) + 1

        print("Mapping finished... # of rows: ", self.row_idx)

    def weight_mapping(self, op_trace):
        for w1_bo, w1 in zip(self.w1_bo, self.w1):
            self.scatter_to_DRAM_all_bank(w1_bo, w1, op_trace)
        for w2_bo, w2 in zip(self.w2_bo, self.w2):
            self.scatter_to_DRAM_all_bank(w2_bo, w2, op_trace)

        print("Weight matrix stored")

    def gating(self):
        self.top_experts = [0, 1]

    def FFN_ref(self):
        expert_outputs = []

        for i in range(1):
            expert_idx = self.top_experts[i]

            w1 = self.w1[expert_idx].view(self.dim_expert, self.dim)
            o1 = (self.x1 * w1).sum(dim=1)

            x2 = o1
            
            w2 = self.w2[expert_idx].view(self.dim, self.dim_expert)
            o2 = (x2 * w2).sum(dim=1)
            
            expert_outputs.append(o2)
        final_output = torch.stack(expert_outputs).sum(dim=0)
        
        return final_output


    def FFN_PIM(self, op_trace):
        # x1 Broadcast
        self.broadcast_to_DRAM_all_bank(self.x1_bo, self.x1, op_trace)

        # x1 * W1 GEMV
        for i in range(self.top_k):
            self.PIM_GEMV_BO(self.x1_bo, self.w1_bo[self.top_experts[i]], self.o1_bo[i], op_trace)

        # o1 All-gather
        self.o1 = []
        for i in range(self.top_k):
            self.o1.append(self.gather_from_DRAM_all_bank(self.o1_bo[i], op_trace))
            self.o1[-1] = self.o1[-1].sum(dim=1)

        # Activation - SwiGLU
        self.x2 = self.o1

        # x2 Scatter
        for i in range(self.top_k):
            self.scatter_to_DRAM_all_bank(self.x2_bo[i], self.x2[i], op_trace)

        # x2 * W2 GEMV
        for i in range(self.top_k):
            self.PIM_GEMV_BO(self.x2_bo[i], self.w2_bo[self.top_experts[i]], self.o2_bo[i], op_trace)

        # o2 All-reduce
        self.o2 = []
        for i in range(self.top_k):
            self.o2.append(self.reduce_from_DRAM_all_bank(self.o2_bo[i], op_trace))
        #print(self.o2[0])

        print("FFN completed")
        return self.o2[0].sum(dim=0)