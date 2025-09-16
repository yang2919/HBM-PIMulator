from function import System

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

        Num_per_row = self.burst_length * self.DRAM_column
        Num_total_bank = self.num_channels * self.num_bankgroups * (self.num_banks // 2)

        self.w1_bo = []
        for w1 in self.w1:
            self.w1_bo.append(self.create_BO(len(w1), hbm, channel, [self.row_idx, 0], self.get_bankop(), True))
            self.row_idx += (w1.shape[0] // Num_per_row) * (w1.shape[1] // Num_total_bank)
        
        self.w2_bo = []
        for w2 in self.w2:
            self.w2_bo.append(self.create_BO(len(w2), hbm, channel, [self.row_idx, 0], self.get_bankop(), True))
            self.row_idx += (w2.shape[0] // Num_total_bank) * (w2.shape[1] // Num_per_row)
        
        self.x1_bo = self.create_BO(len(self.x1), hbm, channel, [self.row_idx, 0], self.get_bankop(), False)
        self.row_idx += self.x1.shape[0] // Num_per_row

        self.o1_bo = []
        for _ in range(self.top_k):
            self.o1_bo.append(self.create_BO(self.w1[0].shape[1], hbm, channel, [self.row_idx, 0], self.get_bankop(), True))
            self.row_idx += 1

        self.x2_bo = []
        for _ in range(self.top_k):
            self.x2_bo.append(self.create_BO(self.w2[0].shape[0], hbm, channel, [self.row_idx, 0], self.get_bankop(), True))
            self.row_idx += 1

        self.o2_bo = self.create_BO(self.w2.shape[1], hbm, channel, [self.row_idx, 0], self.get_bankop(), False)
        self.row_idx += self.w2.shape[1] // Num_per_row
