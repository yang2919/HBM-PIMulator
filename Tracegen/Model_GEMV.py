from function_v02 import System

class ModelGEMV(System):
    def __init__(self, dic_model, args):
        super().__init__(args)
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim

        self.x = dic_model["x"]
        self.w = dic_model["w"]

    def set_mapping(self):
        self.row_idx = 0
        hbm = [0]
        channel = range(self.num_channels)

        self.w_bo = self.create_BO(self.in_dim * self.out_dim, hbm, channel, [self.row_idx, 0], True)
        self.row_idx += self.w_bo.size // self.DRAM_column

        self.o_bo = self.create_BO(self.out_dim * 16, hbm, channel, [self.row_idx, 0], False)
        self.row_idx += (self.o_bo.size // self.DRAM_column)+1

        print("Mapping finished... # of rows: ", self.row_idx)

    def weight_storing(self, op_trace=False):
        self.scatter_to_DRAM_all_bank(self.w_bo, self.w, op_trace)

        print("Weight matrix stored to banks...")

    def GEMV_PIM(self, op_trace):
        # x * W GEMV
        self.PIM_GEMV(self.x, self.w_bo, self.o_bo, op_trace)

        # output All-gather
        self.o = self.gather_from_DRAM_all_bank(self.o_bo, op_trace)

        print("GEMV operation finished...")
        return self.o.sum(dim=0)

    