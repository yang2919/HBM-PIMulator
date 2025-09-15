import torch
from pim import Memory

class Buffer():
    def __init__(self, size, hbm_index, channel_index, start_index, final_index, store_type):
        self.size = size
        self.hbm_index = hbm_index
        self.channel_index = channel_index
        self.start_index = start_index
        self.final_index = final_index
        self.store_type = store_type # true = scatter, false = broadcast

    def get_index(self, column_size, offset):
        start_idx = self.start_index[0] * column_size + self.start_index[1]
        final_idx = start_idx + offset
        return [final_idx // column_size, final_idx % column_size]

class System(Memory):
    def __init__(self, args):
        super().__init__(args)

    def create_BO(self, size, hbm_index, channel_index, start_index, store_type):
        start_idx = start_index[0] * self.DRAM_column + start_index[1]
        if store_type == True: # Scatter
            size = size // (self.num_bankgroups * self.num_banks)
        final_idx = start_idx + size
        final_index = [final_idx // self.DRAM_column, final_idx % self.DRAM_column]

        return Buffer(size, hbm_index, channel_index, start_index, final_index, store_type)

    def broadcast_to_DRAM_all_bank(self, bo, data, op_trace):
        for bg in range(self.num_bankgroups):
            for bk in range(self.num_banks):
                for idx in range(bo.size):
                    row = idx // self.DRAM_column
                    col = idx % self.DRAM_column
                    self.store_to_DRAM_single_bank(bo.hbm_index, bo.channel_index, bg, bk, row, col, 2, data[idx], op_trace)

    def scatter_to_DRAM_all_bank(self, bo, data, op_trace):
        bg_size = bo.size // self.num_bankgroups
        bk_size = bo.size // (self.num_bankgroups * self.num_banks)
        for idx in range(bo.size):
            bg = idx // bg_size
            bk = bg // bk_size
            row = (idx % bk_size) // self.DRAM_column
            col = (idx % bk_size) % self.DRAM_column
            self.store_to_DRAM_single_bank(bo.hbm_index, bo.channel_index, bg, bk, row, col, 2, data[idx], op_trace)

    def gather_from_DRAM_all_bank(self, bo, op_trace):
        data = []
        for bg in range(self.num_bankgroups):
            for bk in range(self.num_banks):
                for idx in range(bo.size):
                    row = idx // self.DRAM_column
                    col = idx % self.DRAM_column
                    data.append(self.load_from_DRAM_single_bank(bo.hbm_index, bo.channel_index, bg, bk, row, col, 2, op_trace))
        return data

    def GEMV_BO_PRE(self, in_bo1, in_bo2, out_bo):
        num_rfs = self.num_grfs // 2
        for iter in range(in_bo1.size // num_rfs):
            for rf in range(num_rfs):
                index = in_bo1.get_index(self.DRAM_column, iter * num_rfs + rf)
                self.PIM_FILL(in_bo1.hbm_index, in_bo1.channel_index, 0, index[0], index[1], rf, True)
            for rf in range(num_rfs):
                index = in_bo2.get_index(self.DRAM_column, iter * num_rfs + rf)
                self.PIM_MAC_RD_BANK(in_bo2.hbm_index, in_bo2.channel_index, 0, index[0], index[1], rf, num_rfs, True)
        for i in range(out_bo.size):
            index = out_bo.get_index(self.DRAM_column, i)
            self.PIM_MOVE(out_bo.hbm_index, out_bo.channel_index, 0, num_rfs, index[0], index[1], True)
        return self.gather_from_DRAM_all_bank(out_bo, True)
    
