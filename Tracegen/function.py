import torch
from pim import Memory

class Buffer():
    def __init__(self, size: int, hbm_index: list, channel_index: list, start_index: list, final_index: list, bank_op: bool , store_type: bool):
        self.size = size
        self.hbm_index = hbm_index
        self.channel_index = channel_index
        self.start_index = start_index
        self.final_index = final_index
        self.bank_op = bank_op # true = even, false = odd
        self.store_type = store_type # true = scatter, false = broadcast

    def get_index(self, column_size, offset):
        start_idx = self.start_index[0] * column_size + self.start_index[1]
        final_idx = start_idx + offset
        return [final_idx // column_size, final_idx % column_size]

class System(Memory):
    def __init__(self, args):
        super().__init__(args)

    def create_BO(self, size: int, hbm_index: list, channel_index: list, start_index: list, bank_op: bool, store_type: bool):
        start_idx = start_index[0] * self.DRAM_column + start_index[1]
        if store_type == True: # Scatter, size per bank
            size = size // (len(hbm_index) * len(channel_index) * self.num_bankgroups * self.num_banks)
        final_idx = start_idx + size
        final_index = [final_idx // self.DRAM_column, final_idx % self.DRAM_column]

        return Buffer(size, hbm_index, channel_index, start_index, final_index, bank_op, store_type)

    def broadcast_to_DRAM_all_bank(self, bo: Buffer, data: list, op_trace: bool):
        for bg in range(self.num_bankgroups):
            for bk in range(self.num_banks):
                if (bk % 2 == 0) is bo.bank_op:
                    for idx in range(bo.size):
                        row, col = bo.get_index(self.DRAM_column, idx)
                        for hbm in bo.hbm_index:
                            for ch in bo.channel_index:
                                self.store_to_DRAM_single_bank(hbm, ch, bg, bk, row, col, 2, data[idx], op_trace)

    def scatter_to_DRAM_all_bank(self, bo: Buffer, data: list, op_trace: bool):
        num_iters = len(bo.hbm_index) * len(bo.channel_index)
        chunk_size = len(data) // num_iters
        i = 0
        for hbm in bo.hbm_index:
            for ch in bo.channel_index:
                p_data = data[i : i + chunk_size]
                bg_size = len(p_data) // self.num_bankgroups
                bk_size = len(p_data) // (self.num_bankgroups * (self.num_banks//2))
                for idx in range(len(p_data)):
                    bg = idx // bg_size
                    bk = (idx % bg_size) // bk_size
                    bk = bk * 2 if bo.bank_op else bk * 2 + 1
                    row, col = bo.get_index(self.DRAM_column, (idx % bk_size))
                    self.store_to_DRAM_single_bank(hbm, ch, bg, bk, row, col, 2, p_data[idx], op_trace)
                i += chunk_size

    def gather_from_DRAM_all_bank(self, bo, op_trace):
        data = []
        for hbm in bo.hbm_index:
            for ch in bo.channel_index:
                for bg in range(self.num_bankgroups):
                    for bk in range(self.num_banks):
                        if (bk % 2 == 0) is bo.bank_op:
                            for idx in range(bo.size):
                                row, col = bo.get_index(self.DRAM_column, idx)
                                data.append(self.load_from_DRAM_single_bank(hbm, ch, bg, bk, row, col, 2, op_trace))
        return data

    def GEMV_BO_PRE(self, in_bo1, in_bo2, out_bo):
        num_rfs = self.num_grfs // 2
        for iter in range(in_bo1.size // num_rfs):
            for rf in range(num_rfs):
                row, col = in_bo1.get_index(self.DRAM_column, iter * num_rfs + rf)
                for hbm in in_bo1.hbm_index:
                    for ch in in_bo1.channel_index:
                        self.PIM_FILL(hbm, ch, 0, row, col, rf, True)
            for rf in range(num_rfs):
                row, col = in_bo2.get_index(self.DRAM_column, iter * num_rfs + rf)
                for hbm in in_bo2.hbm_index:
                    for ch in in_bo2.channel_index:
                        self.PIM_MAC_RD_BANK(hbm, ch, 0, row, col, rf, num_rfs, True)
        for i in range(out_bo.size):
            row, col = out_bo.get_index(self.DRAM_column, i)
            for hbm in out_bo.hbm_index:
                for ch in out_bo.channel_index:
                    self.PIM_MOVE(hbm, ch, 0, num_rfs, row, col, True)
        return self.gather_from_DRAM_all_bank(out_bo, True)
    
