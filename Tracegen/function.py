import torch
from pim import Memory

class Buffer():
    def __init__(self, size: int, hbm_index: list, channel_index: list, start_index: list, final_index: list, store_type: bool):
        self.size = size
        self.hbm_index = hbm_index
        self.channel_index = channel_index
        self.start_index = start_index
        self.final_index = final_index
        self.store_type = store_type # true = scatter, false = broadcast
        self.cols_per_bank = size

    def get_index(self, column_size, offset):
        start_idx = self.start_index[0] * column_size + self.start_index[1]
        final_idx = start_idx + offset
        return [final_idx // column_size, final_idx % column_size]

class System(Memory):
    def __init__(self, args):
        super().__init__(args)

    def create_BO(self, size: int, hbm_index: list, channel_index: list, start_index: list, store_type: bool):
        start_idx = start_index[0] * self.DRAM_column + start_index[1]
        if store_type == True: # Scatter, size per bank
            size = size // (len(hbm_index) * len(channel_index) * self.num_bankgroups * (self.num_banks // 2))
        size //= 16
        
        final_idx = start_idx + size
        
        final_index = [final_idx // self.DRAM_column, final_idx % self.DRAM_column]
        #print(final_index)
        return Buffer(size, hbm_index, channel_index, start_index, final_index, store_type)

    def broadcast_to_DRAM_all_bank(self, bo: Buffer, data: torch.tensor, op_trace: bool):
        chunks = data.view(bo.size, 16)
        for bg in range(self.num_bankgroups):
            for bk in range(self.num_banks // 2):
                for idx in range(bo.size):
                    _bk = bk * 2
                    row, col = bo.get_index(self.DRAM_column, idx)
                    if row >= self.DRAM_row:
                        row %= self.DRAM_row
                        _bk += 1
                    for hbm in bo.hbm_index:
                        for ch in bo.channel_index:
                            self.store_to_DRAM_single_bank(hbm, ch, bg, bk, row, col, 2, chunks[idx].contiguous(), op_trace)

    def scatter_to_DRAM_all_bank(self, bo: Buffer, data: torch.tensor, op_trace: bool):
        num_iters = len(bo.hbm_index) * len(bo.channel_index)
        chunk_size = len(data) // (num_iters * 16)
        chunks = data.view(-1, 16)
        i = 0
        for hbm in bo.hbm_index:
            for ch in bo.channel_index:
                p_data = chunks[i : i + chunk_size]
                bg_size = chunk_size // self.num_bankgroups
                bk_size = chunk_size // (self.num_bankgroups * (self.num_banks//2))
                for idx in range(chunk_size):
                    bg = idx // bg_size
                    bk = (idx % bg_size) // bk_size
                    bk = bk * 2
                    row, col = bo.get_index(self.DRAM_column, (idx % bk_size))
                    if row >= self.DRAM_row:
                        row %= self.DRAM_row
                        bk += 1
                    self.store_to_DRAM_single_bank(hbm, ch, bg, bk, row, col, 2, p_data[idx].contiguous(), op_trace)
                i += chunk_size

    def gather_from_DRAM_all_bank(self, bo: Buffer, op_trace: bool):
        data = []
        for hbm in bo.hbm_index:
            for ch in bo.channel_index:
                for bg in range(self.num_bankgroups):
                    for bk in range(self.num_banks // 2):
                        for idx in range(bo.size):
                            _bk = bk * 2
                            row, col = bo.get_index(self.DRAM_column, idx)
                            if row >= self.DRAM_row:
                                row %= self.DRAM_row
                                _bk += 1
                            data.append(self.load_from_DRAM_single_bank(hbm, ch, bg, bk, row, col, 2, op_trace))
        return torch.stack(data)
    
    def GEMV_BO(self, in_bo1: Buffer, in_bo2: Buffer, out_bo: Buffer, op_trace: bool):
        num_cols_per_bank = in_bo2.size // in_bo1.size
        num_rfs = self.num_grfs // 2
        num_rfs_out = self.num_grfs // 4

        idx_cur_col = 0
        while idx_cur_col < num_cols_per_bank:
            size_cur_col = min(num_rfs_out, num_cols_per_bank - idx_cur_col) 
            for iter in range(in_bo1.size // num_rfs):
                for rf in range(num_rfs):
                    row, col = in_bo1.get_index(self.DRAM_column, iter * num_rfs + rf)
                    bk = 0
                    if row >= self.DRAM_row:
                        row %= self.DRAM_row
                        bk = 1
                    for hbm in in_bo1.hbm_index:
                        for ch in in_bo1.channel_index:
                            
                            self.PIM_FILL(hbm, ch, bk, row, col, rf, op_trace)
                for rf in range(num_rfs):
                    for rf_col in range(size_cur_col):
                        row, col = in_bo2.get_index(self.DRAM_column, iter * num_rfs + rf + rf_col * in_bo1.size)
                        bk = 0
                        if row >= self.DRAM_row:
                            row %= self.DRAM_row
                            bk = 1
                        for hbm in in_bo2.hbm_index:
                            for ch in in_bo2.channel_index:
                                self.PIM_MAC_RD_BANK(hbm, ch, bk, row, col, rf, num_rfs + rf_col, op_trace)

            for i in range(size_cur_col):
                out_idx = idx_cur_col + i
                row, col = out_bo.get_index(self.DRAM_column, out_idx)
                bk = 0
                if row >= self.DRAM_row:
                    row %= self.DRAM_row
                    bk = 1
                for hbm in out_bo.hbm_index:
                    for ch in out_bo.channel_index:
                        self.PIM_MOVE(hbm, ch, bk, num_rfs + i, row, col, op_trace)
            idx_cur_col += num_rfs_out
        # return self.gather_from_DRAM_all_bank(out_bo, op_trace)
    
