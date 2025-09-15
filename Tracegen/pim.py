import torch
torch.multiprocessing.set_sharing_strategy('file_system')

class Bank():
    def __init__(self, args):
        self.DRAM_column = args.DRAM_column
        self.DRAM_row = args.DRAM_row
        self.burst_length = args.burst_length
        self.arrays = 0 if args.only_trace else torch.zeros((self.DRAM_row, self.DRAM_column, self.burst_length), dtype=torch.float16)

class PIM(Bank):
    def __init__(self, args):
        super().__init__(args)
        self.PIM_grf = args.PIM_grf
        self.PIM_srf = args.PIM_srf
        self.burst_length = args.burst_length
        self.grfs = 0 if args.only_trace else torch.zeros(torch.Size([self.PIM_grf, self.burst_length]))
        self.srfs = 0 if args.only_trace else torch.zeros(torch.Size([self.burst_length]))
        self.bank = {}
        self.bank[0] = Bank(args)
        self.bank[1] = Bank(args)

class BankGroup(PIM):
    def __init__(self, args):
        super().__init__(args)
        self.num_banks = args.num_banks
        self.bankgroup = {}
        self.pim = {}
        for _pim in range(self.num_banks // 2):
            self.pim[_pim] = PIM(args)
            self.bankgroup[_pim * 2 + 0] = self.pim[_pim].bank[0]
            self.bankgroup[_pim * 2 + 1] = self.pim[_pim].bank[1]


class Channel(BankGroup):
    def __init__(self, args):
        super().__init__(args)
        self.num_groups = args.num_groups
        self.channel = {}
        for bg in range(self.num_groups):
            self.channel[bg] = BankGroup(args)

class Device(Channel):
    def __init__(self, args):
        super().__init__(args)
        self.num_channels = args.num_channels
        self.HBM = {}
        for channel in range(self.num_channels):
            self.HBM[channel] = Channel(args)

class Memory():
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, args):
        self.DRAM_column = args.DRAM_column
        self.DRAM_row = args.DRAM_row
        self.burst_length = args.burst_length
        self.num_banks = args.num_banks
        self.num_bankgroups = args.num_bankgroups
        self.num_channels = args.num_channels
        self.num_grfs = args.PIM_grf
        self.pim_device = {}
        if not args.only_trace:
            if args.model_parallel:
                for i in range(args.FC_devices):
                    self.pim_device[i] = Device(args)
            else:
                self.pim_device[0] = Device(args)
        self.op_trace = args.op_trace
        self.trace_file = args.trace_file
        self.file = open(self.trace_file, "w")

    def address(self, hbm_index, channel_index, bankgroup_index, bank_index, row_index, col):
        bank_size = self.DRAM_column * self.DRAM_row
        bankgroup_size = bank_size * self.num_bankgroups
        channel_size = bankgroup_size * self.num_banks
        hbm_size = channel_size * self.num_channels
        addr = hbm_index * hbm_size + channel_index * channel_size + bankgroup_index * bankgroup_size + bank_index * bank_size + row_index * self.DRAM_column + col
        return addr
    
    def store_to_DRAM_single_bank(self, hbm_index, channel_index, bankgroup_index, bank_index, row_index, col_index, size, data, op_trace):
        # HBM2 stores with 32B granularity
        if op_trace and hbm_index == 0:
            for i in range((size - 1) // self.burst_length + 1):
                self.file.write("W {}\n".format(self.address(hbm_index, channel_index, bankgroup_index, bank_index, row_index, col_index)))
        self.pim_device[hbm_index].HBM[channel_index].channel[bankgroup_index].bankgroup[bank_index].arrays[row_index][col_index] = data

    def load_from_DRAM_single_bank(self, hbm_index, channel_index, bankgroup_index, bank_index, row_index, col_index, size, op_trace):
        # HBM2 stores with 32B granularity
        if op_trace and hbm_index == 0:
            for i in range((size - 1) // self.burst_length + 1):
                self.file.write("R {}\n".format(self.address(hbm_index, channel_index, bankgroup_index, bank_index, row_index, col_index)))
        return self.pim_device[hbm_index].HBM[channel_index].channel[bankgroup_index].bankgroup[bank_index].arrays[row_index][col_index]

    def PIM_FILL(self, hbm_index, channel_index, bank_index, row_index, col_index, dst_index, op_trace):
        if op_trace and hbm_index == 0:
            self.file.write("PIM FILL {} BANK,{},{} GRF,{}\n".format(channel_index, row_index, col_index, dst_index))
        for bg in range(self.num_bankgroups):
            for _pim in range(self.num_banks // 2):
                A = self.load_from_DRAM_single_bank(hbm_index, channel_index, bg, _pim + bank_index, row_index, col_index, self.burst_length, False)
                self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[dst_index] = A

    def PIM_MOVE(self, hbm_index, channel_index, bank_index, src_index, row_index, col_index, op_trace):
        if op_trace and hbm_index == 0:
            self.file.write("PIM MOVE {} GRF,{} BANK,{},{} \n".format(channel_index, src_index, row_index, col_index))
        for bg in range(self.num_bankgroups):
            for _pim in range(self.num_banks // 2):
                A = self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[src_index]
                self.store_to_DRAM_single_bank(hbm_index, channel_index, bg, _pim + bank_index, row_index, col_index, self.burst_length, A, False)

    def PIM_MAC_RD_BANK(self, hbm_index, channel_index, bank_index, row_index, col_index, src_index, dst_index, op_trace):
        if op_trace and hbm_index == 0:
            self.file.write("PIM MAC {} BANK,{},{} GRF,{} GRF,{}\n".format(channel_index, row_index, col_index, src_index, dst_index))
        for bg in range(self.num_bankgroups):
            for _pim in range(self.num_banks // 2):
                A = self.load_from_DRAM_single_bank(hbm_index, channel_index, bg, _pim + bank_index, row_index, col_index, self.burst_length, False)
                B = self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[src_index]
                self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[dst_index] += self.MAC(A, B, False)

    def PIM_MAC_ONLY_RF(self, hbm_index, channel_index, src1_index, src2_index, dst_index, op_trace):
        if op_trace and hbm_index == 0:
            self.file.write("PIM MAC {} GRF,{} GRF,{} GRF,{}\n".format(channel_index, src1_index, src2_index, dst_index))
        for bg in range(self.num_bankgroups):
            for _pim in range(self.num_banks // 2):
                A = self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[src1_index]
                B = self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[src2_index]
                self.pim_device[hbm_index].HBM[channel_index].channel[bg].pim[_pim].grfs[dst_index] += self.MAC(A, B, False)

    def ADD(self, A, B, profile: bool):
        result = A + B
        return result

    def MAC(self, A, B, profile: bool):
        result = A * B
        return result
    
    def Vector_Vector_Mul_Row(self, A, B, profile: bool):
        n = (A.shape[0] - 1) // self.burst_length + 1
        lst = [self.MAC(
                    A[i*self.burst_length:(i+1)*self.burst_length], 
                    B[i*self.burst_length:(i+1)*self.burst_length], profile) for i in range(n-1)]
        lst.append(self.MAC(A[(n-1)*self.burst_length:], B[(n-1)*self.burst_length:], profile))
        return sum(lst)

    def Vector_Vector_Mul(self, A, B, profile: bool):
        n = (A.shape[0] - 1) // self.DRAM_column + 1
        lst = [self.Vector_Vector_Mul_Row(
                A[i*self.DRAM_column:(i+1)*self.DRAM_column], 
                B[i*self.DRAM_column:(i+1)*self.DRAM_column], profile) for i in range(n-1)]
        lst.append(self.Vector_Vector_Mul_Row(A[(n-1)*self.DRAM_column:], B[(n-1)*self.DRAM_column:], profile))
        if profile:
            print(lst)
        return sum(lst)

    def Vector_Matrix_Mul(self, vector, matrix, profile=False):
        matrix_dim = matrix.shape[1]
        result = []
        for i in range(matrix_dim):
            result.append(self.Vector_Vector_Mul(vector, matrix[:, i], profile))
        return result
    
    def Vector_Matrix_Mul_multithreads(self, vector, matrix):
        assert vector.dim() == 1
        vector_dim = vector.shape[0]
        assert vector_dim == matrix.shape[0]
        matrix_dim = matrix.shape[1]
        matrix_parallel_n = (matrix_dim-1) // self.threads + 1
        pool = torch.multiprocessing.Pool(processes=self.threads)
        arguments = [(vector, matrix[:, i * matrix_parallel_n : (i+1) * matrix_parallel_n], False) for i in range(self.threads)]
        results = pool.starmap(self.Vector_Matrix_Mul, arguments)
        pool.close()
        pool.join()
        for i in range(len(results)):
            results[i] = torch.tensor(results[i])
        return torch.cat(results)


    def Vector_Vector_EWMUL_Row(self, A, B):
        n = (A.shape[0] - 1) // self.burst_length + 1
        lst = [A[i*self.burst_length:(i+1)*self.burst_length] *  
               B[i*self.burst_length:(i+1)*self.burst_length] for i in range(n-1)]
        lst.append(A[(n-1)*self.burst_length:] * B[(n-1)*self.burst_length:])
        return torch.cat(lst)

    def Vector_Vector_EWMUL(self, A, B):
        n = (A.shape[0] - 1) // self.DRAM_column + 1
        lst = [self.Vector_Vector_EWMUL_Row(
                A[i*self.DRAM_column:(i+1)*self.DRAM_column], 
                B[i*self.DRAM_column:(i+1)*self.DRAM_column]) for i in range(n-1)]
        lst.append(self.Vector_Vector_EWMUL_Row(A[(n-1)*self.DRAM_column:], B[(n-1)*self.DRAM_column:]))
        return torch.cat(lst)
    
    def EWADD(self, A, B):
        return A + B

    def Vector_Vector_EWADD_Row(self, A, B):
        n = (A.shape[0] - 1) // self.burst_length + 1
        lst = [self.EWADD(
                    A[i*self.burst_length:(i+1)*self.burst_length], 
                    B[i*self.burst_length:(i+1)*self.burst_length]) for i in range(n-1)]
        lst.append(self.EWADD(A[(n-1)*self.burst_length:], B[(n-1)*self.burst_length:]))
        return torch.cat(lst)

    def Vector_Vector_EWADD(self, A, B):
        if self.op_trace:
            self.file.write("AiM EWADD {} 0 0\n".format(A.shape[-1] // self.burst_length))
        n = (A.shape[-1] - 1) // self.DRAM_column + 1
        lst = [self.Vector_Vector_EWADD_Row(
                A[0][0][i*self.DRAM_column:(i+1)*self.DRAM_column], 
                B[0][0][i*self.DRAM_column:(i+1)*self.DRAM_column]) for i in range(n-1)]
        lst.append(self.Vector_Vector_EWADD_Row(A[0][0][(n-1)*self.DRAM_column:], B[0][0][(n-1)*self.DRAM_column:]))
        return torch.cat(lst).reshape(A.shape)
    