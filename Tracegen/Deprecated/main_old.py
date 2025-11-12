import argparse
import torch

from function_v01 import System 
from Hypothesis1 import ModelMixtral

def build_args():
    parser = argparse.ArgumentParser(description="PIM Memory simulator arguments")

    # DRAM / HBM 구성
    parser.add_argument("--DRAM_column", type=int, default=32)
    parser.add_argument("--DRAM_row", type=int, default=8192)
    parser.add_argument("--burst_length", type=int, default=16)

    parser.add_argument("--num_banks", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=4)         # = BankGroup 개수
    parser.add_argument("--num_bankgroups", type=int, default=4)     # 코드 호환용(=num_groups)
    parser.add_argument("--num_channels", type=int, default=1)

    # PIM 레지스터 파일
    parser.add_argument("--PIM_grf", type=int, default=16)
    parser.add_argument("--PIM_srf", type=int, default=4)

    # Model hyper parameters
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--dim_expert", type=int, default=2048)
    parser.add_argument("--n_expert", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)

    # 실행/트레이스 옵션
    parser.add_argument("--only_trace", action="store_true",
                        help="데이터 배열 생성 없이 트레이스만 기록")
    parser.add_argument("--model_parallel", action="store_true",
                        help="여러 디바이스 병렬 구성")
    parser.add_argument("--FC_devices", type=int, default=1,
                        help="model_parallel일 때 디바이스 수")
    parser.add_argument("--op_trace", type=bool, default=True)
    parser.add_argument("--trace_file", type=str, default="test.trace",
                        help="트레이스 출력 파일 경로")

    # (선택) 멀티프로세싱용 스레드 수
    parser.add_argument("--threads", type=int, default=2)

    args = parser.parse_args()

    # 코드 내부에서 num_groups와 num_bankgroups를 모두 사용하므로 값 일치 보장
    if not hasattr(args, "num_bankgroups") or args.num_bankgroups != args.num_groups:
        args.num_bankgroups = args.num_groups

    return args

def generate_random_fp16_tensor(size):
    return torch.randn(size, dtype=torch.float32).half()

def fill_all_banks_with_random(mem, row=0, col=0):
    num_devices   = len(mem.pim_device)
    num_channels  = mem.num_channels
    num_bgs       = mem.num_bankgroups
    num_banks     = mem.num_banks
    row_idx       = row
    col_idx       = col
    size_per_row  = mem.DRAM_column

    for dev in range(num_devices):
        for ch in range(num_channels):
            for bg in range(num_bgs):
                for bank in range(num_banks):
                    data_fp16 = generate_random_fp16_tensor(16)
                    data = data_fp16

                    mem.store_to_DRAM_single_bank(
                        hbm_index=dev,
                        channel_index=ch,
                        bankgroup_index=bg,
                        bank_index=bank,
                        row_index=row_idx,
                        col_index=col_idx,
                        size=size_per_row,
                        data=data,
                        op_trace=mem.op_trace
                    )

def generate_model_dic(model : str="Mixtral"):
    model_dic = {
        # "Mixtral" : {
        #     "x1" : torch.randn(1024, dtype=torch.float16),
        #     "w1" : {
        #         f"expert{i}": torch.randn(1024 * 8192, dtype=torch.float16) for i in range(8)
        #     },
        #     "w2" : {
        #         f"expert{i}": torch.randn(1024 * 8192, dtype=torch.float16) for i in range(8)
        #     }
        # },
        "Mixtral" : {
            "x1" : torch.zeros(4096, dtype=torch.float16),
            "w1" : {
                f"expert{i}": torch.zeros(4096 * 14336, dtype=torch.float16) for i in range(8)
            },
            "w2" : {
                f"expert{i}": torch.zeros(14336 * 4096, dtype=torch.float16) for i in range(8)
            }
        },
        "Deepseek-MoE-16B" : {
            "x1" : torch.randn(2048, dtype=torch.float16),
            "w1" : {
                f"expert{i}": torch.randn(2048 * 768, dtype=torch.float16) for i in range(1)
            },
            "w2" : {
                f"expert{i}": torch.randn(768 * 2048, dtype=torch.float16) for i in range(1)
            }
        },
        "Qwen" : {
            "x1" : torch.randn(2048, dtype=torch.float16),
            "w1" : {
                f"expert{i}": torch.randn(2048 * 2048, dtype=torch.float16) for i in range(128)
            },
            "w2" : {
                f"expert{i}": torch.randn(2048 * 2048, dtype=torch.float16) for i in range(128)
            }
        }
    }
    return model_dic[model]

def compare_lists(list1, list2, tol: float = 0.2) -> bool:
    results = []
    for t1, t2 in zip(list1, list2):
        results.append(abs(t1.sum() - t2.sum()) < tol)
    
    return all(results)

import torch

def save_diff_indices(t1: torch.Tensor, t2: torch.Tensor, filename="diff.txt", threshold=0.1):
    """
    두 텐서를 원소별로 비교하여 오차가 threshold 이상인 index를 파일에 저장
    
    Args:
        t1, t2: 같은 shape의 torch.Tensor
        filename: 저장할 파일명
        threshold: 허용 오차
    """
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")

    # 절대 오차
    diff = (t1 - t2).abs()

    # threshold 이상인 위치
    mask = diff >= threshold
    indices = mask.nonzero(as_tuple=False)

    with open(filename, "w") as f:
        for idx in indices:
            idx_tuple = tuple(idx.tolist())
            f.write(
                f"Index={idx_tuple}, t1={t1[idx_tuple].item()}, t2={t2[idx_tuple].item()}, diff={diff[idx_tuple].item()}\n"
            )

    print(f"총 {indices.shape[0]}개 차이를 {filename}에 저장했습니다.")


def GEMV_example(args):
    mem = System(args)
    
    print("Memory 객체 생성 완료!")

    # Example GEMV
    input1 = []
    input2 = []
    input1 = generate_random_fp16_tensor(16 * 32)
    input2 = generate_random_fp16_tensor(16 * 32 * mem.num_bankgroups * (mem.num_banks//2) * 8)

    in1_bo = mem.create_BO(len(input1), [0], [0, 1], [0, 0], False)
    in2_bo = mem.create_BO(len(input2), [0], [0, 1], [1, 0], True)
    out_bo = mem.create_BO(len(input2) // len(input1) * 16, [0], [0, 1], [2, 0], True)

    mem.broadcast_to_DRAM_all_bank(in1_bo, input1, True)
    mem.scatter_to_DRAM_all_bank(in2_bo, input2, False)
    out = mem.PIM_GEMV_BO(in1_bo, in2_bo, out_bo)
    torch.set_printoptions(threshold=10) 
    print(out.sum(dim=1), out.shape)
    print("----------------------------------------------")
    in2 = input2.view(-1, len(input1))
    output = input1 * in2
    output = output.sum(dim = 1)
    print(output, output.shape)
    print("----------------------------------------------")

    try:
        mem.file.close()
    except Exception:
        pass

def main():
    # macOS/Windows 호환을 위해 spawn 권장
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 무시

    # 공유 전략(사용자 코드 상단과 동일)
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = build_args()
    #GEMV_example(args)
    torch.manual_seed(1)
    
    #model_dic = generate_model_dic("Deepseek-MoE-16B")
    model_dic = generate_model_dic("Qwen")
    print("Parameter generation finished...")

    torch.set_printoptions(threshold=10)
    #torch.set_printoptions(threshold=float("inf"))

    model = ModelMixtral(model_dic, args)
    model.set_mapping()
    
    print("x1: ", model.x1_bo.size)
    print("w1: ", model.w1_bo[0].size)
    print("o1: ", model.o1_bo[0].size)
    print("x2: ", model.x2_bo[0].size)
    print("w2: ", model.w2_bo[0].size)
    print("o2: ", model.o2_bo[0].size)
    model.weight_mapping(False)
    model.gating()
    #out1 = model.FFN_ref()
    model.FFN_PIM(True)
    #save_diff_indices(out1, out2)
    model.file.write("PIM EXIT")
    model.file.close()

if __name__ == "__main__":
    main()
