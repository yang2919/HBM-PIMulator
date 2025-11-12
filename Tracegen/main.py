import argparse
import torch

from Model_GEMV import ModelGEMV

def build_args():
    parser = argparse.ArgumentParser(description="PIM Memory simulator arguments")

    # DRAM / HBM Config.
    parser.add_argument("--DRAM_column", type=int, default=32)
    parser.add_argument("--DRAM_row", type=int, default=8192)
    parser.add_argument("--burst_length", type=int, default=16)

    parser.add_argument("--num_banks", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=4)         # = BankGroup 개수
    parser.add_argument("--num_bankgroups", type=int, default=4)     # 코드 호환용(=num_groups)
    parser.add_argument("--num_channels", type=int, default=1)

    # PIM Register Config.
    parser.add_argument("--PIM_grf", type=int, default=16)
    parser.add_argument("--PIM_srf", type=int, default=4)

    # LLM Model Hyper Parameters
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--out_dim", type=int, default=2048)

    # Execution/Trace Options
    parser.add_argument("--only_trace", action="store_true")
    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--FC_devices", type=int, default=1)
    parser.add_argument("--op_trace", type=bool, default=True)
    parser.add_argument("--trace_file", type=str, default="test.trace")

    # (선택) 멀티프로세싱용 스레드 수
    parser.add_argument("--threads", type=int, default=2)

    args = parser.parse_args()

    # 코드 내부에서 num_groups와 num_bankgroups를 모두 사용하므로 값 일치 보장
    if not hasattr(args, "num_bankgroups") or args.num_bankgroups != args.num_groups:
        args.num_bankgroups = args.num_groups

    return args

def generate_model_dic(args):
    model_dic = {
        "x" : torch.randn(args.in_dim, dtype=torch.float16),
        "w" : torch.randn(args.in_dim * args.out_dim, dtype=torch.float16),
    }
    return model_dic

def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 무시

    # 공유 전략(사용자 코드 상단과 동일)
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = build_args()
    torch.manual_seed(1)
    
    model_dic = generate_model_dic(args)
    print("Parameter generation finished...")

    torch.set_printoptions(threshold=10)
    #torch.set_printoptions(threshold=float("inf"))

    model = ModelGEMV(model_dic, args)
    model.set_mapping()
    model.weight_storing()
    model.GEMV_PIM(True)
    model.file.write("PIM EXIT")
    model.file.close()  

if __name__ == "__main__":
    main()
