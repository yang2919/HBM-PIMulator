# run_memory.py
import argparse
import torch

# 사용자가 제공한 클래스들이 같은 파일에 없다면 적절히 import 경로를 수정하세요.
# 예: from pim_mem import Memory
from function import System 

def build_args():
    parser = argparse.ArgumentParser(description="PIM Memory simulator arguments")

    # DRAM / HBM 구성
    parser.add_argument("--DRAM_column", type=int, default=16)
    parser.add_argument("--DRAM_row", type=int, default=128)
    parser.add_argument("--burst_length", type=int, default=16)

    parser.add_argument("--num_banks", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=4)         # = BankGroup 개수
    parser.add_argument("--num_bankgroups", type=int, default=4)     # 코드 호환용(=num_groups)
    parser.add_argument("--num_channels", type=int, default=1)

    # PIM 레지스터 파일
    parser.add_argument("--PIM_grf", type=int, default=8)
    parser.add_argument("--PIM_srf", type=int, default=4)

    # 실행/트레이스 옵션
    parser.add_argument("--only_trace", action="store_true",
                        help="데이터 배열 생성 없이 트레이스만 기록")
    parser.add_argument("--model_parallel", action="store_true",
                        help="여러 디바이스 병렬 구성")
    parser.add_argument("--FC_devices", type=int, default=1,
                        help="model_parallel일 때 디바이스 수")
    parser.add_argument("--op_trace", type=bool, default=True)
    parser.add_argument("--trace_file", type=str, default="trace.txt",
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

def main():
    # macOS/Windows 호환을 위해 spawn 권장
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 무시

    # 공유 전략(사용자 코드 상단과 동일)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    args = build_args()

    # 여기까지가 "argument parsing"
    # ------------------------------
    # 아래에서 Memory 객체를 생성합니다.
    mem = System(args)
    
    print("Memory 객체 생성 완료!")

    # Example GEMV
    input1 = []
    input2 = []
    for _ in range(32):
        input1.append(generate_random_fp16_tensor(16))
    for _ in range(32 * mem.num_bankgroups * mem.num_banks):
        input2.append(generate_random_fp16_tensor(16))
    in1_bo = mem.create_BO(len(input1), 0, 0, [0, 0], False)
    in2_bo = mem.create_BO(len(input2), 0, 0, [1, 0], True)
    out_bo = mem.create_BO(mem.num_bankgroups * mem.num_banks, 0, 0, [2, 0], True)
    mem.broadcast_to_DRAM_all_bank(in1_bo, input1, True)
    mem.scatter_to_DRAM_all_bank(in2_bo, input2, True)
    mem.GEMV_BO_PRE(in1_bo, in2_bo, out_bo)
    out = mem.gather_from_DRAM_all_bank(out_bo, True)
    print("----------------------------------------------")
    for iter in out:
        print(iter.sum(), end=' ')
    print("----------------------------------------------")
    for i in range(mem.num_bankgroups * mem.num_banks):
        output = torch.zeros(16)
        for j in range(32):
            output += input1[j] * input2[i * 32 + j]
        print(output.sum())
    # print(f"DRAM_row={args.DRAM_row}, DRAM_column={args.DRAM_column}, "
    #       f"burst_length={args.burst_length}, num_banks={args.num_banks}, "
    #       f"num_groups={args.num_groups}, num_channels={args.num_channels}")
    # # 필요 시 이후 로직 추가
    # fill_all_banks_with_random(mem, 0, 0)
    # fill_all_banks_with_random(mem, 0, 1)
    # print("Random 데이터 저장 완료")
    # mem.PIM_FILL(0, 0, 0, 0, 0, 0, True)
    # mem.PIM_MAC_RD_BANK(0, 0, 0, 0, 1, 0, 1, True)
    # mem.PIM_MOVE(0, 0, 0, 1, 0, 2, True)
    # A = mem.load_from_DRAM_single_bank(0, 0, 0, 0, 0, 0, mem.burst_length, False)
    # B = mem.load_from_DRAM_single_bank(0, 0, 0, 0, 0, 1, mem.burst_length, False)
    # print((A * B))

    # print(mem.load_from_DRAM_single_bank(0, 0, 0, 0, 0, 2, mem.burst_length, False))

    # trace 파일 핸들 정리(생성 직후 종료한다면)
    try:
        mem.file.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
