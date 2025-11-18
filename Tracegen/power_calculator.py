import sys
import argparse
import math

KILO = 1000
MEGA = 1000000
GIGA = 1000000000
FREQ = 2.00 * GIGA

CH_PER_DV = 32.00

commands = ["ACT",
            "ACTA",
            "PRE", 
            "PREA",
            "RD",  
            "WR",  
            "RDA",  
            "WRA",
            "REFab", 
            "REFsb",
            "MAC", 
            "MUL", 
            "ADD",
            "MACRF", 
            "MULRF", 
            "ADDRF",
            "DATA", 
            "CON",
            "TMOD", 
            "RWR"]

isrs =  ["MAC", 
         "MUL", 
         "MAD", 
         "ADD", 
         "MACRF", 
         "MULRF", 
         "MADRF", 
         "ADDRF",
         "MOV", 
         "FILL", 
         "NOP", 
         "JUMP", 
         "EXIT",
         "TMOD_A", 
         "TMOD_P",
         "RWR"]

tRC = 44.5
tBL = 1.25
tCCDL = 1.0

# DRAM_POWER = {  "ACT_STBY": 415,
#                 "PRE_STBY": 317.5,
#                 "ACT": 93.9,
#                 "WR": 915,
#                 "RD": 525}

DRAM_POWER = {  "ACT_STBY": 527.5 / 2.00, # 415,
                "PRE_STBY": 366.3 / 2.00, # 317.5,
                "ACT": 132.6 / 2.00, # 93.9,
                "WR": 1106.3 / 2.00, # 915,
                "RD": 876.3 / 2.00} # 525}

RV_COUNT = 8
# Latency of 1 SIMD operation
SB_RD_CYCLE = 1.00
SB_WR_CYCLE = 1.00
EXP_LANE_CYCLE = 11.00
RV_RMSNorm_CYCLE = 26.00
RV_ROTEmbed_CYCLE = 3.00 / RV_COUNT
RV_SFT_CYCLE_PIPELINE = 16.00 * SB_WR_CYCLE + 2.00 / RV_COUNT + 1.00 * SB_RD_CYCLE
RV_SFT_CYCLE_SINGLE = 16.00 * SB_WR_CYCLE + 2.00 + 1.00 * SB_RD_CYCLE

# TRX: Transaction Engine
# PHY: Physical Interface
# TODO: make sure PHY is not DQ
CTRL_POWER = {  "TRX": 267.7082056,
                "PHY": 381.0445262}
CH_PER_CTRL = 2.00

# pJ/bit
DQ_ENERGY = 5.5
PCIE_ENERGY = 4.4

WORD_SIZE = 256

def command_processor(stat_path):
    file = open(stat_path, 'r')
    lines = file.readlines()
    file.close()
    stat = {}
    for command in commands:
        stat[command] = 0.00
    for isr in isrs:
        stat[isr] = 0.00
    stat["cycles"] = 0.00
    stat["idle_cycles"] = 0.00
    stat["active_cycles"] = 0.00
    stat["precharged_cycles"] = 0.00
    
    for line in lines:
        words = line.split(' ')
        while len(words) > 0 and words[0] == "":
            words.pop(0)

        if "memory_system_cycles" in words[0]:
            assert stat["cycles"] == 0
            stat["cycles"] = float(words[1])
        if "idle_cycles" in words[0]:
            stat["idle_cycles"] += float(words[1])
        if "active_cycles" in words[0]:
            stat["active_cycles"] += float(words[1])
        if "precharged_cycles" in words[0]:
            stat["precharged_cycles"] += float(words[1])

        # Todo: Synchronize with HBM-PIM
        for command in commands:
            if "num_" + command + "_commands" in words[0]:
                stat[command] += float(words[1])
        for isr in isrs:
            if "total_num_AiM_ISR_" + isr + "_requests" in words[0]:
                stat[isr] += float(words[1])
        
    # stat["idle_cycles"] = stat["idle_cycles"] / CH_PER_DV
    # stat["active_cycles"] = stat["active_cycles"] / CH_PER_DV
    # stat["precharged_cycles"] = stat["precharged_cycles"] / CH_PER_DV
    # print(stat["cycles"], stat["idle_cycles"], stat["active_cycles"], stat["precharged_cycles"])
    # ms (average of all channels)
    stat["latency"] = stat["cycles"] * KILO / FREQ
    # ms (average of all channels)
    stat["active_latency"] = stat["active_cycles"] / CH_PER_DV * KILO / FREQ
    # ms (average of all channels)
    stat["precharged_latency"] = stat["precharged_cycles"] / CH_PER_DV * KILO / FREQ
    # % (average of all channels)
    if stat["cycles"] == 0:
        print(stat_path)
    stat["utilization"] = 100.00 - (stat["idle_cycles"] / CH_PER_DV / stat["cycles"]) * 100.00
    return stat

def power_calculator(stat):
    energy = {}

    # TODO: should we use the tRC or tRCD?
    energy["ACT/PRE"] = DRAM_POWER["ACT"] * (stat["ACT"] + 8.00 * stat["ACTA"]) * tRC / GIGA
    energy["RD"] = DRAM_POWER["RD"] * (stat["RD"]) * tBL / GIGA
    energy["WR"] = DRAM_POWER["WR"] * (stat["WR"]) * tBL / GIGA
    energy["PIM"] = DRAM_POWER["RD"] * (8.00 * stat["MAC"] + 8.00 * stat["ADD"] + 8.00 * stat["MUL"] + 8.00 * stat["MOV"] + 8.00 * stat["FILL"]) * tCCDL / GIGA
    energy["ACT_STBY"] = DRAM_POWER["ACT_STBY"] * CH_PER_DV * stat["active_latency"] / KILO
    energy["PRE_STBY"] = DRAM_POWER["PRE_STBY"] * CH_PER_DV * stat["precharged_latency"] / KILO
    energy["DQ"] = DQ_ENERGY * WORD_SIZE * (stat["RD"] + stat["WR"] + stat["RWR"]) / GIGA

    return energy

def get_args():
    parser = argparse.ArgumentParser(description="HBM-PIM Power Calculator")
    parser.add_argument("--mlog", help="path of the main ramulator log", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    mlog = args.mlog

    if args.ch_per_dv != CH_PER_DV:
        CH_PER_DV = args.ch_per_dv
        # latency of pipelining 32 accelerators
        # each having 16 SIMD lanes
        ACCEL_CYCLE = { "EXP": CH_PER_DV * SB_RD_CYCLE + EXP_LANE_CYCLE + SB_WR_CYCLE,
                        "VEC": CH_PER_DV * 2.00 * SB_RD_CYCLE + 1.00 + SB_WR_CYCLE}

    energy_token = {}
    power_alldv = {}
    stat_main = command_processor(mlog)
    PCIE = hidden if CH_PER_BL <= CH_PER_DV else hidden * 10 + fc * 2.00
    energy_main, latency_main = power_calculator(stat_main, PCIE, head, hidden, token, gqa)

    total_ch_used = block * CH_PER_BL
    total_dv_need = 0
    if CH_PER_BL >= CH_PER_DV:
        DV_PER_BL = math.ceil(float(CH_PER_BL) / float(CH_PER_DV))
        assert DV_PER_BL >= 1.00
        PIPE_STAGES = DV / DV_PER_BL
        assert DV % DV_PER_BL == 0
        stat_pim = command_processor(plog)
        # print(stat_main)
        # print(stat_pim)
        energy_pim, latency_pim = power_calculator(stat_pim, PCIE, head, hidden, token, gqa)
        total_dv_need = block * DV_PER_BL
        for comp in energy_main.keys():
            energy_token[comp] = (energy_main[comp] + energy_pim[comp] * (DV_PER_BL - 1.00)) * block
            power_alldv[comp] = (energy_main[comp] + energy_pim[comp] * (DV_PER_BL - 1.00)) * PIPE_STAGES / stat_main["latency"]
    else:
        BL_PER_DV = int(CH_PER_DV / CH_PER_BL)
        total_dv_need = math.ceil(float(block) / float(BL_PER_DV))
        assert total_dv_need <= DV
        for comp in energy_main.keys():
            energy_token[comp] = energy_main[comp] * total_dv_need
            power_alldv[comp] = energy_main[comp] * total_dv_need / stat_main["latency"]
        for comp in latency_main.keys():
            latency_main[comp] = latency_main[comp] * float(BL_PER_DV)
    total_ch_need = total_dv_need * CH_PER_DV

    # print("Configuration:")
    # print("CH/DV,CH-used,CH-needed,DV-needed")
    # print(f"{CH_PER_DV},{total_ch_used},{total_ch_need},{total_dv_need}")

    # print(",\nlatency (ms)")
    # print("pim,RMS,SFT,ROT,Total Acc,Total,utilization(%)")
    total_acc_latency = latency_main["RMSNorm_latency"] + latency_main["Softmax_latency"] + latency_main["RotEmbed_latency"]
    total_latency = stat_main["latency"] + latency_main["RMSNorm_latency"] + latency_main["Softmax_latency"] + latency_main["RotEmbed_latency"]
    print(f"{stat_main['latency']},{latency_main['RMSNorm_latency']},{latency_main['Softmax_latency']},{latency_main['RotEmbed_latency']},{total_acc_latency},{total_latency},{stat_main['utilization']}")
    print(total_acc_latency)

    print(",\nenergy 1 token detailed (mJ):")
    for comp in energy_token.keys():
        print(comp, end=",")
    print()
    for comp in energy_token.keys():
        print(energy_token[comp], end=",")
    print()

    # print(",\nenergy 1 token summary (mJ):") 
    # print("DRAM,ctrl,DQ,DV,PCIe,Total")
    # print(energy_token["ACT/PRE"] + energy_token["RD"] + energy_token["WR"] + energy_token["PIM"] + energy_token["ACT_STBY"] + energy_token["PRE_STBY"] + energy_token["GB_STT"] + energy_token["GB_RD"] + energy_token["GB_WR"], end=",")
    # print(energy_token["MEM_CTR"], end=",")
    # print(energy_token["DQ"], end=",")
    # print(energy_token["IB_STT"] + energy_token["SB_STT"] + energy_token["RED_STT"] + energy_token["EXP_STT"] + energy_token["VEC_STT"] + energy_token["IB_DYN"] + energy_token["SB_DYN"] + energy_token["RV_DYN"] + energy_token["RED_DYN"] + energy_token["EXP_DYN"] + energy_token["VEC_DYN"] + energy_token["DV_CTR"], end=",")
    # print(energy_token["PCIe"], end=",")
    total_energy = 0
    for comp in energy_token.keys():
        total_energy += energy_token[comp]
    print(total_energy)

    # print(",\nenergy 1 query summary (J):", total_energy * 4096 / 1000) 
    # print("energy 1 query summary (J):", total_energy * 4096 / 1000) 

    # print(",\npower all devices detailed (W):")
    # for comp in power_alldv.keys():
    #     print(comp, end=",")
    # print()
    # for comp in power_alldv.keys():
    #     print(power_alldv[comp], end=",")
    # print()

    # print(",\npower all devices summary (W):") 
    # print("DRAM,ctrl,DQ,DV,PCIe,Total")
    # print(power_alldv["ACT/PRE"] + power_alldv["RD"] + power_alldv["WR"] + power_alldv["PIM"] + power_alldv["ACT_STBY"] + power_alldv["PRE_STBY"] + power_alldv["GB_STT"] + power_alldv["GB_RD"] + power_alldv["GB_WR"], end=",")
    # print(power_alldv["MEM_CTR"], end=",")
    # print(power_alldv["DQ"], end=",")
    # print(power_alldv["IB_STT"] + power_alldv["SB_STT"] + power_alldv["RED_STT"] + power_alldv["EXP_STT"] + power_alldv["VEC_STT"] + power_alldv["IB_DYN"] + power_alldv["SB_DYN"] + power_alldv["RV_DYN"] + power_alldv["RED_DYN"] + power_alldv["EXP_DYN"] + power_alldv["VEC_DYN"] + power_alldv["DV_CTR"], end=",")
    # print(power_alldv["PCIe"], end=",")
    total_power = 0
    for comp in power_alldv.keys():
        total_power += power_alldv[comp]
    print(total_power)

    one_device = total_power/DV
    PIM_power = power_alldv["PIM"]/DV
    standby_power = (power_alldv["ACT_STBY"] + power_alldv["PRE_STBY"])/DV
    ACT_PRE_power = power_alldv["ACT/PRE"]/DV
    print("1 device", one_device, "PIM", PIM_power/one_device, "standby", standby_power/one_device, "ACT/PRE", ACT_PRE_power/one_device)
