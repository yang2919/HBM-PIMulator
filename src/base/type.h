#ifndef     RAMULATOR_BASE_TYPE_H
#define     RAMULATOR_BASE_TYPE_H

#include <vector>
#include <unordered_map>
#include <string>
#include <type_traits>


namespace Ramulator {

using Clk_t     = int64_t;            // Clock cycle
using Addr_t    = int64_t;            // Plain address as seen by the OS
using AddrVec_t = std::vector<int>;   // Device address vector as is sent to the device from the controller

enum LOCATE{
    BANK,
    GRF,
    SRF
};

struct Opcode {
    enum : int{
            MAC,
            MUL,
            MAD,
            ADD,
            MACRF,
            MULRF,
            MADRF,
            ADDRF,
            MOV,
            FILL,
            NOP,
            JUMP,
            EXIT,
            TMOD_A,
            TMOD_P,
    };
};


typedef struct POperand_{ // PIM specific address struct
    LOCATE loc;
    int addr;
}POperand;

using POperand_t = std::vector<POperand>;

template<typename T>
using Registry_t = std::unordered_map<std::string, T>;

// From WG21 P2098R1 Proposing std::is_specialization_of
template<class T, template<class...> class Primary>
struct is_specialization_of : std::false_type {};

template<template<class...> class Primary, class... Args>
struct is_specialization_of<Primary<Args...>, Primary> : std::true_type {};

template< class T, template<class...> class Primary>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Primary>::value;

}        // namespace Ramulator


#endif   // RAMULATOR_BASE_TYPE_H