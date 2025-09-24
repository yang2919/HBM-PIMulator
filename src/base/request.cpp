#include "base/request.h"

namespace Ramulator {

Request::Request(Addr_t addr, int type): addr(addr), type_id(type) {};

Request::Request(AddrVec_t addr_vec, int type): addr_vec(addr_vec), type_id(type) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback):
addr(addr), type_id(type), source_id(source_id), callback(callback) {};

Request::Request(Addr_t addr, std::function<void(Request&)> callback):
addr(addr), callback(callback) {};

Request::Request(Addr_t addr, int type, int opcode, POperand_t pop):
addr(addr), type_id(type), operation_id(opcode), poperand(pop) {
    if(poperand[0].loc != LOCATE::BANK && poperand[1].loc != LOCATE::BANK){
        operation_id += 4;
    }
};

Request::Request(int opcode): type_id(Type::PIM), operation_id(opcode) {};

}        // namespace Ramulator

