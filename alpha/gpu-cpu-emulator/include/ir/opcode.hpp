#pragma once

#include <cstdint>
#include <vector>

namespace ir {

// GPU IR: fixed 16-byte records for coalesced global-memory access.
enum class Opcode : uint8_t {
    Nop = 0,
    LoadImm,      // dst_reg, imm64
    MovReg,       // dst_reg, src_reg
    LoadMem,      // dst_reg, addr_reg (optional disp in imm)
    StoreMem,     // addr_reg, src_reg
    BinOpReg,     // dst_reg, src_reg, subop
    BinOpImm,     // dst_reg, imm64, subop
    BinOpMemImm,  // [base+disp], imm
    BinOpMem,     // mem via addr in imm's low bits / base reg
    CmpReg,       // lhs, rhs
    CmpImm,       // lhs, imm
    TestReg,
    Jmp,          // target offset in basic block or abs in imm
    Jcc,          // cc, target
    Push,
    Pop,
    Lea,          // dst, [base + disp]
    Call,         // guest target pc in imm
    Ret,
    Leave,        // mov rsp, rbp; pop rbp
    Hlt,
    Syscall,
    Barrier,      // sync point for parallel translation chunks
};

enum class IrBinOp : uint8_t { Add = 0, Sub, And, Or, Xor };

#pragma pack(push, 1)
struct Instr {
    Opcode op = Opcode::Nop;
    uint8_t width = 8;
    uint8_t dst = 0;
    uint8_t src = 0;
    uint8_t aux = 0;
    uint32_t disp = 0;
    int64_t imm = 0;
};
#pragma pack(pop)

static_assert(sizeof(Instr) == 17, "IR instr must be 17 bytes (packed)");

struct BasicBlock {
    uint64_t guest_pc = 0;
    std::vector<uint64_t> guest_pcs;
    std::vector<Instr> code;
};

struct TranslationUnit {
    std::vector<BasicBlock> blocks;
    uint64_t entry_pc = 0;
};

}  // namespace ir
