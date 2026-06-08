#pragma once

#include <cstdint>
#include <cstring>

namespace x86 {

enum class Reg : uint8_t {
    RAX = 0, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
    COUNT
};

enum class Width : uint8_t { B = 1, W = 2, D = 4, Q = 8 };

enum class OpKind : uint8_t {
    Invalid,
    MovImm,
    MovRegReg,
    MovRegMem,
    MovMemReg,
    BinOpRegReg,   // add/sub/and/or/xor
    BinOpRegImm,
    BinOpMemImm,
    BinOpMemReg,
    CmpRegImm,
    CmpRegReg,
    TestRegReg,
    Jmp,
    Jcc,
    Push,
    Pop,
    Lea,
    Call,
    Ret,
    Leave,
    Nop,
    Hlt,
    Syscall,
    ImulRegMem,
    ShiftRegImm,
};

enum class BinOp : uint8_t { Add, Sub, And, Or, Xor };

struct Operand {
    bool is_mem = false;
    bool rip_rel = false;
    bool has_sib = false;
    bool abs_sib = false;
    Reg reg = Reg::RAX;
    int8_t scale = 1;
    Reg index = Reg::RAX;
    Reg base = Reg::RAX;
    int32_t disp = 0;
    int64_t imm = 0;
    Width width = Width::Q;
};

struct Instruction {
    OpKind kind = OpKind::Invalid;
    BinOp binop = BinOp::Add;
    Operand dst{};
    Operand src{};
    uint8_t cc = 0;       // condition code for Jcc
    uint64_t target = 0;  // absolute jump target
    uint64_t size = 0;    // encoded instruction length
    uint64_t pc = 0;      // guest PC at decode time
    bool zext = false;    // zero-extend on load (movzbl)
    bool sext = false;    // sign-extend on load (movsxd, cdqe)
};

inline const char* reg_name(Reg r) {
    static const char* names[] = {
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
    };
    return names[static_cast<int>(r)];
}

inline uint64_t width_mask(Width w) {
    switch (w) {
        case Width::B: return 0xFF;
        case Width::W: return 0xFFFF;
        case Width::D: return 0xFFFFFFFF;
        case Width::Q: return ~0ULL;
    }
    return ~0ULL;
}

}  // namespace x86
