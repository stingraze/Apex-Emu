#pragma once

#include "types.hpp"
#include <cstdint>

namespace x86 {

// GPU-friendly, 16-byte aligned guest CPU state (one per virtual CPU).
struct alignas(16) CpuState {
    uint64_t gpr[16]{};   // indexed by Reg
    uint64_t rip = 0;
    uint64_t rflags = 0x202;  // IF set
    uint32_t halt = 0;
    uint32_t syscall_pending = 0;
    int64_t  syscall_num = 0;
    int64_t  syscall_ret = 0;
};

// RFLAGS bits used by the emulator.
constexpr uint64_t CF = 1ULL << 0;
constexpr uint64_t ZF = 1ULL << 6;
constexpr uint64_t SF = 1ULL << 7;
constexpr uint64_t OF = 1ULL << 11;

inline void set_flags_arith(CpuState& s, uint64_t result, uint64_t lhs, uint64_t rhs, Width w, bool is_sub) {
    const uint64_t mask = width_mask(w);
    result &= mask;
    lhs &= mask;
    rhs &= mask;

    s.rflags &= ~(CF | ZF | SF | OF);
    if (result == 0) s.rflags |= ZF;

    const unsigned bits = static_cast<unsigned>(w) * 8;
    const uint64_t sign = 1ULL << (bits - 1);
    if (result & sign) s.rflags |= SF;

    if (is_sub) {
        s.rflags |= (lhs < rhs) ? CF : 0;
        const bool overflow = ((lhs ^ rhs) & (lhs ^ result) & sign) != 0;
        if (overflow) s.rflags |= OF;
    } else {
        s.rflags |= (result < lhs) ? CF : 0;
        const bool overflow = ((~(lhs ^ rhs)) & (lhs ^ result) & sign) != 0;
        if (overflow) s.rflags |= OF;
    }
}

}  // namespace x86
