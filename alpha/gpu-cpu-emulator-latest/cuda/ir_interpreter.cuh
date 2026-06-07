#pragma once

#include "ir/opcode.hpp"
#include "x86/state.hpp"

#ifdef __CUDACC__
#define GPUEMU_DEVICE __device__
#define GPUEMU_FORCE_INLINE __forceinline__
#else
#define GPUEMU_DEVICE
#define GPUEMU_FORCE_INLINE inline
#endif

namespace gpuexec {

GPUEMU_DEVICE GPUEMU_FORCE_INLINE uint64_t mask_width(uint8_t w) {
    switch (w) {
        case 1: return 0xFF;
        case 2: return 0xFFFF;
        case 4: return 0xFFFFFFFF;
        default: return ~0ULL;
    }
}

GPUEMU_DEVICE GPUEMU_FORCE_INLINE uint64_t load_mem(const uint8_t* mem, uint64_t addr,
                                                     uint8_t w, uint64_t mem_size) {
    if (addr >= mem_size) return 0;
    switch (w) {
        case 1: return mem[addr];
        case 2: {
            uint16_t v;
            if (addr + 2 > mem_size) return 0;
#ifdef __CUDACC__
            v = *reinterpret_cast<const uint16_t*>(mem + addr);
#else
            std::memcpy(&v, mem + addr, 2);
#endif
            return v;
        }
        case 4: {
            uint32_t v;
            if (addr + 4 > mem_size) return 0;
#ifdef __CUDACC__
            v = *reinterpret_cast<const uint32_t*>(mem + addr);
#else
            std::memcpy(&v, mem + addr, 4);
#endif
            return v;
        }
        default: {
            uint64_t v;
            if (addr + 8 > mem_size) return 0;
#ifdef __CUDACC__
            v = *reinterpret_cast<const uint64_t*>(mem + addr);
#else
            std::memcpy(&v, mem + addr, 8);
#endif
            return v;
        }
    }
}

GPUEMU_DEVICE GPUEMU_FORCE_INLINE void store_mem(uint8_t* mem, uint64_t addr,
                                                uint64_t val, uint8_t w, uint64_t mem_size) {
    if (addr >= mem_size) return;
    const uint64_t m = mask_width(w);
    val &= m;
    switch (w) {
        case 1: mem[addr] = static_cast<uint8_t>(val); break;
        case 2: {
            if (addr + 2 > mem_size) return;
            auto v = static_cast<uint16_t>(val);
#ifdef __CUDACC__
            *reinterpret_cast<uint16_t*>(mem + addr) = v;
#else
            std::memcpy(mem + addr, &v, 2);
#endif
            break;
        }
        case 4: {
            if (addr + 4 > mem_size) return;
            auto v = static_cast<uint32_t>(val);
#ifdef __CUDACC__
            *reinterpret_cast<uint32_t*>(mem + addr) = v;
#else
            std::memcpy(mem + addr, &v, 4);
#endif
            break;
        }
        default: {
            if (addr + 8 > mem_size) return;
#ifdef __CUDACC__
            *reinterpret_cast<uint64_t*>(mem + addr) = val;
#else
            std::memcpy(mem + addr, &val, 8);
#endif
            break;
        }
    }
}

GPUEMU_DEVICE GPUEMU_FORCE_INLINE uint64_t binop_val(ir::IrBinOp op, uint64_t a, uint64_t b) {
    switch (op) {
        case ir::IrBinOp::Add: return a + b;
        case ir::IrBinOp::Sub: return a - b;
        case ir::IrBinOp::And: return a & b;
        case ir::IrBinOp::Or:  return a | b;
        case ir::IrBinOp::Xor: return a ^ b;
    }
    return 0;
}

GPUEMU_DEVICE GPUEMU_FORCE_INLINE void update_flags(x86::CpuState& cpu, uint64_t res,
                                                     uint64_t lhs, uint64_t rhs,
                                                     uint8_t w, ir::IrBinOp op) {
    const uint64_t mask = mask_width(w);
    res &= mask; lhs &= mask; rhs &= mask;
    cpu.rflags &= ~(x86::CF | x86::ZF | x86::SF | x86::OF);

    if (res == 0) cpu.rflags |= x86::ZF;
    const unsigned bits = w * 8;
    const uint64_t sign = 1ULL << (bits - 1);
    if (res & sign) cpu.rflags |= x86::SF;

    const bool is_sub = (op == ir::IrBinOp::Sub);
    if (is_sub) {
        if (lhs < rhs) cpu.rflags |= x86::CF;
        if ((lhs ^ rhs) & (lhs ^ res) & sign) cpu.rflags |= x86::OF;
    } else if (op == ir::IrBinOp::Add) {
        if (res < lhs) cpu.rflags |= x86::CF;
        if ((~(lhs ^ rhs)) & (lhs ^ res) & sign) cpu.rflags |= x86::OF;
    }
}

GPUEMU_DEVICE GPUEMU_FORCE_INLINE uint64_t eff_addr(const x86::CpuState& cpu, const ir::Instr& ins,
                                                     uint8_t base_reg) {
    uint64_t addr = cpu.gpr[base_reg] + static_cast<int32_t>(ins.disp);
    if ((ins.aux & 0xF) == 6) {
        const int scale_bits = (ins.aux >> 4) & 3;
        const uint64_t scale = 1ULL << scale_bits;
        const int idx = (ins.aux & 0x10) ? static_cast<int>(ins.src)
                                         : static_cast<int>(ins.imm);
        addr += cpu.gpr[idx] * scale;
    }
    return addr;
}

GPUEMU_DEVICE GPUEMU_FORCE_INLINE bool eval_cc(uint8_t cc, uint64_t flags) {
    const bool cf = flags & x86::CF;
    const bool zf = flags & x86::ZF;
    const bool sf = flags & x86::SF;
    const bool of = flags & x86::OF;
    switch (cc) {
        case 0x0: return of;                    // O
        case 0x1: return !of;                   // NO
        case 0x2: return cf;                    // C / NA
        case 0x3: return !cf;                   // NC / AE
        case 0x4: return zf;                    // E / Z
        case 0x5: return !zf;                   // NE / NZ
        case 0x6: return cf || zf;              // BE / NA
        case 0x7: return !cf && !zf;            // A / NBE
        case 0x8: return sf;                    // S
        case 0x9: return !sf;                   // NS
        case 0xA: return sf != of;              // P / PE (parity approximated)
        case 0xB: return sf == of;              // NP / PO
        case 0xC: return zf || (sf != of);      // LE / NG
        case 0xD: return !zf && (sf == of);     // G / NLE
        case 0xE: return sf != of;              // L / NGE
        case 0xF: return sf == of;              // GE / NL
        default:  return false;
    }
}

// Execute up to max_steps IR instructions for one vCPU.
// Returns: 0 = running, 1 = halt, 2 = syscall trap
GPUEMU_DEVICE GPUEMU_FORCE_INLINE int exec_ir(x86::CpuState& cpu, uint8_t* mem,
                                               uint64_t mem_size,
                                               const ir::Instr* code, uint32_t code_len,
                                               uint32_t& pc, uint32_t max_steps) {
    for (uint32_t step = 0; step < max_steps; ++step) {
        if (pc >= code_len) {
            cpu.halt = 1;
            return 1;
        }
        const ir::Instr& ins = code[pc++];

        switch (ins.op) {
            case ir::Opcode::Nop:
                break;

            case ir::Opcode::LoadImm:
                cpu.gpr[ins.dst] = static_cast<uint64_t>(ins.imm);
                break;

            case ir::Opcode::MovReg:
                if (ins.aux == 5) {
                    const int32_t v = static_cast<int32_t>(cpu.gpr[ins.src]);
                    cpu.gpr[ins.dst] = static_cast<uint64_t>(static_cast<int64_t>(v));
                } else {
                    cpu.gpr[ins.dst] = cpu.gpr[ins.src] & mask_width(ins.width);
                }
                break;

            case ir::Opcode::LoadMem: {
                uint64_t addr = (ins.aux == 1) ? cpu.rip + static_cast<int32_t>(ins.disp)
                                               : eff_addr(cpu, ins, ins.src);
                uint64_t val = load_mem(mem, addr, ins.width, mem_size);
                if (ins.aux == 4 || ins.width == 1) {
                    val &= 0xFF;
                } else if (ins.aux == 5) {
                    val = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(val)));
                } else if (ins.width == 4) {
                    val = static_cast<uint32_t>(val);
                } else if (ins.width == 2) {
                    val = static_cast<uint16_t>(val);
                }
                cpu.gpr[ins.dst] = val;
                break;
            }

            case ir::Opcode::StoreMem: {
                uint64_t addr;
                if (ins.aux == 1) {
                    addr = cpu.rip + static_cast<int32_t>(ins.disp);
                } else if ((ins.aux & 0xF) == 6) {
                    addr = eff_addr(cpu, ins, ins.dst);
                } else {
                    addr = cpu.gpr[ins.dst] + static_cast<int32_t>(ins.disp);
                }
                const uint64_t val = (ins.aux & 0x10) ? static_cast<uint64_t>(ins.imm)
                                                      : cpu.gpr[ins.src];
                store_mem(mem, addr, val, ins.width, mem_size);
                break;
            }

            case ir::Opcode::BinOpReg: {
                auto op = static_cast<ir::IrBinOp>(ins.aux);
                uint64_t lhs = cpu.gpr[ins.dst];
                uint64_t rhs = cpu.gpr[ins.src];
                uint64_t res = binop_val(op, lhs, rhs) & mask_width(ins.width);
                cpu.gpr[ins.dst] = res;
                update_flags(cpu, res, lhs, rhs, ins.width, op);
                break;
            }

            case ir::Opcode::BinOpImm: {
                auto op = static_cast<ir::IrBinOp>(ins.aux);
                uint64_t lhs = cpu.gpr[ins.dst];
                uint64_t rhs = static_cast<uint64_t>(ins.imm);
                uint64_t res = binop_val(op, lhs, rhs) & mask_width(ins.width);
                cpu.gpr[ins.dst] = res;
                update_flags(cpu, res, lhs, rhs, ins.width, op);
                break;
            }

            case ir::Opcode::BinOpMem: {
                const bool sib = (ins.aux & 0xF) == 6;
                auto op = static_cast<ir::IrBinOp>(sib ? (ins.aux >> 4) & 0x7 : ins.aux & 0xF);
                const uint64_t addr = sib ? eff_addr(cpu, ins, ins.dst)
                                            : cpu.gpr[ins.dst] +
                                                  static_cast<int32_t>(ins.disp);
                uint64_t lhs = load_mem(mem, addr, ins.width, mem_size);
                uint64_t rhs = cpu.gpr[ins.src];
                uint64_t res = binop_val(op, lhs, rhs) & mask_width(ins.width);
                store_mem(mem, addr, res, ins.width, mem_size);
                update_flags(cpu, res, lhs, rhs, ins.width, op);
                break;
            }

            case ir::Opcode::BinOpMemImm: {
                const bool sib = (ins.aux & 0xF) == 6;
                auto op = static_cast<ir::IrBinOp>(ins.src);
                const uint64_t addr = sib ? eff_addr(cpu, ins, ins.dst)
                                            : cpu.gpr[ins.dst] +
                                                  static_cast<int32_t>(ins.disp);
                uint64_t lhs = load_mem(mem, addr, ins.width, mem_size);
                uint64_t rhs = static_cast<uint64_t>(ins.imm);
                uint64_t res = binop_val(op, lhs, rhs) & mask_width(ins.width);
                store_mem(mem, addr, res, ins.width, mem_size);
                update_flags(cpu, res, lhs, rhs, ins.width, op);
                break;
            }

            case ir::Opcode::CmpReg: {
                uint64_t lhs = cpu.gpr[ins.dst];
                uint64_t rhs = cpu.gpr[ins.src];
                if (ins.aux == 7) {
                    rhs = load_mem(mem,
                                   cpu.gpr[ins.src] + static_cast<int32_t>(ins.disp),
                                   ins.width, mem_size);
                } else if (ins.aux == 8) {
                    lhs = load_mem(mem,
                                   cpu.gpr[ins.dst] + static_cast<int32_t>(ins.disp),
                                   ins.width, mem_size);
                }
                uint64_t res = (lhs - rhs) & mask_width(ins.width);
                update_flags(cpu, res, lhs, rhs, ins.width, ir::IrBinOp::Sub);
                break;
            }

            case ir::Opcode::CmpImm: {
                uint64_t lhs = (ins.aux == 7)
                    ? load_mem(mem, cpu.gpr[ins.dst] + static_cast<int32_t>(ins.disp),
                               ins.width, mem_size)
                    : cpu.gpr[ins.dst];
                uint64_t rhs = static_cast<uint64_t>(ins.imm);
                uint64_t res = (lhs - rhs) & mask_width(ins.width);
                update_flags(cpu, res, lhs, rhs, ins.width, ir::IrBinOp::Sub);
                break;
            }

            case ir::Opcode::TestReg: {
                uint64_t lhs = (ins.aux == 7)
                    ? load_mem(mem, cpu.gpr[ins.dst] + static_cast<int32_t>(ins.disp),
                               ins.width, mem_size)
                    : cpu.gpr[ins.dst];
                uint64_t res = (lhs & cpu.gpr[ins.src]) & mask_width(ins.width);
                cpu.rflags &= ~(x86::CF | x86::OF);
                if (res == 0) cpu.rflags |= x86::ZF;
                const uint64_t sign = 1ULL << (ins.width * 8 - 1);
                if (res & sign) cpu.rflags |= x86::SF;
                break;
            }

            case ir::Opcode::Push: {
                cpu.gpr[4] -= 8;  // RSP
                store_mem(mem, cpu.gpr[4], cpu.gpr[ins.src], 8, mem_size);
                break;
            }

            case ir::Opcode::Pop:
                cpu.gpr[ins.dst] = load_mem(mem, cpu.gpr[4], 8, mem_size);
                cpu.gpr[4] += 8;
                break;

            case ir::Opcode::Lea:
                if (ins.aux == 3) {
                    cpu.gpr[ins.dst] = static_cast<uint64_t>(ins.imm);
                } else if ((ins.aux & 0xF) == 6) {
                    cpu.gpr[ins.dst] = eff_addr(cpu, ins, ins.src);
                } else {
                    cpu.gpr[ins.dst] = cpu.gpr[ins.src] + static_cast<int32_t>(ins.disp);
                }
                break;

            case ir::Opcode::Call: {
                const uint32_t ret_pc = pc;
                cpu.gpr[4] -= 8;
                store_mem(mem, cpu.gpr[4], ret_pc, 8, mem_size);
                pc = static_cast<uint32_t>(ins.imm);
                break;
            }

            case ir::Opcode::Ret: {
                pc = static_cast<uint32_t>(load_mem(mem, cpu.gpr[4], 8, mem_size));
                cpu.gpr[4] += 8;
                break;
            }

            case ir::Opcode::Leave:
                cpu.gpr[4] = cpu.gpr[5];
                cpu.gpr[5] = load_mem(mem, cpu.gpr[4], 8, mem_size);
                cpu.gpr[4] += 8;
                break;

            case ir::Opcode::Jmp:
                pc = static_cast<uint32_t>(ins.imm);
                break;

            case ir::Opcode::Jcc:
                if (eval_cc(ins.aux, cpu.rflags)) {
                    pc = static_cast<uint32_t>(ins.imm);
                }
                break;

            case ir::Opcode::Hlt:
                cpu.halt = 1;
                return 1;

            case ir::Opcode::Syscall:
                cpu.syscall_num = static_cast<int64_t>(cpu.gpr[0]);
                cpu.syscall_pending = 1;
                return 2;

            case ir::Opcode::Barrier:
                break;

            default:
                cpu.halt = 1;
                return 1;
        }
    }
    return 0;
}

}  // namespace gpuexec
