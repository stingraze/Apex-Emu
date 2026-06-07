#include "x86/decode.hpp"
#include <cstring>

namespace x86 {

Decoder::Decoder(const uint8_t* code, size_t size, uint64_t base_pc)
    : code_(code), size_(size), base_pc_(base_pc), pc_(base_pc), pos_(0) {}

uint8_t Decoder::fetch8() {
    return need(1) ? code_[pos_++] : 0;
}

uint16_t Decoder::fetch16() {
    if (!need(2)) return 0;
    uint16_t v;
    std::memcpy(&v, code_ + pos_, 2);
    pos_ += 2;
    return v;
}

uint32_t Decoder::fetch32() {
    if (!need(4)) return 0;
    uint32_t v;
    std::memcpy(&v, code_ + pos_, 4);
    pos_ += 4;
    return v;
}

uint64_t Decoder::fetch64() {
    if (!need(8)) return 0;
    uint64_t v;
    std::memcpy(&v, code_ + pos_, 8);
    pos_ += 8;
    return v;
}

bool Decoder::parse_rex(uint8_t& rex) {
    if (need(1) && (code_[pos_] & 0xF0) == 0x40) {
        rex = fetch8();
        return true;
    }
    rex = 0;
    return false;
}

Reg Decoder::decode_reg(uint8_t code, Width w, uint8_t rex, bool rm_field) const {
    code &= 7;
    const int ext = (rm_field ? (rex & 1) : (rex & 4)) ? 8 : 0;
    if (w == Width::B) {
        static const Reg lo8[] = {Reg::RAX, Reg::RCX, Reg::RDX, Reg::RBX,
                                  Reg::RSP, Reg::RBP, Reg::RSI, Reg::RDI};
        return static_cast<Reg>(static_cast<int>(lo8[code]) + ext);
    }
    return static_cast<Reg>(code + ext);
}

bool Decoder::parse_sib(Operand& op, uint8_t rex, uint8_t mod) {
    if (!need(1)) return false;
    uint8_t sib = fetch8();
    op.scale = 1 << (sib >> 6);
    const uint8_t rex_x = static_cast<uint8_t>((rex & 2) ? 0x4 : 0);
    op.index = decode_reg((sib >> 3) & 7, Width::Q, rex_x, true);
    op.base = decode_reg(sib & 7, Width::Q, rex, true);
    if ((sib & 7) == 5 && mod == 0) {
        op.disp = static_cast<int32_t>(fetch32());
    }
    return true;
}

bool Decoder::parse_modrm(Operand& op, Width w, uint8_t rex, uint8_t modrm) {
    uint8_t mod = modrm >> 6;
    uint8_t rm = modrm & 7;
    uint8_t reg = (modrm >> 3) & 7;

    if (!op.is_mem) {
        op.reg = decode_reg(reg, w, rex, false);
        return true;
    }

    op.base = decode_reg(rm, Width::Q, rex, true);
    if (mod == 3) {
        op.is_mem = false;
        op.reg = op.base;
        return true;
    }

    if (rm == 4) {
        op.disp = 0;
        op.has_sib = true;
        if (!parse_sib(op, rex, mod)) return false;
        if (mod == 0) {
            return true;
        }
    } else if (rm == 5 && mod == 0) {
        op.rip_rel = true;
        op.disp = static_cast<int32_t>(fetch32());
        return true;
    } else if (mod == 0) {
        op.disp = 0;
        return true;
    }

    if (mod == 1) op.disp = static_cast<int8_t>(fetch8());
    else op.disp = static_cast<int32_t>(fetch32());
    return true;
}

bool Decoder::parse_modrm(Operand& op, Width w, uint8_t rex) {
    if (!need(1)) return false;
    return parse_modrm(op, w, rex, fetch8());
}

bool Decoder::decode_one(Instruction& out) {
    if (!need(1)) return false;

    const size_t start = pos_;
    out = Instruction{};
    out.pc = pc_;

    uint8_t rex = 0;
    bool in_rex = parse_rex(rex);

    uint8_t b0 = fetch8();
    Width w = Width::Q;
    OpKind kind = OpKind::Invalid;

    // REX.W for 64-bit
    if (in_rex && (rex & 8)) w = Width::Q;

    auto read_imm = [&](Width iw) -> int64_t {
        switch (iw) {
            case Width::B: return static_cast<int8_t>(fetch8());
            case Width::W: return static_cast<int16_t>(fetch16());
            case Width::D: return static_cast<int32_t>(fetch32());
            case Width::Q: return static_cast<int64_t>(fetch64());
        }
        return 0;
    };

    // NOP
    if (b0 == 0x90) {
        out.kind = OpKind::Nop;
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // HLT
    if (b0 == 0xF4) {
        out.kind = OpKind::Hlt;
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    if (b0 == 0x0F) {
        uint8_t b1 = fetch8();
        if (decode_0f(b1, rex, in_rex, out, start)) return true;
    }

    // MOV reg, imm32/64: B8+rd
    if (b0 >= 0xB8 && b0 <= 0xBF) {
        const Width iw = (in_rex && (rex & 8)) ? Width::Q : Width::D;
        out.kind = OpKind::MovImm;
        out.dst.reg = decode_reg(b0 - 0xB8, iw, rex, false);
        out.dst.width = iw;
        out.dst.imm = (iw == Width::Q) ? static_cast<int64_t>(fetch64())
                                       : static_cast<int32_t>(fetch32());
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // PUSH/POP r64: 50-5F
    if (b0 >= 0x50 && b0 <= 0x57) {
        out.kind = OpKind::Push;
        out.src.reg = decode_reg(b0 - 0x50, Width::Q, rex, false);
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }
    if (b0 >= 0x58 && b0 <= 0x5F) {
        out.kind = OpKind::Pop;
        out.dst.reg = decode_reg(b0 - 0x58, Width::Q, rex, false);
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // JMP rel8
    if (b0 == 0xEB) {
        out.kind = OpKind::Jmp;
        out.target = pc_ + 2 + static_cast<int8_t>(fetch8());
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // JMP rel32: E9
    if (b0 == 0xE9) {
        out.kind = OpKind::Jmp;
        out.target = pc_ + 5 + static_cast<int32_t>(fetch32());
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // Group: 83 /0-/7 imm8
    if (b0 == 0x83) {
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        uint8_t op = (modrm >> 3) & 7;
        static const BinOp ops[] = {BinOp::Add, BinOp::Or, BinOp::Add, BinOp::Sub,
                                    BinOp::And, BinOp::Sub, BinOp::Xor, BinOp::Add};
        out.dst.is_mem = true;
        out.dst.width = w;
        parse_modrm(out.dst, w, rex, modrm);
        const int8_t imm8 = static_cast<int8_t>(fetch8());
        if (op == 7) {
            out.kind = OpKind::CmpRegImm;
            out.src.imm = imm8;
        } else if (!out.dst.is_mem) {
            out.kind = OpKind::BinOpRegImm;
            out.binop = ops[op];
            out.dst.reg = out.dst.reg;
            out.src.imm = imm8;
        } else {
            out.kind = OpKind::BinOpMemImm;
            out.binop = ops[op];
            out.src.imm = imm8;
        }
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // 81 /0-/7 imm32
    if (b0 == 0x81) {
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        uint8_t op = (modrm >> 3) & 7;
        static const BinOp ops[] = {BinOp::Add, BinOp::Or, BinOp::Add, BinOp::Sub,
                                    BinOp::And, BinOp::Sub, BinOp::Xor, BinOp::Add};
        if (op == 7) {
            out.kind = OpKind::CmpRegImm;
        } else {
            out.kind = OpKind::BinOpRegImm;
            out.binop = ops[op];
        }
        out.dst.is_mem = true;
        out.dst.width = Width::D;
        parse_modrm(out.dst, Width::D, rex, modrm);
        out.src.imm = static_cast<int32_t>(fetch32());
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // MOV r/m, imm: C7 /0
    if (b0 == 0xC7) {
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        if (((modrm >> 3) & 7) != 0) return false;
        out.kind = OpKind::MovImm;
        out.dst.is_mem = true;
        out.dst.width = w;
        parse_modrm(out.dst, w, rex, modrm);
        out.dst.imm = static_cast<int32_t>(fetch32());
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // MOV r/m64, imm32: C7 /0 with REX.W — use imm32 zero-extended
    if (in_rex && (rex & 8) && b0 == 0xC7) {
        // handled above
    }

    // REX-prefixed opcodes
    if (in_rex) {
        // 48 89 /r MOV r/m64, r64
        if (b0 == 0x89) {
            out.kind = OpKind::MovMemReg;
            out.dst.is_mem = true;
            out.dst.width = Width::Q;
            out.src.is_mem = false;
            out.src.width = Width::Q;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 8B /r MOV r64, r/m64
        if (b0 == 0x8B) {
            out.kind = OpKind::MovRegMem;
            out.dst.is_mem = false;
            out.dst.width = Width::Q;
            out.src.is_mem = true;
            out.src.width = Width::Q;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.dst.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.src, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 01 /r ADD r/m64, r64
        if (b0 == 0x01) {
            out.kind = OpKind::BinOpMemReg;
            out.binop = BinOp::Add;
            out.dst.is_mem = true;
            out.dst.width = Width::Q;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 29 /r SUB r/m64, r64
        if (b0 == 0x29) {
            out.kind = OpKind::BinOpMemReg;
            out.binop = BinOp::Sub;
            out.dst.is_mem = true;
            out.dst.width = Width::Q;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 31 /r XOR r/m64, r64
        if (b0 == 0x31) {
            out.kind = OpKind::BinOpMemReg;
            out.binop = BinOp::Xor;
            out.dst.is_mem = true;
            out.dst.width = Width::Q;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 21 /r AND r/m64, r64
        if (b0 == 0x21) {
            out.kind = OpKind::BinOpMemReg;
            out.binop = BinOp::And;
            out.dst.is_mem = true;
            out.dst.width = Width::Q;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 09 /r OR r/m64, r64
        if (b0 == 0x09) {
            out.kind = OpKind::BinOpMemReg;
            out.binop = BinOp::Or;
            out.dst.is_mem = true;
            out.dst.width = Width::Q;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 01 /r ADD r64, r64 — also 01 /r without mem
        // 48 03 /r ADD r64, r/m64
        if (b0 == 0x03) {
            out.kind = OpKind::BinOpRegReg;
            out.binop = BinOp::Add;
            out.dst.is_mem = false;
            out.src.is_mem = true;
            out.dst.width = Width::Q;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.dst.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.src, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 29 /r SUB r64, r/m64 — 2B
        if (b0 == 0x2B) {
            out.kind = OpKind::BinOpRegReg;
            out.binop = BinOp::Sub;
            out.dst.is_mem = false;
            out.src.is_mem = true;
            out.dst.width = Width::Q;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.dst.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.src, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 39 /r CMP r/m64, r64
        if (b0 == 0x39) {
            out.kind = OpKind::CmpRegReg;
            out.dst.is_mem = true;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 3B /r CMP r64, r/m64
        if (b0 == 0x3B) {
            out.kind = OpKind::CmpRegReg;
            out.dst.is_mem = false;
            out.src.is_mem = true;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.dst.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.src, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 85 /r TEST r64, r64
        if (b0 == 0x85) {
            out.kind = OpKind::TestRegReg;
            out.dst.is_mem = true;
            out.src.is_mem = false;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.src.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.dst, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 8D /r LEA r64, m
        if (b0 == 0x8D) {
            out.kind = OpKind::Lea;
            out.dst.is_mem = false;
            out.src.is_mem = true;
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            uint8_t reg = (modrm >> 3) & 7;
            out.dst.reg = decode_reg(reg, Width::Q, rex, false);
            parse_modrm(out.src, Width::Q, rex, modrm);
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
        // 48 C7 C0 imm32 MOV rax, imm32 (sign-extended)
        if (b0 == 0xC7) {
            if (!need(1)) return false;
            uint8_t modrm = fetch8();
            if ((modrm >> 6) != 3) return false;
            out.kind = OpKind::MovImm;
            out.dst.reg = decode_reg(modrm & 7, Width::Q, rex, true);
            out.dst.width = Width::Q;
            out.dst.imm = static_cast<int32_t>(fetch32());
            out.size = pos_ - start;
            pc_ += out.size;
            return true;
        }
    }

    // Non-REX 89 MOV r/m32, r32
    if (b0 == 0x89) {
        out.kind = OpKind::MovMemReg;
        out.dst.is_mem = true;
        out.dst.width = Width::D;
        out.src.is_mem = false;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.src.reg = decode_reg((modrm >> 3) & 7, Width::D, rex, false);
        parse_modrm(out.dst, Width::D, rex, modrm);
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // 8B MOV r32, r/m32
    if (b0 == 0x8B) {
        out.kind = OpKind::MovRegMem;
        out.dst.is_mem = false;
        out.dst.width = Width::D;
        out.src.is_mem = true;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.dst.reg = decode_reg((modrm >> 3) & 7, Width::D, rex, false);
        parse_modrm(out.src, Width::D, rex, modrm);
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    // 01 ADD r/m, r
    if (b0 == 0x01) {
        out.kind = OpKind::BinOpMemReg;
        out.binop = BinOp::Add;
        out.dst.is_mem = true;
        out.dst.width = Width::D;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.src.reg = decode_reg((modrm >> 3) & 7, Width::D, rex, false);
        parse_modrm(out.dst, Width::D, rex, modrm);
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    }

    if (decode_extended(b0, rex, in_rex, w, out, start)) return true;

    out.kind = OpKind::Invalid;
    out.size = pos_ - start;
    pc_ += out.size;
    return false;
}

std::vector<Instruction> decode_basic_block(const uint8_t* code, size_t size,
                                              uint64_t base_pc, size_t& consumed) {
    std::vector<Instruction> block;
    Decoder dec(code, size, base_pc);
    Instruction insn;
    while (dec.decode_one(insn)) {
        block.push_back(insn);
        if (insn.kind == OpKind::Jmp || insn.kind == OpKind::Jcc ||
            insn.kind == OpKind::Ret || insn.kind == OpKind::Hlt ||
            insn.kind == OpKind::Syscall || insn.kind == OpKind::Invalid) {
            break;
        }
    }
    consumed = dec.pc() - base_pc;
    return block;
}

}  // namespace x86
