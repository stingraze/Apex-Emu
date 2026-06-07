#include "x86/decode.hpp"

namespace x86 {

bool Decoder::decode_0f(uint8_t b1, uint8_t rex, bool in_rex, Instruction& out, size_t start) {
    auto finish = [&]() {
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    };

    if (b1 == 0x05) {
        out.kind = OpKind::Syscall;
        return finish();
    }
    if ((b1 & 0xF0) == 0x80) {
        out.kind = OpKind::Jcc;
        out.cc = b1 & 0xF;
        out.target = pc_ + 6 + static_cast<int32_t>(fetch32());
        return finish();
    }
    if (b1 == 0x85) {
        out.kind = OpKind::Jcc;
        out.cc = 5;
        out.target = pc_ + 6 + static_cast<int32_t>(fetch32());
        return finish();
    }
    if (b1 == 0xB6) {
        out.kind = OpKind::MovRegMem;
        out.dst.is_mem = false;
        out.dst.width = (in_rex && (rex & 8)) ? Width::Q : Width::D;
        out.src.is_mem = true;
        out.src.width = Width::B;
        out.zext = true;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.dst.reg = decode_reg((modrm >> 3) & 7, out.dst.width, rex, false);
        parse_modrm(out.src, Width::B, rex, modrm);
        return finish();
    }
    (void)in_rex;
    return false;
}

bool Decoder::decode_extended(uint8_t b0, uint8_t rex, bool in_rex, Width w, Instruction& out,
                               size_t start) {
    auto finish = [&]() {
        out.size = pos_ - start;
        pc_ += out.size;
        return true;
    };

    if (b0 == 0xE8) {
        out.kind = OpKind::Call;
        const int32_t rel = static_cast<int32_t>(fetch32());
        out.target = pc_ + 5 + rel;
        return finish();
    }
    if (b0 == 0xC3) {
        out.kind = OpKind::Ret;
        return finish();
    }
    if (b0 == 0xC9) {
        out.kind = OpKind::Leave;
        return finish();
    }
    if (b0 >= 0x70 && b0 <= 0x7F) {
        out.kind = OpKind::Jcc;
        out.cc = static_cast<uint8_t>(b0 - 0x70);
        out.target = pc_ + 2 + static_cast<int8_t>(fetch8());
        return finish();
    }

    if (in_rex && b0 == 0x3B) {
        out.kind = OpKind::CmpRegReg;
        out.dst.is_mem = false;
        out.src.is_mem = true;
        out.dst.width = Width::Q;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.dst.reg = decode_reg((modrm >> 3) & 7, Width::Q, rex, false);
        parse_modrm(out.src, Width::Q, rex, modrm);
        return finish();
    }

    if (b0 == 0x88) {
        out.kind = OpKind::MovMemReg;
        out.dst.is_mem = true;
        out.dst.width = Width::B;
        out.src.is_mem = false;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.src.reg = decode_reg((modrm >> 3) & 7, Width::B, rex, false);
        parse_modrm(out.dst, Width::B, rex, modrm);
        return finish();
    }

    if (b0 == 0x84) {
        out.kind = OpKind::TestRegReg;
        out.dst.is_mem = true;
        out.src.is_mem = false;
        out.dst.width = Width::B;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.src.reg = decode_reg((modrm >> 3) & 7, Width::B, rex, false);
        parse_modrm(out.dst, Width::B, rex, modrm);
        return finish();
    }

    if (b0 == 0xC6) {
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.kind = OpKind::MovImm;
        out.dst.is_mem = true;
        out.dst.width = Width::B;
        parse_modrm(out.dst, Width::B, rex, modrm);
        out.dst.imm = static_cast<int8_t>(fetch8());
        return finish();
    }

    if (b0 == 0x8D) {
        out.kind = OpKind::Lea;
        out.dst.is_mem = false;
        out.src.is_mem = true;
        out.dst.width = w;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.dst.reg = decode_reg((modrm >> 3) & 7, out.dst.width, rex, false);
        parse_modrm(out.src, out.dst.width, rex, modrm);
        return finish();
    }

    if (in_rex && (rex & 8) && b0 == 0x98) {
        out.kind = OpKind::MovRegReg;
        out.dst.reg = Reg::RAX;
        out.src.reg = Reg::RAX;
        out.dst.width = Width::Q;
        out.src.width = Width::D;
        out.sext = true;
        return finish();
    }

    if (b0 == 0x63) {
        out.kind = OpKind::MovRegMem;
        out.dst.is_mem = false;
        out.dst.width = Width::Q;
        out.src.is_mem = true;
        out.src.width = Width::D;
        out.sext = true;
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        out.dst.reg = decode_reg((modrm >> 3) & 7, Width::Q, rex, false);
        parse_modrm(out.src, Width::D, rex, modrm);
        return finish();
    }

    // FF /0 INC, FF /1 DEC
    if (b0 == 0xFF) {
        if (!need(1)) return false;
        uint8_t modrm = fetch8();
        const uint8_t op = (modrm >> 3) & 7;
        if (op > 1) return false;
        out.binop = (op == 0) ? BinOp::Add : BinOp::Sub;
        out.dst.is_mem = true;
        out.dst.width = w;
        parse_modrm(out.dst, w, rex, modrm);
        out.src.imm = 1;
        if (!out.dst.is_mem) {
            out.kind = OpKind::BinOpRegImm;
        } else {
            out.kind = OpKind::BinOpMemImm;
        }
        return finish();
    }

    (void)w;
    return false;
}

}  // namespace x86
