#include "translator.hpp"

namespace gpuemu {

namespace {

// aux bit 4 (0x10): StoreMem uses imm field as value
constexpr uint8_t kMemImmStore = 0x10;

void encode_mem_addr(ir::Instr& i, const x86::Operand& op, uint8_t flags) {
    i.dst = static_cast<uint8_t>(op.base);
    i.disp = static_cast<uint32_t>(op.disp);
    uint8_t aux = flags;
    if (op.rip_rel) aux |= 1;
    if (op.has_sib) {
        int scale_bits = 0;
        for (int s = 1; s < op.scale; s <<= 1) scale_bits++;
        aux = static_cast<uint8_t>(6 | (scale_bits << 4) | (flags & kMemImmStore));
        if (flags & kMemImmStore) {
            i.src = static_cast<uint8_t>(op.index);
        } else {
            i.imm = static_cast<int64_t>(op.index);
        }
    } else {
        i.aux = aux;
    }
    i.aux = aux;
}

}  // namespace

ir::IrBinOp Translator::map_binop(x86::BinOp op) {
    switch (op) {
        case x86::BinOp::Add: return ir::IrBinOp::Add;
        case x86::BinOp::Sub: return ir::IrBinOp::Sub;
        case x86::BinOp::And: return ir::IrBinOp::And;
        case x86::BinOp::Or:  return ir::IrBinOp::Or;
        case x86::BinOp::Xor: return ir::IrBinOp::Xor;
    }
    return ir::IrBinOp::Add;
}

void Translator::emit_x86(const x86::Instruction& insn, ir::BasicBlock& bb) {
    auto push = [&](ir::Instr i) { bb.code.push_back(i); };
    auto make = [](ir::Opcode op) {
        ir::Instr i{};
        i.op = op;
        return i;
    };

    switch (insn.kind) {
        case x86::OpKind::Nop:
            push(make(ir::Opcode::Nop));
            break;

        case x86::OpKind::MovImm: {
            if (insn.dst.is_mem) {
                ir::Instr i = make(ir::Opcode::StoreMem);
                i.src = 0;
                i.imm = insn.dst.imm;
                encode_mem_addr(i, insn.dst, kMemImmStore);
                i.width = static_cast<uint8_t>(insn.dst.width);
                push(i);
            } else {
                ir::Instr i = make(ir::Opcode::LoadImm);
                i.dst = static_cast<uint8_t>(insn.dst.reg);
                i.imm = insn.dst.imm;
                i.width = static_cast<uint8_t>(insn.dst.width);
                push(i);
            }
            break;
        }

        case x86::OpKind::MovRegReg: {
            ir::Instr i = make(ir::Opcode::MovReg);
            i.dst = static_cast<uint8_t>(insn.dst.reg);
            i.src = static_cast<uint8_t>(insn.src.reg);
            i.width = static_cast<uint8_t>(insn.dst.width);
            if (insn.sext) i.aux = 5;
            push(i);
            break;
        }

        case x86::OpKind::MovRegMem: {
            ir::Instr i = make(ir::Opcode::LoadMem);
            i.dst = static_cast<uint8_t>(insn.dst.reg);
            i.src = static_cast<uint8_t>(insn.src.base);
            i.disp = static_cast<uint32_t>(insn.src.disp);
            i.aux = insn.src.rip_rel ? 1 : 0;
            if (insn.src.has_sib) {
                i.imm = static_cast<int64_t>(insn.src.index);
                int scale_bits = 0;
                for (int s = 1; s < insn.src.scale; s <<= 1) scale_bits++;
                i.aux = static_cast<uint8_t>(6 | (scale_bits << 4));
            }
            if (insn.zext) {
                i.aux = insn.src.has_sib ? static_cast<uint8_t>(6 | (i.aux & 0xF0)) : 4;
                i.width = 1;
            } else if (insn.sext) {
                i.aux = 5;
                i.width = 4;
            } else {
                i.width = static_cast<uint8_t>(insn.dst.width);
            }
            push(i);
            break;
        }

        case x86::OpKind::MovMemReg: {
            if (!insn.dst.is_mem) {
                ir::Instr i = make(ir::Opcode::MovReg);
                i.dst = static_cast<uint8_t>(insn.dst.reg);
                i.src = static_cast<uint8_t>(insn.src.reg);
                i.width = static_cast<uint8_t>(insn.dst.width);
                push(i);
            } else {
                ir::Instr i = make(ir::Opcode::StoreMem);
                i.src = static_cast<uint8_t>(insn.src.reg);
                encode_mem_addr(i, insn.dst, 0);
                i.width = static_cast<uint8_t>(insn.dst.width);
                push(i);
            }
            break;
        }

        case x86::OpKind::BinOpRegReg: {
            ir::Instr i = make(ir::Opcode::BinOpReg);
            i.dst = static_cast<uint8_t>(insn.dst.reg);
            i.src = static_cast<uint8_t>(insn.src.reg);
            i.aux = static_cast<uint8_t>(map_binop(insn.binop));
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::BinOpRegImm: {
            ir::Instr i = make(ir::Opcode::BinOpImm);
            i.dst = static_cast<uint8_t>(insn.dst.reg);
            i.imm = insn.src.imm;
            i.aux = static_cast<uint8_t>(map_binop(insn.binop));
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::BinOpMemImm: {
            ir::Instr i = make(ir::Opcode::BinOpMemImm);
            i.imm = insn.src.imm;
            i.src = static_cast<uint8_t>(map_binop(insn.binop));
            encode_mem_addr(i, insn.dst, 0);
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::BinOpMemReg: {
            ir::Opcode op = insn.dst.is_mem ? ir::Opcode::BinOpMem : ir::Opcode::BinOpReg;
            ir::Instr i = make(op);
            i.src = static_cast<uint8_t>(insn.src.reg);
            if (insn.dst.is_mem) {
                encode_mem_addr(i, insn.dst, static_cast<uint8_t>(map_binop(insn.binop)));
            } else {
                i.dst = static_cast<uint8_t>(insn.dst.reg);
                i.aux = static_cast<uint8_t>(map_binop(insn.binop));
            }
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::CmpRegImm: {
            ir::Instr i = make(ir::Opcode::CmpImm);
            if (insn.dst.is_mem) {
                i.dst = static_cast<uint8_t>(insn.dst.base);
                i.disp = static_cast<uint32_t>(insn.dst.disp);
                i.aux = 7;
            } else {
                i.dst = static_cast<uint8_t>(insn.dst.reg);
            }
            i.imm = insn.src.imm;
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::CmpRegReg: {
            ir::Instr i = make(ir::Opcode::CmpReg);
            if (insn.src.is_mem) {
                i.dst = static_cast<uint8_t>(insn.dst.reg);
                i.src = static_cast<uint8_t>(insn.src.base);
                i.disp = static_cast<uint32_t>(insn.src.disp);
                i.aux = 7;
            } else if (insn.dst.is_mem) {
                i.dst = static_cast<uint8_t>(insn.dst.base);
                i.disp = static_cast<uint32_t>(insn.dst.disp);
                i.src = static_cast<uint8_t>(insn.src.reg);
                i.aux = 8;
            } else {
                i.dst = static_cast<uint8_t>(insn.dst.reg);
                i.src = static_cast<uint8_t>(insn.src.reg);
            }
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::TestRegReg: {
            ir::Instr i = make(ir::Opcode::TestReg);
            if (insn.dst.is_mem) {
                i.dst = static_cast<uint8_t>(insn.dst.base);
                i.disp = static_cast<uint32_t>(insn.dst.disp);
                i.aux = 7;
            } else {
                i.dst = static_cast<uint8_t>(insn.dst.reg);
            }
            i.src = static_cast<uint8_t>(insn.src.reg);
            i.width = static_cast<uint8_t>(insn.dst.width);
            push(i);
            break;
        }

        case x86::OpKind::Jmp: {
            ir::Instr i = make(ir::Opcode::Jmp);
            i.imm = static_cast<int64_t>(insn.target);
            push(i);
            break;
        }

        case x86::OpKind::Jcc: {
            ir::Instr i = make(ir::Opcode::Jcc);
            i.aux = insn.cc;
            i.imm = static_cast<int64_t>(insn.target);
            push(i);
            break;
        }

        case x86::OpKind::Push: {
            ir::Instr i = make(ir::Opcode::Push);
            i.src = static_cast<uint8_t>(insn.src.reg);
            push(i);
            break;
        }

        case x86::OpKind::Pop: {
            ir::Instr i = make(ir::Opcode::Pop);
            i.dst = static_cast<uint8_t>(insn.dst.reg);
            push(i);
            break;
        }

        case x86::OpKind::Lea: {
            ir::Instr i = make(ir::Opcode::Lea);
            i.dst = static_cast<uint8_t>(insn.dst.reg);
            if (insn.src.rip_rel) {
                i.imm = static_cast<int64_t>(insn.pc + insn.size + insn.src.disp);
                i.aux = 3;
            } else {
                i.src = static_cast<uint8_t>(insn.src.base);
                i.disp = static_cast<uint32_t>(insn.src.disp);
                if (insn.src.has_sib) {
                    i.imm = static_cast<int64_t>(insn.src.index);
                    int scale_bits = 0;
                    for (int s = 1; s < insn.src.scale; s <<= 1) scale_bits++;
                    i.aux = static_cast<uint8_t>(6 | (scale_bits << 4));
                }
            }
            push(i);
            break;
        }

        case x86::OpKind::Call: {
            ir::Instr i = make(ir::Opcode::Call);
            i.imm = static_cast<int64_t>(insn.target);
            push(i);
            break;
        }

        case x86::OpKind::Ret:
            push(make(ir::Opcode::Ret));
            break;

        case x86::OpKind::Leave:
            push(make(ir::Opcode::Leave));
            break;

        case x86::OpKind::Hlt:
            push(make(ir::Opcode::Hlt));
            break;

        case x86::OpKind::Syscall:
            push(make(ir::Opcode::Syscall));
            break;

        default:
            push(make(ir::Opcode::Hlt));
            break;
    }
}

ir::TranslationUnit Translator::translate(const uint8_t* code, size_t size, uint64_t entry_pc) {
    ir::TranslationUnit tu;
    tu.entry_pc = entry_pc;

    size_t offset = 0;
    while (offset < size) {
        size_t consumed = 0;
        auto insns = x86::decode_basic_block(code + offset, size - offset, entry_pc + offset, consumed);
        if (insns.empty() || consumed == 0) break;

        ir::BasicBlock bb;
        bb.guest_pc = entry_pc + offset;
        for (const auto& insn : insns) {
            bb.guest_pcs.push_back(insn.pc);
            emit_x86(insn, bb);
        }
        tu.blocks.push_back(std::move(bb));
        offset += consumed;
    }
    return tu;
}

}  // namespace gpuemu
