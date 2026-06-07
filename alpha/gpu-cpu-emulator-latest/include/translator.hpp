#pragma once

#include "ir/opcode.hpp"
#include "x86/decode.hpp"
#include <cstdint>
#include <vector>

namespace gpuemu {

// Translates guest x86-64 machine code into GPU IR basic blocks.
class Translator {
public:
    ir::TranslationUnit translate(const uint8_t* code, size_t size, uint64_t entry_pc);

private:
    void emit_x86(const x86::Instruction& insn, ir::BasicBlock& bb);
    static ir::IrBinOp map_binop(x86::BinOp op);
};

}  // namespace gpuemu
