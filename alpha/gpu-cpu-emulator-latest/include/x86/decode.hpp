#pragma once

#include "types.hpp"
#include <cstddef>
#include <vector>

namespace x86 {

class Decoder {
public:
    explicit Decoder(const uint8_t* code, size_t size, uint64_t base_pc = 0);

    // Decode one instruction; returns false on failure or end of stream.
    bool decode_one(Instruction& out);

    uint64_t pc() const { return pc_; }

private:
    bool need(size_t n) const { return pos_ + n <= size_; }
    uint8_t fetch8();
    uint16_t fetch16();
    uint32_t fetch32();
    uint64_t fetch64();

    bool parse_rex(uint8_t& rex);
    bool parse_modrm(Operand& op, Width w, uint8_t rex);
    bool parse_modrm(Operand& op, Width w, uint8_t rex, uint8_t modrm);
    bool parse_sib(Operand& op, uint8_t rex, uint8_t mod);
    Reg decode_reg(uint8_t code, Width w, uint8_t rex, bool rm_field = false) const;
    bool decode_extended(uint8_t b0, uint8_t rex, bool in_rex, Width w, Instruction& out,
                         size_t start);
    bool decode_0f(uint8_t b1, uint8_t rex, bool in_rex, Instruction& out, size_t start);

    const uint8_t* code_;
    size_t size_;
    uint64_t base_pc_;
    uint64_t pc_;
    size_t pos_;
    uint8_t rex_ = 0;
    bool has_rex_ = false;
};

// Split guest code into straight-line basic blocks (ends at branch/halt).
std::vector<Instruction> decode_basic_block(const uint8_t* code, size_t size,
                                            uint64_t base_pc, size_t& consumed);

}  // namespace x86
