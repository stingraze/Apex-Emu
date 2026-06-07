#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace virt {

// Guest physical memory with page-granular trap support for MMIO / syscalls.
class GuestMemory {
public:
    static constexpr uint64_t PAGE_SIZE = 4096;
    static constexpr uint64_t DEFAULT_SIZE = 16 * 1024 * 1024;  // 16 MiB

    explicit GuestMemory(uint64_t size = DEFAULT_SIZE);

    void load(uint64_t guest_addr, const void* data, size_t len);
    void store(uint64_t guest_addr, const void* data, size_t len);

    uint8_t read8(uint64_t addr) const;
    uint16_t read16(uint64_t addr) const;
    uint32_t read32(uint64_t addr) const;
    uint64_t read64(uint64_t addr) const;

    void write8(uint64_t addr, uint8_t v);
    void write16(uint64_t addr, uint16_t v);
    void write32(uint64_t addr, uint32_t v);
    void write64(uint64_t addr, uint64_t v);

    uint8_t* host_ptr() { return mem_.data(); }
    const uint8_t* host_ptr() const { return mem_.data(); }
    uint64_t size() const { return mem_.size(); }

private:
    void check(uint64_t addr, size_t len) const;
    std::vector<uint8_t> mem_;
};

}  // namespace virt
