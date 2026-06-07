#include "virt/memory.hpp"

namespace virt {

GuestMemory::GuestMemory(uint64_t size) : mem_(size, 0) {}

void GuestMemory::check(uint64_t addr, size_t len) const {
    if (addr + len > mem_.size()) {
        throw std::out_of_range("guest memory access out of bounds");
    }
}

void GuestMemory::load(uint64_t guest_addr, const void* data, size_t len) {
    check(guest_addr, len);
    std::memcpy(mem_.data() + guest_addr, data, len);
}

void GuestMemory::store(uint64_t guest_addr, const void* data, size_t len) {
    load(guest_addr, data, len);
}

uint8_t GuestMemory::read8(uint64_t addr) const {
    check(addr, 1);
    return mem_[addr];
}

uint16_t GuestMemory::read16(uint64_t addr) const {
    check(addr, 2);
    uint16_t v;
    std::memcpy(&v, mem_.data() + addr, 2);
    return v;
}

uint32_t GuestMemory::read32(uint64_t addr) const {
    check(addr, 4);
    uint32_t v;
    std::memcpy(&v, mem_.data() + addr, 4);
    return v;
}

uint64_t GuestMemory::read64(uint64_t addr) const {
    check(addr, 8);
    uint64_t v;
    std::memcpy(&v, mem_.data() + addr, 8);
    return v;
}

void GuestMemory::write8(uint64_t addr, uint8_t v) {
    check(addr, 1);
    mem_[addr] = v;
}

void GuestMemory::write16(uint64_t addr, uint16_t v) {
    check(addr, 2);
    std::memcpy(mem_.data() + addr, &v, 2);
}

void GuestMemory::write32(uint64_t addr, uint32_t v) {
    check(addr, 4);
    std::memcpy(mem_.data() + addr, &v, 4);
}

void GuestMemory::write64(uint64_t addr, uint64_t v) {
    check(addr, 8);
    std::memcpy(mem_.data() + addr, &v, 8);
}

}  // namespace virt
