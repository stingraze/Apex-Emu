#pragma once

#include "virt/memory.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace guest {

struct LoadedImage {
    uint64_t entry = 0;
    uint64_t code_addr = 0;   // region to translate (executable .text)
    size_t code_size = 0;
    bool ok = false;
    std::string error;
};

// Load a static x86-64 ELF into guest memory.
LoadedImage load_elf(virt::GuestMemory& mem, const uint8_t* data, size_t len);

LoadedImage load_object(virt::GuestMemory& mem, const uint8_t* data, size_t len,
                        uint64_t load_base = 0x100000);

LoadedImage load_guest_file(virt::GuestMemory& mem, const char* path,
                            uint64_t load_base = 0x100000);

}  // namespace guest
