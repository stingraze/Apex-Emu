#include "guest/elf_loader.hpp"
#include "virt/memory.hpp"
#include "x86/decode.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "guest/demo.o";
    virt::GuestMemory mem;
    auto img = guest::load_guest_file(mem, path);
    if (!img.ok) {
        printf("load fail: %s\n", img.error.c_str());
        return 1;
    }
    printf("entry=0x%llx text=0x%llx size=%zu\n",
           (unsigned long long)img.entry,
           (unsigned long long)img.code_addr,
           img.code_size);

    x86::Decoder dec(mem.host_ptr() + img.code_addr, img.code_size, img.code_addr);
    x86::Instruction insn;
    int n = 0, bad = 0;
    while (dec.decode_one(insn)) {
        if (insn.kind == x86::OpKind::Invalid) bad++;
        n++;
    }
    printf("decoded=%d invalid=%d end_pc=0x%llx\n", n, bad,
           (unsigned long long)dec.pc());
    return 0;
}
