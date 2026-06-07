#include "guest/elf_loader.hpp"
#include "virt/memory.hpp"
#include "x86/decode.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "guest/demo.o";
    virt::GuestMemory mem;
    uint64_t entry = 0;
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
    uint64_t last_pc = 0;
    int last_kind = 0;
    size_t last_size = 0;
    while (dec.decode_one(insn)) {
        if (insn.kind == x86::OpKind::Invalid) bad++;
        last_pc = insn.pc;
        last_kind = static_cast<int>(insn.kind);
        last_size = insn.size;
        if (insn.pc >= img.code_addr + 0xa0) {
            printf("pc=0x%llx kind=%d size=%zu\n",
                   (unsigned long long)insn.pc, (int)insn.kind, insn.size);
        }
        n++;
    }
    const uint64_t fail = dec.pc();
    printf("last ok: pc=0x%llx kind=%d size=%zu\n",
           (unsigned long long)last_pc, last_kind, last_size);
    printf("bytes@0xff:");
    for (int i = 0; i < 8; ++i) printf(" %02x", mem.host_ptr()[img.code_addr + 0xff + i]);
    printf("\nfail at 0x%llx bytes:", (unsigned long long)fail);
    for (int i = 0; i < 8; ++i) printf(" %02x", mem.host_ptr()[fail + i]);
    printf("\n");
    printf("decoded=%d invalid=%d end_pc=0x%llx\n", n, bad, (unsigned long long)dec.pc());
    return 0;
}
