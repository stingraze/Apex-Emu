#include "guest/elf_loader.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace guest {

namespace {

constexpr int kEtRel = 1;
constexpr int kEtExec = 2;
constexpr int kPtLoad = 1;
constexpr uint32_t kPfX = 1;
constexpr int kShnUndef = 0;

constexpr int kR_X86_64_64 = 1;
constexpr int kR_X86_64_PC32 = 2;
constexpr int kR_X86_64_32 = 10;
constexpr int kR_X86_64_32S = 11;

struct Elf64Ehdr {
    uint8_t e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf64Phdr {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
};

struct Elf64Shdr {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};

struct Elf64Sym {
    uint32_t st_name;
    uint8_t st_info;
    uint8_t st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
};

struct Elf64Rela {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t r_addend;
};

bool is_elf64(const uint8_t* data, size_t len) {
    if (len < sizeof(Elf64Ehdr)) return false;
    return data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' && data[3] == 'F' &&
           data[4] == 2 && data[5] == 1;
}

uint64_t align_up(uint64_t v, uint64_t a) {
    if (a == 0) return v;
    return (v + a - 1) & ~(a - 1);
}

int sym_bind(uint8_t info) { return info >> 4; }
int sym_type(uint8_t info) { return info & 0xF; }
uint32_t rel_type(uint64_t info) { return static_cast<uint32_t>(info & 0xFFFFFFFF); }
uint32_t rel_sym(uint64_t info) { return static_cast<uint32_t>(info >> 32); }

}  // namespace

LoadedImage load_elf(virt::GuestMemory& mem, const uint8_t* data, size_t len) {
    LoadedImage out;
    if (!is_elf64(data, len)) {
        out.error = "not a valid ELF64 image";
        return out;
    }

    const auto* eh = reinterpret_cast<const Elf64Ehdr*>(data);
    if (eh->e_machine != 0x3E) {
        out.error = "not x86-64";
        return out;
    }

    out.entry = eh->e_entry;

    const auto* ph = reinterpret_cast<const Elf64Phdr*>(data + eh->e_phoff);
    for (uint16_t i = 0; i < eh->e_phnum; ++i) {
        if (ph[i].p_type != kPtLoad) continue;
        if (ph[i].p_offset + ph[i].p_filesz > len) {
            out.error = "program header out of range";
            return out;
        }
        if (ph[i].p_vaddr + ph[i].p_memsz > mem.size()) {
            out.error = "segment does not fit in guest RAM";
            return out;
        }
        std::memset(mem.host_ptr() + ph[i].p_vaddr, 0, ph[i].p_memsz);
        std::memcpy(mem.host_ptr() + ph[i].p_vaddr, data + ph[i].p_offset, ph[i].p_filesz);
    }

    if (eh->e_shoff && eh->e_shnum) {
        const auto* sh = reinterpret_cast<const Elf64Shdr*>(data + eh->e_shoff);
        const auto* shstr = reinterpret_cast<const char*>(data + sh[eh->e_shstrndx].sh_offset);
        for (uint16_t i = 0; i < eh->e_shnum; ++i) {
            const char* name = shstr + sh[i].sh_name;
            if (std::strcmp(name, ".text") == 0) {
                out.code_addr = sh[i].sh_addr;
                out.code_size = static_cast<size_t>(sh[i].sh_size);
                break;
            }
        }
    }

    if (out.code_size == 0) {
        for (uint16_t i = 0; i < eh->e_phnum; ++i) {
            if (ph[i].p_type == kPtLoad && (ph[i].p_flags & kPfX)) {
                out.code_addr = ph[i].p_vaddr;
                out.code_size = static_cast<size_t>(ph[i].p_filesz);
                break;
            }
        }
    }

    if (out.code_size == 0) {
        out.error = "no executable region found";
        return out;
    }

    out.ok = true;
    return out;
}

LoadedImage load_object(virt::GuestMemory& mem, const uint8_t* data, size_t len,
                        uint64_t load_base) {
    LoadedImage out;
    if (!is_elf64(data, len)) {
        out.error = "not a valid ELF64 image";
        return out;
    }

    const auto* eh = reinterpret_cast<const Elf64Ehdr*>(data);
    if (eh->e_type != kEtRel) {
        out.error = "not a relocatable .o file (use load_elf for executables)";
        return out;
    }
    if (eh->e_machine != 0x3E) {
        out.error = "not x86-64";
        return out;
    }
    if (!eh->e_shoff || !eh->e_shnum) {
        out.error = "missing section headers";
        return out;
    }

    const auto* sh = reinterpret_cast<const Elf64Shdr*>(data + eh->e_shoff);
    const auto* shstr = reinterpret_cast<const char*>(data + sh[eh->e_shstrndx].sh_offset);

    std::vector<uint64_t> sec_addr(eh->e_shnum, 0);
    uint64_t cursor = load_base;

    auto find_sec = [&](const char* name) -> int {
        for (uint16_t i = 0; i < eh->e_shnum; ++i) {
            if (std::strcmp(shstr + sh[i].sh_name, name) == 0) return i;
        }
        return -1;
    };

    const int text_idx = find_sec(".text");
    const int rodata_idx = find_sec(".rodata");
    const int data_idx = find_sec(".data");
    const int bss_idx = find_sec(".bss");

    if (text_idx < 0) {
        out.error = "no .text section";
        return out;
    }

    auto place = [&](int idx) {
        if (idx < 0) return;
        const uint64_t al = std::max<uint64_t>(sh[idx].sh_addralign, 1);
        cursor = align_up(cursor, al);
        sec_addr[static_cast<size_t>(idx)] = cursor;
        if (sh[idx].sh_size == 0) return;
        if (cursor + sh[idx].sh_size > mem.size()) {
            out.error = "object sections exceed guest RAM";
            return;
        }
        std::memset(mem.host_ptr() + cursor, 0, sh[idx].sh_size);
        if (sh[idx].sh_type != 8 /*SHT_NOBITS*/) {  // not .bss only
            if (sh[idx].sh_offset + sh[idx].sh_size > len) {
                out.error = "section data out of range";
                return;
            }
            std::memcpy(mem.host_ptr() + cursor, data + sh[idx].sh_offset, sh[idx].sh_size);
        }
        cursor += sh[idx].sh_size;
    };

    place(text_idx);
    place(rodata_idx);
    place(data_idx);
    place(bss_idx);

    if (!out.error.empty()) return out;

    const Elf64Sym* symtab = nullptr;
    const char* strtab = nullptr;
    uint64_t sym_count = 0;

    const int symtab_idx = find_sec(".symtab");
    const int strtab_idx = find_sec(".strtab");
    if (symtab_idx >= 0) {
        symtab = reinterpret_cast<const Elf64Sym*>(data + sh[symtab_idx].sh_offset);
        sym_count = sh[symtab_idx].sh_size / sizeof(Elf64Sym);
        if (strtab_idx >= 0) {
            strtab = reinterpret_cast<const char*>(data + sh[strtab_idx].sh_offset);
        }
    }

    auto sym_value = [&](uint32_t sym_idx) -> uint64_t {
        if (!symtab || sym_idx >= sym_count) return 0;
        const Elf64Sym& s = symtab[sym_idx];
        if (s.st_shndx == kShnUndef) return 0;
        if (s.st_shndx < eh->e_shnum) {
            return sec_addr[s.st_shndx] + s.st_value;
        }
        return s.st_value;
    };

    auto apply_rela = [&](int rela_sec_idx) {
        if (rela_sec_idx < 0) return;
        const uint32_t target_sec = sh[rela_sec_idx].sh_info;
        if (target_sec >= eh->e_shnum) return;

        const auto* rela = reinterpret_cast<const Elf64Rela*>(data + sh[rela_sec_idx].sh_offset);
        const size_t count = sh[rela_sec_idx].sh_size / sizeof(Elf64Rela);
        const uint64_t base = sec_addr[target_sec];

        for (size_t i = 0; i < count; ++i) {
            const uint64_t place = base + rela[i].r_offset;
            const uint32_t type = rel_type(rela[i].r_info);
            const uint32_t sym = rel_sym(rela[i].r_info);
            const int64_t addend = rela[i].r_addend;
            const uint64_t S = sym_value(sym);
            const uint64_t P = place;

            if (place + 8 > mem.size()) continue;

            switch (type) {
                case kR_X86_64_64: {
                    auto* p = reinterpret_cast<uint64_t*>(mem.host_ptr() + place);
                    *p = static_cast<uint64_t>(static_cast<int64_t>(S + addend));
                    break;
                }
                case kR_X86_64_32: {
                    auto* p = reinterpret_cast<uint32_t*>(mem.host_ptr() + place);
                    *p = static_cast<uint32_t>(S + addend);
                    break;
                }
                case kR_X86_64_32S: {
                    auto* p = reinterpret_cast<int32_t*>(mem.host_ptr() + place);
                    *p = static_cast<int32_t>(S + addend);
                    break;
                }
                case kR_X86_64_PC32: {
                    auto* p = reinterpret_cast<int32_t*>(mem.host_ptr() + place);
                    const int64_t v = static_cast<int64_t>(S + addend - P);
                    *p = static_cast<int32_t>(v);
                    break;
                }
                default:
                    break;
            }
        }
    };

    for (uint16_t i = 0; i < eh->e_shnum; ++i) {
        if (sh[i].sh_type == 4 /*SHT_RELA*/) {  // SHT_RELA
            apply_rela(static_cast<int>(i));
        }
    }

    out.code_addr = sec_addr[static_cast<size_t>(text_idx)];
    out.code_size = static_cast<size_t>(sh[text_idx].sh_size);

    if (symtab && strtab) {
        for (uint64_t i = 0; i < sym_count; ++i) {
            if (sym_type(symtab[i].st_info) != 2) continue;  // STT_FUNC
            if (std::strcmp(strtab + symtab[i].st_name, "_start") == 0) {
                out.entry = sym_value(static_cast<uint32_t>(i));
                break;
            }
        }
    }

    if (out.entry == 0) {
        out.entry = out.code_addr;
    }

    out.ok = true;
    return out;
}

static LoadedImage load_file_bytes(virt::GuestMemory& mem, const std::vector<uint8_t>& buf,
                                   uint64_t load_base) {
    if (buf.size() < sizeof(Elf64Ehdr)) {
        LoadedImage out;
        out.error = "file too small";
        return out;
    }
    const auto* eh = reinterpret_cast<const Elf64Ehdr*>(buf.data());
    if (eh->e_type == kEtRel) {
        return load_object(mem, buf.data(), buf.size(), load_base);
    }
    if (eh->e_type == kEtExec) {
        return load_elf(mem, buf.data(), buf.size());
    }
    LoadedImage out;
    out.error = "unsupported ELF type";
    return out;
}

LoadedImage load_elf_file(virt::GuestMemory& mem, const char* path) {
    LoadedImage out;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        out.error = std::string("cannot open ") + path;
        return out;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), {});
    return load_file_bytes(mem, buf, 0x100000);
}

LoadedImage load_guest_file(virt::GuestMemory& mem, const char* path, uint64_t load_base) {
    LoadedImage out;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        out.error = std::string("cannot open ") + path;
        return out;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), {});
    return load_file_bytes(mem, buf, load_base);
}

}  // namespace guest
