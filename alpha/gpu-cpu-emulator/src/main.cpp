#include "emulator.hpp"
#include "x86/state.hpp"
#include <cstdio>
#include <cstring>
#include <string>

#ifdef SIMPLE_OBJ_DEFAULT
static const char* default_simple_obj() { return SIMPLE_OBJ_DEFAULT; }
#else
static const char* default_simple_obj() { return "guest/simple.o"; }
#endif

static void run_guest(const char* obj_path, uint32_t num_vcpus) {
    printf("Loading compiled guest: %s\n", obj_path);
    printf("vCPUs: %u\n\n", num_vcpus);

    gpuemu::Emulator emu;
    uint64_t entry = 0;
    if (!emu.load_guest(obj_path, entry)) {
        fprintf(stderr, "Failed to load guest .o file.\n");
        fprintf(stderr, "Build first: cmake --build build\n");
        return;
    }

    for (uint32_t i = 0; i < num_vcpus; ++i) {
        x86::CpuState init{};
        init.gpr[static_cast<int>(x86::Reg::RDI)] = i;
        emu.add_vcpu(entry, &init);
    }
    emu.translate(entry);

    printf("Backend: %s | entry=0x%llx | IR ops=%zu\n\n",
           emu.using_gpu() ? "CUDA GPU" : "CPU fallback",
           static_cast<unsigned long long>(entry),
           emu.ir().code.size());

    auto result = emu.run(1 << 22);
    printf("\n--- %u guest(s) finished (exit_code=%ld, halted=%d) ---\n",
           num_vcpus, static_cast<long>(result.exit_code), result.halted);
}

static void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s [options] [path/to/guest.o]\n\n", prog);
    printf("Options:\n");
    printf("  -p N, --parallel N   Run N vCPUs in parallel (default: 1)\n");
    printf("  -h, --help           Show this help\n\n");
    printf("Default guest: build/guest/println.o (compiled from guest/println.c)\n");
}

int main(int argc, char** argv) {
    std::string obj = default_simple_obj();
    uint32_t num_vcpus = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--parallel") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", argv[i]);
                return 1;
            }
            num_vcpus = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            if (num_vcpus == 0) num_vcpus = 1;
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
        obj = argv[i];
    }

    printf("GPU x86-64 Emulator — Compiled C Guest\n");
    printf("======================================\n");
    run_guest(obj.c_str(), num_vcpus);
    return 0;
}
