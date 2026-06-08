#include "emulator.hpp"
#include "x86/state.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#ifdef GPUEMU_CUDA
extern "C" const char* gpuemu_cuda_device_name();
#endif

#ifdef SIMPLE_OBJ_DEFAULT
static const char* default_simple_obj() { return SIMPLE_OBJ_DEFAULT; }
#else
static const char* default_simple_obj() { return "build/guest/simple.o"; }
#endif

#ifdef PARALLEL_OBJ_DEFAULT
static const char* default_parallel_obj() { return PARALLEL_OBJ_DEFAULT; }
#else
static const char* default_parallel_obj() { return "build/guest/parallel.o"; }
#endif

#ifdef PRINTLN_OBJ_DEFAULT
static const char* default_guest_obj() { return PRINTLN_OBJ_DEFAULT; }
#else
static const char* default_guest_obj() { return "build/guest/println.o"; }
#endif

#ifdef DEMO_OBJ_DEFAULT
static const char* default_demo_obj() { return DEMO_OBJ_DEFAULT; }
#else
static const char* default_demo_obj() { return "build/guest/demo.o"; }
#endif

#ifdef GPUBENCH_OBJ_DEFAULT
static const char* default_bench_obj() { return GPUBENCH_OBJ_DEFAULT; }
#else
static const char* default_bench_obj() { return "build/guest/gpubench.o"; }
#endif

#ifdef MANDEL_OBJ_DEFAULT
static const char* default_mandel_obj() { return MANDEL_OBJ_DEFAULT; }
#else
static const char* default_mandel_obj() { return "build/guest/mandel.o"; }
#endif

static constexpr uint32_t kMandelWidth = 64;
static constexpr uint32_t kMandelHeight = 32;
static constexpr uint64_t kMandelFbBase = 0x200000UL;

struct BenchResult {
    double seconds = 0.0;
    uint64_t tally = 0;
    bool all_halted = false;
    bool used_gpu = false;
};

static uint64_t sum_exit_codes(gpuemu::Emulator& emu) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < emu.vcpu_count(); ++i) {
        const auto& st = emu.vcpu(i).state();
        if (st.halt) {
            total += static_cast<uint64_t>(st.syscall_ret);
        }
    }
    return total;
}

static bool guest_obj_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static bool setup_mandel_emulator(gpuemu::Emulator& emu, const char* obj_path, uint32_t num_vcpus,
                                  uint32_t rows_per, uint64_t& entry_out) {
    if (!emu.load_guest(obj_path, entry_out)) {
        fprintf(stderr, "Failed to load Mandelbrot guest: %s\n", obj_path);
        return false;
    }
    for (uint32_t i = 0; i < num_vcpus; ++i) {
        x86::CpuState init{};
        init.gpr[static_cast<int>(x86::Reg::RDI)] = i;
        init.gpr[static_cast<int>(x86::Reg::RSI)] = num_vcpus;
        init.gpr[static_cast<int>(x86::Reg::RCX)] = static_cast<uint64_t>(i) * rows_per;
        init.gpr[static_cast<int>(x86::Reg::RDX)] = static_cast<uint64_t>(i + 1) * rows_per;
        emu.add_vcpu(entry_out, &init);
    }
    emu.translate(entry_out);
    if (emu.ir().code.empty()) {
        fprintf(stderr, "No guest code was translated. Check that %s is a valid x86-64 .o\n",
                obj_path);
        return false;
    }
    return true;
}

static void print_mandel_framebuffer(const gpuemu::Emulator& emu, uint32_t rows_per) {
    const uint64_t mem_sz = emu.vcpu_memory_size();
    for (uint32_t y = 0; y < kMandelHeight; ++y) {
        const uint32_t vcpu = y / rows_per;
        const uint8_t* mem = emu.vcpu_memory(vcpu);
        for (uint32_t x = 0; x < kMandelWidth; ++x) {
            const uint64_t addr = kMandelFbBase + static_cast<uint64_t>(y) * kMandelWidth + x;
            char c = ' ';
            if (addr < mem_sz) {
                c = static_cast<char>(mem[addr]);
                if (c == '\0') {
                    c = ' ';
                }
            }
            putchar(c);
        }
        putchar('\n');
    }
}

static void run_mandel(const char* obj_path, uint32_t num_vcpus, uint32_t max_steps) {
    if (!guest_obj_exists(obj_path)) {
        fprintf(stderr,
                "Mandelbrot guest not found: %s\n"
                "Build it first:  cmake -S . -B build && cmake --build build\n",
                obj_path);
        return;
    }

    if (kMandelHeight % num_vcpus != 0) {
        fprintf(stderr,
                "Mandelbrot height %u must be divisible by vCPU count %u "
                "(try -p 1, 2, 4, 8, 16, or 32).\n",
                kMandelHeight, num_vcpus);
        return;
    }

    const uint32_t rows_per = kMandelHeight / num_vcpus;

    printf("\n");
    printf("  Parallel Mandelbrot — %u vCPUs × %u rows each (%ux%u ASCII)\n",
           num_vcpus, rows_per, kMandelWidth, kMandelHeight);
    printf("  Compiled C guest (fixed-point); each vCPU computes its band in isolated RAM.\n\n");

    gpuemu::Emulator emu;
    uint64_t entry = 0;
    if (!setup_mandel_emulator(emu, obj_path, num_vcpus, rows_per, entry)) {
        fprintf(stderr, "Build first: cmake --build build\n");
        return;
    }

    bool gpu_backend = false;
#ifdef GPUEMU_CUDA
    if (emu.using_gpu()) {
        gpu_backend = emu.init_gpu_backend();
    }
#endif

    printf("  Backend      : %s\n", gpu_backend ? "CUDA GPU" : "CPU fallback");
    printf("  IR ops       : %zu\n", emu.ir().code.size());
    printf("  Rendering...\n");

    const auto t0 = std::chrono::steady_clock::now();
    const auto result = emu.run(max_steps);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    printf("  Finished in  : %.3f s (halted=%s)\n\n", seconds, result.halted ? "yes" : "no");

    print_mandel_framebuffer(emu, rows_per);
    printf("\n");
}

static bool setup_emulator(gpuemu::Emulator& emu, const char* obj_path, uint32_t num_vcpus,
                           uint64_t& entry_out) {
    if (!emu.load_guest(obj_path, entry_out)) {
        fprintf(stderr, "Failed to load guest .o file.\n");
        return false;
    }
    for (uint32_t i = 0; i < num_vcpus; ++i) {
        x86::CpuState init{};
        init.gpr[static_cast<int>(x86::Reg::RDI)] = i;
        emu.add_vcpu(entry_out, &init);
    }
    emu.translate(entry_out);
    if (emu.ir().code.empty()) {
        fprintf(stderr, "No guest code was translated. Check that %s is a valid x86-64 .o\n",
                obj_path);
        return false;
    }
    return true;
}

static BenchResult run_timed(gpuemu::Emulator& emu, uint32_t max_steps, bool try_gpu) {
    BenchResult out{};
#ifdef GPUEMU_CUDA
    if (try_gpu && emu.using_gpu()) {
        out.used_gpu = emu.init_gpu_backend();
    }
#else
    (void)try_gpu;
#endif

    const auto t0 = std::chrono::steady_clock::now();
    const auto result = emu.run(max_steps);
    const auto t1 = std::chrono::steady_clock::now();

    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.tally = sum_exit_codes(emu);
    out.all_halted = result.halted;
    return out;
}

static void run_guest(const char* obj_path, uint32_t num_vcpus) {
    printf("Loading compiled guest: %s\n", obj_path);
    printf("vCPUs: %u\n\n", num_vcpus);

    gpuemu::Emulator emu;
    uint64_t entry = 0;
    if (!setup_emulator(emu, obj_path, num_vcpus, entry)) {
        fprintf(stderr, "Build first: cmake --build build\n");
        return;
    }

    bool gpu_backend = false;
#ifdef GPUEMU_CUDA
    if (emu.using_gpu()) {
        gpu_backend = emu.init_gpu_backend();
    }
#endif
    printf("Backend: %s | entry=0x%llx | IR ops=%zu\n\n",
           gpu_backend ? "CUDA GPU" : "CPU fallback",
           static_cast<unsigned long long>(entry),
           emu.ir().code.size());

    auto result = emu.run(1 << 22);
    printf("\n--- %u guest(s) finished (exit_code=%ld, halted=%d) ---\n",
           num_vcpus, static_cast<long>(result.exit_code), result.halted);
}

static void run_showcase(uint32_t num_vcpus, const char* obj_path, uint32_t max_steps) {
    printf("\n");
    printf("  +---------------------------------------------------------------------+\n");
    printf("  |  GPU-NATIVE x86-64: real gcc machine code, parallel on CUDA         |\n");
    printf("  |  One CUDA thread = one full x86-64 CPU (decoded .text -> GPU IR)    |\n");
    printf("  +---------------------------------------------------------------------+\n\n");

    gpuemu::Emulator probe;
    uint64_t entry = 0;
    if (!setup_emulator(probe, obj_path, num_vcpus, entry)) {
        fprintf(stderr, "Build first: cmake --build build\n");
        return;
    }

    const size_t ir_ops = probe.ir().code.size();
    printf("  Guest        : %s\n", obj_path);
    printf("  Entry        : 0x%llx\n", static_cast<unsigned long long>(entry));
    printf("  Translation  : %zu IR ops (host decode once, broadcast to all vCPUs)\n", ir_ops);
    printf("  vCPUs        : %u independent x86-64 guests\n", num_vcpus);
    printf("  Workload     : heavy integer loop per guest; exit(tally) for verify\n");
    printf("\n");

    gpuemu::Emulator cpu_emu;
    cpu_emu.disable_gpu();
    setup_emulator(cpu_emu, obj_path, num_vcpus, entry);
    printf("  [1/2] CPU reference (sequential IR interpreter, %u guests)...\n", num_vcpus);
    const BenchResult cpu = run_timed(cpu_emu, max_steps, false);
    printf("        %.3f s   Σ exit tallies = %llu   halted=%s\n",
           cpu.seconds, static_cast<unsigned long long>(cpu.tally),
           cpu.all_halted ? "yes" : "no");
    if (num_vcpus >= 1) {
        const auto& s0 = cpu_emu.vcpu(0).state();
        printf("        vCPU 0 exit  : %lld\n", static_cast<long long>(s0.syscall_ret));
    }
    if (num_vcpus > 1) {
        const auto& sn = cpu_emu.vcpu(num_vcpus - 1).state();
        printf("        vCPU %u exit : %lld  (id×ITER + checksum; unique per guest)\n",
               num_vcpus - 1, static_cast<long long>(sn.syscall_ret));
    }
    if (num_vcpus <= 8) {
        printf("        per-vCPU     :");
        for (uint32_t i = 0; i < num_vcpus; ++i) {
            printf(" [%u]=%lld", i,
                   static_cast<long long>(cpu_emu.vcpu(i).state().syscall_ret));
        }
        printf("\n");
    }

#ifdef GPUEMU_CUDA
    gpuemu::Emulator gpu_emu;
    setup_emulator(gpu_emu, obj_path, num_vcpus, entry);
    if (gpu_emu.using_gpu()) {
        printf("  GPU device   : %s\n", gpuemu_cuda_device_name());
        printf("  [2/2] GPU parallel (one CUDA thread per x86 guest)...\n");
        const BenchResult gpu = run_timed(gpu_emu, max_steps, true);
        printf("        %.3f s   Σ exit tallies = %llu   halted=%s\n\n",
               gpu.seconds, static_cast<unsigned long long>(gpu.tally),
               gpu.all_halted ? "yes" : "no");

        if (cpu.tally != gpu.tally) {
            printf("  *** VERIFY FAILED: CPU tally %llu != GPU tally %llu ***\n",
                   static_cast<unsigned long long>(cpu.tally),
                   static_cast<unsigned long long>(gpu.tally));
        } else {
            printf("  Verify       : PASS (identical results on CPU and GPU)\n");
        }

        if (gpu.seconds > 0.0 && cpu.seconds > 0.0) {
            const double speedup = cpu.seconds / gpu.seconds;
            const double ir_est = static_cast<double>(ir_ops) * static_cast<double>(max_steps);
            const double ir_per_s = ir_est * static_cast<double>(num_vcpus) / gpu.seconds;
            printf("  Speedup      : %.1fx  (CPU wall / GPU wall)\n", speedup);
            printf("  IR throughput: ~%.2e ops/s across %u vCPUs (upper-bound est.)\n",
                   ir_per_s, num_vcpus);
        }
    } else {
        printf("  [2/2] GPU skipped (no CUDA device)\n");
    }
#else
    printf("  [2/2] GPU skipped (build without CUDA)\n");
#endif

    printf("\n  What you just saw: %u separate gcc-compiled x86-64 programs, same .text,\n",
           num_vcpus);
    printf("  each with its own registers and memory — launched like a CUDA kernel.\n\n");
}

enum class GuestKind { Generic, Mandel, Bench };

struct GuestEntry {
    const char* name;
    const char* (*path_fn)();
    GuestKind kind;
    const char* description;
};

static const GuestEntry kGuests[] = {
    {"simple", default_simple_obj, GuestKind::Generic,
     "Hello world + exit(42)"},
    {"parallel", default_parallel_obj, GuestKind::Generic,
     "One write syscall per vCPU; use -p N for parallel launch"},
    {"println", default_guest_obj, GuestKind::Generic,
     "Arithmetic demo with println() helpers"},
    {"demo", default_demo_obj, GuestKind::Generic,
     "Same as println with sys_write() wrapper"},
    {"mandel", default_mandel_obj, GuestKind::Mandel,
     "Parallel ASCII Mandelbrot (row bands per vCPU)"},
    {"gpubench", default_bench_obj, GuestKind::Bench,
     "CPU vs GPU integer loop benchmark"},
};

static const GuestEntry* find_guest(const char* name) {
    for (const auto& g : kGuests) {
        if (std::strcmp(g.name, name) == 0) {
            return &g;
        }
    }
    return nullptr;
}

static void list_guests() {
    printf("Built-in C/asm guests (compile with: cmake --build build)\n\n");
    printf("  Name       Mode      Description\n");
    printf("  --------   -------   -----------\n");
    for (const auto& g : kGuests) {
        const char* mode = "generic";
        if (g.kind == GuestKind::Mandel) {
            mode = "mandel";
        } else if (g.kind == GuestKind::Bench) {
            mode = "bench";
        }
        printf("  %-10s %-9s %s\n", g.name, mode, g.description);
    }
    printf("\nRun examples:\n");
    printf("  ./build/gpuemu --guest parallel -p 4\n");
    printf("  ./build/gpuemu --guest simple\n");
    printf("  ./build/gpuemu --mandel -p 8\n");
    printf("  ./build/gpuemu --bench -p 1024\n");
    printf("  ./build/gpuemu build/guest/parallel.o -p 4   # explicit .o path\n");
}

static void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s [options] [path/to/guest.o]\n\n", prog);
    printf("Options:\n");
    printf("  --guest NAME         Run a built-in guest (see --list-guests)\n");
    printf("  --list-guests        List built-in guests and example commands\n");
    printf("  --bench              Run CPU vs GPU showcase (guest: gpubench)\n");
    printf("  --mandel             Parallel ASCII Mandelbrot (guest: mandel)\n");
    printf("  -p N, --parallel N   Number of vCPUs (default: 1; bench: 512; mandel: 8)\n");
    printf("  --steps N            Max IR steps per vCPU (mandel default: 256M)\n");
    printf("  -h, --help           Show this help\n\n");
    printf("Examples:\n");
    printf("  %s --guest parallel -p 4\n", prog);
    printf("  %s --guest println\n", prog);
    printf("  %s --mandel -p 8\n", prog);
    printf("  %s --bench -p 1024\n\n", prog);
    printf("Default guest: build/guest/println.o\n");
}

int main(int argc, char** argv) {
    std::string obj = default_guest_obj();
    uint32_t num_vcpus = 1;
    uint32_t max_steps = 64 << 20;
    bool bench_mode = false;
    bool mandel_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--list-guests") == 0) {
            list_guests();
            return 0;
        }
        if (std::strcmp(argv[i], "--guest") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing name for --guest (try --list-guests)\n");
                return 1;
            }
            const char* name = argv[++i];
            const GuestEntry* guest = find_guest(name);
            if (!guest) {
                fprintf(stderr, "Unknown guest '%s'. Run with --list-guests.\n", name);
                return 1;
            }
            obj = guest->path_fn();
            if (guest->kind == GuestKind::Mandel) {
                mandel_mode = true;
                if (num_vcpus == 1) {
                    num_vcpus = 8;
                }
                if (max_steps == (64u << 20)) {
                    max_steps = 256u << 20;
                }
            } else if (guest->kind == GuestKind::Bench) {
                bench_mode = true;
                if (num_vcpus == 1) {
                    num_vcpus = 512;
                }
            }
            continue;
        }
        if (std::strcmp(argv[i], "--bench") == 0) {
            bench_mode = true;
            obj = default_bench_obj();
            if (num_vcpus == 1) {
                num_vcpus = 512;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--mandel") == 0) {
            mandel_mode = true;
            obj = default_mandel_obj();
            if (num_vcpus == 1) {
                num_vcpus = 8;
            }
            if (max_steps == (64u << 20)) {
                max_steps = 256u << 20;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--steps") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for --steps\n");
                return 1;
            }
            max_steps = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--parallel") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", argv[i]);
                return 1;
            }
            num_vcpus = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            if (num_vcpus == 0) {
                num_vcpus = 1;
            }
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
        if (mandel_mode || bench_mode) {
            fprintf(stderr,
                    "Warning: ignoring guest path '%s' (--mandel/--bench supply their own guest)\n",
                    argv[i]);
            continue;
        }
        obj = argv[i];
    }

    printf("GPU x86-64 Emulator — Compiled C Guest\n");
    printf("======================================\n");

    if (bench_mode) {
        run_showcase(num_vcpus, obj.c_str(), max_steps);
    } else if (mandel_mode) {
        run_mandel(obj.c_str(), num_vcpus, max_steps);
    } else {
        run_guest(obj.c_str(), num_vcpus);
    }
    return 0;
}
