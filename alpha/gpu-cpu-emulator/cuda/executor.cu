#include "cuda/ir_interpreter.cuh"
#include <cuda_runtime.h>

namespace gpuexec {

// Parallel x86 emulation: one CUDA thread per virtual CPU.
// Each thread runs translated IR in a time-sliced loop.
__global__ void vcpu_run_kernel(x86::CpuState* vcpus, uint32_t num_vcpus,
                                 uint8_t* guest_mem, uint64_t mem_size,
                                 const ir::Instr* ir_code, uint32_t ir_len,
                                 uint32_t* ir_pcs, uint32_t steps_per_launch) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_vcpus) return;

    x86::CpuState& cpu = vcpus[id];
    uint32_t& pc = ir_pcs[id];

    if (cpu.halt) return;

    const int status = exec_ir(cpu, guest_mem, mem_size, ir_code, ir_len, pc, steps_per_launch);
    (void)status;
}

// Batch kernel: multiple vCPUs execute independent IR streams in parallel.
// ir_offsets[i] / ir_lengths[i] index per-vCPU translated code regions.
__global__ void vcpu_batch_kernel(x86::CpuState* vcpus, uint32_t num_vcpus,
                                   uint8_t* guest_mems, uint64_t mem_size,
                                   const ir::Instr* ir_pool,
                                   const uint32_t* ir_offsets,
                                   const uint32_t* ir_lengths,
                                   uint32_t* ir_pcs, uint32_t steps_per_launch) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_vcpus) return;

    x86::CpuState& cpu = vcpus[id];
    uint32_t& pc = ir_pcs[id];
    if (cpu.halt) return;

    const ir::Instr* code = ir_pool + ir_offsets[id];
    const uint32_t len = ir_lengths[id];
    uint8_t* mem = guest_mems + id * mem_size;

    exec_ir(cpu, mem, mem_size, code, len, pc, steps_per_launch);
}

struct CudaExecutor {
    x86::CpuState* d_vcpus = nullptr;
    uint8_t* d_mem = nullptr;
    ir::Instr* d_ir = nullptr;
    uint32_t* d_pcs = nullptr;
    uint64_t mem_size = 0;
    uint32_t ir_len = 0;
    uint32_t num_vcpus = 0;

    cudaError_t init(uint32_t n, uint64_t mem_sz, const ir::Instr* host_ir, uint32_t ir_count) {
        num_vcpus = n;
        mem_size = mem_sz;
        ir_len = ir_count;

        cudaMalloc(&d_vcpus, n * sizeof(x86::CpuState));
        cudaMalloc(&d_mem, n * mem_sz);
        cudaMalloc(&d_ir, ir_count * sizeof(ir::Instr));
        cudaMalloc(&d_pcs, n * sizeof(uint32_t));

        cudaMemset(d_vcpus, 0, n * sizeof(x86::CpuState));
        cudaMemset(d_mem, 0, n * mem_sz);
        cudaMemcpy(d_ir, host_ir, ir_count * sizeof(ir::Instr), cudaMemcpyHostToDevice);

        std::vector<uint32_t> zeros(n, 0);
        cudaMemcpy(d_pcs, zeros.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
        return cudaGetLastError();
    }

    cudaError_t upload_vcpu(uint32_t idx, const x86::CpuState& cpu, const uint8_t* mem) {
        cudaMemcpy(d_vcpus + idx, &cpu, sizeof(x86::CpuState), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mem + idx * mem_size, mem, mem_size, cudaMemcpyHostToDevice);
        return cudaGetLastError();
    }

    cudaError_t set_pc(uint32_t idx, uint32_t pc) {
        cudaMemcpy(d_pcs + idx, &pc, sizeof(uint32_t), cudaMemcpyHostToDevice);
        return cudaGetLastError();
    }

    cudaError_t download_vcpu(uint32_t idx, x86::CpuState& cpu, uint8_t* mem) {
        cudaMemcpy(&cpu, d_vcpus + idx, sizeof(x86::CpuState), cudaMemcpyDeviceToHost);
        cudaMemcpy(mem, d_mem + idx * mem_size, mem_size, cudaMemcpyDeviceToHost);
        return cudaGetLastError();
    }

    cudaError_t run(uint32_t steps = 4096) {
        const int threads = 256;
        const int blocks = (num_vcpus + threads - 1) / threads;
        vcpu_run_kernel<<<blocks, threads>>>(d_vcpus, num_vcpus, d_mem, mem_size,
                                              d_ir, ir_len, d_pcs, steps);
        return cudaDeviceSynchronize();
    }

    void destroy() {
        cudaFree(d_vcpus);
        cudaFree(d_mem);
        cudaFree(d_ir);
        cudaFree(d_pcs);
    }
};

}  // namespace gpuexec

// C API for host linkage
extern "C" {

struct GpuExecutorHandle {
    gpuexec::CudaExecutor exec;
};

int gpuemu_cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

GpuExecutorHandle* gpuemu_cuda_create(uint32_t num_vcpus, uint64_t mem_size,
                                       const ir::Instr* ir, uint32_t ir_len) {
    auto* h = new GpuExecutorHandle();
    if (h->exec.init(num_vcpus, mem_size, ir, ir_len) != cudaSuccess) {
        delete h;
        return nullptr;
    }
    return h;
}

int gpuemu_cuda_upload(GpuExecutorHandle* h, uint32_t idx,
                        const x86::CpuState* cpu, const uint8_t* mem) {
    return h->exec.upload_vcpu(idx, *cpu, mem) == cudaSuccess ? 0 : -1;
}

int gpuemu_cuda_set_pc(GpuExecutorHandle* h, uint32_t idx, uint32_t pc) {
    return h->exec.set_pc(idx, pc) == cudaSuccess ? 0 : -1;
}

int gpuemu_cuda_run(GpuExecutorHandle* h, uint32_t steps) {
    return h->exec.run(steps) == cudaSuccess ? 0 : -1;
}

int gpuemu_cuda_download(GpuExecutorHandle* h, uint32_t idx,
                          x86::CpuState* cpu, uint8_t* mem) {
    return h->exec.download_vcpu(idx, *cpu, mem) == cudaSuccess ? 0 : -1;
}

void gpuemu_cuda_destroy(GpuExecutorHandle* h) {
    if (h) {
        h->exec.destroy();
        delete h;
    }
}

}  // extern "C"
