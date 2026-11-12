// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/resident_drisc.hpp>

#include <set>

#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig + CreateKernel(DramConfig)
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"

namespace tt::tt_metal::experimental {

struct ResidentDriscKernel {
    IDevice* device = nullptr;
    std::unique_ptr<Program> program;
    bool running = false;
};

namespace {

IDevice* single_device(distributed::MeshDevice& mesh_device) {
    const auto devices = mesh_device.get_devices();
    TT_FATAL(devices.size() == 1, "Resident DRISC kernels support a single-device mesh, got {}", devices.size());
    return devices[0];
}

const Hal& hal_for(distributed::MeshDevice& mesh_device) {
    const auto& hal = MetalContext::instance(mesh_device.impl().get_context_id()).hal();
    TT_FATAL(hal.has_programmable_core_type(HalProgrammableCoreType::DRAM), "DRAM programmable cores not enabled");
    return hal;
}

}  // namespace

std::vector<DriscSender> GetDriscSenders(distributed::MeshDevice& mesh_device) {
    IDevice* device = single_device(mesh_device);
    hal_for(mesh_device);
    std::vector<DriscSender> senders;
    const uint32_t num_banks = static_cast<uint32_t>(device->num_dram_channels());
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        for (const CoreCoord& core : mesh_device.impl().dram_sender_logical_cores(device, bank)) {
            senders.push_back({bank, core, device->virtual_core_from_logical_core(core, CoreType::DRAM)});
        }
    }
    return senders;
}

uint32_t GetDriscL1UnreservedBase(distributed::MeshDevice& mesh_device) {
    return hal_for(mesh_device).get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
}

uint32_t GetDriscL1UnreservedSize(distributed::MeshDevice& mesh_device) {
    return hal_for(mesh_device).get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
}

std::shared_ptr<ResidentDriscKernel> LaunchResidentDriscKernel(
    distributed::MeshDevice& mesh_device,
    const std::string& kernel_path,
    const std::vector<CoreCoord>& cores,
    const std::vector<uint32_t>& compile_args,
    const std::vector<std::vector<uint32_t>>& runtime_args,
    const std::map<std::string, std::string>& defines) {
    IDevice* device = single_device(mesh_device);
    hal_for(mesh_device);
    TT_FATAL(!cores.empty() && cores.size() == runtime_args.size(), "one runtime-arg list per DRISC core");
    std::set<CoreRange> ranges;
    for (const CoreCoord& core : cores) {
        ranges.insert(CoreRange(core));
    }
    auto kernel = std::make_shared<ResidentDriscKernel>();
    kernel->device = device;
    kernel->program = std::make_unique<Program>(CreateProgram());
    const KernelHandle kid = CreateKernel(
        *kernel->program,
        kernel_path,
        CoreRangeSet(ranges),
        DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args, .defines = defines});
    for (size_t i = 0; i < cores.size(); ++i) {
        SetRuntimeArgs(*kernel->program, kid, cores[i], runtime_args[i]);
    }
    // As TensorPrefetcherManager::start: slow-dispatch launch, return while the kernel runs.
    kernel->program->impl().compile(device, /*force_slow_dispatch=*/true);
    ::tt::tt_metal::slow_dispatch::WriteRuntimeArgsToDevice(*device, *kernel->program, /*force_slow_dispatch=*/true);
    ::tt::tt_metal::slow_dispatch::LaunchProgramAsync(*device, *kernel->program, /*force_slow_dispatch=*/true);
    kernel->running = true;
    return kernel;
}

std::vector<uint32_t> ReadDriscL1(
    distributed::MeshDevice& mesh_device, const CoreCoord& logical_core, uint32_t addr, uint32_t num_words) {
    IDevice* device = single_device(mesh_device);
    const Hal& hal = hal_for(mesh_device);
    // DRISC L1 sits at a NoC offset from its private address (0x2000000000 on Blackhole).
    const uint64_t noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED) -
                              hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED) + addr;
    std::vector<uint32_t> out(num_words, 0);
    MetalContext::instance(mesh_device.impl().get_context_id())
        .get_cluster()
        .read_core(
            out.data(),
            num_words * sizeof(uint32_t),
            tt_cxy_pair(device->id(), device->virtual_core_from_logical_core(logical_core, CoreType::DRAM)),
            noc_addr);
    return out;
}

void WaitResidentDriscKernel(ResidentDriscKernel& kernel) {
    if (!kernel.running) {
        return;
    }
    ::tt::tt_metal::slow_dispatch::WaitProgramDone(*kernel.device, *kernel.program);
    kernel.running = false;
}

}  // namespace tt::tt_metal::experimental
