// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Research API: a long-running kernel on the DRAM cores' RISC-Vs (DRISC), launched outside the
// command queue so it runs alongside fast-dispatch (and traced) Tensix programs. Same launch path
// as the tensor prefetcher (TensorPrefetcherManager::start): compile, write runtime args and send
// the go message through slow dispatch, without waiting for the cores to finish. The kernel must
// end on its own (e.g. on a signal from a Tensix kernel); WaitResidentDriscKernel then blocks
// until it has, and must run before the device closes.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

struct ResidentDriscKernel;

// One DRISC that may run a kernel: its DRAM bank, its logical DRAM core (as CreateKernel takes it)
// and its virtual NoC coordinates (as Tensix kernels address it).
struct DriscSender {
    uint32_t bank_id;
    CoreCoord logical_core;
    CoreCoord virtual_core;
};

// The two DRISC senders of every DRAM bank that the tensor prefetcher uses: the bank's free
// subchannel and its NOC1 worker endpoint (the third, the NOC0 worker endpoint, is owned by the
// syseng firmware and runs no DRISC firmware). Kernels on them run on NOC0, so Tensix DRAM traffic,
// which enters these endpoints on NOC1 only, keeps its NOC2AXI path when a kernel switches its
// NOC0 NIU to stream mode.
std::vector<DriscSender> GetDriscSenders(distributed::MeshDevice& mesh_device);

// DRISC L1 available to a kernel: [base, base + size).
uint32_t GetDriscL1UnreservedBase(distributed::MeshDevice& mesh_device);
uint32_t GetDriscL1UnreservedSize(distributed::MeshDevice& mesh_device);

// Launch `kernel_path` on `cores` (logical DRAM cores, NOC0) of a single-device mesh, with
// per-core runtime args (runtime_args[i] for cores[i]). Returns without waiting for the kernel.
std::shared_ptr<ResidentDriscKernel> LaunchResidentDriscKernel(
    distributed::MeshDevice& mesh_device,
    const std::string& kernel_path,
    const std::vector<CoreCoord>& cores,
    const std::vector<uint32_t>& compile_args,
    const std::vector<std::vector<uint32_t>>& runtime_args,
    const std::map<std::string, std::string>& defines);

// Read `num_words` words of a DRISC's L1 at its private address `addr` (host read; for debugging).
std::vector<uint32_t> ReadDriscL1(
    distributed::MeshDevice& mesh_device, const CoreCoord& logical_core, uint32_t addr, uint32_t num_words);

// Block until every core of `kernel` has finished.
void WaitResidentDriscKernel(ResidentDriscKernel& kernel);

}  // namespace experimental
}  // namespace tt::tt_metal
