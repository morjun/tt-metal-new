// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Research bindings for tt::tt_metal::experimental resident DRISC kernels (resident_drisc.hpp):
// a DRAM-core kernel launched outside the command queue that runs alongside traced Tensix programs.

#include "resident_drisc_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/resident_drisc.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::operations::experimental {

namespace {
// The kernel type is opaque in the tt_metal API; Python holds it through this handle.
struct ResidentDriscHandle {
    std::shared_ptr<tt::tt_metal::experimental::ResidentDriscKernel> kernel;
};
}  // namespace

void bind_resident_drisc(nb::module_& mod) {
    namespace ex = tt::tt_metal::experimental;
    using tt::tt_metal::distributed::MeshDevice;

    nb::class_<ResidentDriscHandle>(mod, "ResidentDriscKernel");

    mod.def(
        "drisc_senders",
        [](MeshDevice* mesh_device) {
            std::vector<std::tuple<uint32_t, CoreCoord, CoreCoord>> out;
            for (const auto& s : ex::GetDriscSenders(*mesh_device)) {
                out.emplace_back(s.bank_id, s.logical_core, s.virtual_core);
            }
            return out;
        },
        nb::arg("mesh_device"),
        "(bank, logical DRAM core, virtual NoC core) of the two DRISC senders of every DRAM bank.");

    mod.def(
        "drisc_l1_unreserved",
        [](MeshDevice* mesh_device) {
            return std::make_tuple(
                ex::GetDriscL1UnreservedBase(*mesh_device), ex::GetDriscL1UnreservedSize(*mesh_device));
        },
        nb::arg("mesh_device"),
        "(base, size) of the DRISC L1 a kernel may use.");

    mod.def(
        "launch_resident_drisc_kernel",
        [](MeshDevice* mesh_device,
           const std::string& kernel_path,
           const std::vector<CoreCoord>& cores,
           const std::vector<uint32_t>& compile_args,
           const std::vector<std::vector<uint32_t>>& runtime_args,
           const std::map<std::string, std::string>& defines) {
            return ResidentDriscHandle{
                ex::LaunchResidentDriscKernel(*mesh_device, kernel_path, cores, compile_args, runtime_args, defines)};
        },
        nb::arg("mesh_device"),
        nb::arg("kernel_path"),
        nb::arg("cores"),
        nb::arg("compile_args"),
        nb::arg("runtime_args"),
        nb::arg("defines") = std::map<std::string, std::string>{},
        "Launch a kernel on logical DRAM cores (NOC0) through slow dispatch and return while it runs.");

    mod.def(
        "read_drisc_l1",
        [](MeshDevice* mesh_device, const CoreCoord& logical_core, uint32_t addr, uint32_t num_words) {
            return ex::ReadDriscL1(*mesh_device, logical_core, addr, num_words);
        },
        nb::arg("mesh_device"),
        nb::arg("logical_core"),
        nb::arg("addr"),
        nb::arg("num_words"),
        "Read words of a DRISC's L1 at a private address (debugging).");

    mod.def(
        "wait_resident_drisc_kernel",
        [](ResidentDriscHandle& handle) { ex::WaitResidentDriscKernel(*handle.kernel); },
        nb::arg("kernel"),
        "Block until every core of a resident DRISC kernel has finished.");
}

}  // namespace ttnn::operations::experimental
