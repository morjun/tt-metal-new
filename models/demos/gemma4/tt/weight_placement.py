# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where a Gemma4 weight tensor lives: DRAM (default) or L1/SRAM.

Every Gemma4 weight is loaded with a bare ``ttnn.as_tensor(...,
memory_config=ttnn.DRAM_MEMORY_CONFIG)`` and every weight matmul is a bare
``ttnn.linear(x, w)`` with no matmul program config. That combination means L1
residency for a weight is a *memory-config swap* — no height-sharding, no split
matmul, no concat. This module is the policy object that decides the swap.

Motivation: the it-assistant drafter is small enough that its transformer
weights fit in L1 (~14 MB per device at tp=2 for E2B), so the question "does
pinning draft weights in SRAM speed up drafting?" can be answered directly.
The output embedding does not fit (128 MiB per device, replicated) and falls
back to DRAM by budget rather than by a hardcoded exclusion, so the accounting
in :meth:`WeightPlacement.report` doubles as the capacity measurement.

Default is ``mode="dram"``, and every consumer defaults its ``weight_placement``
kwarg to ``None``, so the target model and existing tests are unaffected.

Sizing caveat: the budget is an explicit knob, not something derived from
``ttnn.get_max_worker_l1_unreserved_size()``. The hardware number (~1.46 MB per
core) is the size of the L1 *bank*, but statically-allocated circular buffers
and L1 buffers share one bank address space, and the CB high-water mark reaches
~1.25 MB per core on the SDPA decode cores. The real ceiling is therefore the
leftover, which is program-dependent. We log the hardware number for reference
and let the operator tune ``GEMMA4_L1_WEIGHT_BUDGET_MB`` against the loud
failure ("Statically allocated circular buffers ... clash with L1 buffers").
"""

import math
import os
import re
from dataclasses import dataclass, field

from loguru import logger

import ttnn

# Bytes per element. bfloat8_b is block-float: 1 mantissa byte per element plus
# one shared exponent byte per 16-element face row.
_BYTES_PER_ELEM = {
    ttnn.bfloat16: 2.0,
    ttnn.bfloat8_b: 1.0625,
    ttnn.bfloat4_b: 0.5625,
    ttnn.float32: 4.0,
    ttnn.uint32: 4.0,
    ttnn.int32: 4.0,
    ttnn.uint16: 2.0,
    ttnn.uint8: 1.0,
}

DEFAULT_BUDGET_MB = 32

_LAYER_RE = re.compile(r"layer_(\d+)")


def _parse_layer_spec(spec):
    """ "0,1,2" or "0-3" or "0-3,7" -> frozenset of ints; empty -> ()."""
    spec = (spec or "").strip()
    if not spec:
        return ()
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return frozenset(out)


def _parse_pin_spec(spec):
    """ "down_proj@0;gate_proj@0-3:3;up_proj@0:3:pc:y4" -> ((substring, layers, per_core_N or None, per_core,
    origin row, spread), ...).

    ``:pc`` allocates the pin per core (HYBRID allocator mode only); ``:y<n>`` shards it on the row-major prefix of
    the grid's rows n and below instead of from row 0 (gemma4-12b record §9.4); ``:s`` spreads it over every grid
    core (see ``shard_l1_width``; record §9.5)."""
    out = []
    for entry in (spec or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        head, *opts = entry.split(":")
        per_core = "pc" in opts
        spread = "s" in opts
        origin = [int(o[1:]) for o in opts if o.startswith("y")]
        pcn = [o for o in opts if o.strip() and o not in ("pc", "s") and not o.startswith("y")]
        sub, _, layers = head.partition("@")
        out.append(
            (
                sub.strip(),
                _parse_layer_spec(layers),
                int(pcn[0]) if pcn else None,
                per_core,
                origin[0] if origin else 0,
                spread,
            )
        )
    return tuple(out)


def _pick_grid(n_tiles, max_x=8, max_y=8):
    """Largest core rectangle whose core count divides n_tiles.

    MUST match DecodeMatmulTuner's grid choice — the matmul validator requires
    per_core_N == in1 shard width in tiles, so a mismatch is a hard TT_FATAL.
    """
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if n_tiles % c == 0 and c > best[0] * best[1]:
                best = (gx, gy)
    return best


def _is_replicating(mesh_mapper) -> bool:
    """True when the mapper puts a full copy of the tensor on every device.

    ``None`` means single-device or an explicit replicate; ``ShardTensor2dMesh``
    (what ``MeshConfig.column_parallel`` / ``row_parallel`` return) splits it.
    """
    if mesh_mapper is None:
        return True
    return "Replicate" in type(mesh_mapper).__name__


def per_device_bytes(shape, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None, num_devices=1) -> int:
    """Bytes this weight occupies on ONE device, after tile padding and sharding."""
    dims = list(shape)
    if layout == ttnn.TILE_LAYOUT and len(dims) >= 2:
        dims[-1] = math.ceil(dims[-1] / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        dims[-2] = math.ceil(dims[-2] / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    elems = 1
    for d in dims:
        elems *= int(d)
    if not _is_replicating(mesh_mapper) and num_devices > 1:
        elems = math.ceil(elems / num_devices)
    return int(math.ceil(elems * _BYTES_PER_ELEM.get(dtype, 2.0)))


@dataclass
class _Entry:
    name: str
    nbytes: int
    placed_l1: bool
    reason: str


@dataclass
class WeightPlacement:
    """Decides DRAM vs L1 per weight tensor, with a per-device byte budget.

    Args:
        mode: ``"dram"`` (default, no behavior change) or ``"l1"``.
        budget_bytes: per-device L1 budget for weights. Defaults to
            ``GEMMA4_L1_WEIGHT_BUDGET_MB`` (default 32 MB).
        label: shows up in the log lines, so a DRAM arm and an L1 arm built in
            the same process stay distinguishable.
    """

    #: ``"dram"`` | ``"l1"`` (interleaved) | ``"l1_sharded"``.
    #: MEASURED: interleaved L1 is worth ~0 (it stripes the weight across all 110
    #: cores, so a compute core still fetches over the NoC). ``l1_sharded`` puts
    #: each core's own N-slice in its own L1 and is worth 42-85% per matmul at the
    #: target's shapes. It REQUIRES the matching explicit program config, so it
    #: only pays off with DecodeMatmulTuner enabled.
    mode: str = "dram"
    budget_bytes: int = None
    label: str = "weights"
    #: Substring allow-list. When non-empty, only weights whose name contains one
    #: of these substrings are eligible for L1. Doubles as the tiering knob (pin
    #: just the MLPs, say) and as the bisect tool when an op misbehaves on an
    #: L1 operand. Env: GEMMA4_L1_ONLY="gate_proj,up_proj".
    only: tuple = ()
    #: Layer allow-list for selective pinning. Empty = every layer is eligible.
    #: Names carry ``layer_<i>/``, so this filters on the parsed index.
    #: Env: GEMMA4_L1_LAYERS="0,1,2" or "0-3".
    layers: tuple = ()
    #: Per-weight pin list, superseding ``only`` / ``layers`` when set. Entries are
    #: ``<substring>[@<layers>][:<per_core_N>]`` separated by ``;``, e.g.
    #: ``down_proj@0;gate_proj@0:3;post_projection:2``. ``per_core_N`` shards the weight on
    #: the row-major prefix of ceil(Nt / per_core_N) cores of the full device grid instead
    #: of the tuner's grid (gemma4-12b record §9.2). A trailing ``:pc`` allocates that pin per core,
    #: which needs the HYBRID allocator mode (record §9.4); ``:s`` spreads it over every grid core (record §9.5).
    #: Env: GEMMA4_L1_PIN.
    pins: tuple = ()
    #: Defer all sharding to an explicit post-load pass. Required for the TARGET,
    #: whose weights also serve prefill: a decode-shaped shard makes prefill's
    #: auto-selected config FATAL, so the reshard must wait until prefill is done
    #: (Gemma4Model.pin_l1_weights). The drafter has no prefill and leaves this off.
    defer: bool = False
    used_bytes: int = 0
    entries: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in ("dram", "l1", "l1_sharded"):
            raise ValueError(f"WeightPlacement.mode must be dram|l1|l1_sharded, got {self.mode!r}")
        if self.budget_bytes is None:
            self.budget_bytes = int(float(os.getenv("GEMMA4_L1_WEIGHT_BUDGET_MB", DEFAULT_BUDGET_MB)) * (1 << 20))
        if not self.only:
            raw = os.getenv("GEMMA4_L1_ONLY", "").strip()
            self.only = tuple(s.strip() for s in raw.split(",") if s.strip()) if raw else ()
        if not self.layers:
            self.layers = _parse_layer_spec(os.getenv("GEMMA4_L1_LAYERS", ""))
        if not self.pins:
            self.pins = _parse_pin_spec(os.getenv("GEMMA4_L1_PIN", ""))
        self._pcn = {}
        self._per_core = set()
        self._origin = {}
        self._spread = set()

    @classmethod
    def from_env(cls, default_mode="dram", **kwargs):
        """Build from ``GEMMA4_WEIGHTS_IN_L1``, scoped by ``GEMMA4_L1_SCOPES``.

        ``sharded`` (recommended) -> WIDTH_SHARDED on the matmul grid;
        ``1``/``l1`` -> interleaved (measured worthless, kept for A/B);
        ``0``/unset -> DRAM.

        **Scoping, and why it defaults to the drafter only.** Both the drafter
        (``label="draft"``) and the 35-layer target (``label="target"``) call this,
        so a bare ``GEMMA4_WEIGHTS_IN_L1=sharded`` used to pin BOTH — and pinning
        the target is structurally broken, not merely unprofitable: its weights
        are consumed by prefill as well as decode, the tuned config only applies at
        ``Mt == 1``, and at prefill ttnn auto-selects a config that does not know
        about the decode-shaped shard. The result is a hard
        ``input B shard width in tiles (2) must equal per_core_N (1)`` during the
        first decode after prefill, which made the knob unusable end to end even
        though the drafter side works. So ``GEMMA4_L1_SCOPES`` defaults to
        ``draft``; set it to ``draft,target`` or ``all`` to opt the target back in.
        Same shape as ``DecodeMatmulTuner.from_env``'s scope list.
        """
        raw = (os.getenv("GEMMA4_WEIGHTS_IN_L1") or "").strip().lower()
        mode = default_mode
        if raw in ("sharded", "l1_sharded", "shard"):
            mode = "l1_sharded"
        elif raw in ("1", "true", "yes", "on", "l1"):
            mode = "l1"
        elif raw in ("0", "false", "no", "off"):
            mode = "dram"
        scope = kwargs.get("label")
        if mode != "dram" and scope:
            allowed = (os.getenv("GEMMA4_L1_SCOPES") or "draft").strip().lower()
            if allowed not in ("all", "1", "true", "yes", "on"):
                if scope not in {t.strip() for t in allowed.split(",")}:
                    logger.info(f"[placement:{scope}] GEMMA4_L1_SCOPES={allowed!r} excludes this scope — staying DRAM")
                    mode = default_mode
        return cls(mode=mode, **kwargs)

    @property
    def enabled(self) -> bool:
        return self.mode in ("l1", "l1_sharded")

    @property
    def sharded(self) -> bool:
        return self.mode == "l1_sharded"

    def _pin_for(self, name):
        """The first GEMMA4_L1_PIN entry matching this weight name, or None."""
        m = _LAYER_RE.search(name or "")
        for pin in self.pins:
            sub, layers = pin[0], pin[1]
            if sub in name and (not layers or (m is not None and int(m.group(1)) in layers)):
                return pin
        return None

    def per_core_n(self, name):
        """per_core_N override for a pinned weight (GEMMA4_L1_PIN), or None for the tuner's grid."""
        return self._pcn.get(name)

    def per_core(self, name):
        """Whether a pinned weight is allocated per core (GEMMA4_L1_PIN ``:pc``)."""
        return name in self._per_core

    def origin_row(self, name):
        """First grid row of a pinned weight's shard prefix (GEMMA4_L1_PIN ``:y<n>``), 0 by default."""
        return self._origin.get(name, 0)

    def spread(self, name):
        """Whether a pinned weight is spread over every grid core (GEMMA4_L1_PIN ``:s``)."""
        return name in self._spread

    def _layer_allowed(self, name):
        if not self.layers:
            return True
        m = _LAYER_RE.search(name or "")
        # Weights outside any layer (projections, CME tables) stay eligible.
        return True if m is None else int(m.group(1)) in self.layers

    def log_hardware_ceiling(self, mesh_device):
        """Log the raw L1 bank capacity. Reference only — see the module docstring."""
        try:
            per_core = ttnn.get_max_worker_l1_unreserved_size()
            grid = mesh_device.compute_with_storage_grid_size()
            cores = grid.x * grid.y
        except Exception as e:  # noqa: BLE001 - diagnostics only, never fatal
            logger.info(f"[placement:{self.label}] could not query L1 capacity: {e}")
            return
        logger.info(
            f"[placement:{self.label}] mode={self.mode} budget={self.budget_bytes/(1<<20):.1f} MB/device; "
            f"hardware L1 = {per_core} B/core x {cores} cores ({grid.x}x{grid.y}) = "
            f"{per_core*cores/(1<<20):.1f} MB/device RAW (CBs share this space; budget is the real limit)"
        )

    def memory_config(
        self, name, shape=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=None, mesh_device=None
    ):
        """Memory config for one weight, recording the decision.

        Returns ``ttnn.DRAM_MEMORY_CONFIG`` unless L1 mode is on AND the tensor
        fits in the remaining per-device budget.
        """
        if not self.enabled:
            return ttnn.DRAM_MEMORY_CONFIG
        if self.sharded and self.defer:
            # Budget is accounted for by the post-prefill pass, not here.
            return ttnn.DRAM_MEMORY_CONFIG
        if shape is None:
            # Dummy-weight paths pass a None tensor; nothing to account for.
            self.entries.append(_Entry(name, 0, False, "unknown shape"))
            return ttnn.DRAM_MEMORY_CONFIG
        if self.pins:
            pin = self._pin_for(name)
            if pin is None:
                self.entries.append(_Entry(name, 0, False, "not in GEMMA4_L1_PIN"))
                return ttnn.DRAM_MEMORY_CONFIG
            if pin[2] is not None:
                self._pcn[name] = pin[2]
            if pin[3]:
                self._per_core.add(name)
            if pin[4]:
                self._origin[name] = pin[4]
            if pin[5]:
                self._spread.add(name)
        elif self.only and not any(s in name for s in self.only):
            self.entries.append(_Entry(name, 0, False, "not in GEMMA4_L1_ONLY"))
            return ttnn.DRAM_MEMORY_CONFIG
        if not self._layer_allowed(name):
            self.entries.append(_Entry(name, 0, False, "layer not in GEMMA4_L1_LAYERS"))
            return ttnn.DRAM_MEMORY_CONFIG

        num_devices = 1
        if mesh_device is not None and hasattr(mesh_device, "get_num_devices"):
            num_devices = mesh_device.get_num_devices()
        nbytes = per_device_bytes(shape, dtype, layout, mesh_mapper, num_devices)

        if self.used_bytes + nbytes <= self.budget_bytes:
            self.used_bytes += nbytes
            self.entries.append(_Entry(name, nbytes, True, "fits"))
            # l1_sharded needs the PER-DEVICE shape to build its shard spec, which
            # is only knowable after the mesh mapper has run. Load to DRAM here and
            # let place_as_tensor reshard; it records the outcome either way.
            return ttnn.DRAM_MEMORY_CONFIG if self.sharded else ttnn.L1_MEMORY_CONFIG

        self.entries.append(
            _Entry(
                name,
                nbytes,
                False,
                f"over budget ({(self.used_bytes+nbytes)/(1<<20):.1f} MB > " f"{self.budget_bytes/(1<<20):.1f} MB)",
            )
        )
        return ttnn.DRAM_MEMORY_CONFIG

    # ── reporting ────────────────────────────────────────────────────────────
    def summary(self):
        l1 = [e for e in self.entries if e.placed_l1]
        dram = [e for e in self.entries if not e.placed_l1]
        return {
            "mode": self.mode,
            "budget_bytes": self.budget_bytes,
            "l1_bytes": sum(e.nbytes for e in l1),
            "l1_count": len(l1),
            "dram_bytes": sum(e.nbytes for e in dram),
            "dram_count": len(dram),
        }

    def report(self, top_n=12) -> str:
        s = self.summary()
        lines = [
            f"[placement:{self.label}] mode={s['mode']} "
            f"L1={s['l1_bytes']/(1<<20):.2f} MB/device over {s['l1_count']} tensors, "
            f"DRAM={s['dram_bytes']/(1<<20):.2f} MB/device over {s['dram_count']} tensors, "
            f"budget={s['budget_bytes']/(1<<20):.1f} MB"
        ]
        rejected = [e for e in self.entries if not e.placed_l1 and e.nbytes > 0]
        if rejected:
            lines.append(f"[placement:{self.label}] fell back to DRAM:")
            for e in sorted(rejected, key=lambda e: -e.nbytes)[:top_n]:
                lines.append(f"    {e.name:<48} {e.nbytes/(1<<20):8.2f} MB  ({e.reason})")
        biggest = sorted([e for e in self.entries if e.placed_l1], key=lambda e: -e.nbytes)[:top_n]
        if biggest:
            lines.append(f"[placement:{self.label}] largest tensors pinned to L1:")
            for e in biggest:
                lines.append(f"    {e.name:<48} {e.nbytes/(1<<20):8.2f} MB")
        return "\n".join(lines)


#: Shared no-op instance so call sites can write ``(placement or DRAM_ONLY)``
#: without allocating a policy object per weight.
DRAM_ONLY = WeightPlacement(mode="dram", label="default")


def resolve(placement):
    """``None`` -> the shared DRAM-only policy."""
    return placement if placement is not None else DRAM_ONLY


def prefix_core_range_set(num_cores, width, origin_row=0):
    """The first ``num_cores`` cores, row-major, of a grid ``width`` cores wide, from (0, ``origin_row``).

    This is the core order the 1D mcast factory walks (``num_cores_with_work`` row-major
    over its grid), so a weight sharded here lines up with that factory's cores.
    """
    rows, rem = divmod(num_cores, width)
    y0 = origin_row
    ranges = []
    if rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord(width - 1, y0 + rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y0 + rows), ttnn.CoreCoord(rem - 1, y0 + rows)))
    return ttnn.CoreRangeSet(set(ranges))


def l1_core_addresses(tensor):
    """{(x, y): L1 address} over a sharded L1 tensor's shard cores; per-core allocations answer per core."""
    cores = ttnn.corerange_to_cores(tensor.memory_config().shard_spec.grid, row_wise=True)
    if tensor.is_per_core_allocated():
        return {(c.x, c.y): tensor.experimental_per_core_buffer_address(c) for c in cores}
    addr = tensor.buffer_address()
    return {(c.x, c.y): addr for c in cores}


def spread_tiles(nt, cores, per_core_n):
    """(first tile, real tiles) per core of a spread pin, row-major: the first ``nt - (per_core_n - 1) * cores``
    cores hold ``per_core_n`` tiles, the others ``per_core_n - 1`` (gemma4-12b record §9.5)."""
    full = nt - (per_core_n - 1) * cores
    if per_core_n < 2 or not 0 < full < cores:
        raise ValueError(f"spread pin: {nt} tiles do not spread over {cores} cores at {per_core_n} / {per_core_n - 1}")
    out, t = [], 0
    for i in range(cores):
        r = per_core_n if i < full else per_core_n - 1
        out.append((t, r))
        t += r
    return out


def _page_copy_program(kernel, dram, cores, pages, l1_addr):
    """One reader per core copying its ``pages`` 2 KiB pages between ``l1_addr`` and rows [i * pages, ...) of the
    ROW_MAJOR DRAM tensor ``dram`` (the lm_head reload's snapshot / reload kernels)."""
    from models.demos.gemma4.tt.lm_head_reload import KDIR, PAGE

    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [dram.buffer_address(), i * pages, pages, l1_addr]
    k = ttnn.KernelDescriptor(
        kernel_source=KDIR + kernel,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]),
        compile_time_args=[PAGE] + list(ttnn.TensorAccessorArgs(dram).get_compile_time_args()),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])


def _spread_l1_width(tensor, mesh_device, k, n, per_core_n):
    """RESEARCH (gemma4-12b record §9.5): a thin-K weight width-sharded at ``per_core_n`` tiles per core on EVERY grid
    core, for the 1D-mcast factory's TT_MM1D_IN1_SPREAD mode. Core i's shard holds its real tiles
    (``spread_tiles``) followed by zero pad tiles; the tensor keeps its logical [.., K, N] shape, so the matmul's
    output is N wide (the factory's writer drops the pad tiles). The bytes are staged through a padded tensor on the
    same shard spec, because ttnn would place the logical columns on the first ceil(Nt / per_core_n) cores only."""
    import torch

    from models.demos.gemma4.tt.lm_head_reload import PAGE

    T = ttnn.TILE_SIZE
    if os.getenv("TT_MM1D_IN1_SPREAD") is None:
        raise RuntimeError("a spread pin needs TT_MM1D_IN1_SPREAD=1, or the stock matmul reads its shards wrong")
    if mesh_device.get_num_devices() != 1 or tensor.dtype != ttnn.bfloat16:
        raise ValueError("spread pins are single-device bf16 only")
    grid_size = mesh_device.compute_with_storage_grid_size()
    cores = grid_size.x * grid_size.y
    layout = spread_tiles(n // T, cores, per_core_n)
    lead = list(tensor.shape)[:-2]
    host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(k, n)
    w = per_core_n * T
    padded = torch.zeros(k, cores * w, dtype=host.dtype)
    for i, (t, r) in enumerate(layout):
        padded[:, i * w : i * w + r * T] = host[:, t * T : (t + r) * T]
    grid = prefix_core_range_set(cores, grid_size.x)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [k, w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    core_list = ttnn.corerange_to_cores(grid, row_wise=True)
    pages = (k // T) * per_core_n * ttnn.TILE_SIZE * ttnn.TILE_SIZE * 2 // PAGE
    stage = ttnn.from_torch(
        padded.reshape(lead + [k, cores * w]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mc,
    )
    snap = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, cores * pages, PAGE // 2]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.generic_op(
        [stage, snap], _page_copy_program("dram_snapshot_pages.cpp", snap, core_list, pages, stage.buffer_address())
    )
    stage.deallocate(True)
    moved = ttnn.allocate_tensor_on_device(tensor.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, mc)
    ttnn.generic_op(
        [snap, moved], _page_copy_program("dram_reload_pages.cpp", snap, core_list, pages, moved.buffer_address())
    )
    snap.deallocate(True)
    full = sum(r == per_core_n for _, r in layout)
    logger.info(
        f"[placement] spread pin {list(tensor.shape)} at {moved.buffer_address()} on {cores} cores: "
        f"{full} x {per_core_n} + {cores - full} x {per_core_n - 1} real tiles"
    )
    return moved


def shard_l1_width(tensor, mesh_device, per_core_n=None, per_core=False, origin_row=0, spread=False):
    """Reshard a loaded weight to L1 WIDTH_SHARDED on the matmul's compute grid.

    Each core ends up holding exactly the N-slice it multiplies, which is what
    makes L1 residency worth 42-85% per matmul (interleaved L1 buys ~0).

    The grid comes from ``derive_decode_1d_config`` — the SAME function the matmul
    tuner uses — because the validator requires
    ``per_core_N == in1_shard_width_tiles`` and a duplicated grid heuristic drifts
    out of sync (it did: "shard width in tiles (2) must equal per_core_N (1)").

    Returns the resharded tensor, or None when the weight does not qualify
    (ROW_MAJOR norms, shapes with no valid grid).

    ``per_core`` (with ``per_core_n``; HYBRID allocator mode only, gemma4-12b record §9.4) allocates the shard per
    core. Stock ops bind a sharded buffer at one address, so the shard must land at the same address on all its
    cores; that is asserted. ``origin_row`` (with ``per_core_n``) starts the prefix at that grid row; the tuner then
    runs the matmul on that offset rectangle (``allowed_worker_cores``).
    """
    from models.demos.gemma4.tt.matmul_tuning import derive_decode_1d_config

    T = ttnn.TILE_SIZE
    if tensor.layout != ttnn.TILE_LAYOUT:
        return None  # norms / embedding tables are not matmul in1
    try:
        local = ttnn.get_device_tensors(tensor)[0].shape if mesh_device.get_num_devices() > 1 else tensor.shape
    except Exception:  # noqa: BLE001
        local = tensor.shape
    if len(local) < 2:
        return None
    k, n = int(local[-2]), int(local[-1])

    if spread:
        if per_core_n is None or per_core or origin_row:
            raise ValueError("a spread pin needs per_core_N, LOCKSTEP and origin row 0 (GEMMA4_L1_PIN <name>:<n>:s)")
        if k % ttnn.TILE_SIZE or n % ttnn.TILE_SIZE:
            return None
        return _spread_l1_width(tensor, mesh_device, k, n, per_core_n)
    if per_core_n is not None:
        # GEMMA4_L1_PIN per_core_N: the row-major prefix of the full device grid; the last
        # shard may be partial. DecodeMatmulTuner reads the grid back from the shard spec.
        if k % T or n % T:
            return None
        grid_size = mesh_device.compute_with_storage_grid_size()
        cores = -(-(n // T) // per_core_n)
        if cores > grid_size.x * (grid_size.y - origin_row):
            return None
        grid = prefix_core_range_set(cores, grid_size.x, origin_row)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [k, per_core_n * T], ttnn.ShardOrientation.ROW_MAJOR),
        )
        if per_core:
            mc.experimental_set_per_core_allocation(True)
        moved = ttnn.to_memory_config(tensor, mc)
        if per_core:
            addrs = set(l1_core_addresses(moved).values())
            if not moved.is_per_core_allocated() or len(addrs) != 1:
                raise RuntimeError(
                    f"per-core pin {list(local)}: per-core={moved.is_per_core_allocated()}, addresses {sorted(addrs)}"
                )
            logger.info(f"[placement] per-core pin {list(local)} at {addrs.pop()} on {cores} cores")
        return moved

    if per_core or origin_row:
        raise ValueError(
            "a per-core or offset pin needs an explicit per_core_N (GEMMA4_L1_PIN <name>:<per_core_N>:...)"
        )
    pc = derive_decode_1d_config(1, k, n)
    if pc is None:
        return None
    gx, gy = pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y
    cores = gx * gy
    shard_w = n // cores
    if shard_w // T != pc.per_core_N:  # would FATAL in the matmul validator
        return None
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [k, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.to_memory_config(tensor, mc)


def _dram_shard_names():
    """GEMMA4_DRAM_SHARD: comma list of weight-name substrings to store DRAM WIDTH_SHARDED (not the target)."""
    raw = os.getenv("GEMMA4_DRAM_SHARD", "").strip()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def dram_shard_width(tensor, device):
    """RESEARCH (gemma4-12b record §9.2a, DRAM-sharded fat-K): re-lay a DRAM-interleaved [.., K, N] matmul weight as
    DRAM WIDTH_SHARDED, one contiguous N / banks slice per DRAM bank, for
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig (DecodeMatmulTuner.linear). Returns None if N does not
    split into whole tiles per bank."""
    k, n = int(tensor.shape[-2]), int(tensor.shape[-1])
    banks = device.dram_grid_size().x
    if n % (ttnn.TILE_SIZE * banks):
        return None
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [k, n // banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.to_memory_config(tensor, mc)


def place_as_tensor(placement, name, torch_tensor, *, device, **kwargs):
    """``_place_as_tensor``, then (research knob GEMMA4_DRAM_SHARD; every placement but the target's) a
    DRAM-interleaved weight whose name matches is re-laid DRAM WIDTH_SHARDED."""
    tensor = _place_as_tensor(placement, name, torch_tensor, device=device, **kwargs)
    names = _dram_shard_names()
    if (
        names
        and device is not None
        and tensor is not None
        and getattr(resolve(placement), "label", "") != "target"
        and any(s in name for s in names)
        and tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        and tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        moved = dram_shard_width(tensor, device)
        if moved is not None:
            logger.info(f"[placement:{resolve(placement).label}] DRAM-sharded {name} {list(tensor.shape)}")
            tensor.deallocate(True)
            return moved
    return tensor


def _place_as_tensor(
    placement,
    name,
    torch_tensor,
    *,
    device,
    dtype,
    layout,
    mesh_mapper=None,
    cache_file_name=None,
    shape=None,
):
    """``ttnn.as_tensor`` that actually honours the placement decision.

    This wrapper exists because of a sharp edge in ``ttnn.as_tensor``: on a
    tensor-cache **hit** it returns
    ``ttnn._ttnn.tensor.load_tensor_flatbuffer(cache_file_name, device=device)``
    (``ttnn/ttnn/operations/core.py``) and **never applies memory_config**. Only
    the cache-miss path calls ``tensor.to(device, memory_config)``. So the first
    run of an L1 arm would place weights in L1 and every subsequent run — with
    the cache warm — would silently put them back in DRAM, turning the A/B into
    a no-op that still looks like it worked.

    We therefore check the buffer type we actually got and relocate if needed.
    """
    placement = resolve(placement)
    if shape is None and torch_tensor is not None:
        shape = torch_tensor.shape
    mem = placement.memory_config(
        name, shape=shape, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper, mesh_device=device
    )
    tensor = ttnn.as_tensor(
        torch_tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mesh_mapper,
        cache_file_name=cache_file_name,
        memory_config=mem,
    )
    if placement.sharded and placement.defer:
        return tensor
    if placement.sharded and mem.buffer_type == ttnn.BufferType.DRAM:
        # memory_config() returns DRAM for sharded mode; a recorded L1 entry means
        # "pin this one". Anything else genuinely stays in DRAM.
        wanted = placement.entries and placement.entries[-1].name == name and placement.entries[-1].placed_l1
        if not wanted:
            return tensor
        try:
            moved = shard_l1_width(
                tensor,
                device,
                per_core_n=placement.per_core_n(name),
                per_core=placement.per_core(name),
                origin_row=placement.origin_row(name),
                spread=placement.spread(name),
            )
            if moved is None:
                # Normal for norms / non-matmul weights: give the budget back and
                # record it, but do not shout about it.
                for entry in reversed(placement.entries):
                    if entry.name == name and entry.placed_l1:
                        entry.placed_l1 = False
                        entry.reason = "not a shardable matmul weight"
                        placement.used_bytes -= entry.nbytes
                        break
                return tensor
            tensor.deallocate(True)
            return moved
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[placement:{placement.label}] could not L1-shard {name}: {str(e)[:110]}")
            for entry in reversed(placement.entries):
                if entry.name == name and entry.placed_l1:
                    entry.placed_l1 = False
                    entry.reason = f"shard failed: {type(e).__name__}"
                    placement.used_bytes -= entry.nbytes
                    break
            return tensor

    if device is None or mem.buffer_type == ttnn.BufferType.DRAM:
        return tensor
    if tensor.memory_config().buffer_type == mem.buffer_type:
        return tensor
    try:
        moved = ttnn.to_memory_config(tensor, mem)
        tensor.deallocate(True)
        return moved
    except Exception as e:  # noqa: BLE001
        # Keep the accounting honest rather than reporting a pin that never happened.
        logger.warning(f"[placement:{placement.label}] could not relocate {name} to L1: {e}")
        for entry in reversed(placement.entries):
            if entry.name == name and entry.placed_l1:
                entry.placed_l1 = False
                entry.reason = f"relocate failed: {type(e).__name__}"
                placement.used_bytes -= entry.nbytes
                break
        return tensor
