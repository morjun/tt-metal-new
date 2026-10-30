# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit program configs for the batch-1 decode linears.

Every Gemma4 weight matmul is a bare ``ttnn.linear(x, w)``, which lets ttnn pick
the program config. MEASURED (`tests/unit/test_matmul_weight_placement.py`): that
automatic choice is **1.1-2.4x off** at the shapes the decode path actually uses,
with the weights in DRAM where they already are:

    shape (M,K,N)          auto      explicit 1D    speedup
    32 x  256 x 1024      5.92 us      4.53 us       1.31x   wqkv / gate / up
    32 x  256 x 1536      6.66 us      5.90 us       1.13x   post_projection
    32 x  512 x  256      7.93 us      4.25 us       1.87x   o_proj
    32 x 1024 x  256     13.79 us      6.64 us       2.08x   down_proj
    32 x 3072 x  256     37.35 us     15.53 us       2.41x   pre_projection
    32 x 1536 x 1536     26.58 us     17.34 us       1.53x   target-scale
    32 x 1536 x 3072     36.99 us     24.73 us       1.50x   target-scale

Why the heuristic misses: `create_simple_matmul_program_config`
(`ttnn/cpp/.../config/matmul_program_config.cpp`) only recomputes the block and
subblock sizes from the real Kt/Mt/Nt when *every* operand is DRAM-interleaved;
otherwise it keeps `in0_block_w = 2` and a generic per-core factor. Even on the
all-DRAM path it prefers a 2D config for shapes where a 1D multicast along N is
much better at M = 1 tile row.

The config below is essentially forced by the shape — grid from Nt, `per_core_N`
from Nt/cores, `per_core_M` = Mt = 1, subblocks 1x1 because each core owns one
output tile. The only free knob is ``in0_block_w``; a sweep over its divisors
picked the largest divisor of Kt <= 8 at every drafter shape.

Default OFF (`enabled=False`), so the target model and existing tests are
unaffected until explicitly opted in.
"""

import os

from loguru import logger

import ttnn


def _env_on(name):
    """``1|true|yes|on`` -> True; unset / anything else -> False.

    A bare ``os.getenv(name)`` truth test makes ``NAME=0`` mean ENABLED, which is
    the opposite of every other knob in this model (``GEMMA4_TUNE_MATMULS``,
    ``GEMMA4_WEIGHTS_IN_L1``, ``GEMMA4_SHARD_ACTIVATIONS``). That silently
    contaminates any A/B run with the variable exported as 0.
    """
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _block_cap(shape=None):
    """Ceiling on ``in0_block_w``. Default 8; ``GEMMA4_MM_BLOCK_CAP`` overrides, and
    ``GEMMA4_MM_BLOCK_CAP_SHAPES="<K>x<N>=<cap>;..."`` overrides it for those (K, N) only
    (gemma4-12b record §9.2: DRAM read rate vs blocking per matmul shape).

    The 8 is a heuristic, not a ttnn rule — the only hard constraints are
    ``Kt % in0_block_w == 0`` and, for a width-sharded in0,
    ``in0_shard_width_tiles % in0_block_w == 0``. MEASURED isolated on
    pre_projection 3072x256 (MEASUREMENT_RECORD.md 9d): the cap costs **-31.3%**
    with a pinned weight (6 -> 12) and **-35.2%** on the all-interleaved default
    (8 -> 32). It is left at 8 because raising it is NOT free:

      * the CBs grow linearly (in0 CB = in0_block_w x 2 tiles, and the in1 CB
        matches it when the weight is interleaved), so blk=48 wants 384 KB/core
        against a ~320 KB budget and collides with weight pinning;
      * it is NOT bit-neutral — reblocking changes the K accumulation order, so
        it moves generated tokens and therefore drafter acceptance, exactly like
        the tuner itself (see ``from_env``'s caveat).

    Note the scope: with ``GEMMA4_TUNE_MATMULS`` unset the tuner runs on the
    TARGET only (``DEFAULT_SCOPES``), so changing this knob moves the 35-layer
    target by default and the drafter only when the tuner is force-enabled.
    """
    if shape is not None:
        for entry in os.getenv("GEMMA4_MM_BLOCK_CAP_SHAPES", "").split(";"):
            key, _, cap = entry.strip().partition("=")
            if key and key == f"{shape[0]}x{shape[1]}":
                return max(1, int(cap))
    try:
        return max(1, int(os.getenv("GEMMA4_MM_BLOCK_CAP", "8")))
    except ValueError:
        return 8


def _grid_override(shape):
    """per_core_N from ``GEMMA4_MM_GRID_SHAPES="<K>x<N>=<per_core_N>;..."`` for a (K, N) matmul, else None.

    Runs a weight that is NOT L1 WIDTH_SHARDED (DRAM interleaved) on the row-major prefix of
    ceil(Nt / per_core_N) cores of the full device grid instead of the tuner's <= 8x8 rectangle
    (gemma4-12b record §9.2: DRAM read rate vs reading-core count).
    """
    for entry in os.getenv("GEMMA4_MM_GRID_SHAPES", "").split(";"):
        key, _, pcn = entry.strip().partition("=")
        if key and key == f"{shape[0]}x{shape[1]}":
            return max(1, int(pcn))
    return None


def _largest_divisor(n, cap=None):
    cap = _block_cap() if cap is None else cap
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _pick_grid(n_tiles, max_x, max_y):
    """Largest core rectangle whose core count divides n_tiles."""
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if n_tiles % c == 0 and c > best[0] * best[1]:
                best = (gx, gy)
    return best


def derive_decode_1d_config(m, k, n, max_x=8, max_y=8, in0_shard_tiles=None):
    """1D multicast config for a batch-1 decode linear, or None if it doesn't apply.

    Returns None (caller keeps ttnn's automatic choice) unless the shape is a
    single tile row and K/N are tile-aligned — the conditions the measured speedup
    was established under.

    ``m`` is the LOGICAL row count and is normally 1 at decode; it occupies one
    padded tile row. Requiring ``m % TILE_SIZE == 0`` here would reject every real
    decode shape, so round up instead.

    ``in0_shard_tiles`` is the width in tiles of each core's in0 shard when the
    ACTIVATION is width-sharded (see activation_sharding.py). The 1D mcast factory
    requires ``in0_shard_width_tiles % in0_block_w == 0``, so the blocking is
    clamped to a divisor of the shard. This matters a lot: MEASURED at K=256,N=1024,
    a 1-tile shard forces in0_block_w=1 and costs **+108%**, while a 4-tile shard
    (in0_block_w=4) costs **+5.7%**.
    """
    T = ttnn.TILE_SIZE
    if k % T or n % T:
        return None
    mt, kt, nt = -(-m // T), k // T, n // T
    if mt != 1:  # decode only; prefill has its own shapes and is not covered here
        return None
    gx, gy = _pick_grid(nt, max_x, max_y)
    cores = gx * gy
    if cores < 2:
        return None
    cap = _block_cap((k, n))
    blk = _largest_divisor(kt, cap)
    if in0_shard_tiles:
        blk = _largest_divisor(in0_shard_tiles, cap=min(cap, in0_shard_tiles))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=blk,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt,
        per_core_N=nt // cores,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _pick_grid_2(n_tiles, k_tiles, max_x=8, max_y=8):
    """Largest rectangle whose core count divides BOTH Nt and Kt.

    `gather_in0` shards the ACTIVATION over K and the weight over N on the same
    grid, so the core count must divide both. This is what confines the mechanism
    to the N=256 shapes.
    """
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if n_tiles % c == 0 and k_tiles % c == 0 and c > best[0] * best[1]:
                best = (gx, gy)
    return best


def derive_decode_1d_gather_config(m, k, n, max_x=8, max_y=8):
    """`gather_in0` ring config, or None when it would cost parallelism.

    Returns None unless the grid that divides both Nt and Kt is the SAME grid the
    mcast config would have used. That single guard is what keeps this off the
    wide-N shapes: at K=256 the ring is capped at Kt=8 cores, so wqkv/gate/up
    (32 cores) and post_projection (48) would lose 4-6x parallelism. Only the
    N=256 shapes — o_proj, down_proj, pre_projection — survive it.

    MEASURED per matmul against the tuned mcast baseline, WITH the in0 reshard
    inside the timed region (`test_matmul_weight_placement.py` arm e+):
        pre_projection 3072x256   15.54 -> 6.48 us   -58.3%
        down_proj      1024x256    6.32 -> 4.30      -32.0%
        o_proj          512x256    4.33 -> 4.18       -3.4%
    """
    T = ttnn.TILE_SIZE
    if k % T or n % T:
        return None
    mt, kt, nt = -(-m // T), k // T, n // T
    if mt != 1:
        return None
    gx, gy = _pick_grid_2(nt, kt, max_x, max_y)
    cores = gx * gy
    if cores < 2:
        return None
    if (gx, gy) != _pick_grid(nt, max_x, max_y):  # would lose N-parallelism
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=kt // cores,  # the factory overwrites this with the shard width
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=mt,
        per_core_N=nt // cores,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
    )


def gather_shard_specs(k, n, max_x=8, max_y=8):
    """(in0 spec, in1 spec, out spec) for the gather config, or None.

    All three must live on the SAME grid: the validator requires
    `in0.shard_spec().grid == in1.shard_spec().grid` when in1 is in L1
    (`matmul_device_operation.cpp:1741-1749`), and the output must be sharded.
    """
    T = ttnn.TILE_SIZE
    if derive_decode_1d_gather_config(1, k, n, max_x, max_y) is None:
        return None
    gx, gy = _pick_grid_2(n // T, k // T, max_x, max_y)
    cores = gx * gy
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})

    def _mc(shape):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    return _mc([T, k // cores]), _mc([k, n // cores]), _mc([T, n // cores])


class DecodeMatmulTuner:
    """Derives and caches decode matmul configs; a no-op passthrough when disabled.

    Shapes are only known at call time (TP splits them), so configs are built
    lazily on first use and cached by (in0 shape, in1 shape) — the same
    lazy-cache pattern as ``RMSNorm._build_sharded_cfg``.
    """

    #: Scopes tuned when GEMMA4_TUNE_MATMULS is unset. `target` only, on measurement:
    #: end to end it is +7.62% (48.38 -> 52.07 tok/s/u on the spec-decode path,
    #: 3 runs/arm against a 0.11% noise floor), because the 35-layer target is ~92%
    #: of a speculative-decoding iteration. The drafter is the other 7.6%, where the
    #: same change is worth ~0 and costs acceptance (it reblocks the K accumulation,
    #: so proposed draft tokens change), so it stays opt-in.
    DEFAULT_SCOPES = frozenset({"target"})

    @classmethod
    def from_env(cls, mesh_device=None, scope="draft"):
        """Build from ``GEMMA4_TUNE_MATMULS``.

        ``1``/``true``/``all`` enables everywhere; ``0``/``off`` disables everywhere;
        a comma list selects scopes (``draft``, ``target``, ``draft,target``) so the
        two can be A/B'd independently. **Unset now means `DEFAULT_SCOPES`, i.e. the
        target only** — it used to mean "off everywhere".

        Caveat, and it is a product decision rather than a perf one: the tuned config
        is not bit-neutral. It reblocks the K accumulation (per-op 0.99988 vs 0.99997
        PCC against fp32 — both correct), and on the target that changes the
        generated text from about the third character. `GEMMA4_TUNE_MATMULS=0`
        restores the previous numerics exactly.
        """
        raw = (os.getenv("GEMMA4_TUNE_MATMULS") or "").strip().lower()
        if raw in ("1", "true", "yes", "on", "all"):
            enabled = True
        elif raw in ("0", "false", "no", "off"):
            enabled = False
        elif raw == "":
            enabled = scope in cls.DEFAULT_SCOPES
        else:
            enabled = scope in {s.strip() for s in raw.split(",")}
        return cls(mesh_device, enabled=enabled, label=scope)

    def __init__(self, mesh_device=None, enabled=False, label="mm"):
        self.enabled = bool(enabled)
        self.label = label
        self._cache = {}
        self._max_x, self._max_y = 8, 8
        if self.enabled and mesh_device is not None:
            grid = mesh_device.compute_with_storage_grid_size()
            self._max_x, self._max_y = min(8, grid.x), min(8, grid.y)

    @staticmethod
    def _in0_shard_tiles(x):
        """Width in tiles of x's per-core shard, or None when x is interleaved."""
        try:
            if not x.is_sharded():
                return None
            spec = x.memory_config().shard_spec
            return int(spec.shape[1]) // ttnn.TILE_SIZE if spec is not None else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _weight_shard_grid(w):
        """(cores, shard width in tiles, grid x, grid y, origin row) of an L1 WIDTH_SHARDED weight on a row-major
        prefix starting at (0, origin row), else None.

        The 1D-mcast factory walks its cores row-major from the rectangle's top-left corner and binds in1 by address
        alone, so a shard grid that is not exactly that prefix would be read silently wrong: it raises instead."""
        if not DecodeMatmulTuner._l1_width_sharded(w):
            return None
        spec = w.memory_config().shard_spec
        ranges = list(spec.grid.ranges())
        if any(r.start.x != 0 for r in ranges):
            return None
        y0 = min(r.start.y for r in ranges)
        gx = max(r.end.x for r in ranges) + 1
        gy = max(r.end.y for r in ranges) - y0 + 1
        cores = spec.grid.num_cores()
        walked = {(i % gx, y0 + i // gx) for i in range(cores)}
        have = {(c.x, c.y) for c in ttnn.corerange_to_cores(spec.grid)}
        if walked != have:
            raise ValueError(
                f"weight shard grid {spec.grid} is not the row-major prefix of {cores} cores from (0, {y0}) that the "
                "1D-mcast matmul walks"
            )
        return cores, int(spec.shape[1]) // ttnn.TILE_SIZE, gx, gy, y0

    def config_for(self, x, w):
        if not self.enabled:
            return None
        shard_tiles = self._in0_shard_tiles(x)
        w_grid = self._weight_shard_grid(w)
        key = (tuple(x.shape), tuple(w.shape), shard_tiles, w_grid)
        if key not in self._cache:
            pc = derive_decode_1d_config(
                int(x.shape[-2]),
                int(x.shape[-1]),
                int(w.shape[-1]),
                self._max_x,
                self._max_y,
                in0_shard_tiles=shard_tiles,
            )
            if pc is not None and w_grid is not None:
                cores, pcn, gx, gy, y0 = w_grid
                tuned = pc.compute_with_storage_grid_size
                if (pcn, cores) != (pc.per_core_N, tuned.x * tuned.y) or y0:
                    # Weight pinned on its own grid (GEMMA4_L1_PIN per_core_N, gemma4-12b record §9.2): run on
                    # the shard's cores. Same in0_block_w, so the K accumulation order is unchanged. A grid starting
                    # at row y0 > 0 (record §9.4) runs on that offset rectangle; its output must be interleaved.
                    offset = (
                        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord(gx - 1, y0 + gy - 1))])
                        if y0
                        else None
                    )
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        in0_block_w=pc.in0_block_w,
                        out_subblock_h=1,
                        out_subblock_w=1,
                        per_core_M=pc.per_core_M,
                        per_core_N=pcn,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                        allowed_worker_cores=offset,
                    )
            elif pc is not None and (pcn := _grid_override((int(x.shape[-1]), int(w.shape[-1])))) is not None:
                full = w.device().compute_with_storage_grid_size()
                cores = -(-(int(w.shape[-1]) // ttnn.TILE_SIZE) // pcn)
                if cores > full.x * full.y:
                    raise ValueError(f"GEMMA4_MM_GRID_SHAPES: {cores} cores do not fit the {full.x}x{full.y} grid")
                # Same in0_block_w as the tuner's config, so only the reading-core count changes.
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(min(cores, full.x), -(-cores // full.x)),
                    in0_block_w=pc.in0_block_w,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=pc.per_core_M,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            self._cache[key] = pc
            # One line per DISTINCT shape (the cache makes this fire once each), so
            # this stays ~10 lines for a 35-layer model and gives the attribution
            # for any measured speedup rather than just a count.
            grid = f" grid={pc.compute_with_storage_grid_size.x}x{pc.compute_with_storage_grid_size.y}" if pc else ""
            blk = f" in0_block_w={pc.in0_block_w} per_core_N={pc.per_core_N}" if pc else ""
            logger.info(
                f"[mm-tune:{self.label}] M={int(x.shape[-2])} K={int(x.shape[-1])} N={int(w.shape[-1])}"
                f" -> {'TUNED' if pc else 'auto'}{grid}{blk}"
            )
        return self._cache[key]

    def stats(self):
        """(tuned, total) shape count — lets a test assert the tuner actually fired."""
        vals = list(self._cache.values())
        return sum(v is not None for v in vals), len(vals)

    @staticmethod
    def _l1_width_sharded(t):
        try:
            mc = t.memory_config()
            return (
                t.is_sharded()
                and mc.buffer_type == ttnn.BufferType.L1
                and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            )
        except Exception:  # noqa: BLE001
            return False

    def _gather_plan(self, x, w):
        """(config, in0_spec, out_spec) when this call can use the ring, else None.

        Gathering is opted into implicitly: it fires only when the WEIGHT is
        already L1 WIDTH_SHARDED, which `WeightPlacement` decides. So the
        allow-list is `GEMMA4_L1_ONLY`, and no separate knob is needed.
        """
        dbg = _env_on("GEMMA4_GATHER_DEBUG")

        def _no(why):
            if dbg:
                logger.info(f"[gather] SKIP {tuple(x.shape)[-1]}x{tuple(w.shape)[-1]}: {why}")
            return None

        if not self.enabled:
            return _no("tuner disabled")
        if not _env_on("GEMMA4_GATHER_IN0"):
            return _no("GEMMA4_GATHER_IN0 not set")
        # ttnn permits gather_in0 with in1 width-sharded, DRAM-INTERLEAVED, or fed via a
        # global CB (matmul_device_operation.cpp:1730-1741). Requiring L1 WIDTH_SHARDED here
        # is THIS codebase's policy, not a ttnn constraint: it encodes MEASUREMENT_RECORD.md
        # §9b, where on E2B the ring LOST with a DRAM weight (5.00 us vs 4.32 mcast) and won
        # only when pinned (2.68). GEMMA4_GATHER_DRAM_WEIGHT=1 lifts the policy so that
        # contrast can be re-measured on other models -- 12B has already reversed the ring's
        # sign once (§6.1), so the E2B result is not assumed to carry. L1 INTERLEAVED stays
        # refused because ttnn itself TT_FATALs on it.
        _allow_dram = _env_on("GEMMA4_GATHER_DRAM_WEIGHT")
        _w_mc = w.memory_config()
        _w_interleaved = _w_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        _w_dram_interleaved = _w_mc.buffer_type == ttnn.BufferType.DRAM and _w_interleaved
        # GEMMA4_GATHER_L1_INTERLEAVED admits an L1-INTERLEAVED weight and NOTHING else, so the
        # ring fires on exactly the relocated layers and the per-layer expected_ring arithmetic
        # still holds. That keeps "ring on a remote-SRAM weight" a clean arm rather than the
        # mixture ANY_WEIGHT gives (it also rings every DRAM weight, priced at +4.43 us each in
        # §6.5). Requires the loosened ttnn check.
        _allow_l1_il = _env_on("GEMMA4_GATHER_L1_INTERLEAVED")
        _w_l1_interleaved = _w_mc.buffer_type == ttnn.BufferType.L1 and _w_interleaved
        # GEMMA4_GATHER_ANY_WEIGHT=1 drops the model-level layout gate entirely and lets
        # ttnn's own validation adjudicate (matmul_device_operation.cpp:1728-1740). Exists so
        # "L1 INTERLEAVED + ring is impossible" can be DEMONSTRATED rather than asserted: the
        # model gate would otherwise refuse first and the TT_FATAL would never fire.
        if _env_on("GEMMA4_GATHER_ANY_WEIGHT"):
            pass
        elif (
            not self._l1_width_sharded(w)
            and not (_allow_dram and _w_dram_interleaved)
            and not (_allow_l1_il and _w_l1_interleaved)
        ):
            return _no(f"weight not L1 WIDTH_SHARDED ({_w_mc.buffer_type.name}/" f"{_w_mc.memory_layout.name})")
        k, n = int(x.shape[-1]), int(w.shape[-1])
        pc = derive_decode_1d_gather_config(1, k, n, self._max_x, self._max_y)
        if pc is None:
            return _no(f"no gather config for K={k} N={n} grid<={self._max_x}x{self._max_y}")
        specs = gather_shard_specs(k, n, self._max_x, self._max_y)
        if specs is None:
            return _no("gather_shard_specs returned None")
        in0_spec, in1_spec, out_spec = specs
        # The weight must sit on exactly the grid the ring expects -- but only when it IS
        # sharded. A DRAM-interleaved weight has no shard_spec; ttnn streams it per block.
        if w.memory_config().shard_spec is not None:
            have, want = list(w.memory_config().shard_spec.shape), list(in1_spec.shard_spec.shape)
            if have != want:
                return _no(f"weight shard shape {have} != ring's {want}")
        if dbg:
            logger.info(f"[gather] TAKE {k}x{n}")
        return pc, in0_spec, out_spec

    def linear(self, x, w, **kwargs):
        """``ttnn.linear`` with a tuned program config when one applies.

        When the weight is L1 width-sharded on the ring grid and the shape is
        grid-neutral, this takes the `gather_in0` path instead: reshard the
        activation over K, run the ring, and hand back an interleaved output so
        call sites are unchanged. MEASURED with both reshards included, this is
        -58.3% on pre_projection and -32.0% on down_proj (arm e+).
        """
        if kwargs.get("program_config") is None and self.enabled and self._dram_width_sharded(w):
            return self._dram_sharded_linear(x, w, **kwargs)
        plan = self._gather_plan(x, w) if kwargs.get("program_config") is None else None
        if plan is not None:
            pc, in0_spec, out_spec = plan
            if kwargs.get("compute_kernel_config") is None:
                kwargs["compute_kernel_config"] = self.compute_config_for(x, w)
            want_mc = kwargs.pop("memory_config", None) or ttnn.DRAM_MEMORY_CONFIG
            xs = x if x.memory_config() == in0_spec else ttnn.to_memory_config(x, in0_spec)
            out = ttnn.linear(xs, w, program_config=pc, memory_config=out_spec, **kwargs)
            if xs is not x:
                xs.deallocate(True)
            back = ttnn.sharded_to_interleaved(out, want_mc)
            out.deallocate(True)
            return back
        if kwargs.get("program_config") is None:
            pc = self.config_for(x, w)
            if pc is not None:
                kwargs["program_config"] = pc
                if kwargs.get("compute_kernel_config") is None:
                    kwargs["compute_kernel_config"] = self.compute_config_for(x, w)
        return ttnn.linear(x, w, **kwargs)

    @staticmethod
    def _dram_width_sharded(w):
        mc = w.memory_config()
        return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED

    def _dram_sharded_linear(self, x, w, **kwargs):
        """RESEARCH (gemma4-12b record §9.2a, DRAM-sharded fat-K): a DRAM WIDTH_SHARDED weight (GEMMA4_DRAM_SHARD,
        weight_placement.dram_shard_width) runs as MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.

        It keeps the 1D-multicast config's in0_block_w, so K is summed in the same blocks: measured bit-identical to
        the 1D-multicast output on every fat-K shape (research_codes/mm_profiling/test_dram_sharded_fatk.py). in0 is
        resharded to L1 WIDTH_SHARDED on the largest core count (<= 32, a row of up to 8 or whole rows of 8) that
        divides Kt and Nt with a shard width that is a multiple of in0_block_w; the sharded output goes back to the
        caller's memory config (DRAM interleaved by default).
        """
        T = ttnn.TILE_SIZE
        k, n = int(x.shape[-1]), int(w.shape[-1])
        key = ("dram_sharded", tuple(x.shape), tuple(w.shape))
        if key not in self._cache:
            mt, kt, nt = -(-int(x.shape[-2]) // T), k // T, n // T
            iw = derive_decode_1d_config(int(x.shape[-2]), k, n, self._max_x, self._max_y).in0_block_w
            cores = max(
                c
                for c in range(1, 33)
                if (c <= 8 or c % 8 == 0) and kt % c == 0 and nt % c == 0 and (kt // c) % iw == 0
            )
            grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(cores, 8) - 1, -(-cores // 8) - 1))}
            )

            def width_sharded(width):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, [T * mt, width], ttnn.ShardOrientation.ROW_MAJOR),
                )

            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=iw, per_core_M=mt, per_core_N=nt // cores, fused_activation=None
            )
            self._cache[key] = (pc, width_sharded(k // cores), width_sharded(n // cores))
            logger.info(
                f"[mm-tune:{self.label}] M={int(x.shape[-2])} K={k} N={n} -> DRAM_SHARDED in0 cores={cores}"
                f" in0_block_w={iw} per_core_N={nt // cores}"
            )
        pc, in0_mc, out_mc = self._cache[key]
        want_mc = kwargs.pop("memory_config", None) or ttnn.DRAM_MEMORY_CONFIG
        if kwargs.get("compute_kernel_config") is None:
            kwargs["compute_kernel_config"] = self.compute_config_for(x, w)
        xs = ttnn.to_memory_config(x, in0_mc)
        out = ttnn.linear(xs, w, program_config=pc, memory_config=out_mc, **kwargs)
        xs.deallocate(True)
        back = ttnn.sharded_to_interleaved(out, want_mc)
        out.deallocate(True)
        return back

    def compute_config_for(self, x, w):
        """Return the compute config ttnn would pick for this matmul without a program config.

        ttnn lowers its default math fidelity from HiFi2 to LoFi as soon as a program config is
        passed (matmul_device_operation.cpp:2688), so a tuned call restates the automatic
        defaults to change only the blocking. Returns None where ttnn's default already matches
        (both inputs BFP8/BFP4, or both FLOAT32). Same change as PR #57993 (8e7e0f40733).
        """
        low_precision = (ttnn.bfloat8_b, ttnn.bfloat4_b)
        if x.dtype in low_precision and w.dtype in low_precision:
            return None
        if x.dtype == ttnn.float32 and w.dtype == ttnn.float32:
            return None
        return ttnn.init_device_compute_kernel_config(
            x.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )


#: Shared disabled instance so call sites can do ``(tuner or DISABLED).linear(...)``.
DISABLED = DecodeMatmulTuner(enabled=False)


def resolve(tuner):
    return tuner if tuner is not None else DISABLED


def lm_head_out_block_w_config(env_name, x, w):
    """Explicit 1D-mcast config for an lm_head call, fixing only ``out_block_w``. Opt-in.

    Returns ``(program_config, compute_kernel_config)``, or ``(None, None)`` when ``env_name`` is
    unset, which leaves the call on ttnn's automatic config.

    The automatic path sizes ``out_block_w`` against the free L1 (lowest occupied L1 address), so
    pinning anything re-blocks these calls as a side effect. Everything except ``out_block_w``
    restates what the automatic path resolves to for these shapes (M = 1 tile row):
    the full compute grid, ``per_core_N = ceil(ceil(N / cores) / 32)``, ``in0_block_w = 2`` when Kt
    is even, ``out_subblock_w`` = largest divisor of ``out_block_w`` <= 8, and HiFi2 with packer L1
    accumulation. ``out_block_w`` only changes how N is split into passes, not any tile's K-sum, so
    the output is bit-identical to the automatic config (gemma4-12b/MEASUREMENT_RECORD.md §5.5.12).
    """
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return None, None
    out_block_w = int(raw)
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    T = ttnn.TILE_SIZE
    k, n = int(w.shape[-2]), int(w.shape[-1])
    per_core_n = -(-(-(-n // cores)) // T)
    # Optional wider per_core_N (e.g. GEMMA4_LMHEAD_DRAFT_PCN for GEMMA4_LMHEAD_DRAFT_OBW): fewer cores
    # carry the matmul, so a composite out_block_w becomes legal when the minimal per_core_N is
    # prime (gemma4-12b MEASUREMENT_RECORD §5.5.14 Gate 2d). Unset = the minimal per_core_N.
    pcn_raw = (os.getenv(env_name.replace("_OBW", "_PCN")) or "").strip()
    if pcn_raw:
        if int(pcn_raw) < per_core_n:
            raise ValueError(f"{env_name.replace('_OBW', '_PCN')}={pcn_raw} < minimal per_core_N {per_core_n}")
        per_core_n = int(pcn_raw)
    if per_core_n % out_block_w:
        raise ValueError(f"{env_name}={out_block_w} must divide per_core_N={per_core_n} (N={n}, {cores} cores)")
    in0_block_w = 2 if (k // T) % 2 == 0 else 1
    # RESEARCH (gemma4-12b record §9.2a, C1): e.g. GEMMA4_LMHEAD_DRAFT_IW overrides in0_block_w. It reblocks the K sum,
    # so the logits are NOT bit-identical to the automatic config; for mechanism tests only.
    iw_raw = (os.getenv(env_name.replace("_OBW", "_IW")) or "").strip()
    if iw_raw:
        in0_block_w = int(iw_raw)
        if (k // T) % in0_block_w:
            raise ValueError(f"{env_name.replace('_OBW', '_IW')}={iw_raw} must divide Kt={k // T}")
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=max(d for d in range(1, 9) if out_block_w % d == 0),
        out_block_h=1,
        out_block_w=out_block_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    logger.info(
        f"[lm-head-obw] {env_name}: N={n} per_core_N={per_core_n} out_block_w={out_block_w} in0_block_w={in0_block_w}"
    )
    return pc, ckc
