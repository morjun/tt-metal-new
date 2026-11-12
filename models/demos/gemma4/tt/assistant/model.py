# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 it-assistant drafter model (EAGLE / Multi-Token-Prediction).

The drafter is a tiny Gemma4 text model (4 layers, hidden 1024) that proposes K
candidate tokens from a single backbone position. Each step:

    inputs_embeds = cat(target_embed(last_token), last_hidden)       # [.., 2*backbone] (scaled embed)
    h = pre_projection(inputs_embeds)                                # [.., hidden]
    for layer in 4 decoder layers:                                   # cross-attend
        h = layer(h, kv = target's last {sliding,full} layer KV)     #   into target KV
    h = norm(h)
    logits      = lm_head(h)                                         # next draft token (argmax)
                  # or, when use_ordered_embeddings (E2B): a Centroid Masked
                  # Embedding head over ~4096 candidates — see masked_embedding.py
    next_hidden = post_projection(h)                                 # recurrent hidden

The decoder layers are ordinary ``Gemma4DecoderLayer``s (MoE disabled) run in
decode mode with ``is_kv_shared=True``: they compute only Q (the K/V weights are
synthesized as zeros and discarded) and the SDPA attends into the *target's* KV
cache for that layer type. ``position_ids`` and the target KV are held fixed
across the K drafter steps — matching HF's
``SinglePositionMultiTokenCandidateGenerator``.

Reference: transformers ``Gemma4AssistantForCausalLM.forward`` and
``generation/candidate_generator.py:SinglePositionMultiTokenCandidateGenerator``.

Constraints (first cut):
  * batch = 1
  * the target must use UNBOUNDED sliding KV caches (``bounded_sliding_kv_cache``
    off) so the drafter's cross-attention reads absolute cache positions without
    a circular-buffer modulo (the assistant attention config doesn't carry one).
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt import lm_head_reload
from models.demos.gemma4.tt.assistant.masked_embedding import Gemma4TTMaskedEmbedder
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import ccl_allgather
from models.demos.gemma4.tt.decode_matmul import (
    DecodeLinear,
    flags_on,
    lm_head_out_block_w_config,
    lm_head_out_block_w_from_env,
)
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


def _inject_zero_kv_weights(state_dict, text_args):
    """Add zero k_proj/v_proj and a unit k_norm for every assistant layer.

    The assistant checkpoint stores no K/V projections (its layers are all
    KV-shared), but ``Gemma4DecoderLayer``'s attention loader expects a full
    fused QKV. We inject zeros for the K/V columns (the split discards them under
    ``is_kv_shared=True``) and a unit k_norm so the loader's unconditional
    ``k_norm.weight`` read succeeds. Mutates and returns ``state_dict``.
    """
    hidden = text_args.hidden_size
    for i in range(text_args.num_hidden_layers):
        cfg = Gemma4AttentionConfig(text_args, i)
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        prefix = f"model.layers.{i}.self_attn"
        if f"{prefix}.k_proj.weight" not in state_dict:
            state_dict[f"{prefix}.k_proj.weight"] = torch.zeros((kv_size, hidden), dtype=torch.bfloat16)
        # Sliding (non-global) layers load a separate v_proj; global layers tie V=K.
        if not cfg.use_kv_tying and f"{prefix}.v_proj.weight" not in state_dict:
            state_dict[f"{prefix}.v_proj.weight"] = torch.zeros((kv_size, hidden), dtype=torch.bfloat16)
        if f"{prefix}.k_norm.weight" not in state_dict:
            state_dict[f"{prefix}.k_norm.weight"] = torch.ones((cfg.head_dim,), dtype=torch.bfloat16)
    return state_dict


class Gemma4AssistantModel:
    def __init__(
        self,
        mesh_device,
        assistant_args,
        target_model,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_local_batch_size=1,
        bounded_sliding_kv_cache=False,
    ):
        self.mesh_device = mesh_device
        self.max_local_batch_size = max_local_batch_size
        self.args = assistant_args
        self.text_args = assistant_args.text_args
        self.target = target_model
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.backbone_hidden_size = assistant_args.backbone_hidden_size
        self.hidden_size = self.text_args.hidden_size
        self.vocab_size = self.text_args.vocab_size
        self.layer_types = list(self.text_args.layer_types)

        # E2B's assistant replaces the dense lm_head with a Centroid Masked
        # Embedding head (31B/12B assistants set this False and use lm_head).
        self.use_cme = bool(assistant_args.use_ordered_embeddings)

        tp = mesh_config.tp if mesh_config else 1
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        # The drafter shares the target's per-layer-type RoPE caches (identical
        # head_dim / theta), so its Q is RoPE'd consistently with the cached K.
        self.rope_caches_2d = target_model.rope_caches_2d

        state_dict = _inject_zero_kv_weights(dict(state_dict), self.text_args)

        # Decoder layers (reuse the target's layer, MoE disabled, KV-shared).
        #
        # Bounded target KV: the drafter cross-attends into the TARGET's caches,
        # so when those sliding layers are a bounded ring the drafter's SDPA has
        # to wrap absolute positions the same way. Its attention configs pick up
        # cache_position_modulo from the same flag the target layers use (the
        # assistant's own sliding_window matches the target's -- 1024 for both
        # 12B and 31B -- so the ring size agrees). It still allocates NO KV of
        # its own: every layer runs is_kv_shared=True.
        #
        # GEMMA4_ASSISTANT_DECODE_1D=1 (opt-in): the drafter's wqkv / o_proj / gate_up / down and its pre_projection
        # on the explicit 1D decode config (decode_matmul.DecodeLinear: K blocked by up to 8, N spread over the whole
        # worker grid), without turning GEMMA4_MLP_DECODE_1D / GEMMA4_ATTN_DECODE_1D on for the target. It changes
        # the drafter's rounding, so only the acceptance can move; the target's output cannot.
        self._decode_1d = os.environ.get("GEMMA4_ASSISTANT_DECODE_1D", "0") == "1"
        scoped = ("GEMMA4_MLP_DECODE_1D", "GEMMA4_ATTN_DECODE_1D") if self._decode_1d else ()
        self.layers = []
        with flags_on(*scoped):
            for i in range(self.text_args.num_hidden_layers):
                layer = Gemma4DecoderLayer(
                    mesh_device=mesh_device,
                    hf_config=self.text_args,
                    state_dict=state_dict,
                    layer_idx=i,
                    ccl_manager=ccl_manager,
                    dtype=dtype,
                    tensor_cache_path=f"{tensor_cache_path}/layer_{i}" if tensor_cache_path else None,
                    mesh_config=mesh_config,
                    max_seq_len=self.text_args.max_seq_len,
                    max_local_batch_size=max_local_batch_size,
                    bounded_sliding_kv_cache=bounded_sliding_kv_cache,
                )
                self.layers.append(layer)

        # Final norm (model.norm)
        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=self.text_args,
            state_dict=substate(state_dict, "model.norm"),
            tensor_cache_path=f"{tensor_cache_path}/final_norm" if tensor_cache_path else None,
            mesh_config=mesh_config,
        )

        # pre_projection (2*backbone -> hidden) and post_projection (hidden ->
        # backbone) are small and kept replicated so hidden stays full-width
        # across TP (matching the layer norms / attention which expect full
        # hidden). lm_head (hidden -> vocab) is column-parallel on vocab and
        # all-gathered, mirroring the target.
        col_mapper = mesh_config.column_parallel(mesh_device) if tp > 1 else None

        def _linear(key, mapper, transpose=True, weight_dtype=None):
            w = state_dict.get(key)
            if w is None:
                return None
            wt = w.transpose(-2, -1) if transpose else w
            wt = wt.unsqueeze(0).unsqueeze(0)
            cache_key = key.replace(".", "_")
            if weight_dtype is not None and weight_dtype != dtype:
                cache_key += f"_{str(weight_dtype).split('.')[-1]}"  # its own cache entry: never the other dtype's
            return ttnn.as_tensor(
                wt,
                device=mesh_device,
                dtype=weight_dtype or dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper if mapper is not None else (replicate if is_mesh else None),
                cache_file_name=get_cache_file_name(tensor_cache_path, cache_key),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.pre_projection = _linear("pre_projection.weight", None)
        self.post_projection = _linear("post_projection.weight", None)
        # CME mode computes only ~4096 of the 262144 logits, so it needs a
        # row-gatherable [V, H] copy of the output embedding instead of the dense
        # [H, V] lm_head — building both would waste 134 MB of DRAM.
        self.masked_embedding = None
        self.lm_head = None
        if self.use_cme:
            self.masked_embedding = Gemma4TTMaskedEmbedder(
                mesh_device=mesh_device,
                assistant_args=assistant_args,
                state_dict=state_dict,
                dtype=dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
            )
        else:
            # lm_head tied to the assistant's own embed_tokens when a separate
            # lm_head.weight isn't stored.
            lm_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
            # The drafter's lm_head is the largest read of a draft step (the vocab projection, 500 MB per chip on the
            # 12B at TP4): GEMMA4_ASSISTANT_LM_HEAD_BFP8=1 keeps it in bfloat8_b (half the bytes; the drafts are
            # proposals the verify checks, so only the acceptance can move, which the parity test measures).
            lm_head_bfp8 = os.environ.get("GEMMA4_ASSISTANT_LM_HEAD_BFP8", "0") == "1"
            self.lm_head = _linear(lm_key, col_mapper, weight_dtype=ttnn.bfloat8_b if lm_head_bfp8 else None)
            if self.lm_head is None:
                raise ValueError("Assistant checkpoint missing lm_head weights")
        if self.pre_projection is None or self.post_projection is None:
            raise ValueError("Assistant checkpoint missing pre_projection / post_projection weights")
        # pre_projection (2 * backbone -> hidden, a long K: 7680 on the 12B) under GEMMA4_ASSISTANT_DECODE_1D:
        # 107 -> 41 us per call. post_projection stays on the automatic config (DecodeLinear's is slower there).
        self._pre_projection_linear = DecodeLinear(
            self.pre_projection,
            mesh_device,
            k=int(self.pre_projection.shape[-2]),
            n=int(self.pre_projection.shape[-1]),
            enabled=self._decode_1d,
        )
        # GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W (opt-in): fix out_block_w of the full-vocab lm_head, which otherwise runs
        # on ttnn's automatic config (decode_matmul.lm_head_out_block_w_config). Single chip only: at TP > 1 the head
        # is a vocab shard, a case this has not been measured on.
        self._lm_head_out_block_w = lm_head_out_block_w_from_env("GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W")
        if self._lm_head_out_block_w is not None and tp > 1:
            raise ValueError(f"GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W supports a single chip only (tp={tp})")
        if self._lm_head_out_block_w is not None and self.use_cme:
            raise ValueError(
                "GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W does not apply to the centroid masked embedding head"
            )
        self._lm_head_cfg = None
        # GEMMA4_ASSISTANT_LOGITS_L1=1 (opt-in): write the lm_head's logits (one tile row of the full vocab, 16 MiB in
        # bf16) to L1 instead of DRAM. It needs the explicit lm_head config: for an L1 output ttnn's automatic config
        # falls back to the plain multi-core matmul (no packer L1 accumulation), which changes the logits; the explicit
        # 1D config keeps the DRAM path's matmul and logits.
        self._logits_l1 = os.environ.get("GEMMA4_ASSISTANT_LOGITS_L1", "0") == "1"
        if self._logits_l1 and self._lm_head_out_block_w is None:
            raise ValueError("GEMMA4_ASSISTANT_LOGITS_L1=1 needs GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W")
        # GEMMA4_ASSISTANT_LM_HEAD_PER_CORE_N (opt-in, with the out_block_w knob): N tiles per core of the DRAM-fed
        # lm_head (the whole head, or the DRAM remainder of a split head).
        pcn = (os.environ.get("GEMMA4_ASSISTANT_LM_HEAD_PER_CORE_N") or "").strip()
        self._lm_head_per_core_n = int(pcn) if pcn else None
        if self._lm_head_per_core_n is not None and self._lm_head_out_block_w is None:
            raise ValueError("GEMMA4_ASSISTANT_LM_HEAD_PER_CORE_N needs GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W")
        self.lm_head_l1 = None
        self._split_lm_head_to_l1(mesh_device, tp)

    def _split_lm_head_to_l1(self, mesh_device, tp):
        """GEMMA4_ASSISTANT_LM_HEAD_L1_COLS (opt-in, default 0): pin the first ``cols`` vocab columns of the lm_head in
        L1, WIDTH_SHARDED over the whole worker grid (``cols / 32 / cores`` N tiles per core), and keep the rest in
        DRAM. The slice runs the 1D-multicast matmul on a local in1 shard with the automatic path's K blocking
        (``in0_block_w`` 2, HiFi2, packer L1 accumulation), so every logit is computed as the unsplit head computes
        it. ``step`` returns the two parts joined as one row-major row (see ``_join_split_logits``).

        Whole tiles per core only (``cols`` a multiple of 32 x cores). The slice sits at the top of L1 on every
        core; an op whose circular buffers reach into it fails ttnn's CB clash check, unless
        GEMMA4_LMHEAD_L1_RELOAD brackets that op (tt/lm_head_reload.py)."""
        cols = int(os.environ.get("GEMMA4_ASSISTANT_LM_HEAD_L1_COLS", "0"))
        if cols <= 0:
            return
        if self.use_cme or tp > 1:
            raise ValueError("GEMMA4_ASSISTANT_LM_HEAD_L1_COLS supports the dense lm_head on a single chip only")
        if self._lm_head_out_block_w is None:
            raise ValueError(
                "GEMMA4_ASSISTANT_LM_HEAD_L1_COLS needs GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W (the DRAM part)"
            )
        T = ttnn.TILE_SIZE
        grid = mesh_device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        k = int(self.lm_head.shape[-2])
        vocab = int(self.lm_head.shape[-1])
        if cols % (T * cores) or cols >= vocab:
            raise ValueError(
                f"GEMMA4_ASSISTANT_LM_HEAD_L1_COLS={cols}: must be a multiple of {T * cores} (whole tiles on each of "
                f"the {cores} cores) and below the vocab {vocab}"
            )
        per_core_n = cols // T // cores
        kt = k // T
        self._lm_head_l1_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            in0_block_w=2 if kt % 2 == 0 else 1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=per_core_n,  # one output block per core: required for an L1-sharded in1
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self._lm_head_l1_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        head = ttnn.slice(self.lm_head, [0, 0, 0, 0], [1, 1, k, cols])
        tail = ttnn.slice(self.lm_head, [0, 0, 0, cols], [1, 1, k, vocab])
        self.lm_head.deallocate(True)
        self.lm_head = tail  # the DRAM remainder keeps the attribute
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
            [k, cols // cores],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.lm_head_l1 = ttnn.to_memory_config(
            head, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)
        )
        head.deallocate(True)
        charge = k * T * per_core_n * self.lm_head_l1.element_size()
        logger.info(
            f"[lm_head-split] {cols}/{vocab} columns in L1 on {grid.x}x{grid.y}={cores} cores, per_core_N={per_core_n}, "
            f"{charge} B per core at address {self.lm_head_l1.buffer_address()}"
        )
        lm_head_reload.register(self, "lm_head_l1", mesh_device)

    def _join_split_logits(self, head, tail):
        """The split head's two parts as one row-major ``[1, 1, 1, vocab]`` DRAM row, in vocab order: each part is
        untilized (which drops the 31 padding rows of its tile), then copied into the joined row by a custom kernel
        (kernels/join_row_bytes.cpp). ``ttnn.concat`` would stage the whole 524 KiB row in L1 on one core, on top of
        the pinned slice there; the kernel uses a 16 KiB scratch per core. The token pick
        (``spec_decode._argmax_last``) takes a row-major row as it is."""

        def untilize(part):
            if part.memory_config().buffer_type != ttnn.BufferType.L1:
                return ttnn.untilize(part, use_multicore=True)
            # ttnn.untilize drops the tile padding only for a DRAM input.
            end = [d - 1 for d in part.shape]
            return ttnn.untilize_with_unpadding(part, end, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True)

        rows = [untilize(head), untilize(tail)]
        head.deallocate(True)
        tail.deallocate(True)
        joined = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, self.vocab_size]),
            rows[0].dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        elem = rows[0].element_size()
        dst_off = 0
        for i, row in enumerate(rows):
            ttnn.generic_op([row, joined], self._join_program(i, row, joined, dst_off))
            dst_off += int(row.shape[-1]) * elem
        for r in rows:
            r.deallocate(True)
        return joined

    _JOIN_CHUNK = 16384  # scratch bytes per core
    _JOIN_CORES = (2, 8)  # cores copying the L1 part's row and the DRAM part's row

    def _join_program(self, part, row, joined, dst_off):
        """Program copying ``row`` into ``joined`` at byte ``dst_off``, split over this part's cores (the first
        cores of the last grid row for part 0, the next ones for part 1)."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        n = self._JOIN_CORES[part]
        first = sum(self._JOIN_CORES[:part])
        cores = [(first + i, grid.y - 1) for i in range(n)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
        nbytes = int(row.shape[-1]) * row.element_size()
        per_core = -(-nbytes // n // 64) * 64
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(cores):
            off = min(i * per_core, nbytes)
            rt[x][y] = [row.buffer_address(), joined.buffer_address(), off, dst_off + off, min(per_core, nbytes - off)]
        ct = [self._JOIN_CHUNK, nbytes, int(joined.shape[-1]) * joined.element_size()]
        ct += list(ttnn.TensorAccessorArgs(row).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(joined).get_compile_time_args())
        kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/gemma4/tt/assistant/kernels/join_row_bytes.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch = ttnn.CBDescriptor(
            total_size=self._JOIN_CHUNK,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=self._JOIN_CHUNK)
            ],
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[scratch])

    def _lm_head_configs(self, x):
        """(program_config, compute_kernel_config) of the lm_head call: (None, None) = ttnn's automatic config."""
        if self._lm_head_out_block_w is None:
            return None, None
        if self._lm_head_cfg is None:
            self._lm_head_cfg = lm_head_out_block_w_config(
                self.mesh_device,
                k=int(self.lm_head.shape[-2]),
                n=int(self.lm_head.shape[-1]),
                out_block_w=self._lm_head_out_block_w,
                x_dtype=x.dtype,
                w_dtype=self.lm_head.dtype,
                per_core_n=self._lm_head_per_core_n,
            )
        return self._lm_head_cfg

    def _raw_token_embed(self, token_tt):
        """Target token embedding of a single token id -> [1,1,1,backbone] TILE.

        Uses the *scaled* embedding (``embed_tokens`` = raw table * sqrt(hidden)).
        HF's ``embed_tokens`` is a ``Gemma4TextScaledWordEmbedding`` that applies
        the ``sqrt(hidden)`` normalizer inside its forward, so the drafter input
        ``cat(get_input_embeddings()(token), hidden)`` carries the *scaled*
        embedding. Feeding the unscaled table (~62x too small) starves the
        ``pre_projection`` token branch and collapses drafter acceptance
        (measured 0.19 unscaled -> 1.44 scaled, matching the HF reference).
        """
        emb = self.target.embed_tokens(token_tt)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    def prepare_rope(self, pos_uint32):
        """The RoPE rows of ``pos_uint32`` ([1, 32] uint32, one position per row) per layer type, gathered once
        for every ``step`` run at these positions: ``{layer_type: (cos, sin, scratch)}``.

        ``cos`` / ``sin`` are the ``[1, 1, 32, head_dim]`` TILE gathers the attention makes per layer otherwise
        (``decode_forward`` under ``rope_presliced``, the target model's own per-type hoist) and ``scratch`` the
        type's batch>1 RoPE workspace (``rope_prepared``: the transposed and head-repeated forms, built by the
        type's first layer and reused by the rest). Same values, fewer programs: a serving record drafts K tokens
        at one position per row, so one gather per type serves its K x 4 layer calls instead of two per layer per
        step. Freed by ``release_rope``.
        """
        prepared = {}
        for lt in self.layer_types:
            if lt in prepared:
                continue
            cos_2d, sin_2d = self.rope_caches_2d[lt]
            cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(pos_uint32, cos_2d, layout=ttnn.TILE_LAYOUT))
            sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(pos_uint32, sin_2d, layout=ttnn.TILE_LAYOUT))
            prepared[lt] = (cos_pos, sin_pos, {})
        return prepared

    @staticmethod
    def release_rope(prepared):
        """Free what ``prepare_rope`` gathered and what the steps built in its scratch (the target's order)."""
        for cos_pos, sin_pos, _scratch in prepared.values():
            cos_pos.deallocate(True)
            sin_pos.deallocate(True)
        for _cos, _sin, scratch in prepared.values():
            for pair in scratch.values():
                for t in pair:
                    t.deallocate(True)
            scratch.clear()

    def step(
        self,
        token_tt,
        target_hidden,
        shared_kv,
        page_tables,
        pos_uint32,
        pos_int32,
        return_logits=True,
        gather_logits=True,
        rope_prepared=None,
    ):
        """One drafter step.

        Args:
            token_tt: [1,1] uint32 last token id.
            target_hidden: [1,1,1,backbone] TILE — the recurrent hidden (target's
                last-token hidden on the first step, then this method's previous
                ``next_hidden``).
            shared_kv: {layer_type: [k_cache, v_cache]} target caches.
            page_tables: {layer_type: page_table} (or a single page_table reused
                for both types in the simple unbounded case).
            pos_uint32: [1,32] uint32 fixed position for RoPE lookup.
            pos_int32: [1] int32 fixed position for SDPA cur_pos.
            gather_logits: when False (TP > 1), return the chip's vocab slice ``[1, 1, rows, vocab / tp]``
                instead of the all-gathered logits (the serving records' sharded argmax). The CME head's
                compact output is replicated and never gathered.
            return_logits: when False, skip the output head + its TP all-gather
                and return ``(None, next_hidden)`` (used to isolate the
                lm_head/CCL cost in timing harnesses).
            rope_prepared: the RoPE rows of ``pos_uint32`` gathered once per
                layer type by ``prepare_rope`` (the serving records run K steps
                at one position per row); None gathers them per layer, as the
                attention does on its own.

        Returns:
            (logits, next_hidden [1,1,1,backbone]). ``logits`` is a dense
            [1,1,rows,vocab] tensor normally, or a compact ``CmeLogits`` pair
            under Centroid Masked Embedding (E2B). ``spec_decode`` dispatches on
            the type in ``_logits_to_host`` / ``_argmax_last``.
        """
        tok_embed = self._raw_token_embed(token_tt)
        inp = ttnn.concat([tok_embed, target_hidden], dim=-1)
        tok_embed.deallocate(True)

        h = self._pre_projection_linear(inp)
        inp.deallocate(True)

        lm_head_reload.drafter_step()  # no-op unless GEMMA4_LMHEAD_L1_RELOAD

        for i, layer in enumerate(self.layers):
            lt = self.layer_types[i]
            pt = page_tables[lt] if isinstance(page_tables, dict) else page_tables
            if rope_prepared is None:
                rope_kwargs = {"rope_mats": self.rope_caches_2d[lt]}
            else:
                cos_pos, sin_pos, scratch = rope_prepared[lt]
                rope_kwargs = {"rope_mats": (cos_pos, sin_pos), "rope_presliced": True, "rope_prepared": scratch}
            h = layer(
                h,
                position_idx=pos_uint32,
                page_table=pt,
                kv_cache=shared_kv[lt],
                is_decode=True,
                token_index=None,
                is_kv_shared=True,
                position_idx_cache=pos_int32,
                **rope_kwargs,
            )

        normed = self.norm.forward(h)
        h.deallocate(True)

        logits = None
        if return_logits:
            if self.use_cme:
                # Replicated weights, ~4096-wide output: no all-gather needed.
                logits = self.masked_embedding.forward(normed)
            elif self.lm_head_l1 is not None:
                logits_mc = ttnn.L1_MEMORY_CONFIG if self._logits_l1 else None
                lm_head_reload.reload()  # no-op unless GEMMA4_LMHEAD_L1_RELOAD
                head = ttnn.linear(
                    normed,
                    self.lm_head_l1,
                    program_config=self._lm_head_l1_pc,
                    compute_kernel_config=self._lm_head_l1_ckc,
                    memory_config=logits_mc,
                )
                pc, ckc = self._lm_head_configs(normed)
                tail = ttnn.linear(
                    normed, self.lm_head, program_config=pc, compute_kernel_config=ckc, memory_config=logits_mc
                )
                logits = self._join_split_logits(head, tail)
            else:
                pc, ckc = self._lm_head_configs(normed)
                logits = ttnn.linear(
                    normed,
                    self.lm_head,
                    program_config=pc,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.L1_MEMORY_CONFIG if self._logits_l1 else None,
                )
                if gather_logits and self.mesh_config is not None and self.mesh_config.tp > 1:
                    logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)

        next_hidden = ttnn.linear(normed, self.post_projection)
        normed.deallocate(True)
        return logits, next_hidden
