"""Standalone 12B drafter on the upstream-main research branch: traced ms per K-step draft (recurrence + token pick).

Port of l1w test_assistant_standalone_l1.py (_TargetStub, _fabricate_shared_kv, _make_fused_k_body,
_time_fused_k_steps). The draft loop mirrors the fused body (spec_decode.py: assistant.step ->
_argmax_last(rows=1) -> ttnn.reshape). The configuration comes from the environment (run_standalone.py sets it
per arm). Every cell prints the drafted token and the SHA-256 of the final hidden state: the identity check
against the anchor (CLAUDE.md).

    python standalone_draft.py <ctx> [max_seq_len] [--rounds N] [--census out.json] [--modes]

--census: one eager draft inside a ttnn graph capture; writes every circular-buffer and L1 buffer allocation
(address, size, cores) to out.json and prints the highest CB end per op. No timing.
--modes: also time the draft without the token pick and without the lm_head.
"""

import argparse
import json
import hashlib
import math
import time

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.unit.test_spec_decode import _kv_to_tt
from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.model import create_rope_caches
from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs
from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
from models.tt_transformers.tt.common import PagedAttentionConfig

ASSISTANT = "google/gemma-4-12B-it-assistant"
K, REPS, BLOCK = 3, 100, 64


class Stub:
    def __init__(self, mesh, mesh_config, text_args, hidden, max_seq_len):
        self.mesh_device, self.mesh_config, self.ccl_manager = mesh, mesh_config, None
        self.hidden_size = hidden
        self.bounded_sliding_kv_cache = False
        self.embed_scale = hidden**0.5
        _, self.rope_caches_2d = create_rope_caches(mesh, text_args._hf_text_config, max_seq_len)
        g = torch.Generator().manual_seed(0)
        table = (torch.randn(text_args.vocab_size, hidden, generator=g) * 0.02).to(torch.bfloat16)
        self.embedding_weight = ttnn.as_tensor(
            table.unsqueeze(0).unsqueeze(0),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def embed_tokens(self, tokens):
        return ttnn.mul(ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16), self.embed_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctx", type=int)
    ap.add_argument("max_seq_len", type=int, nargs="?", default=4096)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--census")
    ap.add_argument("--modes", action="store_true")
    ap.add_argument(
        "--profile-replays", type=int, default=0, help="untimed: N replays, device profiler read after each"
    )
    cli = ap.parse_args()
    ctx, max_seq_len = cli.ctx, cli.max_seq_len
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64_000_000)
    try:
        args = Gemma4AssistantArgs.from_hf_config(Gemma4AssistantArgs.load_hf_config(ASSISTANT))
        args.model_cache_path = args.resolve_model_cache_path(ASSISTANT)
        args.cluster_shape = (1, 1)
        args.max_seq_len = max_seq_len
        text_args = args.text_args
        text_args.max_seq_len = max_seq_len
        mesh_config = MeshConfig(mesh.shape, decode=ModeConfig(tp=1))
        stub = Stub(mesh, mesh_config, text_args, args.backbone_hidden_size, max_seq_len)
        assistant = Gemma4AssistantModel(
            mesh_device=mesh,
            assistant_args=args,
            target_model=stub,
            state_dict=Gemma4AssistantArgs.load_state_dict(ASSISTANT, dummy_weights=False),
            ccl_manager=None,
            dtype=ttnn.bfloat16,
            tensor_cache_path=str(args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 1))),
            mesh_config=mesh_config,
        )

        # Shared KV: cache sized for max_seq_len (as the demo's), first ctx positions filled.
        pac = PagedAttentionConfig(block_size=BLOCK, max_num_blocks=math.ceil(max_seq_len / BLOCK))
        pt = ttnn.from_torch(
            torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, -1),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        type_to_idx = {}
        for i, lt in enumerate(text_args.layer_types):
            type_to_idx.setdefault(lt, i)
        torch.manual_seed(0)
        shared_kv = {}
        for lt, idx in type_to_idx.items():
            cfg = Gemma4AttentionConfig(text_args, idx)
            kvc = init_kv_cache(mesh_device=mesh, config=cfg, paged_attention_config=pac, cache_dtype=ttnn.bfloat16)
            for j in range(2):
                ref = torch.randn(1, cfg.num_key_value_heads, ctx, cfg.head_dim).bfloat16().float()
                fill = _kv_to_tt(ref, mesh, cfg.num_key_value_heads, cfg.num_attention_heads, 1, 1)
                ttnn.experimental.paged_fill_cache(kvc[j], fill, pt, batch_idx=0)
            shared_kv[lt] = kvc
        page_tables = {lt: pt for lt in shared_kv}

        pos = ctx - 1
        token = ttnn.from_torch(
            torch.tensor([[12345]], dtype=torch.int64), device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        )
        pu_host = torch.zeros((1, 32), dtype=torch.int64)
        pu_host[0, 0] = pos
        pu = ttnn.from_torch(pu_host, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
        pi = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32), device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        g = torch.Generator().manual_seed(7)
        hidden = ttnn.from_torch(
            (torch.randn(1, 1, 1, args.backbone_hidden_size, generator=g) * 5.0).bfloat16(),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        spec = SpeculativeDecoder.__new__(SpeculativeDecoder)

        def body(mode):
            tok, h, last = token, hidden, None
            for _ in range(K):
                logits, h = assistant.step(tok, h, shared_kv, page_tables, pu, pi, return_logits=mode != "backbone")
                if mode == "full":
                    last = spec._argmax_last(logits, rows=1)
                    logits.deallocate(True)
                    tok = ttnn.reshape(last, (1, 1))
                elif logits is not None:
                    logits.deallocate(True)
            return last, h

        def timed(mode):
            body(mode)  # compile
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            idx, h = body(mode)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            for _ in range(3):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / REPS * 1e3
            fp = ""
            if mode == "full":
                tok = ttnn.to_torch(idx).flatten()[:1].tolist()
                sha = hashlib.sha256(ttnn.to_torch(h).float().numpy().tobytes()).hexdigest()[:16]
                fp = f" token={tok} hidden_sha={sha}"
            ttnn.release_trace(mesh, tid)
            return ms, fp

        if cli.census:
            body("full")  # compile: the capture below sees cached programs (program.cpp re-reports their CBs)
            ttnn.synchronize_device(mesh)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            body("full")
            graph = ttnn.graph.end_graph_capture()
            rows, op = [], None
            for node in graph:
                t, p = node.get("node_type"), node.get("params", {})
                if t == "function_start":
                    op = p.get("name")
                elif t == "circular_buffer_allocate" and p.get("globally_allocated") == "0":
                    rows.append({"kind": "cb", "op": op, "end": int(p["address"]) + int(p["size"]), **p})
                elif t == "buffer_allocate" and p.get("type") == "L1":
                    rows.append({"kind": "l1", "op": op, **p})
            with open(cli.census, "w") as f:
                json.dump(rows, f, indent=1)
            top = {}
            for r in rows:
                if r["kind"] == "cb":
                    key = (r["op"], r.get("core_range_set"))
                    top[key] = max(top.get(key, 0), r["end"])
            for (name, crs), end in sorted(top.items(), key=lambda kv: -kv[1])[:25]:
                print(f"CENSUS cb_end={end} op={name} cores={crs}", flush=True)
            l1 = sorted({(int(r["address"]), int(r["size"]), r["op"]) for r in rows if r["kind"] == "l1"})
            for addr, size, name in l1[:25]:
                print(f"CENSUS l1_buffer addr={addr} size={size} op={name}", flush=True)
            return
        if cli.profile_replays:  # per-op device profile of traced replays (run under tracy; record §8)
            body("full")
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            idx, h = body("full")
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            for _ in range(cli.profile_replays):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(mesh)
            tok = ttnn.to_torch(idx).flatten()[:1].tolist()
            sha = hashlib.sha256(ttnn.to_torch(h).float().numpy().tobytes()).hexdigest()[:16]
            print(f"PROFILED replays={cli.profile_replays} token={tok} hidden_sha={sha}", flush=True)
            ttnn.release_trace(mesh, tid)
            return
        for rnd in range(cli.rounds):
            ms, fp = timed("full")
            print(
                f"RESULT ctx={ctx} msl={max_seq_len} round={rnd} mode=full ms_per_draft={ms:.4f} "
                f"ms_per_step={ms / K:.4f}{fp}",
                flush=True,
            )
        for mode in ("no_argmax", "backbone") if cli.modes else ():
            ms, _ = timed(mode)
            print(
                f"RESULT ctx={ctx} msl={max_seq_len} mode={mode} ms_per_draft={ms:.4f} ms_per_step={ms / K:.4f}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
