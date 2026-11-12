"""pytest plugin: which L1 buffers are live at each SDPA of the first target verify (record 11.3 follow-up).

    PYTHONPATH=research_codes/mm_profiling:... pytest -p l1_live_plugin ... (L1_LIVE_OUT=<json path>)

Wraps the first SpeculativeDecoder._verify call in a ttnn graph capture and walks the nodes in order, keeping the
set of L1 buffers allocated and not yet freed inside the capture. At every SDPA decode call it records that set
(address, size, layout, the op that allocated it). Buffers allocated before the verify are not seen: they are
reported only through the lowest address in the clash error.
"""

import json
import os


def pytest_configure(config):
    import ttnn
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

    orig = SpeculativeDecoder._verify
    state = {"done": False}

    def wrapped(self, *args, **kwargs):
        if state["done"]:
            return orig(self, *args, **kwargs)
        state["done"] = True
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            return orig(self, *args, **kwargs)
        finally:
            graph = ttnn.graph.end_graph_capture()
            live, snaps, stack = {}, [], []
            for node in graph:
                t, p = node.get("node_type"), node.get("params", {})
                if t == "function_start":
                    stack.append(p.get("name"))
                    if "scaled_dot_product_attention_decode" in (p.get("name") or ""):
                        snaps.append(
                            {"op": p.get("name"), "live_l1": sorted(live.values(), key=lambda b: b["address"])}
                        )
                elif t == "function_end":
                    if stack:
                        stack.pop()
                elif t == "buffer_allocate" and p.get("type") == "L1":
                    live[p["address"]] = {
                        "address": int(p["address"]),
                        "size": int(p["size"]),
                        "max_size_per_bank": int(p.get("max_size_per_bank", 0)),
                        "layout": p.get("layout"),
                        "num_cores": p.get("num_cores"),
                        "ops": list(stack),
                    }
                elif t == "buffer_deallocate" and p.get("type") == "L1":
                    live.pop(p.get("address"), None)
            out = os.environ.get("L1_LIVE_OUT", "l1_live.json")
            with open(out, "w") as f:
                json.dump(snaps, f, indent=1)
            for s in snaps:
                low = s["live_l1"][:6]
                print(
                    f"L1LIVE {s['op']}: {len(s['live_l1'])} live L1 buffers; lowest: "
                    + "; ".join(
                        f"{b['address']} +{b['max_size_per_bank']}/bank ({b['layout']}, {b['ops'][-1] if b['ops'] else '?'})"
                        for b in low
                    ),
                    flush=True,
                )

    SpeculativeDecoder._verify = wrapped
