import time
from typing import NamedTuple

import torch
import torch.utils.benchmark as benchmark
from einops import rearrange
from triton.testing import do_bench
from flash_attn import flash_attn_func, flash_attn_varlen_func
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.version.cuda:
    backendBLAS = "cuBLAS"
elif torch.version.hip:
    backendBLAS = "hipBLAS"

torch.manual_seed(42)

Timing = NamedTuple("timing", [("mean", float)])

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def get_causal_mask(q, k):
    B, h, Lq, _ = q.shape
    _, _, Lk, _ = k.shape
    causal_mask = torch.full((Lq, Lk), float('-inf'), device=q.device, dtype=q.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # upper triangle
    return causal_mask  # shape: (Lq, Lk)


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    # # Warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # time.sleep(1)
    # return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for _ in range(2):
    #         out = func(*args, **kwargs)
    # torch.cuda.current_stream().wait_stream(s)
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     out = func(*args, **kwargs)
    # time_f = benchmark_forward(lambda: graph.replay(), repeats=repeats, verbose=verbose, desc=desc)
    # # return time_f[1].mean
    # return time_f[1]
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=10, rep=repeats))  # ms


def benchmark_forward(fn, *inputs, repeats=10, desc='', verbose=True, **kwinputs):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
        stmt='fn(*inputs, **kwinputs)',
        globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def test_gemm():
    repeats = 50
    dtype = torch.bfloat16
    device = "cuda"
    verbose = False
    m, n = 8192, 8192

    tflops_matmul = {}
    tflops_matmul1 = {}
    for k in [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]:
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(n, k, device=device, dtype=dtype).transpose(-1, -2)
        nFLOPS_matmul = 2 * m * n * k
        time.sleep(2)  # to reduce power throttling
        timing = benchmark_forward(torch.matmul, a, b, desc=backendBLAS, verbose=verbose, repeats=repeats)[1]
        tflops_matmul[k] = nFLOPS_matmul / timing.mean * 1e-12
        print(f"[torch.utils.benchmark] {backendBLAS}, {m = }, {n = }, {k = }: {timing.mean * 1e3:.3f}ms, {tflops_matmul[k]:.1f} TFLOPS")
        time.sleep(2)  # to reduce power throttling
        ms = do_bench(lambda: torch.matmul(a, b), warmup=10, rep=repeats)
        tflops_matmul1[k] = nFLOPS_matmul / ms * 1e-9
        print(f"[triton.test.do_bench]  {backendBLAS}, {m = }, {n = }, {k = }: {ms:.3f}ms, {tflops_matmul1[k]:.1f} TFLOPS")


def test_fa():
    repeats = 50
    dropout_p = 0.0
    dtype = torch.bfloat16
    # dtype = torch.float16
    # dtype = torch.float8_e4m3fn
    dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    device = "cuda"
    verbose = True
    varlen = False
    has_backward = False
    page_size = None
    softcap = 0.0
    dim = 2048

    bs_seqlen_vals = [
        (128, 512),
        (128, 2048),
        (128, 4096),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (32, 4096),
        (16, 512),
        (16, 1024),
        (16, 2048),
        (8, 512),
        (8, 1024),
        (8, 2048),
        (8, 4096),
        (4, 512),
        (4, 1024),
        (4, 2048),
        (4, 4096),
        (4, 8192),
        (2, 16384),
        (1, 32768)
    ]
    # bs_seqlen_vals = [(32, 512), (16, 1024)]
    # bs_seqlen_vals = [(2, 64 * 132)]
    # bs_seqlen_vals = [(4, 8192)]
    # bs_seqlen_vals = [(1, 16 * 1024)]
    time_f = {}
    time_b = {}

    # for headdim in [64, 128, 256]:
    # for headdim in [64, 96, 128, 192]:
    # for headdim in [64, 96, 128, 192, 256]:
    # for headdim in [64, 96, 128]:
    # for headdim in [64, 128, 256]:
    # for headdim in [64, 96, 128, 192, 256]:

    print(f"{'':>40s}{'sdpa':>20s}{'fa2':>20s}")
    print("-" * 80)
    for headdim in [128]:
        nheads = dim // headdim
        nheads_kv = nheads

        for batch_size, seqlen in bs_seqlen_vals:
            window_size = (-1, -1)
            # window_size = (seqlen // 2 - 1, 0)
            seqlen_q = seqlen

            q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen)
            k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen)
            v = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen)

            if varlen:
                q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_(has_backward) for x in [q, k, v]]
                cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
                cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
                # cu_seqlens_q = torch.tensor([0, 248, 249, 250, 251, 252, 253, 254, 255, 256], device=device, dtype=torch.int32)
                # q_unpad = q_unpad[:256]
                # seqlen_q = 256
                # cu_seqlens_q = torch.tensor([0, 376, 377, 378, 379, 380, 381, 382, 383, 384], device=device, dtype=torch.int32)
                # q_unpad = q_unpad[:384]
                # seqlen_q = 384
            if page_size is not None:
                assert seqlen % page_size == 0
                k_paged, v_paged = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]]
                page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                    "(b s) -> b s", s=seqlen // page_size)
            else:
                page_table = None

            attn_mask = get_causal_mask(q, k)
            m1 = time_fwd(
                torch.nn.functional.scaled_dot_product_attention,
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                scale=1.0,
                is_causal=True,
                repeats=repeats,
                verbose=verbose,
                desc="sdpa",
            )
            # out_attn0 = torch.nn.functional.scaled_dot_product_attention(
            #     q,
            #     k,
            #     v,
            #     attn_mask=attn_mask,
            #     dropout_p=dropout_p,
            #     scale=1.0,
            #     is_causal=True
            # )  # (bsz, seqlen, nheads, headdim)
            # print(f"[sdpa] bs = {batch_size}, seqlen = {seqlen}, headdim = {headdim}, time = {m1.mean * 1e3:.3f}ms")

            # out_attn1 = flash_attn_func(q, k, v, dropout_p, window_size=window_size, softmax_scale=1.0, softcap=softcap, causal=True)  # (bsz, seqlen, nheads, headdim)
            # print(torch.allclose(out_attn0, out_attn1, rtol=1e-4))
            if not varlen:
                m0 = time_fwd(
                    flash_attn_func, q, k, v, dropout_p, window_size=window_size, softmax_scale=1.0, softcap=softcap, causal=True,
                    repeats=repeats, verbose=verbose, desc="Fav2"
                )
            else:
                m0 = time_fwd(
                    flash_attn_varlen_func, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, dropout_p, window_size=window_size, softmax_scale=1.0, softcap=softcap, causal=True,
                    repeats=repeats, verbose=verbose, desc="Fav2"
                )
            # print(f"[fa2] bs = {batch_size}, seqlen = {seqlen}, headdim = {headdim}, time = {m0.mean * 1e3:.3f}ms")

            st = f"bs = {batch_size}, seqlen = {seqlen}, headdim = {headdim}"
            print(f"{st:>40s}{m1.mean:>18.5f}ms{m0.mean:>18.5f}ms")


test_fa()
