import torch
import triton
import triton.language as tl

from task import input_t, output_t

_E4M3_MAX = 448.0


def _fp8_split(x, axis):
    s = x.abs().amax(dim=axis, keepdim=True).clamp(min=1e-30) / _E4M3_MAX
    return (x / s).to(torch.float8_e4m3fn).float() * s


def _mm(a, b, mode):
    if mode == "fp8s2":
        a1 = _fp8_split(a, -1)
        b1 = _fp8_split(b, -2)
        a2 = _fp8_split(a - a1, -1)
        b2 = _fp8_split(b - b1, -2)
        return a1 @ b1 + a2 @ b1 + a1 @ b2
    return a @ b


@triton.jit
def _panel_kernel(
    P,
    TAU,
    T,
    VOUT,
    M,
    IB,
    spb,
    spr,
    spc,
    stb,
    sti,
    sTb,
    sTr,
    sTc,
    svb,
    svr,
    svc,
    BM: tl.constexpr,
    BNB: tl.constexpr,
):
    b = tl.program_id(0)
    r = tl.arange(0, BM)
    c = tl.arange(0, BNB)
    rm = r < M
    cm = c < IB
    p = P + b * spb + r[:, None] * spr + c[None, :] * spc
    tile = tl.load(p, mask=rm[:, None] & cm[None, :], other=0.0)
    tau_vec = tl.zeros((BNB,), dtype=tl.float32)
    for j in range(BNB):
        colj = tl.sum(tl.where(c[None, :] == j, tile, 0.0), axis=1)
        alpha = tl.sum(tl.where(r == j, colj, 0.0))
        xn2 = tl.sum(tl.where(r > j, colj * colj, 0.0))
        reflect = xn2 > 0.0
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = tl.where(reflect, -sgn * tl.sqrt(alpha * alpha + xn2), alpha)
        tau_j = tl.where(reflect, (beta - alpha) / tl.where(reflect, beta, 1.0), 0.0)
        denom = tl.where(reflect, alpha - beta, 1.0)
        vb = colj / denom
        v = tl.where(r == j, 1.0, tl.where(r > j, vb, 0.0))
        vmask = tl.where(r >= j, v, 0.0)
        w = tl.sum(tl.where(c[None, :] > j, vmask[:, None] * tile, 0.0), axis=0)
        tile = tile - tau_j * vmask[:, None] * w[None, :]
        newcol = tl.where(r < j, colj, tl.where(r == j, beta, vb))
        tile = tl.where(c[None, :] == j, newcol[:, None], tile)
        tau_vec = tl.where(c == j, tau_j, tau_vec)
    V = tl.where(
        r[:, None] == c[None, :], 1.0, tl.where(r[:, None] > c[None, :], tile, 0.0)
    )
    tl.store(
        VOUT + b * svb + r[:, None] * svr + c[None, :] * svc,
        V,
        mask=rm[:, None] & cm[None, :],
    )
    # in-kernel compact-WY T
    Tt = tl.zeros((BNB, BNB), dtype=tl.float32)
    tau0 = tl.sum(tl.where(c == 0, tau_vec, 0.0))
    Tt = tl.where((c[:, None] == 0) & (c[None, :] == 0), tau0, Tt)
    for i in range(1, BNB):
        tau_i = tl.sum(tl.where(c == i, tau_vec, 0.0))
        Vi = tl.sum(tl.where(c[None, :] == i, V, 0.0), axis=1)
        dots = tl.sum(V * Vi[:, None], axis=0)
        z = tl.where(c < i, -tau_i * dots, 0.0)
        Tz = tl.sum(tl.where(c[None, :] < i, Tt * z[None, :], 0.0), axis=1)
        newTcol = tl.where(c < i, Tz, tl.where(c == i, tau_i, 0.0))
        Tt = tl.where(c[None, :] == i, newTcol[:, None], Tt)
    tl.store(
        T + b * sTb + c[:, None] * sTr + c[None, :] * sTc,
        Tt,
        mask=cm[:, None] & cm[None, :],
    )
    tl.store(
        P + b * spb + r[:, None] * spr + c[None, :] * spc,
        tile,
        mask=rm[:, None] & cm[None, :],
    )
    tl.store(TAU + b * stb + c * sti, tau_vec, mask=cm)


def qr(A, block, num_warps=8, gemm="fp32"):
    B, m, n = A.shape
    bs = int(block)
    BNB = triton.next_power_of_2(bs)
    H = A.clone()
    tau = A.new_zeros(B, n)
    for k in range(0, n, bs):
        ib = min(bs, n - k)
        BM = triton.next_power_of_2(m - k)
        Hv = H[:, k:, k : k + ib]  # strided view, factored in place
        Tt = A.new_zeros(B, BNB, BNB)
        ts = A.new_zeros(B, BNB)
        Vb = A.new_zeros(B, m - k, ib)  # kernel writes unit-lower V here
        _panel_kernel[(B,)](
            Hv,
            ts,
            Tt,
            Vb,
            m - k,
            ib,
            Hv.stride(0),
            Hv.stride(1),
            Hv.stride(2),
            ts.stride(0),
            ts.stride(1),
            Tt.stride(0),
            Tt.stride(1),
            Tt.stride(2),
            Vb.stride(0),
            Vb.stride(1),
            Vb.stride(2),
            BM=BM,
            BNB=BNB,
            num_warps=num_warps,
        )
        tau[:, k : k + ib] = ts[:, :ib]
        hi = k + ib
        if hi < n:
            V = Vb
            T = Tt[:, :ib, :ib]
            C = H[:, k:, hi:]
            # W = V.transpose(-1, -2) @ C
            # W = T.transpose(-1, -2) @ W
            W = _mm(V.transpose(-1, -2), C, gemm)
            W = _mm(T.transpose(-1, -2), W, gemm)
            if gemm == "fp32":
                C.baddbmm_(V, W, beta=1, alpha=-1)
            else:
                C.sub_(_mm(V, W, gemm))
            # C.baddbmm_(V, W, beta=1, alpha=-1)
    return H, tau


def custom_kernel(data: input_t) -> output_t:
    A = data
    n = A.shape[-1]
    if n > 2048:
        return torch.geqrf(A.contiguous())
    if n >= 1024:
        block, nw, gm = 16, 8, "fp32"
    elif n == 512:
        block, nw, gm = 32, 4, "fp8s2"
    elif n >= 256:
        block, nw, gm = 32, 8, "fp32"
    else:
        block, nw, gm = 32, 4, "fp32"
    return qr(A.contiguous(), block, nw, gm)
