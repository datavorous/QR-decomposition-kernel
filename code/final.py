import torch
import triton
import triton.language as tl

from task import input_t, output_t

USE_TWOLEVEL_FOR = set()
_MANT_MASK = ~((1 << 13) - 1)


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


@triton.jit
def _wy_update_tf32(
    V,
    T,
    C,
    M,
    IB,
    NCOL,
    svb,
    svr,
    svc,
    sTb,
    sTr,
    sTc,
    scb,
    scr,
    scc,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BIB: tl.constexpr,
):
    b = tl.program_id(0)
    jt = tl.program_id(1)
    cols = jt * BN + tl.arange(0, BN)
    cmask = cols < NCOL
    ic = tl.arange(0, BIB)
    icm = ic < IB
    Tt = tl.load(
        T + b * sTb + ic[:, None] * sTr + ic[None, :] * sTc,
        mask=icm[:, None] & icm[None, :],
        other=0.0,
    )
    W = tl.zeros((BIB, BN), dtype=tl.float32)
    for r0 in tl.range(0, M, BM):
        rows = r0 + tl.arange(0, BM)
        rmask = rows < M
        Vt = tl.load(
            V + b * svb + rows[:, None] * svr + ic[None, :] * svc,
            mask=rmask[:, None] & icm[None, :],
            other=0.0,
        )
        Ct = tl.load(
            C + b * scb + rows[:, None] * scr + cols[None, :] * scc,
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )
        W = tl.dot(tl.trans(Vt), Ct, acc=W, input_precision="tf32x3")
    W = -tl.dot(Tt, W, input_precision="tf32x3")
    for r0 in tl.range(0, M, BM):
        rows = r0 + tl.arange(0, BM)
        rmask = rows < M
        Vt = tl.load(
            V + b * svb + rows[:, None] * svr + ic[None, :] * svc,
            mask=rmask[:, None] & icm[None, :],
            other=0.0,
        )
        Cp = C + b * scb + rows[:, None] * scr + cols[None, :] * scc
        Ct = tl.load(Cp, mask=rmask[:, None] & cmask[None, :], other=0.0)
        Ct = tl.dot(Vt, W, acc=Ct, input_precision="tf32x3")
        tl.store(Cp, Ct, mask=rmask[:, None] & cmask[None, :])


def _launch_wy_tf32(
    Vb, Tt, C, B, M, ib, NCOL, BIB, BM, BN=64, num_warps=4, num_stages=2
):
    grid = (B, triton.cdiv(NCOL, BN))
    TtT = Tt.transpose(-1, -2)
    _wy_update_tf32[grid](
        Vb,
        TtT,
        C,
        M,
        ib,
        NCOL,
        Vb.stride(0),
        Vb.stride(1),
        Vb.stride(2),
        TtT.stride(0),
        TtT.stride(1),
        TtT.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        BM=BM,
        BN=BN,
        BIB=BIB,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _tf32_hi(x):
    return (x.view(torch.int32) & _MANT_MASK).view(torch.float32)


def _mm3(A, B):
    Ah = _tf32_hi(A)
    Al = A - Ah
    Bh = _tf32_hi(B)
    Bl = B - Bh
    out = torch.bmm(Ah, Bh)
    out = torch.baddbmm(out, Ah, Bl)
    out = torch.baddbmm(out, Al, Bh)
    return out


def qr_single(A, block, num_warps):
    B, m, n = A.shape
    bs = int(block)
    BNB = triton.next_power_of_2(bs)
    use_fused = n != 512
    H = A.clone()
    tau = A.new_zeros(B, n)
    for k in range(0, n, bs):
        ib = min(bs, n - k)
        BM = triton.next_power_of_2(m - k)
        Hv = H[:, k:, k : k + ib]
        Tt = A.new_zeros(B, BNB, BNB)
        ts = A.new_zeros(B, BNB)
        Vb = A.new_zeros(B, m - k, ib)
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
        if torch.count_nonzero(ts[:, :ib]) == 0:
            break
        hi = k + ib
        if hi < n:
            C = H[:, k:, hi:]
            NCOL = n - hi
            if use_fused:
                trailing_m = m - k
                BM_wy = triton.next_power_of_2(min(128, trailing_m))
                _launch_wy_tf32(
                    Vb, Tt, C, B, trailing_m, ib, NCOL, BNB, BM=BM_wy, BN=64
                )
            else:
                V = Vb
                T = Tt[:, :ib, :ib]
                W = V.transpose(-1, -2) @ C
                W = T.transpose(-1, -2) @ W
                C.baddbmm_(V, W, beta=1, alpha=-1)
    return H, tau


def qr_twolevel(A, ib, NB, num_warps):
    B, m, n = A.shape
    ib = int(ib)
    NB = int(NB)
    BNB_i = triton.next_power_of_2(ib)
    H = A.clone()
    tau = A.new_zeros(B, n)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for k0 in range(0, n, NB):
            nb = min(NB, n - k0)
            V_outer = A.new_zeros(B, m - k0, nb)
            inner_T = []
            inner_off = []
            off = 0
            while off < nb:
                cib = min(ib, nb - off)
                kk = k0 + off
                BM_p = triton.next_power_of_2(m - kk)
                Hv = H[:, kk:, kk : kk + cib]
                Tt = A.new_zeros(B, BNB_i, BNB_i)
                ts = A.new_zeros(B, BNB_i)
                Vb = A.new_zeros(B, m - kk, cib)
                _panel_kernel[(B,)](
                    Hv,
                    ts,
                    Tt,
                    Vb,
                    m - kk,
                    cib,
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
                    BM=BM_p,
                    BNB=BNB_i,
                    num_warps=num_warps,
                )
                tau[:, kk : kk + cib] = ts[:, :cib]
                V_outer[:, off:, off : off + cib] = Vb
                blk_end = k0 + nb
                hi_in = kk + cib
                if hi_in < blk_end:
                    Cn = H[:, kk:, hi_in:blk_end]
                    tm = m - kk
                    BM_wy = triton.next_power_of_2(min(128, tm))
                    _launch_wy_tf32(
                        Vb, Tt, Cn, B, tm, cib, blk_end - hi_in, BNB_i, BM=BM_wy, BN=64
                    )
                inner_T.append(Tt[:, :cib, :cib].clone())
                inner_off.append((off, cib))
                off += cib
            G = _mm3(V_outer.transpose(-1, -2).contiguous(), V_outer)
            T_outer = A.new_zeros(B, nb, nb)
            for (o, c), Tj in zip(inner_off, inner_T):
                if o > 0:
                    T_outer[:, :o, o : o + c] = (
                        -(T_outer[:, :o, :o] @ G[:, :o, o : o + c]) @ Tj
                    )
                T_outer[:, o : o + c, o : o + c] = Tj
            hi = k0 + nb
            if hi < n:
                Cbig = H[:, k0:, hi:]
                W = _mm3(V_outer.transpose(-1, -2).contiguous(), Cbig)
                W = _mm3(T_outer.transpose(-1, -2).contiguous(), W)
                Cbig.sub_(_mm3(V_outer, W))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return H, tau


def custom_kernel(data: input_t) -> output_t:
    A = data
    n = A.shape[-1]
    if n > 2048:
        return torch.geqrf(A.contiguous())

    if n in USE_TWOLEVEL_FOR:
        ib = 16 if n >= 1024 else 32
        return qr_twolevel(A.contiguous(), ib, 128, num_warps=8)

    if n >= 1024:
        block, nw = 16, 8
    elif n >= 256:
        block, nw = 32, (4 if n == 512 else 8)
    else:
        block, nw = 32, 4
    return qr_single(A.contiguous(), block, nw)
