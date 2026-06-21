"""
A = QR where R is upper triangular and Q is orthogonal
Q is product of Householder reflectors, each of the form I - tau.v.v^T
each reflector zeroes out everything below diagonal in one column
"""

import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _panel_factor_kernel(
    A_ptr,
    tau_ptr,
    n,
    p0,
    pb,
    stride_ab,
    stride_ar,
    stride_ac,
    stride_tb,
    stride_tc,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    A = A_ptr + bid * stride_ab
    T = tau_ptr + bid * stride_tb
    rows = tl.arange(0, BLOCK_N)
    for jj in range(0, pb):
        col = p0 + jj
        active = (rows >= col) & (rows < n)
        xptr = A + rows * stride_ar + col * stride_ac
        x = tl.load(xptr, mask=active, other=0.0)

        below0 = active & (rows > col)
        tail_sq = tl.sum(tl.where(below0, x * x, 0.0), axis=0)
        x0 = tl.sum(tl.where(rows == col, x, 0.0), axis=0)

        has_refl = tail_sq > 0.0

        xnorm = tl.sqrt(x0 * x0 + tail_sq)
        sign = tl.where(x0 >= 0.0, 1.0, -1.0)
        alpha_full = -sign * xnorm
        alpha = tl.where(has_refl, alpha_full, x0)
        beta = x0 - alpha
        safe = has_refl & (beta != 0.0)
        inv_beta = tl.where(safe, 1.0 / beta, 0.0)

        v = x * inv_beta
        v = tl.where(rows == col, 1.0, v)
        v = tl.where(active & safe, v, 0.0)
        tau = tl.where(safe, -beta / alpha, 0.0)
        tl.store(T + col * stride_tc, tau)

        below = active & (rows > col)
        tl.store(xptr, alpha, mask=(rows == col))
        tl.store(xptr, v, mask=below)

        p_end = p0 + pb

        # can i use some gpu primitive for this?
        # why am i doing sequential?
        # questions:
        # can i update all remaining panel columns in parallel?
        # what primite exists to exploit this?
        for cc in range(col + 1, p_end):
            do = cc < n
            aptr = A + rows * stride_ar + cc * stride_ac
            a = tl.load(aptr, mask=active & do, other=0.0)
            dot = tl.sum(v * a, axis=0)
            a = a - v * (tau * dot)
            tl.store(aptr, a, mask=below | (rows == col))


# boilerplate wrappers, no perf hacks here to make from my side here
def factorPanel(A: torch.Tensor, p0: int, pb: int, tau: torch.Tensor):
    batch, n, _ = A.shape
    BLOCK_N = triton.next_power_of_2(n)
    _panel_factor_kernel[(batch,)](
        A,
        tau,
        n,
        p0,
        pb,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        tau.stride(0),
        tau.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )


def _build_T(V: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    batch, m, pb = V.shape
    dev, dt = V.device, V.dtype
    T = torch.zeros((batch, pb, pb), device=dev, dtype=dt)
    VtV = V.transpose(-1, -2) @ V
    T[:, 0, 0] = taus[:, 0]
    for j in range(1, pb):

        w = VtV[:, :j, j]
        # i see torch and i feel sus
        # are we leaving some perf out here?
        # TODO: investigate
        z = -taus[:, j].unsqueeze(-1) * torch.einsum("bik,bk->bi", T[:, :j, :j], w)

        T[:, :j, j] = z
        T[:, j, j] = taus[:, j]
    return T


def qr(A, block):
    # (batch, rows, columns) but rows == colunms
    batch, n, _ = A.shape

    # we are going to modify the numbers in place
    H = A.clone()

    # its to hold the zeroes
    tau = torch.zeros((batch, n), device=A.device, dtype=A.dtype)

    # shifting window with the columns
    # like column0, column1 .. column32 -> we analyse this at once
    for p0 in range(0, n, block):
        # this is the protection mechanism
        # 100 / 32 is not a whole number after all
        pb = min(block, n - p0)

        # this kicks off the triton kernel
        # takes the current panel (p0, pb)
        # runs fast sequential loop entirely inside teh GPU's SRAM
        # once done, the chunk of H contains R on its diagonal and the mirror vectors v packed below it
        factorPanel(H, p0, pb, tau)

        c_end = p0 + pb
        if c_end >= n:
            continue

        # ai generated section:
        # idea is to extract our mirror vectors out of their
        # compactly kept places inside R
        # FLAG: possible performance hacks can come from here

        V = H[
            :, p0:, p0:c_end
        ].clone()  # this is a massive allocation, can i do something else?
        # this is getting allocated in the HBM, so that is nullyfying my idea of compacting the storage!
        m = V.shape[1]
        ridx = torch.arange(m, device=A.device)
        col = torch.arange(pb, device=A.device)

        # masking hmm
        # what if instead of faling back to Pytorch, i write another kernel
        # to get rid of a new allocation?
        V = V * (ridx[:, None] >= col[None, :])
        V[:, col, col] = 1.0

        # to handle adversarial rank deficient matrices
        # if a column that was already completely zeroed out, it flagged
        # its tau = 0 -> entire mirror vector to be zero
        zero_cols = (tau[:, p0:c_end] == 0.0).unsqueeze(1)
        V = torch.where(zero_cols, torch.zeros_like(V), V)

        # Atrailing <- Atrailing - V(T^T)(V^T)Atrailing

        # bulids the upper triangular T matrix for this specific block
        Tb = _build_T(V, tau[:, p0:c_end])

        # A_tr = H[:, p0:, c_end:]

        # this is a deliberate choice from my side
        # trying to bracket from right to left such that the GPU never has to
        # hold matrices of huge sizes, a tiny small buffer in registers is
        # all that is needed
        C = H[:, p0:, c_end:]
        # W = V.transpose(-1, -2) @ A_tr
        W = V.transpose(-1, -2) @ C
        W = Tb.transpose(-1, -2) @ W
        # H[:, p0:, c_end:] = A_tr - V @ W
        # C = H[:, p0:, c_end:]
        C.baddbmm_(V, W, beta=1, alpha=-1)

    return H, tau


def custom_kernel(data: input_t) -> output_t:
    # this is the tensor (batch, row, column)
    A = data
    n = A.shape[-1]
    block = 64 if n >= 256 else 32
    return qr(A.contiguous(), block)
