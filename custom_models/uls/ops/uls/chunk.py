import warnings

import torch

from fla.modules.l2norm import l2norm_fwd, l2norm_bwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# ASSIGN COEFFICIENT = ETA/(1-ETA*BETA)

def chunk_uls_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor, 
    g: torch.Tensor,
    lamda: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    beta = eta / (1 - eta * lamda)
    g = chunk_local_cumsum(
        g,
        chunk_size = 64,
        cu_seqlens = cu_seqlens
    )
    # WY REPRESENTATIONS
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A = A,
        cu_seqlens = cu_seqlens,
        output_dtype = k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k = k,
        v = v,
        beta = beta,
        A = A,
        g = g,
        cu_seqlens = cu_seqlens
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k = k,
        w = w,
        u = u,
        g = g,
        initial_state = initial_state,
        output_final_state = output_final_state,
        cu_seqlens = cu_seqlens
    )
    o = chunk_fwd_o(
        q = q,
        k = k,
        v = v_new,
        h = h,
        g = g,
        scale = scale,
        cu_seqlens = cu_seqlens
    )
    return g, o, A, final_state

def chunk_uls_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    lamda: torch.Tensor,
    eta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
):
    beta = eta / (1 - eta * lamda)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens
    )

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens
    )

    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens
    )

    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens
    )

    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens
    )

    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    
    return dq, dk, dv, db, dg, dh0

class ChunkULSFunction(torch.autograd.Function):
    
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        lamda: torch.Tensor,
        eta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q_rstd, k_rtsd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rtsd = l2norm_fwd(k)
        
        g, o, A, final_state = chunk_uls_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            lamda=lamda,
            eta=eta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens
        )
        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rtsd,
            v,
            g,
            lamda,
            eta,
            A, 
            initial_state,
            cu_seqlens
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state
    
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, lamda, eta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_uls_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            lamda=lamda,
            eta=eta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # calculate the gradient via the chain rule
        denom = 1.0 - eta * lamda
        denom = denom ** 2
        deta = db / denom
        dlamda = db * (eta ** 2) / denom

        
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), dlamda.to(lamda), deta.to(eta), None, dh0, None, None, None


@torch.compiler.disable
def chunk_uls_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    lamda: torch.Tensor,
    eta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkULSFunction.apply(
        q,
        k,
        v,
        g,
        lamda,
        eta,
        scale,
        initial_state,
        output_final_state, 
        cu_seqlens,
        use_qk_l2norm_in_kernel
    )
    return o, final_state

