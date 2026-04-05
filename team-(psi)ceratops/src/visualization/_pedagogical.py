from __future__ import annotations

import warnings
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

from src.cat_qubit import (
    CatQubitParams,
    build_hamiltonian,
    build_jump_ops,
    build_operators,
    compute_alpha,
)

# ---------------------------------------------------------------------------
# Challenge-style Wigner animations (replicating 1-challenge.ipynb)
# ---------------------------------------------------------------------------


def animate_simple_two_photon(
    alpha: float = 2.0,
    kappa_2: float = 1.0,
    na: int = 15,
    tfinal: float = 3.0,
    nframes: int = 200,
    save_path: str = "figures/challenge_simple_two_photon.mp4",
    fps: int = 10,
) -> str:
    """Wigner animation: cat state formation via simple two-photon dissipation.

    Replicates challenge notebook cell 29. Single-mode system with
    L = sqrt(kappa_2) * (a^2 - alpha^2 * I), starting from vacuum.
    Shows vacuum evolving to odd cat state.

    Ref: Challenge notebook, Cell 29.

    Parameters
    ----------
    alpha : float
        Target cat size.
    kappa_2 : float
        Two-photon loss rate [MHz].
    na : int
        Hilbert space truncation.
    tfinal : float
        Simulation duration [us].
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 path.
    fps : int
        Frames per second.

    Returns
    -------
    str
        Path to saved animation.
    """
    a = dq.destroy(na)
    psi0 = dq.fock(na, 0)
    H = dq.zeros(na)
    loss_op = jnp.sqrt(kappa_2) * (a @ a - alpha**2 * dq.eye(na))

    tsave = jnp.linspace(0, tfinal, nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        result = dq.mesolve(
            H,
            [loss_op],
            psi0,
            tsave,
            options=dq.Options(progress_meter=False, save_states=True),
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update(frame: int) -> None:
        ax.cla()
        dq.plot.wigner(result.states[frame], ax=ax)
        ax.set_title(
            rf"Two-Photon Dissipation | $\alpha$={alpha} | "
            f"t = {float(tsave[frame]):.2f} μs"
        )

    ani = FuncAnimation(fig, update, frames=nframes)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    print(f"[viz] Simple two-photon animation: {save_path}")
    return save_path


def animate_driven_two_photon(
    eps_2: float = 2.0,
    kappa_2: float = 1.0,
    kappa_a: float = 0.0,
    na: int = 15,
    tfinal: float = 3.0,
    nframes: int = 200,
    save_path: str = "figures/challenge_driven_two_photon.mp4",
    fps: int = 10,
) -> str:
    """Wigner animation: decomposed two-photon drive + dissipation.

    Replicates challenge notebook cell 31. Single-mode system with
    H = i*eps_2*(a†^2 - a^2) and L = sqrt(kappa_2)*a^2, starting from vacuum.

    Ref: Challenge notebook, Cell 31.

    Parameters
    ----------
    eps_2 : float
        Two-photon drive strength [MHz].
    kappa_2 : float
        Two-photon dissipation rate [MHz].
    kappa_a : float
        Single-photon loss rate [MHz]. Default 0 (off).
    na : int
        Hilbert space truncation.
    tfinal : float
        Simulation duration [us].
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 path.
    fps : int
        Frames per second.

    Returns
    -------
    str
        Path to saved animation.
    """
    a = dq.destroy(na)
    psi0 = dq.fock(na, 0)

    alpha_est = jnp.sqrt(2 / kappa_2 * (eps_2 - kappa_a / 4))

    H = 1j * eps_2 * a.dag() @ a.dag() - 1j * jnp.conj(eps_2) * a @ a
    loss_a2 = jnp.sqrt(kappa_2) * a @ a
    loss_a = jnp.sqrt(kappa_a) * a
    jump_ops = [loss_a2, loss_a]

    tsave = jnp.linspace(0, tfinal, nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        result = dq.mesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            options=dq.Options(progress_meter=False, save_states=True),
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update(frame: int) -> None:
        ax.cla()
        dq.plot.wigner(result.states[frame], ax=ax)
        ax.set_title(
            rf"Drive + Dissipation | $\varepsilon_2$={eps_2}, "
            rf"$\kappa_2$={kappa_2} | $\alpha_{{est}}$={float(alpha_est):.2f} | "
            f"t = {float(tsave[frame]):.2f} μs"
        )

    ani = FuncAnimation(fig, update, frames=nframes)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    print(f"[viz] Driven two-photon animation: {save_path}")
    return save_path


def animate_full_cat_qubit(
    g2_re: float = 1.0,
    g2_im: float = 0.0,
    eps_d_re: float = 4.0,
    eps_d_im: float = 0.0,
    kappa_b: float = 10.0,
    kappa_a: float = 0.0,
    na: int = 15,
    nb: int = 5,
    tfinal: float = 3.0,
    nframes: int = 200,
    save_path: str = "figures/challenge_full_cat_qubit.mp4",
    fps: int = 10,
    init_fock: int = 1,
) -> str:
    """Wigner animation: full storage+buffer Hamiltonian.

    Replicates challenge notebook cell 33. Two-mode system with the complete
    two-photon exchange Hamiltonian and buffer drive, kappa_a=0 by default.
    Starts from |init_fock>_a|0>_b and partial-traces to show storage Wigner.

    Ref: Challenge notebook, Cell 33. Berdou et al. (2022), arXiv:2204.09128.

    Parameters
    ----------
    g2_re, g2_im : float
        Two-photon coupling.
    eps_d_re, eps_d_im : float
        Buffer drive amplitude.
    kappa_b : float
        Buffer loss rate [MHz].
    kappa_a : float
        Storage loss rate [MHz]. Default 0 (off).
    na, nb : int
        Hilbert space dimensions.
    tfinal : float
        Simulation duration [us].
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 path.
    fps : int
        Frames per second.
    init_fock : int
        Initial Fock state for storage mode. Default 1 (|1>).

    Returns
    -------
    str
        Path to saved animation.
    """
    params = CatQubitParams(na=na, nb=nb, kappa_b=kappa_b, kappa_a=kappa_a)
    a, b = build_operators(params)
    H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
    jump_ops_list = build_jump_ops(a, b, params)

    alpha_est = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)

    psi0 = dq.tensor(dq.fock(na, init_fock), dq.fock(nb, 0))
    tsave = jnp.linspace(0, tfinal, nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        result = dq.mesolve(
            H,
            jump_ops_list,
            psi0,
            tsave,
            options=dq.Options(progress_meter=False, save_states=True),
        )

    states_a = dq.ptrace(result.states, 0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update(frame: int) -> None:
        ax.cla()
        dq.plot.wigner(states_a[frame], ax=ax)
        loss_label = rf"$\kappa_a$={kappa_a}" if kappa_a > 0 else r"$\kappa_a$=0"
        ax.set_title(
            rf"Full Cat Qubit | $\alpha_{{est}}$={float(alpha_est):.2f} | "
            f"{loss_label} | t = {float(tsave[frame]):.2f} μs"
        )

    ani = FuncAnimation(fig, update, frames=nframes)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    print(f"[viz] Full cat qubit animation: {save_path}")
    return save_path


def animate_cat_decay(
    g2_re: float = 1.0,
    g2_im: float = 0.0,
    eps_d_re: float = 4.0,
    eps_d_im: float = 0.0,
    kappa_b: float = 10.0,
    kappa_a: float = 1.0,
    na: int = 15,
    nb: int = 5,
    tfinal: float = 100.0,
    nframes: int = 200,
    save_path: str = "figures/challenge_cat_decay.mp4",
    fps: int = 10,
) -> str:
    """Wigner animation: bit-flip decay under single-photon loss.

    Replicates challenge notebook cell 36. Full two-mode Hamiltonian with
    kappa_a=1 (single-photon loss ON). Starts from |-alpha> (one well)
    and shows decay as the state tunnels to the other well.

    Ref: Challenge notebook, Cell 36. Berdou et al. (2022), arXiv:2204.09128.

    Parameters
    ----------
    g2_re, g2_im : float
        Two-photon coupling.
    eps_d_re, eps_d_im : float
        Buffer drive amplitude.
    kappa_b : float
        Buffer loss rate [MHz].
    kappa_a : float
        Storage loss rate [MHz]. Default 1 (ON for decay).
    na, nb : int
        Hilbert space dimensions.
    tfinal : float
        Simulation duration [us]. 100 for long-timescale decay.
    nframes : int
        Number of animation frames.
    save_path : str
        Output MP4 path.
    fps : int
        Frames per second.

    Returns
    -------
    str
        Path to saved animation.
    """
    params = CatQubitParams(na=na, nb=nb, kappa_b=kappa_b, kappa_a=kappa_a)
    a, b = build_operators(params)
    H = build_hamiltonian(a, b, g2_re, g2_im, eps_d_re, eps_d_im)
    jump_ops_list = build_jump_ops(a, b, params)

    alpha_est = compute_alpha(g2_re, g2_im, eps_d_re, eps_d_im, params)

    # Start from |-alpha> (one well) — NOT vacuum
    psi0 = dq.tensor(dq.coherent(na, -alpha_est), dq.fock(nb, 0))
    tsave = jnp.linspace(0, tfinal, nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*SparseDIAQArray.*converted to a DenseQArray.*"
        )
        result = dq.mesolve(
            H,
            jump_ops_list,
            psi0,
            tsave,
            options=dq.Options(progress_meter=False, save_states=True),
        )

    states_a = dq.ptrace(result.states, 0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def update(frame: int) -> None:
        ax.cla()
        dq.plot.wigner(states_a[frame], ax=ax)
        ax.set_title(
            rf"Bit-Flip Decay | $\kappa_a$={kappa_a} | "
            rf"$\alpha_{{est}}$={float(alpha_est):.2f} | "
            f"t = {float(tsave[frame]):.1f} μs"
        )

    ani = FuncAnimation(fig, update, frames=nframes)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

    print(f"[viz] Cat decay animation: {save_path}")
    return save_path
