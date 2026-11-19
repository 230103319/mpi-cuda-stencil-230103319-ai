import json
import time
import re
import sys
import hashlib
import subprocess
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15


def get_gpu_uuid():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8"
    )
    return output.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected 4 MPI ranks, got {size}"
        )

    # Bind every rank to GPU 0
    cuda.select_device(0)

    # Hardware-bound identity
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)

    instance_seed = derive_student_seed(
        github_user, gpu_uuid
    )

    # Create non-periodic 2x2 Cartesian grid
    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)

    # Find orthogonal neighbors
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": cart_rank,
        "size": size,
        "coords": coords,
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "instance_seed": instance_seed,
        "gpu_uuid": gpu_uuid
    }


# =========================================
# TASK 2: Subdomain Initialization
# =========================================

import numpy as np

def initialize_subdomains(ctx):
    ny, nx = 256, 256
    row_rank, col_rank = ctx["coords"]

    dx = 1.0 / 513.0
    dy = 1.0 / 513.0

    # Rank-local coordinates including ghost cells
    rows = np.arange(ny + 2, dtype=np.float64)
    cols = np.arange(nx + 2, dtype=np.float64)

    y = (row_rank * ny + rows[:, None]) * dy
    x = (col_rank * nx + cols[None, :]) * dx

    # Spatially heterogeneous diffusivity
    kappa = (
        0.16
        + 0.08 * np.sin(2 * np.pi * x)
        * np.cos(2 * np.pi * y)
    )

    # Seed-dependent Gaussian pulses
    rng = np.random.default_rng(ctx["instance_seed"])

    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)

    for _ in range(2):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        amplitude = rng.uniform(0.5, 1.0)

        u += amplitude * np.exp(
            -((x - cx)**2 + (y - cy)**2)
            / (2 * 0.06**2)
        )

    # Localized source term
    sx = rng.uniform(0.3, 0.7)
    sy = rng.uniform(0.3, 0.7)

    source = 0.01 * np.exp(
        -((x - sx)**2 + (y - sy)**2)
        / (2 * 0.04**2)
    )

    # Homogeneous Dirichlet physical boundaries
    if ctx["nbr_up"] == MPI.PROC_NULL:
        u[0, :] = 0.0

    if ctx["nbr_down"] == MPI.PROC_NULL:
        u[-1, :] = 0.0

    if ctx["nbr_left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0

    if ctx["nbr_right"] == MPI.PROC_NULL:
        u[:, -1] = 0.0

    # Transfer rank-local arrays to GPU
    d_u = cuda.to_device(u)
    d_u_new = cuda.to_device(np.zeros_like(u))
    d_kappa = cuda.to_device(kappa)
    d_source = cuda.to_device(source)

    print(
        f"Rank {ctx['rank']} | "
        f"Shape {u.shape} | "
        f"Kappa [{kappa.min():.4f}, "
        f"{kappa.max():.4f}] | "
        f"Max temperature {u.max():.6f}",
        flush=True
    )

    return {
        "d_u": d_u,
        "d_u_new": d_u_new,
        "d_kappa": d_kappa,
        "d_source": d_source,
        "dx": dx,
        "dy": dy,
        "ny": ny,
        "nx": nx
    }

# =========================================
# TASK 3: GPU Halo Exchange
# =========================================

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


@cuda.jit
def pack_halos_kernel(u, buffer, direction, ny, nx):
    i = cuda.grid(1)

    if direction == UP:
        if i < nx:
            buffer[i] = u[1, i + 1]

    elif direction == DOWN:
        if i < nx:
            buffer[i] = u[ny, i + 1]

    elif direction == LEFT:
        if i < ny:
            buffer[i] = u[i + 1, 1]

    elif direction == RIGHT:
        if i < ny:
            buffer[i] = u[i + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, buffer, direction, ny, nx):
    i = cuda.grid(1)

    if direction == UP:
        if i < nx:
            u[0, i + 1] = buffer[i]

    elif direction == DOWN:
        if i < nx:
            u[ny + 1, i + 1] = buffer[i]

    elif direction == LEFT:
        if i < ny:
            u[i + 1, 0] = buffer[i]

    elif direction == RIGHT:
        if i < ny:
            u[i + 1, nx + 1] = buffer[i]


def exchange_halos_async(ctx, d_u):
    cart_comm = ctx["cart_comm"]

    ny, nx = 256, 256
    stream_boundary = cuda.stream()

    neighbors = {
        UP: ctx["nbr_up"],
        DOWN: ctx["nbr_down"],
        LEFT: ctx["nbr_left"],
        RIGHT: ctx["nbr_right"]
    }

    opposite = {
        UP: DOWN,
        DOWN: UP,
        LEFT: RIGHT,
        RIGHT: LEFT
    }

    tags = {
        UP: 100,
        DOWN: 101,
        LEFT: 102,
        RIGHT: 103
    }

    send_device = {}
    recv_device = {}
    send_host = {}
    recv_host = {}

    active = [
        side for side, neighbor in neighbors.items()
        if neighbor != MPI.PROC_NULL
    ]

    # Stage A: Pack halos on GPU and copy to pinned host buffers
    for side in active:
        n = nx if side in (UP, DOWN) else ny

        send_device[side] = cuda.device_array(
            n, dtype=np.float64
        )
        recv_device[side] = cuda.device_array(
            n, dtype=np.float64
        )

        send_host[side] = cuda.pinned_array(
            n, dtype=np.float64
        )
        recv_host[side] = cuda.pinned_array(
            n, dtype=np.float64
        )

        blocks = (n + 127) // 128

        pack_halos_kernel[blocks, 128, stream_boundary](
            d_u, send_device[side], side, ny, nx
        )

        send_device[side].copy_to_host(
            send_host[side], stream=stream_boundary
        )

    stream_boundary.synchronize()

    # Stage B: Non-blocking MPI communication
    requests = []

    for side in active:
        requests.append(
            cart_comm.Irecv(
                recv_host[side],
                source=neighbors[side],
                tag=tags[opposite[side]]
            )
        )

    for side in active:
        requests.append(
            cart_comm.Isend(
                send_host[side],
                dest=neighbors[side],
                tag=tags[side]
            )
        )

    MPI.Request.Waitall(requests)

    # Stage C: Copy received data to GPU and unpack
    for side in active:
        n = nx if side in (UP, DOWN) else ny
        blocks = (n + 127) // 128

        recv_device[side].copy_to_device(
            recv_host[side], stream=stream_boundary
        )

        unpack_halos_kernel[blocks, 128, stream_boundary](
            d_u, recv_device[side], side, ny, nx
        )

    stream_boundary.synchronize()

    print(
        f"Rank {ctx['rank']} | "
        f"Halo exchange completed | "
        f"Neighbors: {len(active)}",
        flush=True
    )

    return len(active)

# =========================================
# TASK 4: Shared-Memory Tiled Stencil
# =========================================

@cuda.jit
def tiled_stencil_kernel(
    u, u_new, kappa, source,
    r_start, r_end, c_start, c_end,
    dt, dx, dy, damping, ny, nx
):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    r = r_start + cuda.blockIdx.y * 16 + ty
    c = c_start + cuda.blockIdx.x * 16 + tx

    sy = ty + 1
    sx = tx + 1

    # 16x16 tile with one-cell halo
    smem = cuda.shared.array((18, 18), dtype=float64)

    # Center values
    if r <= ny + 1 and c <= nx + 1:
        smem[sy, sx] = u[r, c]
    else:
        smem[sy, sx] = 0.0

    # Load top halo
    if ty == 0:
        if r > 0 and c <= nx + 1:
            smem[0, sx] = u[r - 1, c]
        else:
            smem[0, sx] = 0.0

    # Load bottom halo
    if ty == 15:
        if r < ny + 1 and c <= nx + 1:
            smem[17, sx] = u[r + 1, c]
        else:
            smem[17, sx] = 0.0

    # Load left halo
    if tx == 0:
        if c > 0 and r <= ny + 1:
            smem[sy, 0] = u[r, c - 1]
        else:
            smem[sy, 0] = 0.0

    # Load right halo
    if tx == 15:
        if c < nx + 1 and r <= ny + 1:
            smem[sy, 17] = u[r, c + 1]
        else:
            smem[sy, 17] = 0.0

    # Synchronize all threads in the block
    cuda.syncthreads()

    if (
        r >= r_start and r < r_end
        and c >= c_start and c < c_end
    ):
        center = smem[sy, sx]

        # Arithmetic interface averages
        kc = kappa[r, c]

        k_right = 0.5 * (kc + kappa[r, c + 1])
        k_left = 0.5 * (kc + kappa[r, c - 1])
        k_down = 0.5 * (kc + kappa[r + 1, c])
        k_up = 0.5 * (kc + kappa[r - 1, c])

        flux_x = (
            k_right * (smem[sy, sx + 1] - center)
            - k_left * (center - smem[sy, sx - 1])
        ) / (dx * dx)

        flux_y = (
            k_down * (smem[sy + 1, sx] - center)
            - k_up * (center - smem[sy - 1, sx])
        ) / (dy * dy)

        reaction = -damping * center * center * center

        u_new[r, c] = center + dt * (
            flux_x + flux_y + reaction + source[r, c]
        )


def create_halo_buffers(ctx, fields):
    """Allocate reusable GPU and pinned host buffers."""

    neighbors = {
        UP: ctx["nbr_up"],
        DOWN: ctx["nbr_down"],
        LEFT: ctx["nbr_left"],
        RIGHT: ctx["nbr_right"]
    }

    active = [
        side for side in (UP, DOWN, LEFT, RIGHT)
        if neighbors[side] != MPI.PROC_NULL
    ]

    buffers = {
        "active": active,
        "neighbors": neighbors,
        "send_device": {},
        "recv_device": {},
        "send_host": {},
        "recv_host": {}
    }

    ny = fields["ny"]
    nx = fields["nx"]

    for side in active:
        n = nx if side in (UP, DOWN) else ny

        buffers["send_device"][side] = cuda.device_array(
            n, dtype=np.float64
        )
        buffers["recv_device"][side] = cuda.device_array(
            n, dtype=np.float64
        )
        buffers["send_host"][side] = cuda.pinned_array(
            n, dtype=np.float64
        )
        buffers["recv_host"][side] = cuda.pinned_array(
            n, dtype=np.float64
        )

    return buffers


def begin_halo_exchange(ctx, fields, buffers, stream_boundary):
    """Pack halos and start non-blocking MPI messages."""

    ny = fields["ny"]
    nx = fields["nx"]

    opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
    tags = {UP: 100, DOWN: 101, LEFT: 102, RIGHT: 103}

    for side in buffers["active"]:
        n = nx if side in (UP, DOWN) else ny
        blocks = (n + 127) // 128

        pack_halos_kernel[blocks, 128, stream_boundary](
            fields["d_u"], buffers["send_device"][side],
            side, ny, nx
        )

        buffers["send_device"][side].copy_to_host(
            buffers["send_host"][side],
            stream=stream_boundary
        )

    # Host data must be ready before MPI sends
    stream_boundary.synchronize()

    requests = []

    for side in buffers["active"]:
        requests.append(
            ctx["cart_comm"].Irecv(
                buffers["recv_host"][side],
                source=buffers["neighbors"][side],
                tag=tags[opposite[side]]
            )
        )

    for side in buffers["active"]:
        requests.append(
            ctx["cart_comm"].Isend(
                buffers["send_host"][side],
                dest=buffers["neighbors"][side],
                tag=tags[side]
            )
        )

    return requests


def finish_halo_exchange(
    ctx, fields, buffers, stream_boundary, requests
):
    """Finish communication and populate GPU ghost cells."""

    MPI.Request.Waitall(requests)

    ny = fields["ny"]
    nx = fields["nx"]

    for side in buffers["active"]:
        n = nx if side in (UP, DOWN) else ny
        blocks = (n + 127) // 128

        buffers["recv_device"][side].copy_to_device(
            buffers["recv_host"][side],
            stream=stream_boundary
        )

        unpack_halos_kernel[blocks, 128, stream_boundary](
            fields["d_u"], buffers["recv_device"][side],
            side, ny, nx
        )


def launch_stencil_box(
    fields, stream, dt, r_start, r_end, c_start, c_end
):
    """Launch one rectangular tiled stencil region."""

    rows = r_end - r_start
    cols = c_end - c_start

    grid = (
        (cols + 15) // 16,
        (rows + 15) // 16
    )

    tiled_stencil_kernel[grid, (16, 16), stream](
        fields["d_u"],
        fields["d_u_new"],
        fields["d_kappa"],
        fields["d_source"],
        r_start, r_end, c_start, c_end,
        dt, fields["dx"], fields["dy"],
        LAMBDA_DAMP, fields["ny"], fields["nx"]
    )


def step_overlapped(
    ctx, fields, buffers,
    stream_boundary, stream_interior, dt
):
    ny = fields["ny"]
    nx = fields["nx"]

    # Stage A/B: Start halo communication
    requests = begin_halo_exchange(
        ctx, fields, buffers, stream_boundary
    )

    # Interior calculation while MPI messages are in flight
    launch_stencil_box(
        fields, stream_interior, dt,
        2, ny, 2, nx
    )

    # Stage C: Receive halos and unpack on GPU
    finish_halo_exchange(
        ctx, fields, buffers, stream_boundary, requests
    )

    # Four non-overlapping boundary rectangles
    ring_regions = [
        (1, 2, 1, nx + 1),
        (ny, ny + 1, 1, nx + 1),
        (2, ny, 1, 2),
        (2, ny, nx, nx + 1)
    ]

    for r0, r1, c0, c1 in ring_regions:
        launch_stencil_box(
            fields, stream_boundary, dt,
            r0, r1, c0, c1
        )

    # Both streams must finish before buffer swapping
    stream_interior.synchronize()
    stream_boundary.synchronize()

    fields["d_u"], fields["d_u_new"] = (
        fields["d_u_new"], fields["d_u"]
    )

# =========================================
# TASK 5: Reductions, Benchmark & PTX Audit
# =========================================

def reduce_metric(ctx, value, operation):
    send = np.array([value], dtype=np.float64)
    recv = np.zeros(1, dtype=np.float64)

    ctx["cart_comm"].Allreduce(
        send, recv, op=operation
    )
    return float(recv[0])


def step_serialized(ctx, fields, buffers, stream, dt):
    """Communication first, then GPU computation."""

    requests = begin_halo_exchange(
        ctx, fields, buffers, stream
    )

    finish_halo_exchange(
        ctx, fields, buffers, stream, requests
    )

    ny = fields["ny"]
    nx = fields["nx"]

    # Deep interior
    launch_stencil_box(
        fields, stream, dt, 2, ny, 2, nx
    )

    # Four boundary regions
    regions = [
        (1, 2, 1, nx + 1),
        (ny, ny + 1, 1, nx + 1),
        (2, ny, 1, 2),
        (2, ny, nx, nx + 1)
    ]

    for r0, r1, c0, c1 in regions:
        launch_stencil_box(
            fields, stream, dt, r0, r1, c0, c1
        )

    stream.synchronize()

    fields["d_u"], fields["d_u_new"] = (
        fields["d_u_new"], fields["d_u"]
    )


def global_energy(ctx, fields, values):
    local = np.sum(values, dtype=np.float64)
    local *= fields["dx"] * fields["dy"]

    return reduce_metric(ctx, local, MPI.SUM)


def run_benchmark(ctx, mode, dt):
    # Fresh identical initial conditions for each mode
    fields = initialize_subdomains(ctx)
    buffers = create_halo_buffers(ctx, fields)

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    cuda.synchronize()

    initial = fields["d_u"].copy_to_host()[1:-1, 1:-1]
    energy0 = global_energy(ctx, fields, initial)

    trace = [
        {"step": 0, "energy": energy0, "residual_l2": 0.0}
    ]

    if ctx["rank"] == 0:
        print(f"{mode} step 0 | Energy: {energy0:.12e}",
              flush=True)

    start_event = cuda.event(timing=True)
    end_event = cuda.event(timing=True)

    ctx["cart_comm"].Barrier()
    wall_start = time.perf_counter()
    start_event.record(stream_boundary)

    for step in range(1, STEPS + 1):
        # Save previous state at checkpoint steps
        if step % 100 == 0:
            previous = fields["d_u"].copy_to_host()[
                1:-1, 1:-1
            ].copy()

        if mode == "serialized":
            step_serialized(
                ctx, fields, buffers, stream_boundary, dt
            )
        else:
            step_overlapped(
                ctx, fields, buffers,
                stream_boundary, stream_interior, dt
            )

        if step % 100 == 0:
            current = fields["d_u"].copy_to_host()[
                1:-1, 1:-1
            ]

            energy = global_energy(ctx, fields, current)

            local_sq = np.sum(
                (current - previous)**2,
                dtype=np.float64
            )

            global_sq = reduce_metric(
                ctx, local_sq, MPI.SUM
            )

            residual = np.sqrt(
                global_sq * fields["dx"] * fields["dy"]
            )

            trace.append({
                "step": step,
                "energy": float(energy),
                "residual_l2": float(residual)
            })

            if ctx["rank"] == 0:
                print(
                    f"{mode} step {step} | "
                    f"Energy: {energy:.12e} | "
                    f"L2: {residual:.12e}",
                    flush=True
                )

    end_event.record(stream_boundary)
    end_event.synchronize()

    local_wall_ms = (
        time.perf_counter() - wall_start
    ) * 1000.0

    local_cuda_ms = cuda.event_elapsed_time(
        start_event, end_event
    )

    wall_ms = reduce_metric(
        ctx, local_wall_ms, MPI.MAX
    )

    event_ms = reduce_metric(
        ctx, local_cuda_ms, MPI.MAX
    )

    return fields, wall_ms, event_ms, trace


def audit_ptx():
    """Inspect compiled CUDA stencil assembly."""

    signatures = tiled_stencil_kernel.signatures

    if not signatures:
        raise RuntimeError(
            "Stencil kernel was not compiled"
        )

    ptx = tiled_stencil_kernel.inspect_asm(
        signatures[0]
    )

    if isinstance(ptx, dict):
        ptx = next(iter(ptx.values()))

    counts = {
        "st.shared.f64": len(
            re.findall(r"\bst\.shared\.f64\b", ptx)
        ),
        "ld.shared.f64": len(
            re.findall(r"\bld\.shared\.f64\b", ptx)
        ),
        "bar.sync": len(
            re.findall(r"\bbar\.sync\b", ptx)
        ),
        "st.shared_total": len(
            re.findall(r"\bst\.shared\b", ptx)
        ),
        "ld.shared_total": len(
            re.findall(r"\bld\.shared\b", ptx)
        )
    }

    passed = (
        counts["st.shared_total"] > 0
        and counts["ld.shared_total"] > 0
        and counts["bar.sync"] > 0
    )

    return counts, passed


def run_solver(ctx, username):
    # Determine global maximum diffusivity
    reference = initialize_subdomains(ctx)

    local_max = np.max(
        reference["d_kappa"].copy_to_host()[
            1:-1, 1:-1
        ]
    )

    kappa_max = reduce_metric(
        ctx, local_max, MPI.MAX
    )

    dx = reference["dx"]
    dt_max = dx**2 / (4.0 * kappa_max)

    # Conservative timestep
    dt = 0.8 * dx**2 / (4.0 * 0.24)

    assert dt < dt_max, "CFL stability check failed"

    # 400 steps: serialized execution
    serial_fields, serialized_ms, serial_event_ms, serial_trace = (
        run_benchmark(ctx, "serialized", dt)
    )

    # 400 steps: overlapped execution
    overlap_fields, overlapped_ms, overlap_event_ms, overlap_trace = (
        run_benchmark(ctx, "overlapped", dt)
    )

    # Compare locally owned cells, not ghost cells
    serial_result = serial_fields["d_u"].copy_to_host()[
        1:-1, 1:-1
    ]

    overlap_result = overlap_fields["d_u"].copy_to_host()[
        1:-1, 1:-1
    ]

    local_difference = np.max(
        np.abs(serial_result - overlap_result)
    )

    parity_max_diff = reduce_metric(
        ctx, local_difference, MPI.MAX
    )

    if ctx["rank"] == 0:
        counts, audit_passed = audit_ptx()

        speedup = serialized_ms / overlapped_ms

        manifest = {
            "github_username": username,
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["instance_seed"],
            "mpi_ranks": ctx["size"],
            "cartesian_dims": [2, 2],
            "global_shape": [512, 512],
            "local_shape": [258, 258],
            "steps": STEPS,
            "kappa_max": kappa_max,
            "dt": dt,
            "dt_max": dt_max,
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "cuda_event_serialized_ms": serial_event_ms,
            "cuda_event_overlapped_ms": overlap_event_ms,
            "speedup": speedup,
            "parity_max_diff": parity_max_diff,
            "ptx_counts": counts,
            "ptx_audit_passed": audit_passed,
            "energy_trace_serialized": serial_trace,
            "energy_trace_overlapped": overlap_trace
        }

        with open(
            "submission_manifest.json", "w"
        ) as file:
            json.dump(manifest, file, indent=4)

        print("\n===== FINAL TASK 5 RESULTS =====")
        print(f"Serialized: {serialized_ms:.3f} ms")
        print(f"Overlapped: {overlapped_ms:.3f} ms")
        print(f"Speedup: {speedup:.4f}x")
        print(f"Parity max diff: {parity_max_diff:.12e}")
        print(f"Kappa max: {kappa_max:.8f}")
        print(f"dt_max: {dt_max:.12e}")

        print("\nPTX AUDIT:")
        for name, count in counts.items():
            print(f"{name}: {count}")

        print(f"PTX audit passed: {audit_passed}")

        passed = (
            audit_passed
            and parity_max_diff < 1e-12
        )

        if passed:
            print("TASK 5 COMPLETED")
        else:
            print("TASK 5 NEEDS REVIEW")

        print("Manifest: submission_manifest.json")

if __name__ == "__main__":
    username = sys.argv[1]
    ctx = init_distributed_context(username)

    print(
        f"Rank {ctx['rank']} | "
        f"Coords {ctx['coords']} | "
        f"Up {ctx['nbr_up']} | "
        f"Down {ctx['nbr_down']} | "
        f"Left {ctx['nbr_left']} | "
        f"Right {ctx['nbr_right']}",
        flush=True
    )

    if ctx["rank"] == 0:
        print(f"GPU UUID: {ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED: {ctx['instance_seed']}")
        print("TASK 1 COMPLETED")

    fields = initialize_subdomains(ctx)

    if ctx["rank"] == 0:
        print("TASK 2 COMPLETED")
      

    

    exchanged = exchange_halos_async(
        ctx, fields["d_u"]
    )

    total_exchanges = ctx["cart_comm"].allreduce(
        exchanged, op=MPI.SUM
    )

    if ctx["rank"] == 0:
        print(
            f"TASK 3 COMPLETED | "
            f"Directed exchanges: {total_exchanges}"
        )
    
    # =========================================
    # TASK 4 TEST
    # =========================================

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    buffers = create_halo_buffers(ctx, fields)

    # Stable timestep below the diffusion CFL limit
    dt = 0.8 * fields["dx"]**2 / (4.0 * 0.24)

    before = fields["d_u"].copy_to_host()

    # Execute one overlapped timestep
    step_overlapped(
        ctx, fields, buffers,
        stream_boundary, stream_interior, dt
    )

    after = fields["d_u"].copy_to_host()

    old_interior = before[1:-1, 1:-1]
    new_interior = after[1:-1, 1:-1]

    assert np.isfinite(new_interior).all(), (
        "Invalid numerical values detected"
    )

    change = np.max(
        np.abs(new_interior - old_interior)
    )

    print(
        f"Rank {ctx['rank']} | "
        f"Stencil finished | "
        f"Max change: {change:.10e}",
        flush=True
    )

    ctx["cart_comm"].Barrier()

    if ctx["rank"] == 0:
        print(f"Stable timestep dt: {dt:.10e}")
        print("TASK 4 COMPLETED")
            # Run full Task 5 benchmark
    run_solver(ctx, username)
