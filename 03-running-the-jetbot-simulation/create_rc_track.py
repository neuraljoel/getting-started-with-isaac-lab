# create_rc_track.py
#
# Usage (from IsaacLab folder with env_isaaclab activated):
#   .\isaaclab.bat -p source\standalone\rc_track\create_rc_track.py --experience isaaclab.python.rendering.kit
#
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# -------------------------------------------------------------------------
# 1) Launch Isaac Sim (SimulationApp) BEFORE importing isaaclab.sim
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Create RC track world with Jetbot + FPV camera.")

parser.add_argument(
    "--out_usd",
    type=str,
    default=None,
    help=(
        "Path where the world USD will be saved. "
        "If not set, defaults to <repo_root>/assets/worlds/rc_track_world_jetbot_fpv.usd"
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------------
# 2) Now it's safe to import Isaac Lab / Omniverse modules
# -------------------------------------------------------------------------

from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils

import omni.kit.actions.core as actions # For switching lights
from pxr import Usd, UsdGeom # For hiding Geometry mesh

from rc_track_params import TRACK_CFG
from jetbot_cfg import JETBOT_CFG


# -------------------------------------------------------------------------
# Helpers: ground + track
# -------------------------------------------------------------------------

def create_ground(sim: SimulationContext):
    """Spawn a flat black ground plane."""
    cfg_ground = sim_utils.GroundPlaneCfg(
        color=(0.0, 0.0, 0.0),  # pure black
        size=(20.0, 20.0),
    )
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)


def _spawn_dash(
    name: str,
    center_xy: tuple[float, float],
    yaw: float,
    length: float,
    width: float,
    height: float,
):
    """Spawn a thin white cuboid dash at (x,y) with heading yaw."""
    x, y = center_xy
    cfg = sim_utils.CuboidCfg(
        size=(length, width, height),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.2
        ),
        collision_props=None,
        rigid_props=None,
    )
    half = yaw * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    cfg.func(
        f"/World/Track/{name}",
        cfg,
        translation=(x, y, height * 0.5 + 0.001),
        orientation=(qw, 0.0, 0.0, qz),
    )


def _build_centerline_points(num_points: int = 800) -> np.ndarray:
    """
    Build a smooth closed polyline representing the rounded-rectangle centerline.
    Returns array [N,2] of (x,y).
    """
    L = TRACK_CFG["length"]
    W = TRACK_CFG["width"]
    R = TRACK_CFG["corner_radius"]

    assert R * 2 <= min(L, W), "corner_radius too large for given length/width"

    half_L = L * 0.5
    half_W = W * 0.5

    points = []
    seg_n = num_points // 8

    # 1) top straight
    xs = np.linspace(-half_L + R, half_L - R, seg_n, endpoint=False)
    ys = np.full_like(xs, half_W)
    points.append(np.stack([xs, ys], axis=-1))

    # 2) top-right arc
    cx, cy = (half_L - R, half_W - R)
    thetas = np.linspace(math.pi / 2, 0.0, seg_n, endpoint=False)
    points.append(np.stack([cx + R * np.cos(thetas), cy + R * np.sin(thetas)], axis=-1))

    # 3) right straight
    ys = np.linspace(half_W - R, -half_W + R, seg_n, endpoint=False)
    xs = np.full_like(ys, half_L)
    points.append(np.stack([xs, ys], axis=-1))

    # 4) bottom-right arc
    cx, cy = (half_L - R, -half_W + R)
    thetas = np.linspace(0.0, -math.pi / 2, seg_n, endpoint=False)
    points.append(np.stack([cx + R * np.cos(thetas), cy + R * np.sin(thetas)], axis=-1))

    # 5) bottom straight
    xs = np.linspace(half_L - R, -half_L + R, seg_n, endpoint=False)
    ys = np.full_like(xs, -half_W)
    points.append(np.stack([xs, ys], axis=-1))

    # 6) bottom-left arc
    cx, cy = (-half_L + R, -half_W + R)
    thetas = np.linspace(-math.pi / 2, -math.pi, seg_n, endpoint=False)
    points.append(np.stack([cx + R * np.cos(thetas), cy + R * np.sin(thetas)], axis=-1))

    # 7) left straight
    ys = np.linspace(-half_W + R, half_W - R, seg_n, endpoint=False)
    xs = np.full_like(ys, -half_L)
    points.append(np.stack([xs, ys], axis=-1))

    # 8) top-left arc
    cx, cy = (-half_L + R, half_W - R)
    thetas = np.linspace(-math.pi, -3 * math.pi / 2, seg_n, endpoint=False)
    points.append(np.stack([cx + R * np.cos(thetas), cy + R * np.sin(thetas)], axis=-1))

    pts = np.concatenate(points, axis=0)
    pts = np.vstack([pts, pts[0:1, :]])  # close loop
    return pts


def create_track():
    """Create dashed line track along the centerline."""
    dash_len = TRACK_CFG["dash_length"]
    dash_gap = TRACK_CFG["dash_gap"]
    line_w = TRACK_CFG["line_width"]
    h = TRACK_CFG["line_height"]

    pitch = dash_len + dash_gap  # distance between starts of consecutive dashes

    pts = _build_centerline_points(num_points=800)  # [N+1,2] (closed)
    num_pts = pts.shape[0]
    seg_vecs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = float(seg_lens.sum())

    dash_index = 0
    s_along = 0.0
    next_dash_s = 0.0

    for i in range(num_pts - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        seg_len = float(seg_lens[i])
        if seg_len <= 1e-6:
            continue

        while next_dash_s <= s_along + seg_len and next_dash_s <= total_len:
            t = (next_dash_s - s_along) / seg_len
            t = min(max(t, 0.0), 1.0)
            p = p0 + t * (p1 - p0)

            tangent = (p1 - p0) / seg_len
            yaw = math.atan2(float(tangent[1]), float(tangent[0]))

            name = f"dash_{dash_index:04d}"
            _spawn_dash(name, (float(p[0]), float(p[1])), yaw, dash_len, line_w, h)
            dash_index += 1

            next_dash_s += pitch

        s_along += seg_len

    print(f"Created {dash_index} dashes along track.")


# -------------------------------------------------------------------------
# Jetbot + FPV camera as USD prims
# -------------------------------------------------------------------------

def spawn_jetbot():
    """Spawn the Jetbot USD on top of the track, aligned with it."""
    print(f"Spawning Jetbot from {JETBOT_CFG.usd_path}")

    # Track centerline: top straight segment at y = +width/2
    x = 0.0
    y = -TRACK_CFG["width"] * 0.5    # center of top edge
    z = 0.03                        # tweak if it floats/sinks slightly

    # Heading along +X (parallel to the top straight)
    yaw = 0.0
    half = yaw * 0.5
    qw = math.cos(half)
    qz = math.sin(half)

    JETBOT_CFG.func(
        "/World/Jetbot",
        JETBOT_CFG,
        translation=(x, y, z),
        orientation=(qw, 0.0, 0.0, qz),
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.reset()

    # Switch lighting to Camera Light
    action_registry = actions.get_action_registry()
    action = action_registry.get_action(
        "omni.kit.viewport.menubar.lighting",  # extension id
        "set_lighting_mode_camera",            # action name
    )
    if action is not None:
        action.execute()
    # --------------------------------------------------

    create_ground(sim)

    # Hide the Geometry mesh
    stage = sim.get_initial_stage()
    geom_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/Environment/Geometry")
    if geom_prim and geom_prim.IsValid():
        UsdGeom.Imageable(geom_prim).MakeInvisible()
    # --------------------------------------------
    
    
    create_track()
    spawn_jetbot()

    # Default output path if not overridden
    if args_cli.out_usd is not None:
        out_path = Path(args_cli.out_usd)
    else:
        repo_root = Path(__file__).resolve().parents[3]  # ...\IsaacLab
        out_path = repo_root / "assets" / "worlds" / "rc_track_world_jetbot_fpv.usd"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage = sim.get_initial_stage()
    print(f"Saving world USD to: {out_path}")
    stage.GetRootLayer().Export(str(out_path))

    print("World saved. Close the Isaac Sim window to exit.")
    while simulation_app.is_running():
        sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
