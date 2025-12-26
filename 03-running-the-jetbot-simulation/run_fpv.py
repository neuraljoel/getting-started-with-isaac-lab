# run_fpv.py
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os
import numpy as np
import cv2
from PIL import Image

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from omni.usd import get_context
from omni.isaac.dynamic_control import _dynamic_control as dynamic_control

# ---- Switch viewport lighting to Camera Light ----
import omni.kit.actions.core as actions

from rc_track_params import TRACK_CFG


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Jetbot FPV line follower (raw FPV feed)")
    parser.add_argument(
        "--world_usd",
        type=str,
        default=r"C:\Users\joelm\Desktop\isaaclab\IsaacLab\assets\worlds\rc_track_world_jetbot_fpv.usd",
    )
    parser.add_argument("--save_fpv_frames", action="store_true")
    parser.add_argument("--frames_dir", type=str, default="outputs/fpv")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--world_speed", type=int, default=1)
    parser.add_argument("--debug_vis", action="store_true")

    # Perturbation duration (seconds)
    parser.add_argument(
        "--perturb_duration",
        type=float,
        default=1.0,
        help="Duration of random steering perturbation in seconds.",
    )
    args, _ = parser.parse_known_args()
    return args


# -----------------------------------------------------------------------------
# Jetbot driver
# -----------------------------------------------------------------------------
class SimpleJetbotDriver:
    def __init__(self, art_path="/World/Jetbot"):
        dc = dynamic_control.acquire_dynamic_control_interface()
        self.dc = dc
        self.art = dc.get_articulation(art_path)
        if not self.art:
            raise RuntimeError(f"Articulation not found: {art_path}")

        self.left = dc.find_articulation_dof(self.art, "left_wheel_joint")
        self.right = dc.find_articulation_dof(self.art, "right_wheel_joint")
        if self.left is None or self.right is None:
            raise RuntimeError("Wheel joints not found")

        # Root body for pose queries
        self.root = dc.get_articulation_root_body(self.art)
        print("[INFO] Jetbot driver initialized")

    def set(self, lv, rv):
        self.dc.wake_up_articulation(self.art)
        self.dc.set_dof_velocity_target(self.left, lv)
        self.dc.set_dof_velocity_target(self.right, rv)

    def get_xy(self):
        """Return Jetbot (x, y) in world coordinates."""
        pose = self.dc.get_rigid_body_pose(self.root)
        return float(pose.p.x), float(pose.p.y)


# -----------------------------------------------------------------------------
# Line detection + control (NO CROPPING)
# -----------------------------------------------------------------------------
def compute_line_offset(rgb, thr=140):
    """
    Uses the ENTIRE image. No ROI cropping.
    Returns normalized horizontal offset in [-1, 1], where positive means line is to the RIGHT of image center.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > thr
    if not np.any(mask):
        return None
    _, xs = np.where(mask)
    cx = xs.mean()
    w = rgb.shape[1]
    offset = (cx - (w / 2)) / (w / 2)
    return float(offset)


class LineFollower:
    def __init__(self, driver, dt, perturb_duration):
        self.driver = driver
        self.kp = 3.5
        self.base = 9.0
        self.max_turn = 4.0
        self.search_speed = 2.5

        # max magnitude of random perturbation turn
        self.max_perturb = 1.0

        # --- random perturbation setup ---
        self.dt = dt
        self.perturb_duration = float(perturb_duration)
        self.perturb_steps_total = max(1, int(self.perturb_duration / dt))
        self.perturb_steps_left = 0

        # ONE fixed turn value per perturbation episode
        self.current_perturb_turn = 0.0
        self.was_in_zone = False  # for lap-based triggering

        # Track dimensions (rounded rectangle centered at origin)
        self.length = TRACK_CFG["length"]
        self.width = TRACK_CFG["width"]

        # Failsafe threshold for line offset
        self.offset_fail_thresh = 0.9

    def _on_top_straight(self, x, y):
        """
        Return True if Jetbot is on (roughly) the top straight segment.
        Track is centered at (0,0):
        - top straight is around y = +width/2
        - limit x so we don't trigger near corners
        """
        half_L = 0.5 * self.length
        half_W = 0.5 * self.width

        # Tune these margins as needed
        y_ok = y > half_W * 0.6  # near the top
        x_ok = abs(x) < half_L * 0.8  # central part, away from arcs
        return y_ok and x_ok

    def step(self, rgb):
        """
        Run one control step.

        Returns:
            offset: normalized line offset in [-1,1] or None
            steer: steering term in rad/s (wheel-speed offset)
            mode: "auto" or "perturbation"
        """
        # Default telemetry
        offset = None
        steer = 0.0
        mode = "auto"

        # Get current pose
        x, y = self.driver.get_xy()
        in_zone = self._on_top_straight(x, y)

        # Detect a fresh entry into the zone (once per pass)
        if in_zone and not self.was_in_zone and self.perturb_steps_left == 0:
            self.perturb_steps_left = self.perturb_steps_total
            # choose ONE random turn value for this whole perturb episode
            self.current_perturb_turn = float(np.random.uniform(-self.max_perturb, self.max_perturb))

        self.was_in_zone = in_zone

        # If we are currently perturbing, steer with fixed random turn
        if self.perturb_steps_left > 0:
            if rgb is not None:
                offset = compute_line_offset(rgb)
                # Failsafe: if line is lost or offset is too large, abort perturbation
                if offset is None or abs(offset) > self.offset_fail_thresh:
                    self.perturb_steps_left = 0
                    self.current_perturb_turn = 0.0
                else:
                    self.perturb_steps_left -= 1
                    steer = float(np.clip(self.current_perturb_turn, -self.max_turn, self.max_turn))
                    mode = "perturbation"
                    lv = self.base + steer
                    rv = self.base - steer
                    self.driver.set(lv, rv)
                    return offset, steer, mode
            else:
                # No vision info → be conservative: end perturbation early
                self.perturb_steps_left = 0
                self.current_perturb_turn = 0.0

        # --- normal line-follow logic below ---
        if rgb is None:
            # Camera not ready → slow forward
            self.driver.set(self.base, self.base)
            return offset, steer, mode  # offset=None, steer=0, mode="auto"

        offset = compute_line_offset(rgb)
        if offset is None:
            # Lost line → search (spin)
            self.driver.set(-self.search_speed, self.search_speed)
            # treat as auto mode, steer not really defined → leave as 0
            return offset, steer, mode

        # POSITIVE offset (line on right) → POSITIVE turn → lv > rv → turn right
        steer = self.kp * offset
        steer = float(np.clip(steer, -self.max_turn, self.max_turn))
        lv = self.base + steer
        rv = self.base - steer
        self.driver.set(lv, rv)

        # mode stays "auto"
        return offset, steer, mode


# -----------------------------------------------------------------------------
# Debug window (exact FPV feed)
# -----------------------------------------------------------------------------
def show_debug(rgb, offset, steer, mode):
    vis = rgb.copy()
    h, w, _ = vis.shape

    # Draw center line
    cx0 = w // 2
    cv2.line(vis, (cx0, 0), (cx0, h), (0, 255, 0), 2)

    if offset is not None:
        cx = int(cx0 + offset * (w / 2))
        cv2.line(vis, (cx, 0), (cx, h), (0, 0, 255), 2)

    text_offset = f"offset={offset:.3f}" if offset is not None else "offset=None"
    text_steer = f"steer={steer:.3f}"
    text_mode = f"mode={mode}"

    cv2.putText(
        vis,
        text_offset,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        text_steer,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        text_mode,
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imshow("FPV Debug", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


# -----------------------------------------------------------------------------
# Frame saving
# -----------------------------------------------------------------------------
def save_frame(rgb, directory, idx):
    os.makedirs(directory, exist_ok=True)
    Image.fromarray(rgb).save(os.path.join(directory, f"fpv_{idx:06d}.png"))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    print("[INFO] Loading world:", args.world_usd)
    get_context().open_stage(args.world_usd)

    world = World()
    world.reset()

    action_registry = actions.get_action_registry()
    action = action_registry.get_action(
        "omni.kit.viewport.menubar.lighting",  # extension id
        "set_lighting_mode_camera",  # action name
    )
    if action is not None:
        action.execute()

    # --------------------------------------------------
    # Physics step (approximate, used for random-perturbation timing)
    dt = 1.0 / 60.0

    # EXACT camera prim path
    camera = Camera(
        prim_path="/World/Jetbot/chassis/new_fpv",
        resolution=(1280, 720),
        frequency=60,
    )
    camera.initialize()

    driver = SimpleJetbotDriver()
    controller = LineFollower(driver, dt, args.perturb_duration)

    step = 0
    saved = 0

    print("[INFO] Running… (close Isaac Sim window to exit)")
    try:
        PHYSICS_STEPS_PER_FRAME = args.world_speed  # try 2, 4, 8, 10+

        while simulation_app.is_running():
            # speed up physics
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                world.step(render=False)

            # render once
            world.step(render=True)

            rgb = None
            try:
                rgba = camera.get_rgba()
                if rgba.ndim == 3 and rgba.shape[2] >= 3:
                    rgb = np.array(rgba[:, :, :3], dtype=np.uint8)
            except Exception:
                rgb = None

            # Steering + telemetry
            offset, steer, mode = controller.step(rgb)

            # Debug window
            if args.debug_vis and rgb is not None:
                show_debug(rgb, offset, steer, mode)

            # Save frames
            if args.save_fpv_frames and rgb is not None:
                if step % args.save_interval == 0:
                    save_frame(rgb, args.frames_dir, saved)
                    saved += 1

            step += 1

    finally:
        print(f"[INFO] Saved frames: {saved}")
        if args.debug_vis:
            cv2.destroyAllWindows()
        simulation_app.close()


if __name__ == "__main__":
    main()
