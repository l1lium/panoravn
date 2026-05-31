"""
Webots per-drone controller — hardcoded lawnmower trajectory.

The controller runs in Supervisor mode so it can directly set its own
translation at each waypoint rather than fighting physics with a PID
controller.  Motors are kept at zero; the drone is repositioned to each
GPS target, the camera renders for a few steps, then the frame is captured.

Requires the Mavic2Pro node to have `supervisor TRUE` (set by
swarm_supervisor.py when injecting the VRML at startup).
"""

import json
import math
import os
import sys

_VENV_SITE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "venv", "lib", "python3.12", "site-packages")
)
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

try:
    from controller import Supervisor
except ImportError:
    raise ImportError(
        "Webots 'controller' module not found. "
        "Ensure PYTHONPATH includes Webots' lib/python directory."
    )

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise ImportError(f"Missing dependency: {exc}. Install opencv-python and numpy.")

try:
    import zmq
except ImportError:
    raise ImportError("pyzmq not found. Install it: pip install pyzmq")

MOTOR_NAMES = [
    "front left propeller",
    "front right propeller",
    "rear left propeller",
    "rear right propeller",
]

RENDER_STEPS = 3

GIMBAL_PITCH = 0.0
FOV_DEG      = 84.0

def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh)

def _resolve_config_path() -> str:
    controller_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(controller_dir, "..", "..", "configs", "simulation.yaml")
    )

def _resolve_frames_dir(relative_path: str) -> str:
    """Resolve a potentially relative disk_frames_dir to an absolute path.

    Webots sets the controller CWD to the controller directory, so relative
    paths in simulation.yaml would land under controllers/mavic_controller/.
    We resolve relative to the project root (three levels up from this file).
    """
    if os.path.isabs(relative_path):
        return relative_path
    controller_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(controller_dir, "..", "..", ".."))
    return os.path.normpath(os.path.join(project_root, relative_path))

def _parse_assignment(raw: str) -> "tuple[str, list[dict]] | None":
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if "drone_id" not in data or "waypoints" not in data:
        return None
    drone_id: str         = data["drone_id"]
    waypoints: list[dict] = data["waypoints"]
    if not isinstance(drone_id, str) or not drone_id:
        raise ValueError("drone_id must be a non-empty string")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("waypoints must be a non-empty list")
    return drone_id, waypoints

def _build_zmq_socket(config: dict) -> zmq.Socket:
    transport = config.get("transport", {})
    endpoint  = transport.get("zmq_endpoint", "tcp://localhost:5555")
    sndhwm    = int(transport.get("zmq_sndhwm", 50))
    context   = zmq.Context()
    socket    = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, sndhwm)
    socket.connect(endpoint)
    return socket

def _make_zmq_sender(drone_id: str, socket: zmq.Socket):
    def send(meta_bytes: bytes, jpeg_bytes: bytes) -> None:
        try:
            socket.send_multipart([meta_bytes, jpeg_bytes], flags=zmq.NOBLOCK)
        except zmq.Again:
            frame_id = json.loads(meta_bytes).get("frame_id", "?")
            print(f"[{drone_id}] ZMQ HWM reached — frame {frame_id} dropped.", flush=True)
    return send

def _make_disk_sender(drone_id: str, frames_dir: str):
    drone_dir = os.path.join(frames_dir, drone_id)
    os.makedirs(drone_dir, exist_ok=True)

    def send(meta_bytes: bytes, jpeg_bytes: bytes) -> None:
        frame_id = json.loads(meta_bytes)["frame_id"]
        with open(os.path.join(drone_dir, f"{frame_id}.json"), "wb") as f:
            f.write(meta_bytes)
        with open(os.path.join(drone_dir, f"{frame_id}.jpg"), "wb") as f:
            f.write(jpeg_bytes)
    return send

def run() -> None:
    supervisor = Supervisor()
    timestep   = int(supervisor.getBasicTimeStep())

    config_path   = _resolve_config_path()
    config        = _load_yaml(config_path)
    sim_cfg       = config.get("simulation", {})
    transport_cfg = config.get("transport", {})

    use_zmq: bool = transport_cfg.get("mode", "zmq") == "zmq"

    print("[controller] Waiting for waypoint assignment...", flush=True)
    drone_id:  str        = ""
    waypoints: list[dict] = []
    while supervisor.step(timestep) != -1:
        try:
            result = _parse_assignment(supervisor.getCustomData())
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[controller] FATAL: malformed customData — {exc}", flush=True)
            sys.exit(1)
        if result is not None:
            drone_id, waypoints = result
            break
    print(f"[{drone_id}] Loaded {len(waypoints)} waypoints.", flush=True)

    camera = supervisor.getDevice("camera")
    camera.enable(timestep)
    cam_w: int = camera.getWidth()
    cam_h: int = camera.getHeight()
    print(f"[{drone_id}] Camera: {cam_w}×{cam_h}px.", flush=True)

    gps = supervisor.getDevice("gps")
    gps.enable(timestep)

    imu = supervisor.getDevice("inertial unit")
    imu.enable(timestep)

    for name in MOTOR_NAMES:
        m = supervisor.getDevice(name)
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    camera_pitch = supervisor.getDevice("camera pitch")
    if camera_pitch is not None:
        camera_pitch.setPosition(0.0)
    else:
        print(f"[{drone_id}] WARNING: 'camera pitch' device not found.", flush=True)

    if use_zmq:
        try:
            zmq_socket = _build_zmq_socket(config)
            send_frame = _make_zmq_sender(drone_id, zmq_socket)
        except Exception as exc:
            print(f"[{drone_id}] ZMQ init failed ({exc}); falling back to disk.", flush=True)
            disk_dir   = _resolve_frames_dir(transport_cfg.get("disk_frames_dir", "data/sim_frames"))
            send_frame = _make_disk_sender(drone_id, disk_dir)
    else:
        disk_dir   = _resolve_frames_dir(transport_cfg.get("disk_frames_dir", "data/sim_frames"))
        send_frame = _make_disk_sender(drone_id, disk_dir)

    cruise_speed = float(sim_cfg.get("cruise_speed_m_s", 10.0))
    dwell_ms     = int(sim_cfg.get("dwell_time_ms", 500))
    dwell_steps  = max(RENDER_STEPS, round(dwell_ms / timestep))

    self_node         = supervisor.getSelf()
    translation_field = self_node.getField("translation")
    rotation_field    = self_node.getField("rotation")

    frame_idx = 0
    print(
        f"[{drone_id}] Starting scan ({len(waypoints)} waypoints, "
        f"speed={cruise_speed:.0f} m/s, dwell={dwell_ms} ms).",
        flush=True,
    )

    prev_pos = list(translation_field.getSFVec3f())
    running  = True

    for wp_idx, wp in enumerate(waypoints):
        if not running:
            break
        tx, ty, tz = wp["target"]

        dx   = tx - prev_pos[0]
        dy   = ty - prev_pos[1]
        dz   = tz - prev_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        travel_steps = max(1, round(dist / cruise_speed * 1000.0 / timestep))

        heading = math.atan2(dy, dx) if (abs(dx) > 0.01 or abs(dy) > 0.01) else 0.0
        travel_rot = [0.0, 0.0, 1.0, heading]

        for si in range(travel_steps):
            frac = (si + 1) / travel_steps
            translation_field.setSFVec3f([
                prev_pos[0] + frac * dx,
                prev_pos[1] + frac * dy,
                prev_pos[2] + frac * dz,
            ])
            rotation_field.setSFRotation(travel_rot)
            self_node.resetPhysics()
            if supervisor.step(timestep) == -1:
                running = False
                break

        if not running:
            break

        translation_field.setSFVec3f([tx, ty, tz])
        rotation_field.setSFRotation(travel_rot)
        self_node.resetPhysics()

        for _ in range(dwell_steps):
            if supervisor.step(timestep) == -1:
                running = False
                break

        if not running:
            break

        gps_vals = gps.getValues()
        rpy      = imu.getRollPitchYaw()
        sim_time = supervisor.getTime()

        raw_img = camera.getImage()
        bgra    = np.frombuffer(raw_img, dtype=np.uint8).reshape((cam_h, cam_w, 4))
        bgr     = bgra[:, :, :3]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frame_id = f"{drone_id}_{frame_idx:06d}"
            meta = {
                "drone_id":     drone_id,
                "frame_id":     frame_id,
                "timestamp":    sim_time,
                "gps":          {"x": gps_vals[0], "y": gps_vals[1], "z": gps_vals[2]},
                "imu":          {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]},
                "gimbal_pitch": GIMBAL_PITCH,
                "fov_deg":      FOV_DEG,
                "image_wh":     [cam_w, cam_h],
                "encoding":     "jpeg",
            }
            send_frame(json.dumps(meta).encode(), buf.tobytes())
            print(
                f"[{drone_id}] WP {wp_idx + 1}/{len(waypoints)}"
                f" ({tx:.0f}, {ty:.0f}, {tz:.0f})m"
                f" | frame {frame_id} (t={sim_time:.1f}s)",
                flush=True,
            )
            frame_idx += 1
        else:
            print(f"[{drone_id}] JPEG encode failed at WP {wp_idx}; skipping.", flush=True)

        prev_pos = [tx, ty, tz]

    supervisor.setCustomData(json.dumps({"drone_id": drone_id, "status": "DONE"}))
    print(f"[{drone_id}] Mission complete — {frame_idx} frames captured.", flush=True)

    while supervisor.step(timestep) != -1:
        pass

if __name__ == "__main__":
    run()
