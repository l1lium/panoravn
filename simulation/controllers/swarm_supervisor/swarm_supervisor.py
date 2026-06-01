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
        "Ensure Webots is installed and PYTHONPATH includes its lib/python directory."
    )

def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it in the Webots Python environment: "
            "pip install pyyaml"
        )
    with open(path) as fh:
        return yaml.safe_load(fh)

def _resolve_config_path() -> str:
    controller_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(controller_dir, "..", "..", "configs", "simulation.yaml")
    )

def _compute_lawnmower_grid(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    altitude: float,
    row_spacing: float,
    col_spacing: float,
) -> list[dict]:
    waypoints: list[dict] = []
    y = y_min
    row = 0
    while y <= y_max + 1e-6:
        xs: list[float] = []
        x = x_min
        while x <= x_max + 1e-6:
            xs.append(x)
            x += col_spacing
        if row % 2 == 1:
            xs = list(reversed(xs))
        for xi in xs:
            waypoints.append({
                "target": [round(xi, 3), round(y, 3), round(altitude, 3)],
                "task": "scan",
            })
        y += row_spacing
        row += 1
    return waypoints

def _build_drone_assignments(config: dict) -> list[dict]:
    sim = config["simulation"]
    map_cfg = config["map"]

    num_drones: int = sim["num_drones"]
    altitude: float = sim["altitude_m"]
    overlap: float = sim["overlap_ratio"]
    fov_deg: float = sim["camera_fov_deg"]

    x_min: float = map_cfg["x_min"]
    x_max: float = map_cfg["x_max"]
    y_min: float = map_cfg["y_min"]
    y_max: float = map_cfg["y_max"]

    strip_width = (x_max - x_min) / num_drones
    footprint_side = 2.0 * altitude * math.tan(math.radians(fov_deg) / 2.0)
    row_spacing = footprint_side * (1.0 - overlap)
    col_spacing = row_spacing

    assignments: list[dict] = []
    for i in range(num_drones):
        sx_min = x_min + i * strip_width
        sx_max = x_min + (i + 1) * strip_width
        waypoints = _compute_lawnmower_grid(
            sx_min, sx_max, y_min, y_max, altitude, row_spacing, col_spacing
        )
        assignments.append({
            "drone_id": f"drone_{i}",
            "waypoints": waypoints,
        })
        print(
            f"[supervisor] drone_{i}: strip x=[{sx_min:.1f}, {sx_max:.1f}], "
            f"{len(waypoints)} waypoints",
            flush=True,
        )
    return assignments

def _get_terrain_height(supervisor: "Supervisor", world_x: float, world_y: float) -> float:
    terrain = supervisor.getFromDef("TERRAIN")
    if terrain is None:
        return 0.0

    t = terrain.getField("translation").getSFVec3f()
    origin_x, origin_y = t[0], t[1]

    shape = terrain.getField("children").getMFNode(0)
    grid  = shape.getField("geometry").getSFNode()

    x_dim = grid.getField("xDimension").getSFInt32()
    y_dim = grid.getField("yDimension").getSFInt32()
    x_sp  = grid.getField("xSpacing").getSFFloat()
    y_sp  = grid.getField("ySpacing").getSFFloat()
    hf    = grid.getField("height")

    gx = max(0.0, min(x_dim - 1.0, (world_x - origin_x) / x_sp))
    gy = max(0.0, min(y_dim - 1.0, (world_y - origin_y) / y_sp))
    ix, iy = int(gx), int(gy)
    fx, fy = gx - ix, gy - iy
    ix1 = min(ix + 1, x_dim - 1)
    iy1 = min(iy + 1, y_dim - 1)

    def h(col: int, row: int) -> float:
        return hf.getMFFloat(row * x_dim + col)

    return (h(ix,  iy ) * (1 - fx) * (1 - fy) +
            h(ix1, iy ) * fx       * (1 - fy) +
            h(ix,  iy1) * (1 - fx) * fy       +
            h(ix1, iy1) * fx       * fy)

def _inject_drones(supervisor: "Supervisor", config: dict, timestep: int) -> None:
    map_cfg = config["map"]
    sim_cfg = config["simulation"]
    num_drones: int = sim_cfg["num_drones"]
    x_min: float = map_cfg["x_min"]
    x_max: float = map_cfg["x_max"]
    y_start: float = map_cfg["y_min"] + 5.0
    strip_width = (x_max - x_min) / num_drones

    group_node = supervisor.getFromDef("DRONE_GROUP")
    if group_node is None:
        print(
            "[supervisor] ERROR: DEF 'DRONE_GROUP' not found. "
            "Regenerate drone_swarm.wbt with generate_world.py.",
            flush=True,
        )
        sys.exit(1)

    children_field = group_node.getField("children")

    for i in range(num_drones):
        cx = x_min + (i + 0.5) * strip_width
        vrml = (
            f'DEF DRONE_{i} Mavic2Pro {{\n'
            f'  name "drone_{i}"\n'
            f'  translation {cx:.3f} {y_start:.1f} 10.0\n'
            f'  supervisor TRUE\n'
            f'  controller "mavic_controller"\n'
            f'  customData ""\n'
            f'}}'
        )
        children_field.importMFNodeFromString(-1, vrml)
        print(f"[supervisor] Injected DRONE_{i} at X={cx:.1f}, Y={y_start:.1f}, Z=10.0.", flush=True)

    supervisor.step(timestep)

    for i in range(num_drones):
        cx = x_min + (i + 0.5) * strip_width
        node = supervisor.getFromDef(f"DRONE_{i}")
        if node is None:
            print(f"[supervisor] WARNING: DRONE_{i} not found for grounding.", flush=True)
            continue
        terrain_z = _get_terrain_height(supervisor, cx, y_start)
        ground_z  = terrain_z + 0.5
        node.getField("translation").setSFVec3f([cx, y_start, ground_z])
        node.resetPhysics()
        print(
            f"[supervisor] Grounded DRONE_{i} at ({cx:.1f}, {y_start:.1f}, {ground_z:.2f}m)"
            f" — terrain={terrain_z:.2f}m.",
            flush=True,
        )

def run() -> None:
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    config_path = _resolve_config_path()
    if not os.path.isfile(config_path):
        print(
            f"[supervisor] ERROR: config not found at {config_path!r}. "
            "Ensure simulation/configs/simulation.yaml exists.",
            flush=True,
        )
        sys.exit(1)

    config = _load_yaml(config_path)
    _inject_drones(supervisor, config, timestep)
    assignments = _build_drone_assignments(config)
    num_drones = config["simulation"]["num_drones"]

    drone_nodes = []
    for assign in assignments:
        drone_id = assign["drone_id"]
        def_name = drone_id.upper()
        node = supervisor.getFromDef(def_name)
        if node is None:
            print(
                f"[supervisor] WARNING: DEF '{def_name}' not found in world. "
                "Check that the .wbt file defines nodes with matching DEF names.",
                flush=True,
            )
            drone_nodes.append(None)
            continue

        payload = json.dumps({
            "drone_id": assign["drone_id"],
            "waypoints": assign["waypoints"],
        })
        node.getField("customData").setSFString(payload)
        drone_nodes.append(node)
        print(
            f"[supervisor] Assigned {len(assign['waypoints'])} waypoints "
            f"to {drone_id} (DEF {def_name}).",
            flush=True,
        )

    done_flags = [False] * num_drones
    print(f"[supervisor] All waypoints distributed. Monitoring {num_drones} drones.", flush=True)

    while supervisor.step(timestep) != -1:
        for i, node in enumerate(drone_nodes):
            if done_flags[i] or node is None:
                continue

            raw = node.getField("customData").getSFString()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if parsed.get("status") == "DONE":
                done_flags[i] = True
                print(
                    f"[supervisor] drone_{i} reported DONE "
                    f"(sim time {supervisor.getTime():.1f}s).",
                    flush=True,
                )

        if all(done_flags[i] for i in range(num_drones)):
            print("[supervisor] All drones complete. Pausing simulation.", flush=True)
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break

if __name__ == "__main__":
    run()
