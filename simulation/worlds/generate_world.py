import math
import random
import os

SEED = 42
MAP_HALF = 400
GRID_N = 96
HILL_AMP = 12.0
ROAD_WIDTH = 6.0
NUM_TREE_CLUSTERS = 30
TREES_PER_CLUSTER = 28
CLUSTER_RADIUS = 35.0
NUM_BUILDINGS = 30

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "drone_swarm.wbt")

random.seed(SEED)

def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a, b, t):
    return a + t * (b - a)

def _grad(h, x, y):
    h &= 3
    if h == 0: return  x + y
    if h == 1: return -x + y
    if h == 2: return  x - y
    return              -x - y

_PERM = list(range(256))
random.shuffle(_PERM)
_PERM += _PERM

def perlin2(x, y):
    xi, yi = int(math.floor(x)) & 255, int(math.floor(y)) & 255
    xf, yf = x - math.floor(x), y - math.floor(y)
    u, v = _fade(xf), _fade(yf)
    aa = _PERM[_PERM[xi    ] + yi    ]
    ab = _PERM[_PERM[xi    ] + yi + 1]
    ba = _PERM[_PERM[xi + 1] + yi    ]
    bb = _PERM[_PERM[xi + 1] + yi + 1]
    return _lerp(
        _lerp(_grad(aa, xf,     yf    ), _grad(ba, xf - 1, yf    ), u),
        _lerp(_grad(ab, xf,     yf - 1), _grad(bb, xf - 1, yf - 1), u),
        v
    )

def fbm(x, y, octaves=5, persistence=0.5, lacunarity=2.0):
    value, amp, freq = 0.0, 1.0, 1.0
    for _ in range(octaves):
        value += perlin2(x * freq, y * freq) * amp
        amp   *= persistence
        freq  *= lacunarity
    return value

def build_heightmap():
    heights = []
    for row in range(GRID_N):
        for col in range(GRID_N):
            nx = col / (GRID_N - 1) * 4.0
            ny = row / (GRID_N - 1) * 4.0
            h = fbm(nx + 0.3, ny + 0.7) * HILL_AMP
            world_x = (col / (GRID_N - 1) - 0.5) * 2 * MAP_HALF
            if abs(world_x) < ROAD_WIDTH * 0.8:
                h *= max(0.0, (abs(world_x) - ROAD_WIDTH * 0.3) / (ROAD_WIDTH * 0.5))
            world_y = (row / (GRID_N - 1) - 0.5) * 2 * MAP_HALF
            if abs(world_y) < ROAD_WIDTH * 0.8:
                h *= max(0.0, (abs(world_y) - ROAD_WIDTH * 0.3) / (ROAD_WIDTH * 0.5))
            heights.append(round(h, 4))
    return heights

def height_at(world_x, world_y, heights):
    col_f = (world_x / MAP_HALF + 1.0) * 0.5 * (GRID_N - 1)
    row_f = (world_y / MAP_HALF + 1.0) * 0.5 * (GRID_N - 1)
    col_f = max(0.0, min(GRID_N - 1.001, col_f))
    row_f = max(0.0, min(GRID_N - 1.001, row_f))
    c0, r0 = int(col_f), int(row_f)
    tc, tr = col_f - c0, row_f - r0
    def h(r, c): return heights[r * GRID_N + c]
    return _lerp(_lerp(h(r0, c0), h(r0, c0+1), tc),
                 _lerp(h(r0+1, c0), h(r0+1, c0+1), tc), tr)

def rand_pos(margin=20):
    x = random.uniform(-MAP_HALF + margin, MAP_HALF - margin)
    y = random.uniform(-MAP_HALF + margin, MAP_HALF - margin)
    return x, y

def away_from_road(margin=20):
    road_clear = ROAD_WIDTH * 2 + margin
    while True:
        x, y = rand_pos()
        if abs(x) > road_clear and abs(y) > road_clear:
            return x, y

def fmt_vec3(x, y, z):
    return f"{x:.3f} {y:.3f} {z:.3f}"

_EXTERNPROTOS = """\
IMPORTABLE EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/dji/mavic/protos/Mavic2Pro.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/trees/protos/SimpleTree.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/trees/protos/Pine.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/trees/protos/Oak.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/road/protos/StraightRoadSegment.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/buildings/protos/BungalowStyleHouse.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/buildings/protos/ResidentialBuilding.proto"
"""

_GRASS_TEXTURE = (
    "https://raw.githubusercontent.com/cyberbotics/webots/R2025a"
    "/projects/default/worlds/textures/grass.jpg"
)

def write_header(f):
    f.write('#VRML_SIM R2025a utf8\n')
    f.write(_EXTERNPROTOS)
    f.write('WorldInfo {\n')
    f.write('  title "Drone Swarm Panorama World"\n')
    f.write('  info [\n')
    f.write('    "800x800 m procedurally generated terrain (ENU coordinates)."\n')
    f.write(f'    "X=East [{-MAP_HALF},{MAP_HALF}]  Y=North [{-MAP_HALF},{MAP_HALF}]  Z=Up terrain [0,{HILL_AMP:.0f}]."\n')
    f.write('  ]\n')
    f.write('  basicTimeStep 8\n')
    f.write('  coordinateSystem "ENU"\n')
    f.write('}\n')

def write_viewpoint(f):
    f.write('Viewpoint {\n')
    f.write('  position 0 0 900\n')
    f.write('}\n')

def write_background(f):
    f.write('TexturedBackground {\n')
    f.write('  texture "noon_cloudy_countryside"\n')
    f.write('}\n')
    f.write('TexturedBackgroundLight {\n')
    f.write('  texture "noon_cloudy_countryside"\n')
    f.write('}\n')

def write_terrain(f, heights):
    cell = (MAP_HALF * 2) / (GRID_N - 1)
    f.write('DEF TERRAIN Solid {\n')
    f.write(f'  translation {-MAP_HALF:.1f} {-MAP_HALF:.1f} 0\n')
    f.write('  children [\n')
    f.write('    Shape {\n')
    f.write('      appearance PBRAppearance {\n')
    f.write('        baseColorMap ImageTexture {\n')
    f.write(f'          url "{_GRASS_TEXTURE}"\n')
    f.write('        }\n')
    f.write('        roughness 1\n')
    f.write('        metalness 0\n')
    f.write('        textureTransform TextureTransform {\n')
    f.write('          scale 20 20\n')
    f.write('        }\n')
    f.write('      }\n')
    f.write('      geometry ElevationGrid {\n')
    f.write(f'        xDimension {GRID_N}\n')
    f.write(f'        yDimension {GRID_N}\n')
    f.write(f'        xSpacing {cell:.4f}\n')
    f.write(f'        ySpacing {cell:.4f}\n')
    f.write('        height [\n')
    rows = [heights[r * GRID_N:(r + 1) * GRID_N] for r in range(GRID_N)]
    for row in rows:
        f.write('          ' + ' '.join(str(v) for v in row) + '\n')
    f.write('        ]\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('  ]\n')
    f.write('  boundingObject ElevationGrid {\n')
    f.write(f'    xDimension {GRID_N}\n')
    f.write(f'    yDimension {GRID_N}\n')
    f.write(f'    xSpacing {cell:.4f}\n')
    f.write(f'    ySpacing {cell:.4f}\n')
    f.write('    height [\n')
    for row in rows:
        f.write('      ' + ' '.join(str(v) for v in row) + '\n')
    f.write('    ]\n')
    f.write('  }\n')
    f.write('}\n')

def write_roads(f):
    seg_len = 40.0
    num_segs = int(MAP_HALF * 2 / seg_len)
    half = MAP_HALF - 10

    for i in range(num_segs):
        y_start = -half + i * seg_len
        f.write('StraightRoadSegment {\n')
        f.write(f'  name "road_ns_{i}"\n')
        f.write(f'  translation 0 {y_start:.1f} 0.05\n')
        f.write('  rotation 0 0 1 1.5708\n')
        f.write(f'  length {seg_len:.1f}\n')
        f.write(f'  width {ROAD_WIDTH:.1f}\n')
        f.write('  numberOfLanes 2\n')
        f.write('  lines []\n')
        f.write('  rightBorder FALSE\n')
        f.write('  leftBorder FALSE\n')
        f.write('}\n')

    for i in range(num_segs):
        x_start = -half + i * seg_len
        f.write('StraightRoadSegment {\n')
        f.write(f'  name "road_ew_{i}"\n')
        f.write(f'  translation {x_start:.1f} 0 0.05\n')
        f.write('  rotation 0 0 1 0\n')
        f.write(f'  length {seg_len:.1f}\n')
        f.write(f'  width {ROAD_WIDTH:.1f}\n')
        f.write('  numberOfLanes 2\n')
        f.write('  lines []\n')
        f.write('  rightBorder FALSE\n')
        f.write('  leftBorder FALSE\n')
        f.write('}\n')

def write_forests(f, heights):
    tree_protos = ['SimpleTree', 'Pine', 'Oak']
    node_counter = 0
    for cluster_idx in range(NUM_TREE_CLUSTERS):
        cx, cy = away_from_road(margin=30)
        for t in range(TREES_PER_CLUSTER):
            angle  = random.uniform(0, math.tau)
            radius = random.uniform(0, CLUSTER_RADIUS)
            tx = cx + math.cos(angle) * radius
            ty = cy + math.sin(angle) * radius
            tx = max(-MAP_HALF + 5, min(MAP_HALF - 5, tx))
            ty = max(-MAP_HALF + 5, min(MAP_HALF - 5, ty))
            tz = height_at(tx, ty, heights)
            scale  = random.uniform(0.6, 1.8)
            proto  = random.choice(tree_protos)
            rot_z  = random.uniform(0, math.tau)

            f.write(f'{proto} {{\n')
            if node_counter > 0:
                f.write(f'  name "{proto.lower()}_{node_counter}"\n')
            f.write(f'  translation {fmt_vec3(tx, ty, tz)}\n')
            f.write(f'  rotation 0 0 1 {rot_z:.4f}\n')
            if proto == 'SimpleTree':
                f.write(f'  height {scale * 6:.2f}\n')
                f.write('  enableBoundingObject FALSE\n')
            f.write('}\n')
            node_counter += 1

def write_buildings(f, heights):
    building_protos = ['BungalowStyleHouse', 'ResidentialBuilding']
    placed = 0
    attempts = 0
    while placed < NUM_BUILDINGS and attempts < 500:
        attempts += 1
        side     = random.choice([-1, 1])
        bx       = side * random.uniform(ROAD_WIDTH + 3, ROAD_WIDTH + 30)
        by       = random.uniform(-MAP_HALF + 20, MAP_HALF - 20)
        bz       = height_at(bx, by, heights)
        proto    = random.choice(building_protos)
        rot_z    = random.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        tag      = "bungalow" if "Bungalow" in proto else "residential"
        f.write(f'{proto} {{\n')
        f.write(f'  name "{tag}_{placed}"\n')
        f.write(f'  translation {fmt_vec3(bx, by, bz)}\n')
        f.write(f'  rotation 0 0 1 {rot_z:.4f}\n')
        f.write('}\n')
        placed += 1

def write_supervisor(f):
    f.write('Robot {\n')
    f.write('  name "swarm_supervisor"\n')
    f.write('  supervisor TRUE\n')
    f.write('  controller "swarm_supervisor"\n')
    f.write('  translation 0 0 1\n')
    f.write('  children [\n')
    f.write('  ]\n')
    f.write('}\n')

def write_drone_placeholder(f):
    f.write('\n')
    f.write('# =============================================================\n')
    f.write('# DRONE INJECTION POINT  (ENU: X=East, Y=North, Z=Up/altitude)\n')
    f.write('# The swarm_supervisor injects Mavic2Pro nodes here at runtime\n')
    f.write('# via importMFNodeFromString.  IMPORTABLE EXTERNPROTO above is\n')
    f.write('# required for this to work.\n')
    f.write('#\n')
    f.write('# Manual entry example:\n')
    f.write('#   Mavic2Pro {\n')
    f.write('#     name "drone_0"\n')
    f.write('#     translation -133 -195 0.5\n')
    f.write('#     controller "mavic_controller"\n')
    f.write('#   }\n')
    f.write('# =============================================================\n')
    f.write('DEF DRONE_GROUP Group {\n')
    f.write('  children [\n')
    f.write('  ]\n')
    f.write('}\n')

def generate():
    print("Building heightmap …")
    heights = build_heightmap()

    print(f"Writing world to {OUTPUT_PATH} …")
    with open(OUTPUT_PATH, 'w') as f:
        write_header(f)
        write_viewpoint(f)
        write_background(f)
        write_terrain(f, heights)
        write_roads(f)
        write_forests(f, heights)
        write_buildings(f, heights)
        write_supervisor(f)
        write_drone_placeholder(f)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Done. File size: {size / 1024:.1f} KB")
    print()
    print("World coordinate bounds (ENU):")
    print(f"  X (East):  [{-MAP_HALF}, {MAP_HALF}] m")
    print(f"  Y (North): [{-MAP_HALF}, {MAP_HALF}] m")
    print(f"  Z (Up):    ~[0, {HILL_AMP}] m  (hills, Perlin fBm)")
    print(f"  Grid:      {GRID_N}×{GRID_N} ElevationGrid, {MAP_HALF*2/(GRID_N-1):.2f} m/cell")
    print()
    print("Suggested simulation.yaml updates for 4 drones at 20 m altitude:")
    print("  num_drones: 4")
    print("  altitude_m: 20.0")
    print("  x_min: -100.0   x_max: 100.0")
    print("  y_min: -300.0   y_max: 300.0")

if __name__ == '__main__':
    generate()
