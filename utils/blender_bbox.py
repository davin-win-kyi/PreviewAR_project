# tools/blender_bbox.py
import bpy
import sys
import json
from mathutils import Vector

# Conversion constants
INCHES_PER_M = 39.37007874015748
INCHES_PER_CM = 0.3937007874015748
INCHES_PER_MM = 0.03937007874015748

def import_glb(path: str):
    # Reset scene then import GLB
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)

def scene_world_bbox():
    """
    Computes world-space bounding box over all MESH objects in the scene.
    Returns (min_v, max_v) in Blender units (typically meters after glTF import).
    """
    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        for v in mesh.vertices:
            w = obj_eval.matrix_world @ v.co
            min_v.x = min(min_v.x, w.x)
            min_v.y = min(min_v.y, w.y)
            min_v.z = min(min_v.z, w.z)
            max_v.x = max(max_v.x, w.x)
            max_v.y = max(max_v.y, w.y)
            max_v.z = max(max_v.z, w.z)

        obj_eval.to_mesh_clear()

    return min_v, max_v

def parse_args(argv):
    """
    Usage:
      blender -b -P tools/blender_bbox.py -- model.glb
      blender -b -P tools/blender_bbox.py -- model.glb --gt L W H --gt_units in|cm|mm|m
    """
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender -b -P tools/blender_bbox.py -- model.glb "
            "[--gt L W H] [--gt_units in|cm|mm|m]"
        )

    idx = argv.index("--") + 1
    if idx >= len(argv):
        raise SystemExit("Missing model.glb after --")

    glb_path = argv[idx]
    rest = argv[idx + 1 :]

    gt = None
    gt_units = "in"

    i = 0
    while i < len(rest):
        if rest[i] == "--gt":
            if i + 3 >= len(rest):
                raise SystemExit("--gt requires 3 numbers: L W H")
            gt = (float(rest[i+1]), float(rest[i+2]), float(rest[i+3]))
            i += 4
            continue
        if rest[i] == "--gt_units":
            if i + 1 >= len(rest):
                raise SystemExit("--gt_units requires one of: in, cm, mm, m")
            gt_units = rest[i+1].lower().replace(".", "")
            i += 2
            continue
        i += 1

    return glb_path, gt, gt_units

def inches_per_unit(u: str):
    u = (u or "").strip().lower().replace(".", "")
    if u in ("in", "inch", "inches"):
        return 1.0
    if u in ("cm", "centimeter", "centimeters"):
        return INCHES_PER_CM
    if u in ("mm", "millimeter", "millimeters"):
        return INCHES_PER_MM
    if u in ("m", "meter", "meters", "metre", "metres"):
        return INCHES_PER_M
    return None

def sorted_desc(t):
    a = sorted(list(t), reverse=True)
    return (a[0], a[1], a[2])

def best_fit_uniform_scale(d, g):
    # s* = (g·d)/(d·d)
    num = g[0] * d[0] + g[1] * d[1] + g[2] * d[2]
    den = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    return (num / den) if den > 0 else 0.0

def main():
    glb_path, gt, gt_units = parse_args(sys.argv)

    import_glb(glb_path)
    mn, mx = scene_world_bbox()
    size = mx - mn  # Blender units (usually meters for glTF)

    # GLB absolute extents in inches (x,y,z then sorted)
    glb_abs_in_xyz = (
        abs(size.x) * INCHES_PER_M,
        abs(size.y) * INCHES_PER_M,
        abs(size.z) * INCHES_PER_M,
    )
    glb_abs_sorted = sorted_desc(glb_abs_in_xyz)

    # If no GT provided, just return GLB absolute
    if gt is None:
        out = {
            "glb_absolute": {
                "a": glb_abs_sorted[0],
                "b": glb_abs_sorted[1],
                "c": glb_abs_sorted[2],
            },
            "glb_relative": None,
            "actual_absolute": None,
            "actual_relative": None,
        }
        print(json.dumps(out))
        return

    mult = inches_per_unit(gt_units)
    if mult is None:
        raise SystemExit(f"Unsupported gt_units: {gt_units}")

    actual_abs_in = (gt[0] * mult, gt[1] * mult, gt[2] * mult)
    actual_sorted = sorted_desc(actual_abs_in)

    # Relative: best-fit uniform scaling of GLB -> GT
    s_star = best_fit_uniform_scale(glb_abs_sorted, actual_sorted)
    glb_rel = (
        s_star * glb_abs_sorted[0],
        s_star * glb_abs_sorted[1],
        s_star * glb_abs_sorted[2],
    )

    out = {
        "glb_absolute": {
            "a": glb_abs_sorted[0],
            "b": glb_abs_sorted[1],
            "c": glb_abs_sorted[2],
        },
        "glb_relative": {
            "a": glb_rel[0],
            "b": glb_rel[1],
            "c": glb_rel[2],
            "scale": s_star,
        },
        "actual_absolute": {
            "a": actual_sorted[0],
            "b": actual_sorted[1],
            "c": actual_sorted[2],
        },
        "actual_relative": {
            "a": actual_sorted[0],
            "b": actual_sorted[1],
            "c": actual_sorted[2],
        },
        # Optional debug: keep unsorted xyz extents too
        "debug": {
            "glb_abs_in_xyz": {"x": glb_abs_in_xyz[0], "y": glb_abs_in_xyz[1], "z": glb_abs_in_xyz[2]},
            "gt_units": gt_units,
        },
    }

    print(json.dumps(out))

if __name__ == "__main__":
    main()