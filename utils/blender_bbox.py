# tools/blender_bbox_largest.py
import bpy
import sys
import json
from mathutils import Vector

INCHES_PER_M = 39.37007874015748


def import_glb(path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)


def object_world_bbox(obj, depsgraph):
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    if mesh is None or len(mesh.vertices) == 0:
        obj_eval.to_mesh_clear()
        return None, None

    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))

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


def sorted_desc(t):
    a = sorted(list(t), reverse=True)
    return (a[0], a[1], a[2])


def parse_args(argv):
    if "--" not in argv:
        raise SystemExit("Usage: blender -b -P tools/blender_bbox_largest.py -- model.glb")

    idx = argv.index("--") + 1
    if idx >= len(argv):
        raise SystemExit("Missing model.glb after --")

    return argv[idx]


def largest_mesh_bbox():
    depsgraph = bpy.context.evaluated_depsgraph_get()

    best = None
    best_volume = -1.0

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        mn, mx = object_world_bbox(obj, depsgraph)
        if mn is None or mx is None:
            continue

        size = mx - mn
        volume = abs(size.x * size.y * size.z)

        if volume > best_volume:
            best_volume = volume
            best = {
                "name": obj.name,
                "min": mn,
                "max": mx,
                "size": size,
                "volume": volume,
            }

    if best is None:
        raise RuntimeError("No mesh objects with vertices were found in the scene.")

    return best


def main():
    glb_path = parse_args(sys.argv)
    import_glb(glb_path)

    best = largest_mesh_bbox()
    size = best["size"]

    glb_abs_in_xyz = (
        abs(size.x) * INCHES_PER_M,
        abs(size.y) * INCHES_PER_M,
        abs(size.z) * INCHES_PER_M,
    )
    glb_abs_sorted = sorted_desc(glb_abs_in_xyz)

    out = {
        "largest_mesh_name": best["name"],
        "glb_absolute": {
            "a": glb_abs_sorted[0],
            "b": glb_abs_sorted[1],
            "c": glb_abs_sorted[2],
        },
        "debug": {
            "glb_abs_in_xyz": {
                "x": glb_abs_in_xyz[0],
                "y": glb_abs_in_xyz[1],
                "z": glb_abs_in_xyz[2],
            },
            "bbox_min_m": {
                "x": best["min"].x,
                "y": best["min"].y,
                "z": best["min"].z,
            },
            "bbox_max_m": {
                "x": best["max"].x,
                "y": best["max"].y,
                "z": best["max"].z,
            },
            "bbox_size_m": {
                "x": best["size"].x,
                "y": best["size"].y,
                "z": best["size"].z,
            },
            "bbox_volume_m3": best["volume"],
        },
    }

    print(json.dumps(out))


if __name__ == "__main__":
    main()