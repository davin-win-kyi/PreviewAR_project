# blender_grid_view_glbs.py
import json
import math
import sys
from pathlib import Path

import bpy


# ---------------------------
# Scene helpers
# ---------------------------
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Purge orphan data (safe to run a few times)
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except Exception:
            break


def ensure_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def link_objects_to_collection(objs, coll: bpy.types.Collection):
    for obj in objs:
        # unlink from master collection if needed
        for c in obj.users_collection:
            c.objects.unlink(obj)
        coll.objects.link(obj)


def add_label(text: str, location, scale=0.35):
    bpy.ops.object.text_add(location=location)
    t = bpy.context.object
    t.data.body = text
    t.scale = (scale, scale, scale)

    # Rotate text to face upward (so it reads from top view)
    t.rotation_euler[0] = math.radians(90)
    return t


def import_glb(filepath: str):
    # Import GLB/GLTF; imported objects are typically selected afterwards
    bpy.ops.import_scene.gltf(filepath=filepath)
    imported = list(bpy.context.selected_objects)
    return imported


def bbox_world(objs):
    """Compute combined world-space bbox of a set of objects."""
    minx = miny = minz = float("inf")
    maxx = maxy = maxz = float("-inf")

    for obj in objs:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            continue
        # bound_box is in local space; transform to world space
        for corner in obj.bound_box:
            v = obj.matrix_world @ bpy.mathutils.Vector(corner)
            minx = min(minx, v.x); miny = min(miny, v.y); minz = min(minz, v.z)
            maxx = max(maxx, v.x); maxy = max(maxy, v.y); maxz = max(maxz, v.z)

    if minx == float("inf"):
        return None

    return (minx, miny, minz), (maxx, maxy, maxz)


def move_objects(objs, dx, dy, dz=0.0):
    for obj in objs:
        obj.location.x += dx
        obj.location.y += dy
        obj.location.z += dz


# ---------------------------
# Main: layout grid
# ---------------------------
def layout_glbs_in_grid(
    glb_items,
    *,
    per_row=4,
    spacing=4.0,
    clear_first=True,
    add_text_labels=True,
    label_height=0.0,
    collection_name="RAG_RESULTS",
):
    if clear_first:
        clear_scene()

    coll = ensure_collection(collection_name)

    for i, item in enumerate(glb_items):
        path = item.get("path_to_glb")
        if not path:
            continue

        glb_path = Path(path)
        if not glb_path.exists():
            print(f"[skip] missing file: {glb_path}")
            continue

        name = item.get("clean_name") or glb_path.stem
        score = item.get("score")
        score_str = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        label = f"{name}\nscore: {score_str}"

        # import
        bpy.ops.object.select_all(action="DESELECT")
        imported = import_glb(str(glb_path))

        if not imported:
            print(f"[skip] imported nothing: {glb_path}")
            continue

        # put imported into collection
        link_objects_to_collection(imported, coll)

        row = i // per_row
        col = i % per_row
        x = col * spacing
        y = -row * spacing

        # Move imported objects to grid cell
        move_objects(imported, x, y, 0.0)

        # Add label near the object
        if add_text_labels:
            # try to place label at top-left of bbox if possible
            bbox = bbox_world(imported)
            if bbox:
                (minx, miny, minz), (maxx, maxy, maxz) = bbox
                label_loc = (minx, miny, maxz + 0.2)
            else:
                label_loc = (x, y, label_height)

            t = add_label(label, label_loc, scale=0.35)
            link_objects_to_collection([t], coll)

    print(f"Done. Imported {len(glb_items)} candidate GLBs.")


def read_glb_items_from_json(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Accept either:
    # 1) full report: {"glbs":[...], ...}
    # 2) output of extract_glb_results(): {"glbs":[...], ...}
    # 3) raw list: [...]
    if isinstance(data, dict) and "glbs" in data and isinstance(data["glbs"], list):
        return data["glbs"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized JSON schema in {json_path}")


# ---------------------------
# Entry point
# ---------------------------
def main():
    # Usage:
    # blender --python blender_grid_view_glbs.py -- outputs/relevant_glbs.json
    argv = sys.argv
    if "--" in argv:
        json_arg = argv[argv.index("--") + 1] if len(argv) > argv.index("--") + 1 else None
    else:
        json_arg = None

    json_path = Path(json_arg) if json_arg else Path("outputs/relevant_glbs.json")
    glb_items = read_glb_items_from_json(json_path)

    layout_glbs_in_grid(
        glb_items,
        per_row=4,
        spacing=5.0,
        clear_first=True,
        add_text_labels=True,
    )

if __name__ == "__main__":
    main()