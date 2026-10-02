"""
Dataset registration for domain-generalized urban-scene segmentation.

All target-domain labels are converted ONCE to Cityscapes 19-class train_id
PNGs in a sibling `labels_19class/` directory, then datasets are registered
against the converted directory. The conversion is idempotent — it skips
files that already exist in the destination.

Why offline conversion: detectron2 reads `sem_seg_file_name` as-is. The
`label_map` field in MetadataCatalog is informational only and does NOT
remap labels at load time. The labels on disk must already be in
{0..18, 255} for the SemSegEvaluator to produce meaningful mIoU.

After import, run `verify_datasets.py` to confirm that each registered
dataset's labels contain only the expected integer values.
"""

import os
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog


# ============================================================================
# 19-class Cityscapes label space
# ============================================================================

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]

CITYSCAPES_COLORS = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
]
IGNORE_LABEL = 255


# ============================================================================
# Lookup tables for label conversion
# Each LUT is a length-256 uint8 array: LUT[src_id] = dst_train_id.
# Anything not explicitly mapped defaults to IGNORE_LABEL (255).
# ============================================================================

def _build_lut(mapping: dict, default: int = IGNORE_LABEL) -> np.ndarray:
    """Build a 256-entry uint8 LUT from a {src_id: dst_id} dict."""
    lut = np.full(256, default, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    lut[IGNORE_LABEL] = IGNORE_LABEL
    return lut


# ---- BDD100K ---------------------------------------------------------------
# BDD100K's labels/sem_seg/masks/ contains 19-class train_id PNGs in {0..18, 255}.
# Identity LUT — included only for the same offline-pass code path.
BDD_LUT = _build_lut({i: i for i in range(19)})


# ---- GTA5 (Playing for Data) -----------------------------------------------
# Original labels are RGB-encoded with the Cityscapes color palette
# (3-channel PNGs). We convert via the inverse palette below to 19-class
# train_id. If your `labels/` directory instead contains grayscale eval_id
# PNGs, the 2D-branch in `_convert_one_label` handles that via GTA5_EVAL_LUT.

# eval_id (0..33) -> train_id (0..18, 255)
GTA5_EVAL_LUT = _build_lut({
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    31: 16, 32: 17, 33: 18,
})

# RGB tuple (Cityscapes palette) -> train_id (0..18)
CITYSCAPES_RGB_TO_TRAIN_ID = {tuple(c): i for i, c in enumerate(CITYSCAPES_COLORS)}


# ---- Mapillary Vistas v1.2 -------------------------------------------------
# Canonical 66-class -> 19-class mapping. Cross-reference against
# RobustNet's `datasets/mapillary_labels.py` if anything looks off.
MAPILLARY_V12_TO_19 = {
    13: 0,    # Construction--Flat--Road
    24: 0,    # Construction--Flat--Service Lane     -> road
    41: 0,    # Marking--Crosswalk--Zebra            -> road  (some setups keep as 255)
    2:  1,    # Construction--Flat--Sidewalk
    15: 1,    # Construction--Flat--Pedestrian Area  -> sidewalk
    17: 2,    # Construction--Structure--Building
    6:  3,    # Construction--Barrier--Wall
    3:  4,    # Construction--Barrier--Fence
    45: 5,    # Object--Support--Pole
    47: 5,    # Object--Support--Utility Pole        -> pole
    48: 6,    # Object--Traffic Light
    50: 7,    # Object--Traffic Sign (Front)
    30: 8,    # Nature--Vegetation
    29: 9,    # Nature--Terrain
    27: 10,   # Nature--Sky
    19: 11,   # Human--Person
    20: 12,   # Human--Rider--Bicyclist              -> rider
    21: 12,   # Human--Rider--Motorcyclist           -> rider
    22: 12,   # Human--Rider--Other Rider            -> rider
    55: 13,   # Object--Vehicle--Car
    61: 14,   # Object--Vehicle--Truck
    54: 15,   # Object--Vehicle--Bus
    58: 16,   # Object--Vehicle--On Rails            -> train
    57: 17,   # Object--Vehicle--Motorcycle
    52: 18,   # Object--Vehicle--Bicycle
}
mapillary_lut = _build_lut(MAPILLARY_V12_TO_19)


# ============================================================================
# Offline label conversion
# ============================================================================

def _rgb_to_train_id(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert an RGB-encoded Cityscapes-palette label image to train_id.
    Unknown colors are mapped to IGNORE_LABEL.
    rgb_arr: (H, W, 3) uint8.   returns: (H, W) uint8.
    """
    h, w, _ = rgb_arr.shape
    out = np.full((h, w), IGNORE_LABEL, dtype=np.uint8)
    flat = rgb_arr.reshape(-1, 3)
    # Pack RGB into a single int for fast comparison
    packed = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2].astype(np.int32)
    out_flat = out.reshape(-1)
    for color, tid in CITYSCAPES_RGB_TO_TRAIN_ID.items():
        c = (color[0] << 16) | (color[1] << 8) | color[2]
        out_flat[packed == c] = tid
    return out


def _convert_one_label(src_path: str, dst_path: str, lut_2d: np.ndarray,
                       allow_rgb: bool) -> None:
    """Read a label PNG, convert to train_id, save as single-channel uint8 PNG."""
    arr = np.array(Image.open(src_path))

    if arr.ndim == 3:
        # RGB-encoded label (e.g. GTA5 original release).
        if not allow_rgb:
            raise RuntimeError(
                f"{src_path}: unexpected 3-channel label image. "
                "Set allow_rgb=True for datasets that store labels as RGB."
            )
        out = _rgb_to_train_id(arr[..., :3])
    else:
        # 2D indexed label image (BDD masks/, Mapillary indexed PNG, GTA5 eval_id).
        out = lut_2d[arr]

    Image.fromarray(out, mode="L").save(dst_path)


def _convert_directory(src_dir: str, dst_dir: str, lut_2d: np.ndarray,
                       allow_rgb: bool = False, desc: str = "converting",
                       strip_suffix: str = "") -> int:
    """Idempotently convert every PNG in src_dir to train_id PNGs in dst_dir.

    Args:
        strip_suffix: if non-empty, remove this suffix from each filename's
                      stem before saving. Lets us normalize names like
                      '<id>_train_id.png' -> '<id>.png' so labels pair with
                      images of the same basename.
    """
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)

    src_files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    todo = []
    for sp in src_files:
        stem, _ = os.path.splitext(os.path.basename(sp))
        if strip_suffix and stem.endswith(strip_suffix):
            stem = stem[: -len(strip_suffix)]
        dp = os.path.join(dst_dir, stem + ".png")
        if not os.path.exists(dp):
            todo.append((sp, dp))

    if not todo:
        return len(src_files)

    for sp, dp in tqdm(todo, desc=f"  {desc}", unit="img"):
        _convert_one_label(sp, dp, lut_2d, allow_rgb=allow_rgb)
    return len(src_files)


# ============================================================================
# Loader helpers (point at converted label dirs; image dir is unchanged)
# ============================================================================

def _build_dicts(image_dir: str, label_dir: str,
                 image_ext: str, height: int, width: int) -> list:
    """Pair every <image>.<ext> in image_dir with <image>.png in label_dir."""
    dicts = []
    if not (os.path.isdir(image_dir) and os.path.isdir(label_dir)):
        return dicts
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])
    for idx, fname in enumerate(img_files):
        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".png")
        if not os.path.exists(lbl_path):
            continue
        rec = {"file_name": img_path, "sem_seg_file_name": lbl_path, "image_id": idx}
        if height > 0 and width > 0:
            rec["height"] = height
            rec["width"] = width
        dicts.append(rec)
    return dicts


def _set_meta(name: str) -> None:
    MetadataCatalog.get(name).set(
        stuff_classes=CITYSCAPES_CLASSES,
        stuff_colors=CITYSCAPES_COLORS,
        evaluator_type="sem_seg",
        ignore_label=IGNORE_LABEL,
    )


# ============================================================================
# Per-dataset registration
# ============================================================================

DATASETS_ROOT = os.getenv("DETECTRON2_DATASETS", "datasets")


def register_cityscapes() -> None:
    """Detectron2 auto-registers Cityscapes on import. We just confirm it's there.
    Don't touch its metadata — `evaluator_type` is correctly `cityscapes_sem_seg`
    and the 19-class names are already set by the builtin registration."""
    root = os.path.join(DATASETS_ROOT, "cityscapes")
    if not os.path.exists(root):
        print(f"[cityscapes] not found at {root}")
        return
    for split in ["train", "val"]:
        name = f"cityscapes_fine_sem_seg_{split}"
        if name in DatasetCatalog.list():
            print(f"[cityscapes] {name} already registered by detectron2 builtin")
        else:
            print(f"[cityscapes] WARNING: {name} not pre-registered. "
                  "Make sure detectron2 was imported before this module.")


def register_bdd100k() -> None:
    root = os.path.join(DATASETS_ROOT, "bdd100k")
    if not os.path.exists(root):
        print(f"[bdd100k] not found at {root}")
        return

    print("[bdd100k] preparing labels...")
    for split in ["train", "val"]:
        # Prefer the official 10k split, fall back to flat structure.
        img_dir = os.path.join(root, f"images/10k/{split}")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root, f"images/{split}")
        src_lbl = os.path.join(root, f"labels/sem_seg/masks/{split}")
        if not os.path.exists(src_lbl):
            src_lbl = os.path.join(root, f"labels/{split}")
        dst_lbl = os.path.join(root, f"labels_19class/{split}")

        n = _convert_directory(src_lbl, dst_lbl, BDD_LUT,
                               allow_rgb=False, desc=f"bdd-{split}",
                               strip_suffix="_train_id")
        if n == 0:
            print(f"[bdd100k] no labels found for {split}, skipping")
            continue

        name = f"bdd100k_sem_seg_{split}"
        DatasetCatalog.register(
            name, lambda i=img_dir, l=dst_lbl:
            _build_dicts(i, l, ".jpg", 720, 1280)
        )
        _set_meta(name)
        print(f"[bdd100k] registered {name} ({n} samples)")



def _load_gta5_validated(img_dir, label_dir):
    from PIL import Image as PILImg
    dicts, skipped = [], 0
    if not (os.path.isdir(img_dir) and os.path.isdir(label_dir)):
        return dicts
    for idx, fname in enumerate(sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))):
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".png")
        if not os.path.exists(lbl_path):
            continue
        iw, ih = PILImg.open(img_path).size
        lw, lh = PILImg.open(lbl_path).size
        if (iw, ih) != (lw, lh):
            skipped += 1
            continue
        dicts.append({"file_name": img_path, "sem_seg_file_name": lbl_path,
                      "height": ih, "width": iw, "image_id": idx})
    print(f"[gta5] {len(dicts)} valid pairs, {skipped} skipped (size mismatch)")
    return dicts

def register_gta5() -> None:
    root = os.path.join(DATASETS_ROOT, "gta5")
    if not os.path.exists(root):
        print(f"[gta5] not found at {root}")
        return

    print("[gta5] preparing labels...")
    img_dir = os.path.join(root, "images")
    src_lbl = os.path.join(root, "labels")
    dst_lbl = os.path.join(root, "labels_19class")

    # GTA5 from the Playing-for-Data release uses RGB-encoded labels with
    # the Cityscapes palette. If your labels are eval_id grayscale instead,
    # `_convert_one_label`'s 2D branch will still work via GTA5_EVAL_LUT —
    # but `allow_rgb=True` is required for the original release.
    n = _convert_directory(src_lbl, dst_lbl, GTA5_EVAL_LUT,
                           allow_rgb=True, desc="gta5")
    if n == 0:
        print("[gta5] no labels found")
        return

    # Single dataset, used as val for cross-domain eval. Using the FULL set
    # — the paper does not subsample, and 500 random images is not a
    # representative slice.
    name = "gta5_sem_seg_val"
    DatasetCatalog.register(
        name, lambda i=img_dir, l=dst_lbl:
        _load_gta5_validated(i, l)
    )
    _set_meta(name)
    print(f"[gta5] registered {name} ({n} samples)")



MAPILLARY_NAME_TO_19 = {
    "construction--flat--road": 0, "construction--flat--service-lane": 0,
    "marking--continuous--dashed": 0, "marking--discrete--crosswalk-zebra": 0,
    "construction--flat--sidewalk": 1, "construction--flat--pedestrian-area": 1,
    "construction--structure--building": 2, "construction--barrier--wall": 3,
    "construction--barrier--fence": 4, "object--support--pole": 5,
    "object--support--utility-pole": 5, "object--traffic-light": 6,
    "object--traffic-light--general-single": 6,
    "object--traffic-light--general-upright": 6,
    "object--traffic-light--pedestrians": 6,
    "object--traffic-sign--front": 7, "object--traffic-sign--direction-front": 7,
    "nature--vegetation": 8, "nature--terrain": 9, "nature--sky": 10,
    "human--person": 11, "human--person--individual": 11,
    "human--rider--bicyclist": 12, "human--rider--motorcyclist": 12,
    "human--rider--other-rider": 12, "object--vehicle--car": 13,
    "object--vehicle--truck": 14, "object--vehicle--bus": 15,
    "object--vehicle--on-rails": 16, "object--vehicle--motorcycle": 17,
    "object--vehicle--bicycle": 18,
}

def _load_mapillary_lut(mapillary_root):
    for cfg_name in ("config_v2.0.json", "config_v1.2.json", "config.json"):
        cfg_path = os.path.join(mapillary_root, cfg_name)
        if os.path.exists(cfg_path):
            break
    else:
        print("[mapillary] WARNING: no config*.json found")
        return _build_lut({})
    with open(cfg_path) as f:
        cfg = json.load(f)
    classes = cfg.get("labels", cfg.get("classes", []))
    mapping = {i: MAPILLARY_NAME_TO_19[e.get("name", "")]
               for i, e in enumerate(classes) if e.get("name", "") in MAPILLARY_NAME_TO_19}
    print(f"[mapillary] {cfg_name}: matched {len(mapping)}/{len(classes)} classes -> 19-class")
    return _build_lut(mapping)

def register_mapillary_vistas() -> None:
    root = os.path.join(DATASETS_ROOT, "mapillary_vistas")
    if not os.path.exists(root):
        print(f"[mapillary] not found at {root}")
        return

    print("[mapillary] preparing labels...")
    mapillary_lut = _load_mapillary_lut(root)
    for _n in ("mapillary_vistas_sem_seg_train", "mapillary_vistas_sem_seg_val"):
        if _n in DatasetCatalog.list():
            DatasetCatalog.remove(_n)
            print(f"[mapillary] removed existing registration for {_n}")
        if _n in MetadataCatalog.list():
            md = MetadataCatalog.get(_n)
            md.__dict__.clear()
            md.__dict__["name"] = _n
    for split, subdir in [("train", "training"), ("val", "validation")]:
        img_dir = os.path.join(root, subdir, "images")
        src_lbl = os.path.join(root, subdir, "labels")
        dst_lbl = os.path.join(root, subdir, "labels_19class")

        n = _convert_directory(src_lbl, dst_lbl, mapillary_lut,
                               allow_rgb=False, desc=f"mapillary-{split}")
        if n == 0:
            print(f"[mapillary] no labels for {split}, skipping")
            continue

        name = f"mapillary_vistas_sem_seg_{split}"
        # Mapillary images vary in size; we let detectron2's mapper handle
        # actual H/W at read time. The values below are not strictly used.
        DatasetCatalog.register(
            name, lambda i=img_dir, l=dst_lbl:
            _build_dicts(i, l, ".jpg", 0, 0)
        )
        _set_meta(name)
        print(f"[mapillary] registered {name} ({n} samples)")




def _load_acdc_sem_seg(img_root, lbl_root, conditions=None):
    """Load ACDC dataset across one or more adverse conditions.
    
    Args:
        img_root: path to rgb_anon_trainvaltest/rgb_anon
        lbl_root: path to gt_trainval/gt
        conditions: list of conditions to include, e.g. ["fog","night","rain","snow"]
                    None means all four.
    """
    import glob as _glob
    if conditions is None:
        conditions = ["fog", "night", "rain", "snow"]
    
    dicts = []
    for cond in conditions:
        img_files = sorted(_glob.glob(
            os.path.join(img_root, cond, "val", "**", "*_rgb_anon.png"),
            recursive=True))
        for img_path in img_files:
            frame_id = os.path.basename(img_path).replace("_rgb_anon.png", "")
            scene    = os.path.basename(os.path.dirname(img_path))
            lbl_path = os.path.join(lbl_root, cond, "val", scene,
                                    frame_id + "_gt_labelTrainIds.png")
            if not os.path.exists(lbl_path):
                continue
            dicts.append({
                "file_name":         img_path,
                "sem_seg_file_name": lbl_path,
                "image_id":          len(dicts),
            })
    print(f"[acdc] loaded {len(dicts)} samples from conditions: {conditions}")
    return dicts


def register_acdc():
    root = os.path.join(DATASETS_ROOT, "ACDC")
    if not os.path.exists(root):
        print(f"[acdc] not found at {root}")
        return

    img_root = os.path.join(root, "rgb_anon_trainvaltest", "rgb_anon")
    lbl_root = os.path.join(root, "gt_trainval", "gt")

    # Register all-conditions val (the standard DG eval split)
    name_all = "acdc_sem_seg_val"
    DatasetCatalog.register(
        name_all,
        lambda i=img_root, l=lbl_root: _load_acdc_sem_seg(i, l)
    )
    _set_meta(name_all)
    print(f"[acdc] registered {name_all}")

    # Register per-condition val sets for fine-grained analysis
    for cond in ["fog", "night", "rain", "snow"]:
        name = f"acdc_{cond}_sem_seg_val"
        DatasetCatalog.register(
            name,
            lambda i=img_root, l=lbl_root, c=cond: _load_acdc_sem_seg(i, l, [c])
        )
        _set_meta(name)
        print(f"[acdc] registered {name}")

# ============================================================================
# Auto-register
# ============================================================================

print("\n=== Registering DG datasets (19-class Cityscapes label space) ===")
register_cityscapes()
register_bdd100k()
register_gta5()
register_mapillary_vistas()
register_acdc()
print("=== Registration complete ===\n")