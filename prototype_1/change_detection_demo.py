"""
Structural change detection prototype: T1 vs T2 point clouds -> aligned -> changes -> records -> report.

Pipeline
  1. Build a labelled truss bridge (T1) and a later scan (T2) with 3 injected changes:
     a removed diagonal, an added object, and a displaced vertical. T2 is also moved,
     rotated and noisy, like a real second scan.
  2. Align: FPFH features + RANSAC (coarse), then point-to-plane ICP (fine).
     A second ICP pass uses only stable points, so the changes don't bias the alignment.
  3. Detect: cloud-to-cloud distances in both directions + DBSCAN clustering.
  4. Interpret: each cluster becomes a structured change record (type, component, size, confidence).
  5. Evaluate against the injected ground truth: precision, recall, F1, point IoU, alignment error.
  6. Report: sentences generated only from records, each citing its record ID; optional LLM
     (OpenAI) with a check that every sentence cites a valid record.

Run:  pip install open3d matplotlib numpy  &&  python change_detection_demo.py
Outputs go to ./outputs (figures, change_records.json, report.md, metrics.json).
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
VOXEL = 0.10            # m, working resolution
CHANGE_THR = 0.25       # m, distance above which a point counts as changed


# ---------------------------------------------------------------- 1. synthetic data
def member(p, q, r=0.06, density=900):
    """Sample points on the surface of a cylinder (a structural member) from p to q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p; L = np.linalg.norm(d); d /= L
    a = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u); v = np.cross(d, u)
    n = int(density * L)
    t = rng.uniform(0, L, n); th = rng.uniform(0, 2 * np.pi, n)
    return p + np.outer(t, d) + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))


def box(center, size, n=3000):
    c, s = np.asarray(center, float), np.asarray(size, float)
    pts = rng.uniform(-0.5, 0.5, (n, 3)); f = rng.integers(0, 3, n)
    pts[np.arange(n), f] = rng.choice([-0.5, 0.5], n)          # push onto a face
    return c + pts * s


def truss(shift_vertical=None):
    """Two parallel Pratt trusses + deck beams. Returns {component_name: points}."""
    parts = {}
    xs = np.arange(0, 13, 2.0); H = 2.0
    for side, y in (("north", 0.0), ("south", 3.0)):
        for i in range(len(xs) - 1):
            parts[f"{side} bottom chord {i+1}"] = member([xs[i], y, 0], [xs[i+1], y, 0])
            if 0 < i < len(xs) - 2:
                parts[f"{side} top chord {i}"] = member([xs[i], y, H], [xs[i+1], y, H])
        parts[f"{side} end post W"] = member([xs[0], y, 0], [xs[1], y, H])
        parts[f"{side} end post E"] = member([xs[-1], y, 0], [xs[-2], y, H])
        for i in range(1, len(xs) - 1):
            top = [xs[i], y, H]
            if shift_vertical == (side, i):
                top = [xs[i] + 0.45, y, H]                        # displaced member
            parts[f"{side} vertical {i}"] = member([xs[i], y, 0], top)
        for i in range(1, len(xs) - 2):
            a, b = ([xs[i], y, H], [xs[i+1], y, 0]) if xs[i] < 6 else ([xs[i], y, 0], [xs[i+1], y, H])
            parts[f"{side} diagonal {i}"] = member(a, b)
    for i, x in enumerate(xs):
        parts[f"floor beam {i+1}"] = member([x, 0, 0], [x, 3.0, 0])
    return parts


def stack(parts):
    names = list(parts)
    pts = np.vstack([parts[n] for n in names])
    lab = np.concatenate([[k] * len(parts[n]) for k, n in enumerate(names)])
    return pts, lab, names


t1_parts = truss()
t1_pts, t1_lab, names = stack(t1_parts)

# T2 = T1 with three real changes
t2_parts = truss(shift_vertical=("south", 4))
removed_name = "north diagonal 4"
del t2_parts[removed_name]
added_pts = box([5.0, 1.5, 0.6], [0.8, 0.8, 1.2])               # e.g. equipment left on the deck
t2_pts = np.vstack(list(t2_parts.values()) + [added_pts])
t2_pts += rng.normal(0, 0.01, t2_pts.shape)                     # 1 cm scanner noise

GT = [  # ground truth for evaluation
    {"type": "removed", "component": removed_name, "center": t1_parts[removed_name].mean(0)},
    {"type": "added", "component": "unknown object on deck", "center": added_pts.mean(0)},
    {"type": "altered", "component": "south vertical 4", "center": t1_parts["south vertical 4"].mean(0)},
]

# the second scan sits in its own coordinate frame
def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
T_true = np.eye(4); T_true[:3, :3] = rot(0.05, -0.04, 0.6); T_true[:3, 3] = [3.0, -1.5, 0.8]
t2_scan = t2_pts @ T_true[:3, :3].T + T_true[:3, 3]


# ---------------------------------------------------------------- 2. registration
def pcd(p):
    c = o3d.geometry.PointCloud(); c.points = o3d.utility.Vector3dVector(p); return c


def prep(c):
    d = c.voxel_down_sample(VOXEL)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 3, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 6, max_nn=100))
    return d, f


src, tgt = pcd(t2_scan), pcd(t1_pts)
src_d, src_f = prep(src); tgt_d, tgt_f = prep(tgt)
reg = o3d.pipelines.registration
ransac = reg.registration_ransac_based_on_feature_matching(
    src_d, tgt_d, src_f, tgt_f, True, VOXEL * 2.0,
    reg.TransformationEstimationPointToPoint(False), 3,
    [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(VOXEL * 2.0)],
    reg.RANSACConvergenceCriteria(400000, 0.999))
icp = reg.registration_icp(src_d, tgt_d, VOXEL * 1.5, ransac.transformation,
                           reg.TransformationEstimationPointToPlane())


def distances(a, b):
    """For each point of a, distance to the nearest point of b."""
    return np.asarray(pcd(a).compute_point_cloud_distance(pcd(b)))


def apply(T, p): return p @ T[:3, :3].T + T[:3, 3]

# robust second pass: re-run ICP on stable points only
aligned = apply(icp.transformation, t2_scan)
stable = distances(aligned, t1_pts) < CHANGE_THR
s_d = pcd(t2_scan[stable]).voxel_down_sample(VOXEL)
icp2 = reg.registration_icp(s_d, tgt_d, VOXEL * 1.0, icp.transformation,
                            reg.TransformationEstimationPointToPlane())
T_est = icp2.transformation
aligned = apply(T_est, t2_scan)

T_err = T_est @ T_true                       # should be identity
rot_err = np.degrees(np.arccos(np.clip((np.trace(T_err[:3, :3]) - 1) / 2, -1, 1)))
trans_err = np.linalg.norm(T_err[:3, 3])


# ---------------------------------------------------------------- 3. change detection
d_t2_to_t1 = distances(aligned, t1_pts)      # big -> new in T2 (added / moved-to)
d_t1_to_t2 = distances(t1_pts, aligned)      # big -> gone from T1 (removed / moved-from)


def clusters(points, mask, eps=0.3, min_pts=60):
    if mask.sum() == 0: return []
    lab = np.array(pcd(points[mask]).cluster_dbscan(eps=eps, min_points=min_pts))
    idx = np.where(mask)[0]
    return [idx[lab == k] for k in range(lab.max() + 1)]


new_cl = clusters(aligned, d_t2_to_t1 > CHANGE_THR)
gone_cl = clusters(t1_pts, d_t1_to_t2 > CHANGE_THR)


# ---------------------------------------------------------------- 4. change records
def component_of(idx):
    """Name the T1 component most of these T1 points belong to (stands in for a BIM / labelled model)."""
    k = np.bincount(t1_lab[idx]).argmax()
    return names[k], float((t1_lab[idx] == k).mean())


def nearest_component(points):
    d = distances(points, t1_pts); near = np.asarray(pcd(t1_pts).compute_point_cloud_distance(pcd(points)))
    k = np.bincount(t1_lab[near < np.percentile(near, 2)]).argmax()
    return names[k]


records, used_new = [], set()
for g in gone_cl:
    comp, purity = component_of(g)
    c_gone = t1_pts[g].mean(0)
    match = None                             # a nearby "new" cluster means the part moved
    for j, n in enumerate(new_cl):
        if j not in used_new and np.linalg.norm(aligned[n].mean(0) - c_gone) < 1.2:
            match = j; break
    if match is not None:
        used_new.add(match)
        shift = float(np.linalg.norm(aligned[new_cl[match]].mean(0) - c_gone))
        records.append({"type": "altered", "component": comp, "center": c_gone,
                        "displacement_m": round(shift, 2), "n_points": int(len(g)),
                        "confidence": round(0.5 + 0.5 * purity, 2)})
    else:
        records.append({"type": "removed", "component": comp, "center": c_gone,
                        "length_m": round(float(np.ptp(t1_pts[g], 0).max()), 2), "n_points": int(len(g)),
                        "confidence": round(0.5 + 0.5 * purity, 2)})
for j, n in enumerate(new_cl):
    if j in used_new: continue
    ext = np.ptp(aligned[n], 0)
    records.append({"type": "added", "component": "unknown object", "center": aligned[n].mean(0),
                    "size_m": [round(float(e), 2) for e in ext], "n_points": int(len(n)),
                    "near": nearest_component(aligned[n]), "confidence": 0.8})
for i, r in enumerate(records):
    r["id"] = f"C-{i+1:02d}"
    r["location_m"] = [round(float(v), 2) for v in r.pop("center")]


# ---------------------------------------------------------------- 5. evaluation
def match_gt(records, gt, tol=1.0):
    tp, hit = 0, set()
    for r in records:
        for k, g in enumerate(gt):
            if k not in hit and r["type"] == g["type"] and np.linalg.norm(np.array(r["location_m"]) - g["center"]) < tol:
                tp += 1; hit.add(k); break
    return tp


tp = match_gt(records, GT); fp = len(records) - tp; fn = len(GT) - tp
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)

# point-level IoU for the removed member (predicted changed T1 points vs true member points)
true_removed = t1_lab == names.index(removed_name)
pred_gone = d_t1_to_t2 > CHANGE_THR
iou_removed = (true_removed & pred_gone).sum() / max((true_removed | pred_gone).sum(), 1)

metrics = {"alignment": {"ransac_fitness": round(ransac.fitness, 3), "icp_rmse_m": round(icp2.inlier_rmse, 4),
                         "rotation_error_deg": round(float(rot_err), 3), "translation_error_m": round(float(trans_err), 4)},
           "detection": {"TP": tp, "FP": fp, "FN": fn, "precision": round(prec, 3),
                         "recall": round(rec, 3), "F1": round(f1, 3), "IoU_removed_member": round(float(iou_removed), 3)}}


# ---------------------------------------------------------------- 6. report (grounded)
SEVERITY = {"removed": "high", "altered": "medium", "added": "low"}   # rule floor set by engineers


def template_report(records):
    lines = []
    for r in sorted(records, key=lambda r: ["high", "medium", "low"].index(SEVERITY[r["type"]])):
        sev = SEVERITY[r["type"]]
        if r["type"] == "removed":
            s = f"The {r['component']} (about {r['length_m']} m) is missing in the second scan"
        elif r["type"] == "altered":
            s = f"The {r['component']} appears displaced by about {r['displacement_m']} m"
        else:
            s = f"A new object of about {' x '.join(map(str, r['size_m']))} m appears near the {r['near']}"
        flag = " Sent to an engineer for review." if r["confidence"] < 0.85 else ""
        lines.append(f"- **{sev.upper()}** · {s} [{r['id']}], confidence {r['confidence']}.{flag}")
    return "\n".join(lines)


def llm_report(records):
    """Optional: let an LLM write the summary, but only from the records, and check every citation."""
    from openai import OpenAI
    prompt = ("You write bridge inspection summaries. Use ONLY these change records. Every sentence must end "
              "with the record id in brackets, e.g. [C-01]. Severity floor (never lower it): " + json.dumps(SEVERITY) +
              "\nRecords:\n" + json.dumps(records, indent=1))
    txt = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
                                           ).choices[0].message.content
    valid = {r["id"] for r in records}
    for sent in [s for s in txt.replace("\n", " ").split(".") if len(s.strip()) > 20]:
        cited = {c.strip("[] ") for c in sent.split("[")[1:]}
        assert cited and cited <= valid | {""}, f"Ungrounded sentence rejected: {sent}"
    return txt


report = "# Inspection change report (T1 → T2)\n\n" + template_report(records)
if os.environ.get("OPENAI_API_KEY"):
    try: report += "\n\n## LLM summary (grounded, citations checked)\n\n" + llm_report(records)
    except Exception as e: report += f"\n\n_LLM summary skipped: {e}_"

json.dump(records, open(f"{OUT}/change_records.json", "w", encoding="utf-8"), indent=2)
json.dump(metrics, open(f"{OUT}/metrics.json", "w", encoding="utf-8"), indent=2)
open(f"{OUT}/report.md", "w", encoding="utf-8").write(report)


# ---------------------------------------------------------------- figures
def show(ax, p, c, s=0.3, a=1.0):
    k = rng.choice(len(p), min(len(p), 20000), replace=False)
    ax.scatter(p[k, 0], p[k, 1], p[k, 2], c=c, s=s, alpha=a, linewidths=0)


def style(ax, title):
    ax.set_title(title, fontsize=12); ax.set_box_aspect((13, 4.5, 3)); ax.view_init(22, -62); ax.set_axis_off()


fig = plt.figure(figsize=(18, 5.5))
ax = fig.add_subplot(131, projection="3d"); show(ax, t1_pts, "#303870"); show(ax, t2_scan, "#E2793A")
style(ax, "1 · Before alignment (T1 navy, T2 orange)")
ax = fig.add_subplot(132, projection="3d"); show(ax, t1_pts, "#303870"); show(ax, aligned, "#E2793A", a=0.5)
style(ax, f"2 · After RANSAC + ICP (rot. error {rot_err:.2f}°)")
ax = fig.add_subplot(133, projection="3d"); show(ax, aligned, "#C9CBD6", s=0.2)
col = {"removed": "#D62728", "added": "#2CA02C", "altered": "#E2A11B"}
for r, g in zip([r for r in records if r["type"] != "added"], gone_cl):
    show(ax, t1_pts[g], col[r["type"]], s=2)
for j, n in enumerate(new_cl):
    show(ax, aligned[n], col["added"] if j not in used_new else col["altered"], s=2)
for r in records:
    x, y, z = r["location_m"]; ax.text(x, y, z + 1.0, r["id"], fontsize=10, weight="bold")
style(ax, "3 · Detected changes: removed red · added green · altered amber")
plt.tight_layout(); plt.savefig(f"{OUT}/pipeline.png", dpi=140)

print(json.dumps(metrics, indent=2)); print(); print(report)
