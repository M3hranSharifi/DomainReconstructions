# WebApplication.py

import time
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import binary_closing, label as cc_label

import alphashape
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale as shp_scale


# ============================================================
#                    APP SETTINGS
# ============================================================
APP_TITLE = "CNN-Ready CFD Reconstruction Toolkit"

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "Logo.png"

import base64
from pathlib import Path
import streamlit as st

TEAM_DIR = BASE_DIR / "team"


def _img_to_data_uri(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    ext = img_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        mime = "application/octet-stream"

    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ============================================================
#                    CSS STYLING
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        /* =========================
           Sidebar base
           ========================= */
        [data-testid="stSidebar"] {
            background-color: #932636;
        }

        /* Sidebar labels, headings, captions: white */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: #ffffff !important;
        }

        /* =========================
           Inputs (number, text): light bg + black text
           ========================= */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background-color: rgba(255,255,255,0.92) !important;
            color: #111111 !important;
            border: 1px solid rgba(255,255,255,0.55) !important;
        }

        /* Sidebar buttons (general) */
        [data-testid="stSidebar"] button {
            background-color: rgba(255,255,255,0.18) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
            border-radius: 10px !important;
        }

        /* =========================
           Selectbox: light bg + black text (linear, NaN, ...)
           ========================= */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.92) !important;
            border: 1px solid rgba(255,255,255,0.55) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #111111 !important;
        }

        /* Dropdown menu (portal) */
        [data-baseweb="menu"] {
            background-color: rgba(255,255,255,0.98) !important;
        }
        [data-baseweb="menu"] * {
            color: #111111 !important;
        }

        /* Optional: hide caret inside selectbox */
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            caret-color: transparent !important;
        }

        /* =========================
           File uploader dropzone: light bg + black text
           ========================= */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(255,255,255,0.92) !important;
            border: 1px solid rgba(255,255,255,0.55) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #111111 !important;
        }

        /* Dropzone internal button (Browse files): black text */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background-color: rgba(255,255,255,0.95) !important;
            color: #111111 !important;
            border: 1px solid rgba(0,0,0,0.10) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
            color: #111111 !important;
        }

        /* =========================
           Uploaded file row (Data2 1.0MB): white text + icons not broken
           ========================= */
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"]{
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *{
            color: #ffffff !important;
        }

        /* Icons: avoid forcing fill on everything (prevents broken squares) */
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg{
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg *{
            stroke: currentColor !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg *[fill="none"]{
            fill: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg *[fill]:not([fill="none"]){
            fill: currentColor !important;
        }

        /* Remove button (X): override generic sidebar button styling */
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button{
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button:hover{
            background: rgba(255,255,255,0.12) !important;
        }

        /* =========================
           Checkbox styling: match +/- button style
           ========================= */
        [data-testid="stSidebar"] [data-baseweb="checkbox"] div[role="checkbox"]{
            border: 1px solid rgba(255,255,255,0.35) !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="checkbox"] div[role="checkbox"][aria-checked="false"]{
            background-color: rgba(255,255,255,0.92) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="checkbox"] div[role="checkbox"][aria-checked="true"]{
            background-color: rgba(255,255,255,0.18) !important;
            border-color: rgba(255,255,255,0.35) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="checkbox"] svg,
        [data-testid="stSidebar"] [data-baseweb="checkbox"] svg *{
            fill: #ffffff !important;
            stroke: #ffffff !important;
            color: #ffffff !important;
        }
        
        /* =========================
   Sidebar colored blocks (per section)
   ========================= */

/* marker is invisible, فقط برای انتخاب CSS */
.sb-marker { display: none; }

/* expander base style (applies to all section blocks) */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.26) !important;
  box-shadow: 0 10px 24px rgba(0,0,0,0.14) !important;
}

/* expander header */
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  padding: 10px 12px !important;
  font-weight: 800 !important;
  letter-spacing: 0.2px !important;
  color: #ffffff !important;
}

/* expander content */
[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"]{
  padding: 10px 12px 12px 12px !important;
}

/* =========================
   Color mapping using :has(marker) + next sibling expander
   ========================= */

/* CNN Grid Construction and Interpolation */
[data-testid="stSidebar"] .element-container:has(.sb-marker-grid) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-grid) + .element-container [data-testid="stExpander"] details {
  background: rgba(46,160,214,0.22) !important;
}

/* Distance-Based Method */
[data-testid="stSidebar"] .element-container:has(.sb-marker-dist) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-dist) + .element-container [data-testid="stExpander"] details {
  background: rgba(255,193,7,0.22) !important;
}

/* Classical Alpha-Shape Method */
[data-testid="stSidebar"] .element-container:has(.sb-marker-classic) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-classic) + .element-container [data-testid="stExpander"] details {
  background: rgba(0,200,83,0.22) !important;
}

/* Adaptive Alpha-Shape Method */
[data-testid="stSidebar"] .element-container:has(.sb-marker-adapt) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-adapt) + .element-container [data-testid="stExpander"] details {
  background: rgba(156,39,176,0.22) !important;
}

/* Alpha-Shapes Sampling Control */
[data-testid="stSidebar"] .element-container:has(.sb-marker-sample) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-sample) + .element-container [data-testid="stExpander"] details {
  background: rgba(255,87,34,0.22) !important;
}

/* Boundary Inflation Refinement */
[data-testid="stSidebar"] .element-container:has(.sb-marker-inflate) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-inflate) + .element-container [data-testid="stExpander"] details {
  background: rgba(96,125,139,0.26) !important;
}

/* Ghost Fraction (GF) Metric */
[data-testid="stSidebar"] .element-container:has(.sb-marker-gf) + .element-container [data-testid="stExpander"] summary,
[data-testid="stSidebar"] .element-container:has(.sb-marker-gf) + .element-container [data-testid="stExpander"] details {
  background: rgba(233,30,99,0.22) !important;
}

   /* =========================
   Sidebar colored expanders (stable header color)
   ========================= */

.sb-marker { display: none; }

/* expander container */
[data-testid="stSidebar"] [data-testid="stExpander"] details{
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.26) !important;
  box-shadow: 0 10px 24px rgba(0,0,0,0.14) !important;
  background: transparent !important; /* مهم */
}

/* header (summary) همیشه رنگ خودش را نگه دارد */
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary{
  padding: 10px 12px !important;
  font-weight: 800 !important;
  letter-spacing: 0.2px !important;
  color: #ffffff !important;
  background: var(--sb-head, rgba(255,255,255,0.10)) !important;
}

/* وقتی باز است هم همان رنگ بماند */
[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary{
  background: var(--sb-head, rgba(255,255,255,0.10)) !important;
}

/* حالت hover هم رنگ را عوض نکند */
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover{
  background: var(--sb-head, rgba(255,255,255,0.10)) !important;
}

/* بدنه expander */
[data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"]{
  padding: 10px 12px 12px 12px !important;
  background: var(--sb-body, rgba(255,255,255,0.06)) !important;
}

.team-tooltip a{
  color: #ffffff !important;
  text-decoration: none !important;
}
.team-tooltip a:hover{
  text-decoration: underline !important;
}

/* tooltip پیش فرض کلیک نگیرد تا hover خراب نشود */
.team-tooltip{
  pointer-events: none;
  cursor: pointer;
}

/* وقتی hover شد، کلیک فعال شود */
.team-avatar-wrap:hover .team-tooltip{
  pointer-events: auto;
}
    
        /* Optional: hide menu/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )


def header_with_logo():
    col_left, col_right = st.columns([6, 1.5], vertical_alignment="center")

    with col_left:
        st.markdown(f"<h1 style='margin: 0 0 6px 0;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    
        st.markdown(
            """
            <div style="margin-top: 2px;">
              <ul style="
                  margin: 0;
                  padding-left: 18px;
                  line-height: 1.15;
                  font-size: 0.9rem;
                ">
                <li><b>Developed at:</b> TECNUN (School of Engineering), University of Navarra, San Sebastián, Spain</li>
                <li><b>Research group:</b> TFED, Thermal and Fluids</li>
                <li><b>Supervision:</b> Prof. Gorka Sánchez Larraona and Prof. Alejandro Rivas Nieto</li>
                <li><b>Developer:</b> Eng. Mehran Sharifi (PhD Candidate, Mechanical Engineering)</li>
                <li><b>Timeframe:</b> September 2025 to February 2026</li>
                <li><b>Contact Us:</b> Msharifi@unav.es, Gsanchez@unav.es, Arivas@unav.es</li>
                <hr style="margin: 20px 0 0 0; border: 0; border-top: 1px solid rgba(255,255,255,0.35);" />
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.caption("Logo not found")
            st.caption(f"Expected at: {LOGO_PATH}")

def _to_npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _to_npz_bytes(**kwargs):
    buf = io.BytesIO()
    np.savez_compressed(buf, **kwargs)
    return buf.getvalue()

def _mask_csv_string(mask_u8):
    s = io.StringIO()
    np.savetxt(s, mask_u8, fmt="%d", delimiter=",")
    return s.getvalue()
# ============================================================
#                 COMMON HELPER FUNCTIONS
# ============================================================
def build_grid(x, y, nx=500, ny=500, bounds=None):
    if bounds is None:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
    else:
        xmin, xmax = bounds["xmin"], bounds["xmax"]
        ymin, ymax = bounds["ymin"], bounds["ymax"]

    xi = np.linspace(xmin, xmax, int(nx))
    yi = np.linspace(ymin, ymax, int(ny))
    Xg, Yg = np.meshgrid(xi, yi)
    return Xg, Yg, xmin, xmax, ymin, ymax


def mean_edge_length_from_delaunay(points):
    tri = Delaunay(points)
    edges = set()
    for a, b, c in tri.simplices:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    edges = np.array(list(edges), dtype=int)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return float(np.mean(lengths)), tri


def alpha_polygon_custom(points, alpha_factor=2.0):
    points = np.asarray(points, float)
    if len(points) < 4:
        raise ValueError("Not enough points for alpha shape")

    h_mean, tri = mean_edge_length_from_delaunay(points)
    alpha = max(alpha_factor / (h_mean + 1e-12), 1e-12)

    simplices = points[tri.simplices]
    a = np.linalg.norm(simplices[:, 0] - simplices[:, 1], axis=1)
    b = np.linalg.norm(simplices[:, 1] - simplices[:, 2], axis=1)
    c = np.linalg.norm(simplices[:, 2] - simplices[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None)) + 1e-15
    circum_r = (a * b * c) / (4.0 * area)
    keep = circum_r < (1.0 / alpha)

    from collections import defaultdict
    edge_count = defaultdict(int)
    kept_simplices = tri.simplices[keep]
    for t in kept_simplices:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            e = tuple(sorted((t[i], t[j])))
            edge_count[e] += 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        hull_idx = np.unique(tri.convex_hull.flatten())
        poly = points[hull_idx]
        c0 = poly.mean(axis=0)
        ang = np.arctan2(poly[:, 1] - c0[1], poly[:, 0] - c0[0])
        poly = poly[np.argsort(ang)]
        if not np.allclose(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0]])
        return poly

    adj = defaultdict(list)
    for i, j in boundary_edges:
        adj[i].append(j)
        adj[j].append(i)

    start = next(iter(adj))
    loop = [start]
    prev, curr = None, start
    for _ in range(10 * len(boundary_edges)):
        nbrs = adj[curr]
        nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
        if nxt is None or nxt == start:
            break
        loop.append(nxt)
        prev, curr = curr, nxt

    poly = points[np.array(loop)]
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def expand_polygon_coords(poly_coords, factor=1.0):
    poly_coords = np.asarray(poly_coords, float)
    if factor is None or abs(factor - 1.0) < 1e-15:
        return poly_coords

    poly_shp = Polygon(poly_coords)
    if not poly_shp.is_valid:
        poly_shp = poly_shp.buffer(0)

    poly_expanded = shp_scale(poly_shp, xfact=factor, yfact=factor, origin="center")
    if not poly_expanded.is_valid:
        poly_expanded = poly_expanded.buffer(0)

    return np.asarray(poly_expanded.exterior.coords, float)


def compute_mask_stats(mask):
    mask = mask.astype(bool)
    _, num_cc = cc_label(mask)
    mask_count = int(mask.sum())
    active_fraction = float(mask_count / mask.size)
    return dict(num_components=int(num_cc), active_fraction=active_fraction, active_cells=mask_count)


def point_recall(mask, x, y, xmin, xmax, ymin, ymax, nx, ny):
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    i = ((x - xmin) / dx).astype(int)
    j = ((y - ymin) / dy).astype(int)
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)
    inside = mask[j, i]
    return float(inside.sum() / inside.size)


def ghost_fraction(mask, dist_map, r0):
    mask = mask.astype(bool)
    extra = mask & (dist_map > r0)
    extra_cells = int(extra.sum())
    active_cells = int(mask.sum())
    if active_cells == 0:
        return np.nan, 0
    return float(extra_cells / active_cells), extra_cells


def compare_to_reference(mask, ref_mask):
    m = mask.astype(bool)
    r = ref_mask.astype(bool)
    inter = m & r
    union = m | r
    inter_count = int(inter.sum())
    union_count = int(union.sum())
    m_count = int(m.sum())
    r_count = int(r.sum())

    iou = inter_count / union_count if union_count > 0 else np.nan
    precision = inter_count / m_count if m_count > 0 else np.nan
    recall = inter_count / r_count if r_count > 0 else np.nan
    return float(iou), float(precision), float(recall)


def mask_distance(points, Xg, Yg, threshold, closing_size, dist_map=None):
    if dist_map is None:
        tree = cKDTree(points)
        grid_points = np.column_stack((Xg.ravel(), Yg.ravel()))
        dist, _ = tree.query(grid_points, k=1)
        dist_map_local = dist.reshape(Xg.shape)
    else:
        dist_map_local = dist_map

    mask = dist_map_local < threshold
    mask_closed = binary_closing(mask, structure=np.ones((closing_size, closing_size)))
    return mask_closed


def mask_alpha_custom(points, Xg, Yg, alpha_factor, expand_factor=1.0):
    poly = alpha_polygon_custom(points, alpha_factor=alpha_factor)
    poly = expand_polygon_coords(poly, factor=expand_factor)

    path = MplPath(poly)
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    mask_flat = path.contains_points(pts)
    return mask_flat.reshape(Xg.shape)


def mask_alpha_lib(points, Xg, Yg, alpha_value, expand_factor=1.0):
    geom = alphashape.alphashape(points, alpha_value)

    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        raise TypeError(f"Unexpected geometry type from alphashape: {type(geom)}")

    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    mask_flat = np.zeros(pts.shape[0], dtype=bool)

    for poly in polys:
        poly_use = poly
        if expand_factor is not None and abs(expand_factor - 1.0) > 1e-15:
            poly_use = shp_scale(poly_use, xfact=expand_factor, yfact=expand_factor, origin="center")
            if not poly_use.is_valid:
                poly_use = poly_use.buffer(0)

        coords = np.array(poly_use.exterior.coords)
        path = MplPath(coords)
        mask_flat |= path.contains_points(pts)

    return mask_flat.reshape(Xg.shape)


def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_masked_field(F_masked, xmin, xmax, ymin, ymax, title, x_pts=None, y_pts=None):
    fig, ax = plt.subplots(figsize=(3, 2))
    cmap_f = plt.cm.rainbow.copy()
    cmap_f.set_bad(color="white")

    im = ax.imshow(
        F_masked,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap_f,
        aspect="auto"
    )
    fig.colorbar(im, ax=ax, label="Value")

    if x_pts is not None and y_pts is not None:
        ax.scatter(x_pts, y_pts, s=2, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    return fig


def read_uploaded_file(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), sep=r"\s+", engine="python", comment="#")
    except Exception:
        return pd.read_csv(io.BytesIO(raw_bytes))


def interpolate_field(points, Xg, Yg, values, method):
    return griddata(points, values, (Xg, Yg), method=method)

def minmax_normalize_on_mask(F, mask):

    F = np.asarray(F, dtype=float)
    mask = mask.astype(bool)

    valid = mask & np.isfinite(F)
    if not np.any(valid):
        return F, np.nan, np.nan

    fmin = float(np.min(F[valid]))
    fmax = float(np.max(F[valid]))
    denom = fmax - fmin

    F_out = np.array(F, copy=True)
    if (not np.isfinite(denom)) or abs(denom) < 1e-15:
        F_out[valid] = 0.0
        return F_out, fmin, fmax

    F_out[valid] = (F_out[valid] - fmin) / denom
    return F_out, fmin, fmax

# ============================================================
#                       STREAMLIT APP
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_css()
header_with_logo()

with st.sidebar:
    st.header("Welcome!")
    uploaded = st.file_uploader("Upload Your Data File (ASCII Format)")

    st.markdown('<div class="sb-marker sb-marker-grid"></div>', unsafe_allow_html=True)
    with st.expander("CNN Grid Construction and Interpolation", expanded=False):
        st.markdown(r"**Grid Resolution in X**  ($n_x$)")
        nx = st.number_input(
        "nx", min_value=1, max_value=1_000_000, value=1000, step=1,
        key="nx", label_visibility="collapsed"
        )

        st.markdown(r"**Grid Resolution in Y**  ($n_y$)")
        ny = st.number_input(
        "ny", min_value=1, max_value=1_000_000, value=1000, step=1,
        key="ny", label_visibility="collapsed"
        )
        interp_method = st.selectbox("Interpolation Method", ["linear", "nearest", "cubic"], index=0)
        mask_value_choice = st.selectbox("Outside-Mask Value", ["NaN", "-1.0"], index=1)
        mask_value = np.nan if mask_value_choice == "NaN" else -1.0
        st.caption("Note: Use NaN Mainly for Cleaner Plotting (Visualization).")

    
    st.markdown('<div class="sb-marker sb-marker-grid"></div>', unsafe_allow_html=True)    
    with st.expander("Normalization", expanded=False):
        field_postproc = st.radio("Field Processing",["Interpolation Only (Keep Original Scale)", "Normalize Values Inside Mask (0–1)"],index=0)
        do_normalize = (field_postproc == "Normalize Values Inside Mask (0–1)")
        st.caption(r"Note: Normalization Uses Min-Max Scaling of  $\phi^{\mathrm{norm}}=\frac{\phi-\phi_{\min}}{\phi_{\max}-\phi_{\min}}$")

    st.markdown('<div class="sb-marker sb-marker-dist"></div>', unsafe_allow_html=True)
    with st.expander("Distance-Based Method", expanded=False):

        st.markdown(r"**Threshold Parameter**  ($\tau$)")
        distance_threshold = st.number_input(
        "tau", min_value=0.0, value=0.01, step=0.0000001, format="%.6f",
        key="distance_threshold", label_visibility="collapsed"
        )

        st.caption(r"Note: Setting $\tau=\min(\Delta x,\Delta y)$ Is Typically Effective for Most Geometries and Does Not Require Tuning.")

        st.markdown(r"**Structuring Element Size**  ($s$)")
        closing_size = st.number_input(
        "s", min_value=1, max_value=1000000, value=3, step=1,
        key="closing_size", label_visibility="collapsed"
        )

        st.caption("Note: The Default Value s=3 Is Effective for Most Geometries. Changing s Is Generally Unnecessary Unless the Mask Becomes Perforated.")

    st.markdown('<div class="sb-marker sb-marker-classic"></div>', unsafe_allow_html=True)
    with st.expander("Classical Alpha-Shape Method", expanded=False):
        st.markdown(r"**Classical Alpha-Shape Parameter**  ($\alpha$)")
        alpha_value_lib = st.number_input(
        "alpha", min_value=0.0001, value=10.0, step=0.000001, format="%.6f",
        key="alpha_value_lib", label_visibility="collapsed"
        )
        st.caption("Note: The α Value Is Strongly Geometry-Dependent and Typically Requires Tuning.")

    st.markdown('<div class="sb-marker sb-marker-adapt"></div>', unsafe_allow_html=True)
    with st.expander("Adaptive Alpha-Shape Method", expanded=False):
        st.markdown(r"**Adaptive Alpha-Shape Parameter**  ($\beta$)")
        alpha_factor_custom = st.number_input(
        "beta", min_value=0.0001, value=1.0, step=0.000001, format="%.6f",
        key="alpha_factor_custom", label_visibility="collapsed"
        )

        st.caption(r"Note: $\beta=1$ Is Typically Effective for Most Geometries and Does Not Require Tuning.")

    st.markdown('<div class="sb-marker sb-marker-sample"></div>', unsafe_allow_html=True)
    with st.expander("Alpha-Shapes Sampling Control", expanded=False):
        max_poly_points = st.number_input("Maximum Points for Alpha Polygons (0 = Use All Points)", min_value=0, value=0, step=1)
        st.caption("Note: This Upper Limit Is Intended Only for Large Files with a Very High Number of Points.")
        
    st.markdown('<div class="sb-marker sb-marker-inflate"></div>', unsafe_allow_html=True)
    with st.expander("Boundary Inflation Refinement", expanded=False):
        st.markdown(r"**Boundary Expansion Factor**  ($\eta$)")
        poly_expand_factor = st.number_input(
        "beta_b", min_value=1.0, value=1.0000, step=0.0001, format="%.4f",
        key="poly_expand_factor", label_visibility="collapsed"
        )
        st.caption("Note: Setting $\eta$ to Values Greater Than 1 Is Recommended When Points Near the Boundaries Remain Uncovered.")

        
    st.markdown('<div class="sb-marker sb-marker-gf"></div>', unsafe_allow_html=True)
    with st.expander("Ghost Fraction (GF) Metric", expanded=False):
        st.markdown(r"**Reference Radius Factor**  ($\rho$)")
        r0_factor = st.number_input(
        "rho", min_value=0.0001, value=1.0, step=0.0001, format="%.4f",
        key="r0_factor", label_visibility="collapsed"
        )
        st.caption("Note: $\\rho=1$ Is Typically Effective for Most Geometries and Does Not Require Tuning.")

    run_distance = st.checkbox("Distance-Based Mask", value=False)
    run_alpha_lib = st.checkbox("Classical Alpha-Shape Mask", value=False)
    run_alpha_custom = st.checkbox("Adaptive Alpha-Shape Mask", value=False)
    show_points = st.checkbox("Show Scattered Points on Plots", value=False)

    run_btn = st.button("Build Masks and Report Metrics", type="primary")


if uploaded is None:
    st.error("First Upload a File from the Sidebar.")
    st.stop()

df = read_uploaded_file(uploaded)

st.subheader("Dataset Overview")
st.dataframe(df.head(6), use_container_width=True)

# Coordinate columns
default_x = pick_column(df, ["x-coordinate", "x", "X", "x_coord", "xc"])
default_y = pick_column(df, ["y-coordinate", "y", "Y", "y_coord", "yc"])

c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    x_col = st.selectbox(
        "X Column",
        df.columns.tolist(),
        index=df.columns.get_loc(default_x) if default_x in df.columns else 0
    )
with c2:
    y_col = st.selectbox(
        "Y Column",
        df.columns.tolist(),
        index=df.columns.get_loc(default_y) if default_y in df.columns else min(1, len(df.columns) - 1)
    )
with c3:
    st.write("")
    st.write("")
    st.write(f"Rows: {len(df)}")

x = df[x_col].to_numpy(float)
y = df[y_col].to_numpy(float)
points_all = np.column_stack([x, y])

# Field selection (multi)
st.subheader("Output Fields")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
field_candidates = [c for c in numeric_cols if c not in [x_col, y_col]]

default_mode = "Use Physical Fields from Data" if len(field_candidates) > 0 else "Use Constant Unit Field (Mask and Geometry Only)"
field_mode = st.radio(
    "Choose Data Fields for Visualization and Export.",
    ["Use Physical Fields from Data", "Use Constant Unit Field (Mask and Geometry Only)"],
    index=0 if default_mode == "Use Physical Fields from Data" else 1
)

selected_fields = []
if field_mode == "Use Physical Fields from Data":
    if len(field_candidates) == 0:
        st.error("No Physical Field Columns Found Besides X and Y. Switch to Constant Unit Field.")
        st.stop()

    selected_fields = st.multiselect(
        "Select One or More Fields",
        options=field_candidates,
        default=[field_candidates[0]]
    )
    if len(selected_fields) == 0:
        st.error("Select at Least One Field or Switch to Constant Unit Field.")
        st.stop()
else:
    selected_fields = ["__ONES__"]

# Subsample points for alpha methods
rng = np.random.default_rng(0)
if max_poly_points and max_poly_points > 0 and len(points_all) > max_poly_points:
    idx = rng.choice(len(points_all), size=int(max_poly_points), replace=False)
    points_poly = points_all[idx]
else:
    points_poly = points_all

if run_btn:
    with st.spinner("Running Reconstruction Pipeline. Please Wait...."):

        # -------------------------
        # Shared pipeline timing (common for all methods)
        # -------------------------
        t_shared0 = time.perf_counter()

        # Grid
        Xg, Yg, xmin, xmax, ymin, ymax = build_grid(x, y, nx=int(nx), ny=int(ny))

        # Distance map
        tree = cKDTree(points_all)
        grid_points = np.column_stack((Xg.ravel(), Yg.ravel()))
        dist_to_pts, _ = tree.query(grid_points, k=1)
        dist_map = dist_to_pts.reshape(Xg.shape)

        # r0
        dists_pts, _ = tree.query(points_all, k=2)
        mean_nn_dist = float(np.mean(dists_pts[:, 1]))
        r0 = float(r0_factor * mean_nn_dist)

        # Build selected field grids (interpolation or constant ones)
        field_grids = {}
        if field_mode == "Use Constant Unit Field (Mask and Geometry Only)":
            field_grids["__ONES__"] = np.ones_like(Xg, dtype=float)
        else:
            for col in selected_fields:
                vals = df[col].to_numpy(float)
                field_grids[col] = interpolate_field(points_all, Xg, Yg, vals, interp_method)

        t_shared1 = time.perf_counter()
        shared_time = t_shared1 - t_shared0
        
        st.markdown(fr"""
        - **Mean NN Distance ($\overline{{d^{{NN}}}}$):** {mean_nn_dist:.6g} m
        - **Reference Radius ($r_0$):** {r0:.6g} m
        - **Shared Preprocessing Time (CNN Grid Generation, Distance Map Computation, Reference Radius Calculation, Interpolation):** {shared_time:.4f} s
        """)

        # -------------------------
        # Build masks (mask-only time)
        # -------------------------
        results = {}

        if run_distance:
            t0 = time.perf_counter()
            m = mask_distance(points_all, Xg, Yg, distance_threshold, int(closing_size), dist_map=dist_map)
            t1 = time.perf_counter()
            results["Distance-Based"] = {"mask": m, "time_mask": t1 - t0}

        if run_alpha_lib:
            t0 = time.perf_counter()
            m = mask_alpha_lib(points_poly, Xg, Yg, alpha_value_lib, expand_factor=poly_expand_factor)
            t1 = time.perf_counter()
            results["Classical Alpha-Shape"] = {"mask": m, "time_mask": t1 - t0}

        if run_alpha_custom:
            t0 = time.perf_counter()
            m = mask_alpha_custom(points_poly, Xg, Yg, alpha_factor_custom, expand_factor=poly_expand_factor)
            t1 = time.perf_counter()
            results["Adaptive Alpha-Shape"] = {"mask": m, "time_mask": t1 - t0}

        if len(results) == 0:
            st.error("No Method Selected.")
            st.stop()

        # Reference mask for IoU metrics
        ref_mask = results["Classical Alpha-Shape"]["mask"] if "Classical Alpha-Shape" in results else None

        # -------------------------
        # Per-method postprocess timing:
        # metrics + building export artifacts (NPY/CSV/NPZ + masked fields)
        # -------------------------
        rows = []
        for name, res in results.items():
            t_post0 = time.perf_counter()

            mask = res["mask"].astype(bool)
            mask_u8 = mask.astype(np.uint8)

            # metrics
            stats = compute_mask_stats(mask)
            pr = point_recall(mask, x, y, xmin, xmax, ymin, ymax, int(nx), int(ny))
            gf, extra_cells = ghost_fraction(mask, dist_map, r0)

            if ref_mask is None or name == "Classical Alpha-Shape":
                iou, prec, rec = (np.nan, np.nan, np.nan) if name != "Classical Alpha-Shape" else (1.0, 1.0, 1.0)
            else:
                iou, prec, rec = compare_to_reference(mask, ref_mask)

            # prepare exports once (so UI part does not recompute)
            exports = {}
            exports["mask_u8"] = mask_u8
            exports["mask_npy"] = _to_npy_bytes(mask_u8)
            exports["mask_csv"] = _mask_csv_string(mask_u8)

            # masked fields (reuse for NPZ + plotting + per-field NPY)
            masked_fields = {}
            norm_info = {}  # optional

            if field_mode == "Use Constant Unit Field (Mask and Geometry Only)":
                F0 = field_grids["__ONES__"]
                Fm = np.array(F0, copy=True)
                Fm[~mask] = mask_value
                masked_fields["ones"] = Fm
            else:
                for col in selected_fields:
                    F0 = field_grids[col]

                    if do_normalize:
                        F0n, fmin, fmax = minmax_normalize_on_mask(F0, mask)
                        norm_info[col] = (fmin, fmax)
                    else:
                        F0n = F0
                        norm_info[col] = (np.nan, np.nan)

                    Fm = np.array(F0n, copy=True)
                    Fm[~mask] = mask_value
                    masked_fields[col] = Fm

            exports["masked_fields"] = masked_fields
            exports["masked_field_npy"] = {k: _to_npy_bytes(v) for k, v in masked_fields.items()}

            # NPZ pack
            pack = {
                "Xg": Xg.astype(np.float64),
                "Yg": Yg.astype(np.float64),
                "mask": mask_u8,
            }
            for k, v in masked_fields.items():
                pack[f"field_{k}"] = v

            exports["npz"] = _to_npz_bytes(**pack)

            t_post1 = time.perf_counter()
            time_post = t_post1 - t_post0

            res["time_post"] = time_post
            res["time_total"] = res["time_mask"] + time_post + shared_time
            res["exports"] = exports
            res["norm_info"] = norm_info

            rows.append({
                "Method": name,
                "Mask Time (s)": round(res["time_mask"], 4),
                "Processing Time (s)": round(time_post, 4),
                "Total Time (s)": round(res["time_total"], 4),

                "N_av": stats["active_cells"],
                "AVF (%)": round(stats["active_fraction"] * 100, 3),
                "N_c": stats["num_components"],
                "PR": round(pr, 6),
                "GF": None if np.isnan(gf) else round(gf, 6),
                "N_gc": extra_cells,
                "IoU": None if np.isnan(iou) else round(iou, 6),
                "Pre": None if np.isnan(prec) else round(prec, 6),
                "Rec": None if np.isnan(rec) else round(rec, 6),
            })

        summary_df = pd.DataFrame(rows)
        st.subheader("Reconstruction Assessment Metrics")
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("""
        - **Mask Time** Refers Only to the Time Required to Generate the Mask.
        - **Processing Time** Includes Reconstruction Assessment Metrics, Masking Fields, Normalization, and Preparation of All Export Files.
        - **Total Time** Is the Sum of Shared Preprocessing Time, Mask Time, and Processing Time.
        """)
        
        summary_csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Current Table (.CSV)",
            data=summary_csv,
            file_name="Metrics Report.csv",
            mime="text/csv"
        )

        # ------------------------------------------------------------
        # Visual and downloads (now reusing precomputed exports)
        # ------------------------------------------------------------
        st.subheader("Reconstruction Outcomes")
        method_names = list(results.keys())
        method_tabs = st.tabs(method_names)

        for method_name, method_tab in zip(method_names, method_tabs):
            with method_tab:
                ex = results[method_name]["exports"]
                mask_u8 = ex["mask_u8"]
                mask = mask_u8.astype(bool)

                st.markdown("#### Generated Mask")

                st.download_button(
                    label=f"Export Mask (.NPY) [{method_name}]",
                    data=ex["mask_npy"],
                    file_name=f"Mask_{method_name}.npy",
                    mime="application/octet-stream"
                )

                st.download_button(
                    label=f"Export Mask (.CSV) [{method_name}]",
                    data=ex["mask_csv"],
                    file_name=f"Mask_{method_name}.csv",
                    mime="text/csv"
                )

                # field tabs
                field_display_names = ["ones"] if field_mode == "Use Constant Unit Field (Mask and Geometry Only)" else selected_fields
                field_tabs = st.tabs(field_display_names)

                for field_name, field_tab in zip(field_display_names, field_tabs):
                    with field_tab:
                        key = "ones" if field_mode == "Use Constant Unit Field (Mask and Geometry Only)" else field_name
                        F_masked = ex["masked_fields"][key]

                        fig = plot_masked_field(
                            F_masked, xmin, xmax, ymin, ymax,
                            title=f"{method_name} | field: {key}",
                            x_pts=x if show_points else None,
                            y_pts=y if show_points else None
                        )
                        st.pyplot(fig, clear_figure=True, use_container_width=False)

                        st.markdown("#### Masked Field")
                        st.download_button(
                            label=f"Export Masked Field (.NPY) [{method_name}]",
                            data=ex["masked_field_npy"][key],
                            file_name=f"Masked Field_{key}_{method_name}.npy",
                            mime="application/octet-stream"
                        )

                st.markdown("#### CNN-Ready Data (X, Y, Generated Mask, Masked Fields)")
                st.download_button(
                    label=f"Export CNN-Ready Data (.NPZ) [{method_name}]",
                    data=ex["npz"],
                    file_name=f"CNN-Ready Data_{method_name}.npz",
                    mime="application/octet-stream"
                )
                
        st.markdown(f"""
        Thank You for Using the CNN-Ready CFD Reconstruction Toolkit. If You Publish Any Results Obtained Using This Application, Please Do Cite Our Accompanying Paper:<br>
        **“Novel Distance-Based Masking and Adaptive α-Shape Methods for CNN-Ready Reconstruction of Arbitrary 2D CFD Flow Domains”**
        """, unsafe_allow_html=True)

else:
    st.error("Set Parameters, Then Click Build Masks and Report Metrics.")





