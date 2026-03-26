"""
Generate one sample of each stimulus type for the README gallery.
Outputs images and flow visualizations to samples/.

Uses only numpy and opencv (no scikit-image dependency).

Usage:
    python generate_samples.py
"""

import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.viz_flow import viz_flow
from utils.flow_io import flow_write

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
HEIGHT, WIDTH = 436, 1024
VX, VY = 5, 3  # px/frame


def save_pair(name, frame, flow_u, flow_v):
    """Save an image frame and its flow visualization."""
    os.makedirs(OUTDIR, exist_ok=True)
    img_path = os.path.join(OUTDIR, f'{name}_image.png')
    flow_vis_path = os.path.join(OUTDIR, f'{name}_flow.png')
    flow_flo_path = os.path.join(OUTDIR, f'{name}_flow.flo')

    cv2.imwrite(img_path, cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_GRAY2RGB))
    I_flow = viz_flow(flow_u, flow_v)
    cv2.imwrite(flow_vis_path, cv2.cvtColor(I_flow, cv2.COLOR_RGB2BGR))
    flow_write(flow_flo_path, flow_u, flow_v)
    print(f"  Saved: {img_path}")


def disk_coords(cy, cx, radius, shape):
    """Generate filled circle coordinates (replacement for skimage.draw.disk)."""
    Y, X = np.ogrid[:shape[0], :shape[1]]
    mask = (Y - cy)**2 + (X - cx)**2 <= radius**2
    return np.where(mask)


def circle_perimeter_coords(cy, cx, radius, shape):
    """Generate circle perimeter coordinates using cv2."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=1)
    return np.where(mask > 0)


def line_coords(r0, c0, r1, c1):
    """Generate line coordinates using cv2."""
    # Use a temporary image to rasterize the line
    rmin = min(r0, r1)
    rmax = max(r0, r1)
    cmin = min(c0, c1)
    cmax = max(c0, c1)
    h = rmax - rmin + 20
    w = cmax - cmin + 20
    offset_r = max(0, rmin - 10)
    offset_c = max(0, cmin - 10)
    mask = np.zeros((max(r0, r1) + 20, max(c0, c1) + 20), dtype=np.uint8)
    cv2.line(mask, (c0, r0), (c1, r1), 255, thickness=1)
    ys, xs = np.where(mask > 0)
    return ys, xs


def polygon_coords(rr, cc, shape):
    """Generate filled polygon coordinates using cv2."""
    pts = np.array(list(zip(cc.astype(int), rr.astype(int))), dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return np.where(mask > 0)


def generate_grating(Height, Width, sinSf, sinDirection, velocity):
    """Generate sinusoidal grating (5 temporal frames)."""
    yc = np.linspace(0, Height, Height, endpoint=False) - np.floor(Height / 2)
    xc = np.linspace(0, Width, Width, endpoint=False) - np.floor(Width / 2)
    t = np.linspace(0, 5, 5, endpoint=False)
    XX, YY, tt = np.meshgrid(xc, yc, t)
    YY = -YY
    sinTf = velocity * sinSf
    res = np.cos(2 * np.pi * sinSf * np.cos(np.deg2rad(sinDirection)) * XX +
                 2 * np.pi * sinSf * np.sin(np.deg2rad(sinDirection)) * YY -
                 2 * np.pi * sinTf * tt)
    res = res * 0.5 + 0.5
    return res


def generate_plaid(Height, Width, sf1, dir1, vel1, sf2, dir2, vel2):
    """Generate plaid pattern (sum of two gratings)."""
    g1 = generate_grating(Height, Width, sf1, dir1, vel1)
    g2 = generate_grating(Height, Width, sf2, dir2, vel1)
    return g1 * 0.5 + g2 * 0.5


# --- Sample generators ---

def sample_circle_full():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    yc, xc = disk_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1, 42, frame.shape)
    frame[yc, xc] = 200
    flow_u[yc, xc] = VX
    flow_v[yc, xc] = VY
    save_pair('circle_full', frame, flow_u, flow_v)


def sample_circle_perimeter():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    yc, xc = circle_perimeter_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1, 42, frame.shape)
    frame[yc, xc] = 200
    flow_u[yc, xc] = VX
    flow_v[yc, xc] = VY
    save_pair('circle_perimeter', frame, flow_u, flow_v)


def sample_line():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    length = 30
    orientation = 60
    y_off = int(round((length / 2) * np.sin(np.radians(-orientation))))
    x_off = int(round((length / 2) * np.cos(np.radians(-orientation))))
    cy, cx = HEIGHT // 2 - 1, WIDTH // 2 - 1
    yc, xc = line_coords(cy - y_off, cx - x_off, cy + y_off, cx + x_off)
    frame[yc, xc] = 200
    flow_u[yc, xc] = VX
    flow_v[yc, xc] = VY
    save_pair('line', frame, flow_u, flow_v)


def sample_rectangle():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    length = 50
    aspect_ratio = 1.5
    orientation = 45
    y1 = int(round((length / 2) * np.sin(np.radians(-orientation))))
    x1 = int(round((length / 2) * np.cos(np.radians(-orientation))))
    y2 = int(round((aspect_ratio * length / 2) * np.sin(np.radians(-(orientation + 90)))))
    x2 = int(round((aspect_ratio * length / 2) * np.cos(np.radians(-(orientation + 90)))))
    cy, cx = HEIGHT // 2 - 1, WIDTH // 2 - 1
    rtyc = np.array([cy - y1 - y2, cy - y1 + y2, cy + y1 + y2, cy + y1 - y2], dtype=np.float64)
    rtxc = np.array([cx - x1 - x2, cx - x1 + x2, cx + x1 + x2, cx + x1 - x2], dtype=np.float64)
    yc, xc = polygon_coords(rtyc, rtxc, frame.shape)
    frame[yc, xc] = 200
    flow_u[yc, xc] = VX
    flow_v[yc, xc] = VY
    save_pair('rectangle', frame, flow_u, flow_v)


def sample_grating():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    radius = 100
    freq = 1.0 / 20
    orientation = 0
    speed = 3.0
    grating = generate_grating(HEIGHT, WIDTH, freq, orientation, speed)
    yc, xc = disk_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1, radius, frame.shape)
    frame[yc, xc] = 200 * grating[yc, xc, 2]  # center frame
    flow_u[yc, xc] = speed * np.cos(np.deg2rad(orientation))
    flow_v[yc, xc] = -speed * np.sin(np.deg2rad(orientation))
    save_pair('grating', frame, flow_u, flow_v)


def sample_plaid():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    radius = 100
    freq1, freq2 = 1.0 / 15, 1.0 / 25
    orient1, orient2 = 30, 90
    speed1, speed2 = 2.0, 3.0
    plaid = generate_plaid(HEIGHT, WIDTH, freq1, orient1, speed1, freq2, orient2, speed2)
    yc, xc = disk_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1, radius, frame.shape)
    frame[yc, xc] = 200 * plaid[yc, xc, 2]
    # IoC velocity
    denom = np.sin(np.deg2rad(orient2) - np.deg2rad(orient1))
    if abs(denom) > 1e-6:
        ioc_vx = (speed1 * np.sin(np.deg2rad(orient2)) - speed2 * np.sin(np.deg2rad(orient1))) / denom
        ioc_vy = -1 * (speed1 * np.cos(np.deg2rad(orient2)) - speed2 * np.cos(np.deg2rad(orient1))) / denom
    else:
        ioc_vx = 0.5 * (speed1 * np.cos(np.deg2rad(orient1)) + speed2 * np.cos(np.deg2rad(orient2)))
        ioc_vy = -0.5 * (speed1 * np.sin(np.deg2rad(orient1)) + speed2 * np.sin(np.deg2rad(orient2)))
    flow_u[yc, xc] = ioc_vx
    flow_v[yc, xc] = ioc_vy
    save_pair('plaid', frame, flow_u, flow_v)


def sample_hybrid_plaid():
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    radius = 100
    grating_offset = 50
    freq1, freq2 = 1.0 / 15, 1.0 / 25
    orient1, orient2 = 30, 90
    speed1, speed2 = 2.0, 3.0
    g1 = generate_grating(HEIGHT, WIDTH, freq1, orient1, speed1)
    g2 = generate_grating(HEIGHT, WIDTH, freq2, orient2, speed2)
    # Left grating
    yc_l, xc_l = disk_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1 - grating_offset, radius, frame.shape)
    frame[yc_l, xc_l] = 200 * g1[yc_l, xc_l, 2]
    flow_u[yc_l, xc_l] = speed1 * np.cos(np.deg2rad(orient1))
    flow_v[yc_l, xc_l] = -speed1 * np.sin(np.deg2rad(orient1))
    # Right grating
    yc_r, xc_r = disk_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1 + grating_offset, radius, frame.shape)
    frame[yc_r, xc_r] = 200 * g2[yc_r, xc_r, 2]
    flow_u[yc_r, xc_r] = speed2 * np.cos(np.deg2rad(orient2))
    flow_v[yc_r, xc_r] = -speed2 * np.sin(np.deg2rad(orient2))
    save_pair('hybrid_plaid', frame, flow_u, flow_v)


if __name__ == '__main__':
    print(f"Generating samples in {OUTDIR}/")
    print(f"  Image size: {WIDTH}x{HEIGHT}, velocity: VX={VX}, VY={VY}")
    print()

    print("[1/7] Circle (filled)")
    sample_circle_full()
    print("[2/7] Circle (perimeter)")
    sample_circle_perimeter()
    print("[3/7] Line")
    sample_line()
    print("[4/7] Rectangle")
    sample_rectangle()
    print("[5/7] Grating")
    sample_grating()
    print("[6/7] Plaid (IoC)")
    sample_plaid()
    print("[7/7] Hybrid Plaid")
    sample_hybrid_plaid()

    print()
    print(f"Done! 7 sample pairs saved to {OUTDIR}/")
