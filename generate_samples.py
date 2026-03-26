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


def sample_barber_pole():
    """Barber pole: grating behind an elongated rectangular aperture.

    The grating's component velocity is normal to its orientation, but the
    perceived (barber pole) velocity shifts along the aperture's long axis.
    We save both ground truths: component and barber-pole predicted.
    """
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_comp = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_comp = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_bp = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_bp = np.zeros([HEIGHT, WIDTH], dtype=np.float64)

    rect_length, rect_width = 250, 50
    grating_orient = 60  # degrees
    aper_orient = 0  # horizontal aperture
    freq = 1.0 / 20
    speed = 3.0

    grating = generate_grating(HEIGHT, WIDTH, freq, grating_orient, speed)

    # Rectangular aperture
    half_l, half_w = rect_length / 2.0, rect_width / 2.0
    cos_a = np.cos(np.deg2rad(-aper_orient))
    sin_a = np.sin(np.deg2rad(-aper_orient))
    cy, cx = HEIGHT // 2 - 1, WIDTH // 2 - 1
    corners_x = np.array([-half_l, half_l, half_l, -half_l])
    corners_y = np.array([-half_w, -half_w, half_w, half_w])
    rot_x = (corners_x * cos_a - corners_y * sin_a + cx).astype(int)
    rot_y = (corners_x * sin_a + corners_y * cos_a + cy).astype(int)
    mask = np.zeros([HEIGHT, WIDTH], dtype=np.uint8)
    pts = np.array(list(zip(rot_x, rot_y)), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    yc, xc = np.where(mask > 0)

    frame[yc, xc] = 200 * grating[yc, xc, 2]

    # Component velocity (normal to grating)
    comp_vx = speed * np.cos(np.deg2rad(grating_orient))
    comp_vy = -speed * np.sin(np.deg2rad(grating_orient))
    flow_u_comp[yc, xc] = comp_vx
    flow_v_comp[yc, xc] = comp_vy

    # Barber pole velocity (along aperture axis)
    ax_x = np.cos(np.deg2rad(aper_orient))
    ax_y = -np.sin(np.deg2rad(aper_orient))
    angle_diff = np.deg2rad(grating_orient - aper_orient)
    cos_diff = np.cos(angle_diff)
    bp_speed = speed / cos_diff if abs(cos_diff) > 1e-6 else 0
    flow_u_bp[yc, xc] = bp_speed * ax_x
    flow_v_bp[yc, xc] = bp_speed * ax_y

    save_pair('barber_pole_component', frame, flow_u_comp, flow_v_comp)
    save_pair('barber_pole_perceived', frame, flow_u_bp, flow_v_bp)


def sample_barber_plaid():
    """Barber plaid: plaid pattern behind an elongated rectangular aperture."""
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_ioc = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_ioc = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_bp = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_bp = np.zeros([HEIGHT, WIDTH], dtype=np.float64)

    rect_length, rect_width = 250, 50
    aper_orient = 0
    freq1, freq2 = 1.0 / 15, 1.0 / 25
    orient1, orient2 = 30, 90
    speed1, speed2 = 2.0, 3.0

    plaid = generate_plaid(HEIGHT, WIDTH, freq1, orient1, speed1, freq2, orient2, speed2)

    # Rectangular aperture
    half_l, half_w = rect_length / 2.0, rect_width / 2.0
    cy, cx = HEIGHT // 2 - 1, WIDTH // 2 - 1
    corners_x = np.array([-half_l, half_l, half_l, -half_l])
    corners_y = np.array([-half_w, -half_w, half_w, half_w])
    mask = np.zeros([HEIGHT, WIDTH], dtype=np.uint8)
    pts = np.array(list(zip((corners_x + cx).astype(int), (corners_y + cy).astype(int))), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    yc, xc = np.where(mask > 0)

    frame[yc, xc] = 200 * plaid[yc, xc, 2]

    # IoC velocity
    denom = np.sin(np.deg2rad(orient2) - np.deg2rad(orient1))
    if abs(denom) > 1e-6:
        ioc_vx = (speed1 * np.sin(np.deg2rad(orient2)) - speed2 * np.sin(np.deg2rad(orient1))) / denom
        ioc_vy = -1 * (speed1 * np.cos(np.deg2rad(orient2)) - speed2 * np.cos(np.deg2rad(orient1))) / denom
    else:
        ioc_vx = 0.5 * (speed1 * np.cos(np.deg2rad(orient1)) + speed2 * np.cos(np.deg2rad(orient2)))
        ioc_vy = -0.5 * (speed1 * np.sin(np.deg2rad(orient1)) + speed2 * np.sin(np.deg2rad(orient2)))

    flow_u_ioc[yc, xc] = ioc_vx
    flow_v_ioc[yc, xc] = ioc_vy

    # Barber pole: project IoC onto aperture axis
    ax_x = np.cos(np.deg2rad(aper_orient))
    ax_y = -np.sin(np.deg2rad(aper_orient))
    proj = ioc_vx * ax_x + ioc_vy * ax_y
    flow_u_bp[yc, xc] = proj * ax_x
    flow_v_bp[yc, xc] = proj * ax_y

    save_pair('barber_plaid_ioc', frame, flow_u_ioc, flow_v_ioc)
    save_pair('barber_plaid_perceived', frame, flow_u_bp, flow_v_bp)


def sample_transparent_dots():
    """Transparent motion: two overlapping dot populations moving in different directions."""
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_s1 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_s1 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_s2 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_s2 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)

    np.random.seed(42)
    num_dots = 200
    dot_radius = 3
    margin = 80
    speed1, speed2 = 5, 5
    dir1, dir2 = 0, 180  # opposite directions

    vx1 = speed1 * np.cos(np.deg2rad(dir1))
    vy1 = -speed1 * np.sin(np.deg2rad(dir1))
    vx2 = speed2 * np.cos(np.deg2rad(dir2))
    vy2 = -speed2 * np.sin(np.deg2rad(dir2))

    half = num_dots // 2
    cx_s1 = np.random.randint(margin, WIDTH - margin, half)
    cy_s1 = np.random.randint(margin, HEIGHT - margin, half)
    cx_s2 = np.random.randint(margin, WIDTH - margin, half)
    cy_s2 = np.random.randint(margin, HEIGHT - margin, half)

    # Surface 1
    for i in range(half):
        cv2.circle(frame, (cx_s1[i], cy_s1[i]), dot_radius, 200, -1)
        dot_mask = np.zeros([HEIGHT, WIDTH], dtype=np.uint8)
        cv2.circle(dot_mask, (cx_s1[i], cy_s1[i]), dot_radius, 255, -1)
        dy, dx = np.where(dot_mask > 0)
        flow_u_s1[dy, dx] = vx1
        flow_v_s1[dy, dx] = vy1

    # Surface 2
    for i in range(half):
        cv2.circle(frame, (cx_s2[i], cy_s2[i]), dot_radius, 120, -1)
        dot_mask = np.zeros([HEIGHT, WIDTH], dtype=np.uint8)
        cv2.circle(dot_mask, (cx_s2[i], cy_s2[i]), dot_radius, 255, -1)
        dy, dx = np.where(dot_mask > 0)
        flow_u_s2[dy, dx] = vx2
        flow_v_s2[dy, dx] = vy2

    save_pair('transparent_dots_surface1', frame, flow_u_s1, flow_v_s1)
    save_pair('transparent_dots_surface2', frame, flow_u_s2, flow_v_s2)


def sample_transparent_gratings():
    """Transparent gratings: two overlapping gratings with separate per-surface GT."""
    frame = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_s1 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_s1 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_u_s2 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)
    flow_v_s2 = np.zeros([HEIGHT, WIDTH], dtype=np.float64)

    radius = 100
    freq1, freq2 = 1.0 / 15, 1.0 / 25
    orient1, orient2 = 30, 120
    speed1, speed2 = 2.0, 3.0

    g1 = generate_grating(HEIGHT, WIDTH, freq1, orient1, speed1)
    g2 = generate_grating(HEIGHT, WIDTH, freq2, orient2, speed2)

    yc, xc = disk_coords(HEIGHT // 2 - 1, WIDTH // 2 - 1, radius, frame.shape)
    frame[yc, xc] = 200 * (0.5 * g1[yc, xc, 2] + 0.5 * g2[yc, xc, 2])

    flow_u_s1[yc, xc] = speed1 * np.cos(np.deg2rad(orient1))
    flow_v_s1[yc, xc] = -speed1 * np.sin(np.deg2rad(orient1))
    flow_u_s2[yc, xc] = speed2 * np.cos(np.deg2rad(orient2))
    flow_v_s2[yc, xc] = -speed2 * np.sin(np.deg2rad(orient2))

    save_pair('transparent_gratings_surface1', frame, flow_u_s1, flow_v_s1)
    save_pair('transparent_gratings_surface2', frame, flow_u_s2, flow_v_s2)


if __name__ == '__main__':
    print(f"Generating samples in {OUTDIR}/")
    print(f"  Image size: {WIDTH}x{HEIGHT}, velocity: VX={VX}, VY={VY}")
    print()

    print("[1/11] Circle (filled)")
    sample_circle_full()
    print("[2/11] Circle (perimeter)")
    sample_circle_perimeter()
    print("[3/11] Line")
    sample_line()
    print("[4/11] Rectangle")
    sample_rectangle()
    print("[5/11] Grating")
    sample_grating()
    print("[6/11] Plaid (IoC)")
    sample_plaid()
    print("[7/11] Hybrid Plaid")
    sample_hybrid_plaid()
    print("[8/11] Barber Pole")
    sample_barber_pole()
    print("[9/11] Barber Plaid")
    sample_barber_plaid()
    print("[10/11] Transparent Motion (Dots)")
    sample_transparent_dots()
    print("[11/11] Transparent Motion (Gratings)")
    sample_transparent_gratings()

    print()
    print(f"Done! 11 sample sets saved to {OUTDIR}/")
