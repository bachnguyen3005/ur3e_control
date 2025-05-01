import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# === Inputs ===
P0 = np.array([0.17445, -0.26278, 0.04])   # start [x, y, z], cubes position
P2 = np.array([-0.315, 0.022, 0.04])       # target [x, y, z], drop off position
C  = np.array([0.036, -0.17, 0.04])       # cylinder obstacle center [x, y, z], ciricle position
Q0 = np.array([0, 0, 0, 1])                # quaternion [qx, qy, qz, qw]

# === Parameters ===
safeMargin              = 0.05              # extra clearance (m)
projectedCylinderRadius = 0.025             # cylinder radius (m)
R                       = projectedCylinderRadius + safeMargin
zConstant               = P0[2]             # keep constant Z

base_center = np.array([0.0, 0.0])  # robot-base at origin (XY only)
base_R      = 0.18                 # 175 mm radius → 350 mm Ø

# === Helper functions ===
def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def check_collision(p0_xy, p1_xy, c_xy, radius):
    """True if the segment p0→p1 comes within 'radius' of c_xy."""
    v = p1_xy - p0_xy
    w = c_xy   - p0_xy
    t = np.clip(np.dot(w, v) / np.dot(v, v), 0.0, 1.0)
    closest = p0_xy + t*v
    return np.linalg.norm(c_xy - closest) < radius

def compute_tangent_path(p0_xy, p2_xy, c_xy, radius, n_intermediate=1):
    """
    Return a smooth detour [p0, T1, mid…, T2, p2] around circle at c_xy.
    Ensures T1 is the tangent closest to p0 to avoid any 'bounce'.
    """
    # angles from center to endpoints
    θ0 = math.atan2(p0_xy[1]-c_xy[1], p0_xy[0]-c_xy[0])
    θ2 = math.atan2(p2_xy[1]-c_xy[1], p2_xy[0]-c_xy[0])
    d0, d2 = np.linalg.norm(p0_xy - c_xy), np.linalg.norm(p2_xy - c_xy)
    α0 = math.acos(radius / d0)
    α2 = math.acos(radius / d2)

    cand0 = [θ0 + α0, θ0 - α0]
    cand2 = [θ2 + α2, θ2 - α2]

    # pick the pair whose absolute angular sweep is smallest
    best = float('inf')
    sel0 = sel2 = None
    for a0 in cand0:
        for a2 in cand2:
            sweep = wrap_to_pi(a2 - a0)
            if abs(sweep) < best:
                best, sel0, sel2 = abs(sweep), a0, a2

    # compute the two tangent points
    T1 = c_xy + radius * np.array([math.cos(sel0), math.sin(sel0)])
    T2 = c_xy + radius * np.array([math.cos(sel2), math.sin(sel2)])

    # if T1 is farther from p0 than T2, swap them (so we always go to the closer one first)
    if np.linalg.norm(p0_xy - T1) > np.linalg.norm(p0_xy - T2):
        T1, T2 = T2, T1
        sel0, sel2 = sel2, sel0

    # now sample a few points along the shorter arc from sel0→sel2
    Δθ = wrap_to_pi(sel2 - sel0)
    ts = np.linspace(0, 1, n_intermediate+2)  # includes endpoints
    arc_pts = [c_xy + radius * np.array([math.cos(sel0 + t*Δθ),
                                         math.sin(sel0 + t*Δθ)])
               for t in ts]

    # build [p0, arc_pts..., p2]
    return np.vstack([p0_xy] + arc_pts + [p2_xy])

def circles_intersect(c1, r1, c2, r2):
    return np.linalg.norm(c1 - c2) < (r1 + r2)

def projection_t(p0, p1, c):
    v = p1 - p0
    w = c  - p0
    return np.clip(np.dot(w, v) / np.dot(v, v), 0.0, 1.0)

def compute_detour_sequence(p0, p2, obstacles):
    hits = []
    for c_xy, radius, name in obstacles:
        if check_collision(p0, p2, c_xy, radius):
            hits.append((projection_t(p0, p2, c_xy), c_xy, radius, name))
    if not hits:
        return np.vstack([p0, p2])

    hits.sort(key=lambda x: x[0])
    waypoints = [p0]
    current = p0

    for _, c_xy, radius, name in hits:
        print(f"Collision with {name} → detouring around it")
        detour = compute_tangent_path(current, p2, c_xy, radius, n_intermediate=1)
        # detour = [current, T1, mid..., T2, p2]; keep only T1, mid…, T2
        waypoints.extend(detour[1:-1])
        current = detour[-2]  # last arc‐point before p2

    waypoints.append(p2)
    return np.vstack(waypoints)

# === Main flow ===
P0_xy, P2_xy = P0[:2], P2[:2]
C_xy         = C[:2]
obstacles    = [
    (C_xy,        R,      'cylinder'),
    (base_center, base_R, 'robot base'),
]

if circles_intersect(C_xy, R, base_center, base_R):
    print("⚠️ Warning: cylinder and base no-go zones overlap!")

pts_xy = compute_detour_sequence(P0_xy, P2_xy, obstacles)

# === Save CSV ===
N    = pts_xy.shape[0]
xyz  = np.hstack([pts_xy, np.full((N,1), zConstant)])
quat = np.tile(Q0, (N,1))
df   = pd.DataFrame(np.hstack([xyz, quat]),
                    columns=['x','y','z','qx','qy','qz','qw'])
df.to_csv('scripts/waypointsMatrix.csv', index=False)
print("Saved waypointsMatrix.csv with columns:", df.columns.tolist())

# === Plot ===
plt.figure()
plt.axis('equal')
plt.grid(True)
plt.plot(P0_xy[0], P0_xy[1], 'go',  label='Start')
plt.plot(P2_xy[0], P2_xy[1], 'rs',  label='Target')
plt.plot(pts_xy[:,0], pts_xy[:,1], 'b-o', label='Waypoints')
plt.gca().add_patch(plt.Circle((C_xy[0], C_xy[1]), R,
                               linestyle='--', fill=False, color='r',
                               label='Cylinder no-go'))
plt.gca().add_patch(plt.Circle((0,0), base_R,
                               linestyle='--', fill=False, color='m',
                               label='Base no-go'))
plt.legend(loc='best')
plt.title('Collision-Avoidance Waypoints')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()