import pybullet as p
import pybullet_data
from pathlib import Path

client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(str(Path("urdf").resolve()))

robot = p.loadURDF(str(Path("urdf/rex.urdf").resolve()), [0, 0, 0])

# Get AABB (axis-aligned bounding box) of entire robot
aabb_min, aabb_max = p.getAABB(robot, physicsClientId=client)
print(f"Width  (X): {aabb_max[0]-aabb_min[0]:.3f} m")
print(f"Length (Y): {aabb_max[1]-aabb_min[1]:.3f} m")
print(f"Height (Z): {aabb_max[2]-aabb_min[2]:.3f} m")

# Per-link bounding boxes
n = p.getNumJoints(robot)
for i in range(-1, n):  # -1 = base link
    info = p.getJointInfo(robot, i) if i >= 0 else None
    name = info[1].decode() if info else "base_link"
    mn, mx = p.getAABB(robot, i)
    print(f"  {name:45s} z: {mn[2]:.3f} → {mx[2]:.3f} m")

p.disconnect(client)
