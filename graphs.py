import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import imageio
import math

def is_not_nan(value):
    return not math.isnan(value)

nx, ny, nz = 40, 40, 40
nit = 50
c = 1
dx, dy, dz = 2 / (nx - 1), 2 / (ny - 1), 2 / (nz - 1)

x, y, z = np.linspace(0, 2, nx), np.linspace(0, 2, ny), np.linspace(0, 2, nz)

X, Y, Z = np.meshgrid(x, y, z)


#u = np.load("Results/speed_x2D1.npy")
#v = np.load("Results/speed_y2D1.npy")
#print(u.shape)
#u = np.ones((ny, nx))
#v = np.ones((ny, nx))

#p = np.load("Results/pressure2D1.npy")
#p = np.linspace(0, 300, 200*200).reshape(200, 200).T
#print(p)

u = np.load("speed_x.npy")
v = np.load("speed_y.npy")
w = np.load("speed_z.npy")
p = np.load("pressure.npy")

def get_force(x,y):
    p_lift = 0
    p_drag = 0

    if is_not_nan(p[x-1,y]):
        p_lift += dx*p[x-1,y]

    if is_not_nan(p[x+1,y]):
        p_lift -= dx*p[x+1,y]

    if is_not_nan(p[x,y-1]):
        p_drag += dy*p[x,y-1]

    if is_not_nan(p[x,y+1]):
        p_drag -= dy*p[x,y+1]

    return (p_lift, p_drag)

image_path = "Ala.bmp"
bmp_collision = imageio.v2.imread(image_path)

wing = []

"""
for x in range(100):
    for y in range(100):
        if not np.any(bmp_collision[x][y]):
            wing.append((-x-((ny-100)//2),y+((nx-100)//3)))

for x,y in wing:
    u[x,y],v[x,y],p[x,y] = None,None,0


f_x, f_y = 0,0
for x,y in wing:
        fx,fy = get_force(x, y)
        f_x += fx
        f_y += fy

print(f_x,f_y)
for i in range(len(p)):
    for j in range(len(p[0])):
        if p[i,j] > 300:
            p[i,j] = 300
        if p[i,j] < -300:
            p[i,j] = -300"""

#fig = plt.figure(figsize=(10,8), dpi=100)
# plotting velocity field
#c = 16
#plt.contourf(X[150:400], Y[150:400], p[150:400], alpha = 0.75, cmap=cm.viridis)
#plt.colorbar(label = "Pressure [Pa]")
#plt.quiver(X[::c,::c], Y[::c,::c], u[::c,::c], v[::c,::c], (u*u+v*v)[::c,::c])
"""plt.streamplot(X[150:400], Y[150:400], u[150:400], v[150:400], color='k', linewidth=1, arrowsize=1, density=3)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.savefig("Wing_trace2D.png")

plt.show()"""

"""
threshold = 1
mask = np.abs(u - threshold) > 0.5

# Apply the mask to your data
u_masked = np.ma.array(u, mask=~mask)
v_masked = np.ma.array(v, mask=~mask)
w_masked = np.ma.array(w, mask=~mask)


fig = pyplot.figure(figsize = (11,7), dpi=100)
ax = pyplot.figure().add_subplot(projection='3d')
ax.quiver(X[::3, ::3, ::3], Y[::3, ::3, ::3], Z[::3, ::3, ::3], u_masked[::3, ::3, ::3], v_masked[::3, ::3, ::3], w_masked[::3, ::3, ::3], length = 0.2, cmap="Reds");
"""
fig = plt.figure(figsize = (11,7), dpi=100)
c = 16
#plt.contourf(X[::,::,c], Y[::,::,c], p[::,::,c], alpha = 0.75, cmap=cm.viridis)
#plt.colorbar(label = "Pressure [Pa]")
plt.quiver(X[::,::,c], Y[::,::,c], u[::,::,c], v[::,::,c], (u*u+v*v)[::,::,c])
#plt.streamplot(X[150:400], Y[150:400], u[150:400], v[150:400], color='k', linewidth=1, arrowsize=1, density=3)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.savefig("Wing_pressure3D.png")
plt.show()

