import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import imageio
import trimesh


nx = 400
ny = 400
nt = 500
nit = 50

height, width = 4,4
dx = height / (nx - 1)
dy = width / (ny - 1)
x = np.linspace(0, width, nx)
y = np.linspace(0, height, ny)
X, Y = np.meshgrid(x, y)


rho = 1.21
#nu = .156
nu = .16
dt = .00001
error = .00001
speed = 1


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))


def build_up_b(b, rho, dt, u, v, dx, dy):

    b[1:-1, 1:-1] = (rho * (1 / dt *
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b


def pressure_poisson(p, dx, dy, b, stepcount):
    pn = np.empty_like(p)
    pn = p.copy()

    pseudotime = max(5, nit-stepcount)

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                          b[1:-1,1:-1])

        p[:, -1] = p[:, 0] = p[0,:] = p[-1,:] = 0 # dp/dx = 0 at x = 2


    return p


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    # for n in range(nt):

    udiff = 1
    stepcount = 0
    for _ in range(10):

    # while abs(udiff) > error:
        print(stepcount)
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b, stepcount)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = speed
        u[:, 0]  = speed
        u[:, -1] = speed
        u[-1, :] = speed
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0

        u[bmp_collision] = 0
        v[bmp_collision] = 0
        """
        for x in range(100):
            for y in range(100):
                if not np.any(bmp_collision[x][y]):
                    u[x+((ny-100)//2),y+((nx-100)//3)] = 0
                    v[x+((ny-100)//2),y+((nx-100)//3)] = 0
        """
        udiff = ((np.sum(u) - np.sum(un))+(np.sum(v) - np.sum(vn))) / (np.sum(u)+np.sum(v))
        stepcount += 1

    print(stepcount)
    print(f"Diff: {udiff}")
    print(f"Speed:  {speed}")
    print(f"Error: {error}")
    print(f"Nu: {nu}")
    print(f"dt: {dt}")
    print(f"nx,ny,dx,dy,height,width: ", end="")
    print(nx,ny,dx,dy,height,width)
    return u, v, p


mesh = trimesh.load_mesh('Wing3.stl')
grid = np.array(np.meshgrid(x, y, y)).T.reshape(-1,3)
bmp_collision = mesh.contains(grid).reshape(np.zeros((nx,nx,nx)).shape)[0]


#image_path = "Ala.bmp"
#bmp_collision = imageio.v2.imread(image_path)
#print(bmp_collision)
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

np.save('speed_x2D.npy', u)
np.save('speed_y2D.npy', v)
np.save('pressure2D.npy', p)

u[bmp_collision] = None
v[bmp_collision] = None
p[bmp_collision] = None

fig = plt.figure(figsize=(10,8), dpi=100)
# plotting velocity field
c = 3
plt.contourf(X, Y, p, alpha = 0.75, cmap=cm.viridis)
plt.colorbar()
plt.quiver(X[::c,::c], Y[::c,::c], u[::c,::c], v[::c,::c], (u*u+v*v)[::c,::c])
plt.xlabel('X')
plt.ylabel('Y')

plt.savefig("Wing_pressure2D.png")

plt.show()
