import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# Grid variables
nx, ny, nz = 40, 40, 40
nit = 50
c = 1
dx, dy, dz = 2 / (nx - 1), 2 / (ny - 1), 2 / (nz - 1)

x, y, z = np.linspace(0, 2, nx), np.linspace(0, 2, ny), np.linspace(0, 2, nz)

X, Y, Z = np.meshgrid(x, y, z)


##physical variables
rho = 1
nu = .1
F = 1
dt = .0001

#initial conditions
u = np.zeros((ny, nx, nz))
un = np.zeros((ny, nx, nz))

v = np.zeros((ny, nx, nz))
vn = np.zeros((ny, nx, nz))

w = np.zeros((ny, nx, nz))
wn = np.zeros((ny, nx, nz))


p = np.ones((ny, nx, nz))
pn = np.ones((ny, nx, nz))

b = np.zeros((ny, nx, nz))


def build_up_b(rho, dt, dx, dy, dz, u, v, w):
    """ Calculates the 'b' part of the PPE, simplifies things """
    b = np.zeros_like(u)
    b[1:-1, 1:-1, 1:-1] =\
        (-1 * rho *(
        ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 0:-2]) / (2 * dx))**2 +
        ((v[1:-1, 2:, 1:-1] - v[1:-1, 0:-2, 1:-1]) / (2 * dy))**2 +
        ((w[2:, 1:-1, 1:-1] - w[0:-2, 1:-1, 1:-1]) / (2 * dz))**2 +

        2 * ((u[1:-1, 2:, 1:-1] - u[1:-1, 0:-2, 1:-1]) / (2 * dy) *
                (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, 0:-2]) / (2 * dx)) +
        2 * ((u[2:, 1:-1, 1:-1] - u[0:-2, 1:-1, 1:-1]) / (2 * dz) *
                (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 0:-2]) / (2 * dx)) +
        2 * ((v[2:, 1:-1, 1:-1] - v[0:-2, 1:-1, 1:-1]) / (2 * dz) *
                (w[1:-1, 2:, 1:-1] - w[1:-1, 0:-2, 1:-1]) / (2 * dy))
        )
        )

    return b


def pressure_poisson_periodic(p, dx, dy, dz):
    """ Returns the pressure on very point of the grid after iterating in pseudo-time """
    pn = np.empty_like(p)
    pseudotime = max(5, nit-stepcount) # Less are needed later in the simulation

    for q in range(pseudotime):
        pn = p.copy()
        d = 2*(dx**2*dy**2 + dx**2*dz**2 + dy**2*dz**2)
        p[1:-1, 1:-1, 1:-1] =\
            (((pn[1:-1, 1:-1, 2:] + pn[1:-1, 1:-1, 0:-2]) * (dy**2*dz**2) +
            (pn[2:, 1:-1, 1:-1] + pn[0:-2, 1:-1, 1:-1]) * (dy**2*dx**2) +
            (pn[1:-1, 2:, 1:-1] + pn[1:-1, 0:-2, 1:-1]) * (dx**2*dz**2)) / (d) -
            ((dx**2 * dy**2 * dz**2)/ (d)) * b[1:-1, 1:-1, 1:-1])

    p[0,::,::] = p[::,0,::] = p[::,::,0] = p[-1,::,::] = p[::,-1,::] = p[::,::,-1] = 0


    return p

# Load the mesh and calculate a bitmap from it
mesh = trimesh.load_mesh('untitled.stl')
grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
bitmap = mesh.contains(grid).reshape(u.shape)

udiff = 1
stepcount = 0


while udiff > .00001: # Calculates speed and pressure until it reaches a steady state
    un, vn, wn = u.copy(), v.copy(), w.copy()

    b = build_up_b(rho, dt, dx, dy, dz, u, v, w)
    p = pressure_poisson_periodic(p, dx, dy, dz)

    u[1:-1, 1:-1, 1:-1] =\
        un[1:-1, 1:-1, 1:-1] -\
        un[1:-1, 1:-1, 1:-1]  *  (dt/dx) * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 0:-2]) -\
        vn[1:-1, 1:-1, 1:-1]  *  (dt/dy) * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 0:-2, 1:-1]) -\
        wn[1:-1, 1:-1, 1:-1]  *  (dt/dz) * (un[1:-1, 1:-1, 1:-1] - un[0:-2, 1:-1, 1:-1]) -\
        (dt/(2*dx*rho)) * (p[1:-1, 1:-1, 2:]-p[1:-1, 1:-1, 0:-2]) +\
        nu * ((dt/dx**2)* (un[1:-1, 1:-1, 2:]-2*un[1:-1, 1:-1, 1:-1]+un[1:-1, 1:-1, 0:-2])+\
                (dt/dy**2)* (un[1:-1, 2:, 1:-1]-2*un[1:-1, 1:-1, 1:-1]+un[1:-1, 0:-2, 1:-1])+\
                (dt/dz**2)* (un[2:, 1:-1, 1:-1]-2*un[1:-1, 1:-1, 1:-1]+un[0:-2, 1:-1, 1:-1]))


    v[1:-1, 1:-1, 1:-1] =\
        vn[1:-1, 1:-1, 1:-1] -\
        un[1:-1, 1:-1, 1:-1]  *  (dt/dx) * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, 0:-2]) -\
        vn[1:-1, 1:-1, 1:-1]  *  (dt/dy) * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 0:-2, 1:-1]) -\
        wn[1:-1, 1:-1, 1:-1]  *  (dt/dz) * (vn[1:-1, 1:-1, 1:-1] - vn[0:-2, 1:-1, 1:-1]) -\
        (dt/(2*dy*rho)) * (p[1:-1, 2:, 1:-1]-p[1:-1, 0:-2, 1:-1]) +\
        nu * ((dt/dx**2)* (vn[1:-1, 1:-1, 2:]-2*vn[1:-1, 1:-1, 1:-1]+vn[1:-1, 1:-1, 0:-2])+\
                (dt/dy**2)* (vn[1:-1, 2:, 1:-1]-2*vn[1:-1, 1:-1, 1:-1]+vn[1:-1, 0:-2, 1:-1])+\
                (dt/dz**2)* (vn[2:, 1:-1, 1:-1]-2*vn[1:-1, 1:-1, 1:-1]+vn[0:-2, 1:-1, 1:-1]))

    w[1:-1, 1:-1, 1:-1] =\
        wn[1:-1, 1:-1, 1:-1] -\
        un[1:-1, 1:-1, 1:-1]  *  (dt/dx) * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, 0:-2]) -\
        vn[1:-1, 1:-1, 1:-1]  *  (dt/dy) * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 0:-2, 1:-1]) -\
        wn[1:-1, 1:-1, 1:-1]  *  (dt/dz) * (wn[1:-1, 1:-1, 1:-1] - wn[0:-2, 1:-1, 1:-1]) -\
        (dt/(2*dz*rho)) * (p[2:, 1:-1, 1:-1]-p[0:-2, 1:-1, 1:-1]) +\
        nu * ((dt/dx**2)* (wn[1:-1, 1:-1, 2:]-2*wn[1:-1, 1:-1, 1:-1]+wn[1:-1, 1:-1, 0:-2])+\
                (dt/dy**2)* (wn[1:-1, 2:, 1:-1]-2*wn[1:-1, 1:-1, 1:-1]+wn[1:-1, 0:-2, 1:-1])+\
                (dt/dz**2)* (wn[2:, 1:-1, 1:-1]-2*wn[1:-1, 1:-1, 1:-1]+wn[0:-2, 1:-1, 1:-1]))

    # BC

    u[0,::,::] = u[::,0,::] = u[::,::,0] = u[-1,::,::] = u[::,-1,::] = u[::,::,-1] = 10
    v[0,::,::] = v[::,0,::] = v[::,::,0] = v[-1,::,::] = v[::,-1,::] = v[::,::,-1] = 0
    w[0,::,::] = w[::,0,::] = w[::,::,0] = w[-1,::,::] = w[::,-1,::] = w[::,::,-1] = 0

    u[bitmap] = 0
    v[bitmap] = 0
    w[bitmap] = 0

    udiff = ((np.sum(u) - np.sum(un))+(np.sum(v) - np.sum(vn))+(np.sum(w) - np.sum(wn))) / (np.sum(u)+np.sum(v)+np.sum(w))
    stepcount += 1


print(stepcount)

# Saving the data to create the graphs and all
np.save('speed_x.npy', u)
np.save('speed_y.npy', v)
np.save('speed_z.npy', w)
np.save('pressure.npy', p)




