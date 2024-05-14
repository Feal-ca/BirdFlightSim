import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

##variable declarations
nx, ny, nz = 40, 40, 40
nit = 50
c = 1
dx, dy, dz = 2 / (nx - 1), 2 / (ny - 1), 2 / (nz - 1)

x, y, z = numpy.linspace(0, 2, nx), numpy.linspace(0, 2, ny), numpy.linspace(0, 2, nz)

X, Y, Z = numpy.meshgrid(x, y, z)


##physical variables
rho = 1
nu = .1
F = 1
dt = .001

#initial conditions
u = numpy.zeros((ny, nx, nz))
un = numpy.zeros((ny, nx, nz))

v = numpy.zeros((ny, nx, nz))
vn = numpy.zeros((ny, nx, nz))

w = numpy.zeros((ny, nx, nz))
wn = numpy.zeros((ny, nx, nz))


p = numpy.ones((ny, nx, nz))
pn = numpy.ones((ny, nx, nz))

b = numpy.zeros((ny, nx, nz))

def build_up_b(rho, dt, dx, dy, dz, u, v, w):
    b = numpy.zeros_like(u)
    b[1:-1, 1:-1, 1:-1] = (-1* rho *
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

    return b


def pressure_poisson_periodic(p, dx, dy, dz):
    pn = numpy.empty_like(p)

    for q in range(nit):
        pn = p.copy()
        d = 2*(dx**2*dy**2 + dx**2*dz**2 + dy**2*dz**2)
        p[1:-1, 1:-1, 1:-1] = (((pn[1:-1, 1:-1, 2:] + pn[1:-1, 1:-1, 0:-2]) * (dy**2*dz**2) +
                                (pn[2:, 1:-1, 1:-1] + pn[0:-2, 1:-1, 1:-1]) * (dy**2*dx**2) +
                                (pn[1:-1, 2:, 1:-1] + pn[1:-1, 0:-2, 1:-1]) * (dx**2*dz**2)) / (d) -

                                ((dx**2 * dy**2 * dz**2)/ (d)) * b[1:-1, 1:-1, 1:-1])
    return p

udiff = 1
stepcount = 0

while udiff > .001:
    un, vn, wn = u.copy(), v.copy(), w.copy()

    b = build_up_b(rho, dt, dx, dy, dz, u, v, w)
    p = pressure_poisson_periodic(p, dx, dy, dz)

    u[1:-1, 1:-1, 1:-1] = un[1:-1, 1:-1, 1:-1] -\
                        un[1:-1, 1:-1, 1:-1]  *  (dt/dx) * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 0:-2]) -\
                        vn[1:-1, 1:-1, 1:-1]  *  (dt/dy) * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 0:-2, 1:-1]) -\
                        wn[1:-1, 1:-1, 1:-1]  *  (dt/dz) * (un[1:-1, 1:-1, 1:-1] - un[0:-2, 1:-1, 1:-1]) -\
                        (dt/(2*dx*rho)) * (p[1:-1, 1:-1, 2:]-p[1:-1, 1:-1, 0:-2]) +\
                        nu * ((dt/dx**2)* (un[1:-1, 1:-1, 2:]-2*un[1:-1, 1:-1, 1:-1]+un[1:-1, 1:-1, 0:-2])+\
                              (dt/dy**2)* (un[1:-1, 2:, 1:-1]-2*un[1:-1, 1:-1, 1:-1]+un[1:-1, 0:-2, 1:-1])+\
                              (dt/dz**2)* (un[2:, 1:-1, 1:-1]-2*un[1:-1, 1:-1, 1:-1]+un[0:-2, 1:-1, 1:-1])) +\
                        F*dt

    v[1:-1, 1:-1, 1:-1] = vn[1:-1, 1:-1, 1:-1] -\
                        un[1:-1, 1:-1, 1:-1]  *  (dt/dx) * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, 0:-2]) -\
                        vn[1:-1, 1:-1, 1:-1]  *  (dt/dy) * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 0:-2, 1:-1]) -\
                        wn[1:-1, 1:-1, 1:-1]  *  (dt/dz) * (vn[1:-1, 1:-1, 1:-1] - vn[0:-2, 1:-1, 1:-1]) -\
                        (dt/(2*dy*rho)) * (p[1:-1, 2:, 1:-1]-p[1:-1, 0:-2, 1:-1]) +\
                        nu * ((dt/dx**2)* (vn[1:-1, 1:-1, 2:]-2*vn[1:-1, 1:-1, 1:-1]+vn[1:-1, 1:-1, 0:-2])+\
                              (dt/dy**2)* (vn[1:-1, 2:, 1:-1]-2*vn[1:-1, 1:-1, 1:-1]+vn[1:-1, 0:-2, 1:-1])+\
                              (dt/dz**2)* (vn[2:, 1:-1, 1:-1]-2*vn[1:-1, 1:-1, 1:-1]+vn[0:-2, 1:-1, 1:-1]))

    w[1:-1, 1:-1, 1:-1] = wn[1:-1, 1:-1, 1:-1] -\
                        un[1:-1, 1:-1, 1:-1]  *  (dt/dx) * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, 0:-2]) -\
                        vn[1:-1, 1:-1, 1:-1]  *  (dt/dy) * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 0:-2, 1:-1]) -\
                        wn[1:-1, 1:-1, 1:-1]  *  (dt/dz) * (wn[1:-1, 1:-1, 1:-1] - wn[0:-2, 1:-1, 1:-1]) -\
                        (dt/(2*dz*rho)) * (p[2:, 1:-1, 1:-1]-p[0:-2, 1:-1, 1:-1]) +\
                        nu * ((dt/dx**2)* (wn[1:-1, 1:-1, 2:]-2*wn[1:-1, 1:-1, 1:-1]+wn[1:-1, 1:-1, 0:-2])+\
                              (dt/dy**2)* (wn[1:-1, 2:, 1:-1]-2*wn[1:-1, 1:-1, 1:-1]+wn[1:-1, 0:-2, 1:-1])+\
                              (dt/dz**2)* (wn[2:, 1:-1, 1:-1]-2*wn[1:-1, 1:-1, 1:-1]+wn[0:-2, 1:-1, 1:-1]))


    u[20:30, 20:30, 20:30] = 0
    v[20:30, 20:30, 20:30] = 0
    w[20:30, 20:30, 20:30] = 0

    udiff = (numpy.sum(u) - numpy.sum(un)) / numpy.sum(u)
    stepcount += 1


"""fig = pyplot.figure(figsize = (11,7), dpi=100)
ax = pyplot.figure().add_subplot(projection='3d')
ax.quiver(X[::3, ::3, ::3], Y[::3, ::3, ::3], Z[::3, ::3, ::3], u[::3, ::3, ::3], v[::3, ::3, ::3], w[::3, ::3, ::3], length = 0.2, cmap="Reds");"""

fig = pyplot.figure(figsize=(11, 7), dpi=100)
s = 25

contour = pyplot.contourf(X[::, ::, s], Y[::, ::, s], p[::, ::, s], cmap=cm.viridis)
quiver = pyplot.quiver(X[::, ::, s], Y[::, ::, s], u[::, ::, s], v[::, ::, s], w[::, ::, s])
pyplot.colorbar(contour)

pyplot.show()

