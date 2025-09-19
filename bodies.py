import numpy as np
import matplotlib.pyplot as plt


AU2M = 1.496e11
YR2SEC = 3600 * 24 * 365.25
DAY2SEC = 3600 * 24

# F = G M m / r**2 = m * a
G_CONST = 6.67430e-11  # m**3 / kg / s**2


def push(arr: np.array, vec3: np.array):
    arr[:-1, :] = arr[1:, :]
    arr[-1, :] = vec3


def extrapolate(arr, t_future):
    result = np.zeros(3)
    for n in range(3):
        p = np.polyfit(np.arange(arr.shape[0]), arr[:, n], 2)
        result[n] = np.polyval(p, arr.shape[0] - 1 + t_future)
    return result


def r_func_plot(x, y, r_func, ax):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    ax.plot(r_func(r) * np.cos(theta), r_func(r) * np.sin(theta))


class Body:

    mass = None  # in kg

    def __init__(self):
        self._posns = np.zeros((1, 3))
        self.vel = np.zeros((3, 3))

    @property
    def pos(self):
        return self._posns[-1, :]
    
    @property
    def posns(self):
        return self._posns
    
    def pos_au(self):
        return self._posns / AU2M
    
    def dist(self, other: "Body"):
        return np.linalg.norm(self.pos - other.pos)

    def step(self, delt, other: list["Body"]):
        a = np.zeros(3)
        for o in other:
            force_vec = (o.pos - self.pos)
            force_vec /= np.linalg.norm(force_vec)
            d2 = self.dist(o)**2
            a_this = force_vec * G_CONST * o.mass / d2
            a += a_this

        # This extrapolation seems to work for errors introduced
        # with larger time steps in the successive approximation.
        # I can't really rationalize why it works though. :( 
        v = extrapolate(self.vel, 0.5)

        self.pos_tmp = self.pos + v * delt + 0.5 * a * delt**2
        push(self.vel, self.vel[-1] + a * delt)

    def finalize_step(self):
        self._posns = np.vstack([self._posns, self.pos_tmp])

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class Earth(Body):

    mass = 5.972e24
    r = 1 * AU2M

    def __init__(self):
        super().__init__()
        self._posns = np.array([[self.r, 0, 0]])
        self.vel = np.array([0, 2 * np.pi * self.r / YR2SEC, 0])
        self.vel = np.vstack([
            self.vel, self.vel, self.vel
        ])


class Moon(Body):

    mass = 7.34767309e22
    r = 3.844e8

    def __init__(self):
        super().__init__()
        self._posns = np.array([[1 * AU2M, self.r, 0]])
        del_v = 2 * np.pi * self.r / (27.3 * DAY2SEC)
        print(f"rel vel km/s: {del_v / 1000}")
        print(f"% of earth mass: {self.mass / Earth.mass * 100:.2f}")
        self.vel = np.array([del_v, 2 * np.pi * Earth.r / YR2SEC, 0])
        self.vel = np.vstack([
            self.vel, self.vel, self.vel
        ])


class Sun(Body):

    mass = 1.988475e30


if __name__ == '__main__':

    t = 0

    DELT = 1 * DAY2SEC * 0.1

    e = Earth()
    s = Sun()
    m = Moon()

    while t < 1 * YR2SEC:
    # while t < 30 * DAY2SEC:

        e.step(DELT, [s, m])
        m.step(DELT, [s, e])

        for o in [e, m]:
            o.finalize_step()

        t += DELT

    # plt.ion()

    plt.plot(e.pos_au()[:, 0], e.pos_au()[:, 1], label='Earth')
    plt.plot(e.pos_au()[-1, 0], e.pos_au()[-1, 1], '*')

    plt.plot(m.pos_au()[:, 0], m.pos_au()[:, 1], label='Moon')
    plt.plot(m.pos_au()[-1, 0], m.pos_au()[-1, 1], '*')
    
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    m_rel_e = (m.posns - e.posns) / Moon.r

    plt.figure()
    plt.plot(m_rel_e[:, 0], m_rel_e[:, 1])
    plt.plot(m_rel_e[-1, 0], m_rel_e[-1, 1], 'x')

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.axis('equal')
    plt.grid(True)
    plt.title('Moon orbit')

    f, ax = plt.subplots()
    func = lambda r: np.polyval(np.polyfit([.99, 1.01], [.5, 1.5], 1), r)
    r_func_plot(e.pos_au()[:, 0], e.pos_au()[:, 1], 
                func, ax)
    r_func_plot(m.pos_au()[:, 0], m.pos_au()[:, 1], 
                func, ax)
    
    plt.axis('equal')
    plt.grid(True)
    
    plt.show()