import numpy as np
import matplotlib.pyplot as plt


AU2M = 1.496e11
YR2SEC = 3600 * 24 * 365.25
DAY2SEC = 3600 * 24

# F = G M m / r**2 = m * a
G_CONST = 6.67430e-11  # m**3 / kg / s**2


class Body:

    mass = None  # in kg

    def __init__(self):
        self._posns = np.zeros((1, 3))
        self.vel = np.zeros(3)

    @property
    def pos(self):
        return self._posns[-1, :]
    
    # @property
    # def vel(self):
    #     return self._vs[-1, :]
    
    def pos_au(self):
        return self._posns / AU2M
    
    def dist(self, other: "Body"):
        return np.linalg.norm(self.pos - other.pos)

    def step(self, delt, other: list["Body"]):
        a = np.zeros(3)
        for o in other:
            force_vec = -self.pos / np.linalg.norm(self.pos)
            d2 = self.dist(o)**2
            a += force_vec * G_CONST * o.mass / d2

        pos = self.pos + self.vel * delt + 0.5 * a * delt**2
        self.vel = self.vel + a * delt

        self._posns = np.vstack([self._posns, pos])
        # self._vs = np.vstack([self._vs, vel])



class Earth(Body):

    mass = 5.972e24
    r = 1 * AU2M

    def __init__(self):
        super().__init__()
        self._posns = np.array([[self.r, 0, 0]])
        self.vel = np.array([0, 2 * np.pi * self.r / YR2SEC, 0])


class Sun(Body):

    mass = 1.988475e30


if __name__ == '__main__':

    t = 0

    DELT = 1 * DAY2SEC * .01

    e = Earth()
    s = Sun()

    while t < 2 * YR2SEC:

        e.step(DELT, [s])

        t += DELT

    plt.ion()

    plt.plot(e.pos_au()[:, 0], e.pos_au()[:, 1])
    plt.axis('equal')
    plt.grid(True)