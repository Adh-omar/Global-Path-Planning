# https://stackoverflow.com/questions/57065080/draw-perpendicular-line-of-fixed-length-at-a-point-of-another-line
import math
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def norm(self):
        return self.dot(self) ** 0.5

    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)

    def lengthto(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def perp(self):
        if self.y == 0:
            return Vector(1,-self.x/0.001)
        return Vector(1, -self.x / self.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def scalar_div(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)

    def sqr_magnitude(self):
        return (self.x ** 2 + self.y ** 2)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def __str__(self):
        return f'({self.x}, {self.y})'
