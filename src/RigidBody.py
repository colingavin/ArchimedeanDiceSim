import numpy
import numpy.linalg
from copy import copy
import csv
import sys
import random


class Body(object):
    def __init__(self, pts, mass, pos=numpy.zeros(3), rot=numpy.identity(3), p=numpy.zeros(3), l=numpy.zeros(3), inertia=None):
        super(Body, self).__init__()
        self.pts = pts
        self.mass = float(mass)
        self.pos = pos
        self.rot = rot
        self.p = p
        self.l = l
        if inertia is None:
            self.computeInertiaTensor()
        else:
            self.bodyInertia = inertia
            self.invBodyInertia = numpy.linalg.inv(self.bodyInertia)
        self.updateKineticQuantities()

    def computeInertiaTensor(self):
        ptMass = self.mass / len(self.pts)
        ident = numpy.identity(3)
        self.bodyInertia = sum([ptMass * (numpy.inner(pt, pt) * ident - numpy.outer(pt, pt)) for pt in self.pts])
        self.invBodyInertia = numpy.linalg.inv(self.bodyInertia)

    def updateKineticQuantities(self):
        self.vel = 1/self.mass * self.p
        self.inertia = numpy.dot(numpy.dot(self.rot, self.bodyInertia), numpy.transpose(self.rot))
        self.angVel = numpy.dot(numpy.linalg.inv(self.inertia), self.l)

    def timestep(self, force, torque, dt):
        self.pos = self.pos + dt * self.vel
        infAngVel = self.infinitesimalize(self.angVel)
        self.rot = self.normalizeRotation(self.rot + dt * numpy.dot(infAngVel, self.rot))
        self.p = self.p + dt * force
        self.l = self.l + dt * torque
        self.updateKineticQuantities()

    def worldspacePoints(self):
        return [numpy.dot(self.rot, pt) + self.pos for pt in self.pts]

    def infinitesimalize(self, vect):
        return numpy.array([[0, -vect[2], vect[1]],
                            [vect[2], 0, -vect[0]],
                            [-vect[1], vect[0], 0]])

    def normalizeRotation(self, m):
        (u, s, v) = numpy.linalg.svd(m)
        return numpy.dot(u, v)

    def kineticEnergy(self):
        # p^2 / (2m) + 1/2 (w . (I w))
        return pow(numpy.linalg.norm(self.p), 2) / (2 * self.mass) + 1/2 * numpy.dot(self.angVel, numpy.dot(self.angVel, self.inertia))

    def __str__(self):
        return "Body(pos = %s,\n\trot = %s,\n\tp = %s,\n\tl = %s)" % (str(self.pos), str(self.rot), str(self.p), str(self.l))


class World(object):
    def __init__(self, gravity, die, groundRestitution, output=None):
        super(World, self).__init__()
        self.die = die
        self.gravity = gravity
        self.setOutput(output)
        self.restitution = groundRestitution

    def setOutput(self, output):
        if output is not None:
            self.outputCSV = csv.writer(output)
        else:
            self.outputCSV = None

    # finds a point on the die intersecting the ground if it exists
    def intersectionPoint(self):
        worldPts = self.die.worldspacePoints()
        for pt in worldPts:
            if pt[2] < 0.0:
                return pt
        return None

    def runSimulation(self, avgLength=30, stoppingAvg=2, maxIterations=1000):
        rolling = []
        # run until the rolling average of steps / collision is too small
        # this happens when the body has entered sliding motion
        iterations = 0
        while iterations < maxIterations and (len(rolling) < avgLength or (float(sum(rolling)) / len(rolling)) > stoppingAvg):
            # run upto the next collision
            steps, contactPt = self.runUptoCollision()
            
            # apply the correction force for the collision
            self.collisionCorrect(contactPt)

            # updated the rolling average
            rolling.append(steps)
            if len(rolling) > avgLength:
                rolling.pop(0)
            iterations += 1

    def runUptoCollision(self):
        torque = numpy.zeros(3)
        steps = 0
        while True:
            steps += 1

            # step the die forward
            self.die.timestep(self.gravity, torque, 0.01)

            # check to see if any vertex of the die is intersecting the ground
            hitPt = self.intersectionPoint()
            if hitPt is not None:
                return steps, hitPt

            # output the configuration data
            self.writeState()

    def collisionCorrect(self, contactPt):
        pointDisplacement = contactPt - self.die.pos
        contactPointVel = self.die.vel + numpy.cross(self.die.angVel, pointDisplacement)
        relVel = contactPointVel[2]
        
        # compute impulse
        n = numpy.array([0,0,1])
        term1 = 1 / self.die.mass
        term2 = numpy.dot(n, numpy.cross(numpy.dot(numpy.linalg.inv(self.die.inertia), numpy.cross(pointDisplacement, n)), pointDisplacement))
        j = (-(1 + self.restitution) * relVel) / (term1 + term2)
        
        # apply the impluse
        force = j * n
        self.die.p = self.die.p + force
        self.die.l = self.die.l + numpy.cross(pointDisplacement, force)

        self.die.pos = self.die.pos + numpy.array([0,0,-contactPt[2]])

        #don't forget to update velocity, etc.
        self.die.updateKineticQuantities()

    # returns the ordered indices of the upward face
    def determineFaceUp(self):
        worldPts = self.die.worldspacePoints()
        indexedPts = list(enumerate(worldPts))
        indexedPts.sort(lambda a, b: cmp(b[1][2], a[1][2]))
        upperIndices = map(lambda a: a[0], indexedPts[0:4])
        upperIndices.sort()
        return upperIndices

    def writeState(self):
        if self.outputCSV is not None:
            self.outputCSV.writerow(list(self.die.pos) + list(self.die.rot.flat))


def main():
    cubePoints = [numpy.array(pt) for pt in [[-1,-1,-1], [-1,1,-1], [1,1,-1], [1,-1,-1], [-1,-1,1], [-1,1,1], [1,1,1], [1,-1,1]]];
    rectPoints = [numpy.array(pt) for pt in [[-2,-1,-1], [-2,1,-1], [2,1,-1], [2,-1,-1], [-2,-1,1], [-2,1,1], [2,1,1], [2,-1,1]]];
    octPoints = [numpy.array(pt) for pt in [[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[-1,0,0],[1,0,0]]];
    b = Body(rectPoints, 1, pos=numpy.array([0,0,10]), p=numpy.array([0.1,0.1,0]), l=numpy.array([0.1,1.1,0.1]))
    gravity = numpy.array([0, 0, -0.3])
    w = World(gravity, b, 0.3, None)
    #w.runSimulation()
    #w.determineFaceUp()
    s = Sampler(w, 1000, 2, 2, sys.stdout)
    s.run()

if __name__ == '__main__':
    main()
