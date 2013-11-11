import numpy
from copy import copy
import csv
import random
import sys

from ExperimentSetup import simulationMain


class Sampler(object):
    def __init__(self, world, faceCenters, samples, linearMomentumMax, angularMomentumMax, output):
        super(Sampler, self).__init__()
        self.world = world
        self.samples = samples
        self.linearMomentumMax = linearMomentumMax
        self.angularMomentumMax = angularMomentumMax
        self.faceCenters = faceCenters
        if output is not None:
            self.outputCSV = csv.writer(output)
        else:
            self.outputCSV = None

    def sampleVector(self, maxNorm, n=3):
        vect = numpy.array([random.uniform(-maxNorm, maxNorm) for i in xrange(0, n)])
        if numpy.linalg.norm(vect) > maxNorm:
            return self.sampleVector(maxNorm, n)
        else:
            return vect

    # sample a random element of SO(3)
    def sampleOrientation(self):
        # sample a unit quaternion
        vect = self.sampleVector(1, 4)
        vect = vect / numpy.linalg.norm(vect)
        # convert it into a rotation matrix
        # see http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        (a, b, c, d) = tuple(vect)
        return numpy.array([
            [a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
            [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b],
            [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2]])

    def run(self):
        diePrototype = copy(self.world.die)
        for i in xrange(0, self.samples):
            self.world.die = copy(diePrototype)
            self.world.die.p = self.sampleVector(self.linearMomentumMax)
            self.world.die.l = self.sampleVector(self.angularMomentumMax)
            self.world.die.rot = self.sampleOrientation()
            self.world.die.updateKineticQuantities()
            self.world.runSimulation()
            self.writeResult(self.determineFaceUp())

    def determineFaceUp(self):
        maxHeight = 0
        maxIdx = -1
        for idx in range(len(self.faceCenters)):
            center = numpy.dot(self.world.die.rot, self.faceCenters[idx]) + self.world.die.pos
            if center[2] > maxHeight:
                maxHeight = center[2]
                maxIdx = idx
        return [maxIdx]

    def writeResult(self, row):
        if self.outputCSV is not None:
            self.outputCSV.writerow(row)
            sys.stdout.flush()


def main(experiment, output):
    sampler = Sampler(experiment.world, 
                      experiment.faceCenters,
                      experiment.iterations,
                      experiment.maxLinearMomentum,
                      experiment.maxAngularMomentum,
                      output)
    sampler.run()

if __name__ == '__main__':
    simulationMain(main)
