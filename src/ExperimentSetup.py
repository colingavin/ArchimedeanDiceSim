import numpy
import json
import argparse
import sys
import inspect

from RigidBody import Body, World

class Experiment(object):
    def __init__(self, specification):
        super(Experiment, self).__init__()
        self.specification = json.loads(specification)
        vertexVectors = map(numpy.array, self.specification["vertices"])
        die = Body( vertexVectors,
                    self.specification["mass"],
                    pos = numpy.array(self.specification["initialPos"]),
                    inertia = self.specification["inertiaTensor"])
        self.world = World(numpy.array(self.specification["gravity"]), die, self.specification["groundRestitution"])
        self.iterations = self.specification["iterations"]
        self.maxLinearMomentum = self.specification["maxLinearMomentum"]
        self.maxAngularMomentum = self.specification["maxAngularMomentum"]
        self.faceCenters = map(numpy.array, self.specification["faceCenters"])


def simulationMain(main, extraArgs=None):
    parser = argparse.ArgumentParser(description="Run a single physical simulation of a non-standard dice")
    parser.add_argument("specification", type=argparse.FileType())
    parser.add_argument("--out", type=argparse.FileType('wb'), dest="outputFile")
    if extraArgs is not None:
        extraArgs(parser)

    args = parser.parse_args()

    output = args.outputFile if args.outputFile is not None else sys.stdout
    experiment = Experiment(args.specification.read())

    if len(inspect.getargspec(main)[0]) > 2:
        main(experiment, output, args)
    else:
        main(experiment, output)

    # cleanup
    if args.specification is not None:
        args.specification.close()
    
    if args.outputFile is not None:
        args.outputFile.close()
