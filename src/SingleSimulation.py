import numpy

from ExperimentSetup import simulationMain


def main(experiment, output):
    experiment.world.setOutput(output)
    experiment.world.die.l = numpy.array([0.001,0.001,0.001])
    experiment.world.die.updateKineticQuantities()
    experiment.world.runSimulation()

if __name__ == '__main__':
    simulationMain(main)
