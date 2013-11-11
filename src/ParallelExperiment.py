import argparse
import numpy
import multiprocessing
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run multiple monte-carlo simulations in parallel")
    parser.add_argument("specification")
    parser.add_argument("--out", type=argparse.FileType('wb'), dest="outputFile")
    parser.add_argument("--n", type=int, dest="threads", default=lambda: multiprocessing.cpu_count())

    args = parser.parse_args()

    output = args.outputFile if args.outputFile is not None else sys.stdout

    procs = []
    for i in range(args.threads):
        proc = subprocess.Popen(["python", "ExperimentRunner.py", args.specification], stdout=output)
        procs.append(proc)

    for proc in procs:
        proc.wait()

    if args.outputFile is not None:
        args.outputFile.close()


if __name__ == '__main__':
    main()
