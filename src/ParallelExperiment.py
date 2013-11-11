import argparse
import numpy
import multiprocessing
import subprocess
import sys
import os.path


def main():
    parser = argparse.ArgumentParser(description="Run multiple monte-carlo simulations in parallel")
    parser.add_argument("specifications", nargs="+")
    parser.add_argument("--outdir", dest="outputDir")
    parser.add_argument("--n", type=int, dest="threads", default=lambda: multiprocessing.cpu_count())

    args = parser.parse_args()

    for spec in args.specifications:
        output = open(os.path.join(args.outputDir, os.path.basename(spec) + ".csv"), 'wb') if args.outputDir is not None else sys.stdout
        procs = []
        for i in range(args.threads):
            proc = subprocess.Popen(["python", "ExperimentRunner.py", spec], stdout=output)
            procs.append(proc)

        for proc in procs:
            proc.wait()

        if args.outputDir is not None:
            output.close()


if __name__ == '__main__':
    main()
