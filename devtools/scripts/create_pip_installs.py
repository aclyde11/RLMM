import yaml
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='devtools/conda-envs/test_env.yaml')
args = parser.parse_args()

with open(args.i, 'r') as f:
    res = yaml.load(f, Loader=yaml.FullLoader)

installs = res['dependencies'][-1]['pip']
subprocess.run(["pip install " + " ".join(installs)], shell=True)

