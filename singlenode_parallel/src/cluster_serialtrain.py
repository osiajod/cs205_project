import argparse
import subprocess
import os.path
import math


def dispatch(out_file, err_file, cmd, go, num_cores=1, num_nodes=1, max_hours=1, memory_in_gb=16):
    """
    Populates 'runscript.sh' file to run 'dqn_original.py' file
    on cluster's GPU partition for 'max_hours' hours with 1 node, 1 core, and 32GB memory
    """
    with open('runscript.sh', 'w+') as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -n {num_cores}                 # Number of cores
#SBATCH -N {num_nodes}                 # Ensure that all cores are on one machine
#SBATCH -t {format_time(max_hours)}           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to
#SBATCH --gres=gpu           # number of GPUs (here 1; see also --gres=gpu:n)
#SBATCH --mem={gb_to_mb(memory_in_gb)}          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o {out_file}  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e {err_file}  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/5.0.1-fasrc01  # Load modules
module load cudnn/7.6.5.32_cuda10.1-fasrc01
module load cuda/10.0.130-fasrc01
source activate gpt2  # Switch to correct conda environment
{cmd}  # Run code
"""
        )
    if go:
        subprocess.call(['sbatch', 'runscript.sh'])


def format_time(total_hours):
    '''Converts hours to D-HH:MM format.'''
    days = total_hours // 24
    frac_hour, hours = math.modf(total_hours % 24)
    minutes = math.ceil(frac_hour * 60.0)
    if minutes == 60:
        hours += 1
        minutes = 0
    if hours == 24:
        hours = 0
        days += 1
    return f'{int(days)}-{int(hours):02d}:{int(minutes):02d}'


def gb_to_mb(gb):
    '''Converts gb to mb'''
    mb = int(gb * 1000)
    return mb


def print_red(string):
    print('\033[1;31;40m' + string + '\033[0;37;40m')


def print_yellow(string):
    print('\033[1;33;40m' + string + '\033[0;37;40m')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                        help="""
                        (str) Base name for output and error files to which SLURM writes results, 
                        and ID for storing checkpoints and samples.
                        """)
    parser.add_argument('dataset', type=str,
                        help='(str) Path to dataset for training')
    parser.add_argument('restore_from', type=str,
                        help='(str) Either "latest", "fresh", or a path to a checkpoint file')
    parser.add_argument('--sample_every', default=100, type=int,
                        help='(int) How often to generate samples (every N steps)')
    parser.add_argument('--save_every', default=1000, type=int,
                        help='(int) How often to create model checkpoint (every N steps)')
    parser.add_argument('--go', action='store_true',
                        help='(flag) Submits jobs to cluster if present. Default disabled')
    parser.add_argument('--num_cores', default=1, type=int,
                        help='(int) Number of cores to run on')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='(int) Number of nodes to run on')
    parser.add_argument('--hours', default=1., type=float,
                        help='(float) Wall clock time to request on SLURM')
    parser.add_argument('--gb_memory', default=16., type=float,
                        help='(float) Memory (in GB) to request')
    args = parser.parse_args()

    basename = args.run_name
    out_file = basename + '.txt'
    err_file = basename + '.err.txt'
    cmd = f'python3 train.py --dataset {args.dataset} --restore_from {args.restore_from} --run_name {args.run_name}\
    --sample_every {args.sample_every} --save_every {args.save_every}'

    # If file for a configuration exists, skip over that configuration
    if os.path.exists(out_file) or os.path.exists(err_file):
        print_red(f'{basename} (already exists; skipping)')

    else:
        # Otherwise, generate and run script on cluster
        # Populates 'runscript.sh' file to run specified file
        # on cluster's GPU partition with specified number of nodes, cores, and memory
        # Dispatches 'runscript.sh' to SLURM if '--go' flag was specified in CLI
        print(basename)
        dispatch(out_file=out_file,
                 err_file=err_file,
                 cmd=cmd,
                 go=args.go,
                 num_cores=args.num_cores,
                 num_nodes=args.num_nodes,
                 max_hours=args.hours,
                 memory_in_gb=args.gb_memory)

    if not args.go:
        print_yellow('''
*** This was just a test! No jobs were actually dispatched.
*** If the output looks correct, re-run with the "--go" argument.''')
        print(flush=True)


if __name__ == '__main__':
    main()
