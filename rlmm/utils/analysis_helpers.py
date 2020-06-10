import glob


def create_pbsa_analysis():
    with open("run.sh", 'w') as f:
        files = map(lambda x: x.split("/")[0], glob.glob("*/traj.dcd"))
        first_run = True
        pids = []
        for folder in files:
            if first_run:
                f.write(f"cd {folder}\n")
                first_run = False
            else:
                f.write(f"cd ../{folder}\n")
            f.write(f"ante-MMPBSA.py -p com_{folder}.prmtop -l noslig.prmtop -r nosapo.prmtop -n :UNL\n")
            f.write(
                f"mpiexec -np 4 MMPBSA.py.MPI -cp com_{folder}.prmtop  -lp noslig.prmtop -rp nosapo.prmtop -y traj.dcd -i ../../mmpbsa_input.txt  -prefix gbsa_ &\n")
            f.write(f"process_id{folder}=$!\n")
            pids.append(f"wait $process_id{folder}")
        for pid in pids:
            f.write(pid + "\n")
