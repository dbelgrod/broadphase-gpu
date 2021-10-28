
import os
MESH_FN = "/home/dbelgrod/dataall/cloth-ball/bf_parallel"
BIN = "/home/dbelgrod/bruteforce-gpu/build/GPUBF_bin"
CPUBIN = "/home/dbelgrod/bruteforce-gpu/build/cpusweep_bin"
MESH_FOLDER = "/home/dbelgrod/dataset/UNC-Dynamic-Scene-Benchmarks/cloth-ball"
PREFIX = "cloth_ball"
EXT = ".ply"

def main():
    files = os.listdir(MESH_FOLDER)
    fn = [f for f in files if "fn" in files]
    for i in range(90,91):
        ee = os.path.join(MESH_FN, f'{i}ee_mma.json')
        vf = os.path.join(MESH_FN, f'{i}vf_mma.json')
        fn = lambda itr: os.path.join(MESH_FOLDER, PREFIX + str(itr) + EXT)
        cmd = f'{BIN} {fn(i)} {fn(i+1)} -c {ee} {vf}'
        os.system(f'echo {cmd}')
        os.system(cmd)


if __name__ == "__main__":
    main()