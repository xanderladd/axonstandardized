
import os
import sys
import numpy as np
import subprocess
import shutil
import csv
import json
import re
import fileinput


def main():
    num_trials=int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    len_of_trials = len(str(num_trials))
    volt_sandbox = "./volts_sandbox_setup/sbatch_run.slr"
    score_sandbox = "./score_volts_efficent_sandbox/sbatch_score.slr"

    textToSearch = '#SBATCH --array 1-'
    if num_trials != 0:
        textToReplace = '#SBATCH --array 1-' + str(num_trials) + '\n'
    elif num_trials == 0 and num_nodes <= 1:
        print("num trials set to 0 so using multithread parallelism instead of multi-node... jobs will run on one node only")
        textToReplace = '##SBATCH --array 1- \n'
    else:
        textToReplace = '#SBATCH --array 1-' + str(num_nodes) + '\n'

    tempFile = open( volt_sandbox, 'r+' )

    for line in fileinput.input( volt_sandbox ):
        if re.match(r'.*?#SBATCH --array 1-.',line):
             print('Match Found')
             tempFile.write(line.replace(line, textToReplace))
             #print(line.replace(line, textToReplace))
        else:
            tempFile.write(line)

    tempFile2 = open( score_sandbox, 'r+' )

    for line in fileinput.input( score_sandbox ):
        if re.match(r'.*?#SBATCH --array 1-.',line):
            print('Match Found')
            textToReplaceThrottle = '#SBATCH --array 1-' + str(num_trials) + '%40\n'
            tempFile2.write(line.replace(line, textToReplaceThrottle))
            #print(line.replace(line, textToReplace))
        else:
            tempFile2.write(line)

    tempFile2.close()



if __name__ == '__main__':
    main()
