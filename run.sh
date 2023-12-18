#!/bin/bash
#SBATCH -J trial
#SBATCH -o test-%j.print
#SBATCH -e test-%j.terminal
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --mem=512GB
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:2


#source ~/.bashrc

#conda activate rixin

clash
proxy
cd -
python run.py -c config/chat_demo.yaml