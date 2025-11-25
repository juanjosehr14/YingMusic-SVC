#!/bin/bash

source='/user-fs/chenzihao/chengongyu/svc/seed-vc/test/good_vocals/Legend of a Hungry Wolf__vocals.mp3'
accompany='/user-fs/chenzihao/chengongyu/svc/seed-vc/test/good_vocals/Legend of a Hungry Wolf__instrumental.mp3'
target="/user-fs/chenzihao/chengongyu/svc/seed-vc/test/timbres/孙燕姿.wav"
diffusion_step=100
fp16="True"
config='./configs/YingMusic-SVC.yml'
checkpoint='/user-fs/chenzihao/chengongyu/svc/seed-vc/runs/zxyRL/11.18/all8_737.pth'
expname="DEB"
cuda="0"


python my_inference.py \
    --source "${source}" \
    --target "${target}" \
    --diffusion-steps "${diffusion_step}" \
    --checkpoint "${checkpoint}" \
    --expname "${expname}" \
    --cuda "${cuda}" \
    --fp16 "${fp16}" \
    --accompany "${accompany}" \
    --config "${config}"





