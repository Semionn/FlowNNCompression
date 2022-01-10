#!/usr/bin/env bash
dev_name="gpu5"
scp -r experiments/*.py "$dev_name:~/jupyter/FlowNNCompression/experiments"
scp -r *.ipynb "$dev_name:~/jupyter/FlowNNCompression/"

update_docker_temp_dir_code="cp -r /home/spolyakov/jupyter/FlowNNCompression/* /home/s.polyakov/code/temp_dir/FlowNNCompression/"
ssh $dev_name "$update_docker_temp_dir_code"
