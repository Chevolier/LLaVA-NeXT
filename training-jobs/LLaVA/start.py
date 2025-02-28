import os
import json
import socket

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    os.environ['NODE_NUMBER'] = str(len(hosts))

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["FI_LOG_LEVEL"] = "WARN"

    os.system("chmod +x ./finetune-llava-video.sh")
    os.system("/bin/bash -c ./finetune-llava-video.sh")