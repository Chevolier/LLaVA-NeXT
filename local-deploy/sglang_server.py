from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process

vision_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path /home/ec2-user/SageMaker/efs/Models/LLaVA-Video-7B-Qwen2 \
    --chat-template=chatml-llava --port 30008 
"""
"""
python3 -m sglang.launch_server --model-path /home/ec2-user/SageMaker/efs/Projects/LLaVA-NeXT/checkpoints/llava-video-task-full-2025-03-19-05-52-43/checkpoint-100 \
    --chat-template=chatml-llava --port 30008 
"""
#     """
# python3 -m sglang.launch_server --model-path /home/ec2-user/SageMaker/efs/Models/Qwen2.5-VL-7B-Instruct \
#     --chat-template=qwen2-vl --port 37008
# """
)

wait_for_server(f"http://localhost:{port}")