import os
import json
from dotenv import load_dotenv
import openai
from prompts import *
import numpy as np

from Chat import AgentClient
from Toolbox_det import ImageProcessingToolBoxes_det

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# Set the environment variable for the OpenAI API key
load_dotenv()
# api_key = your_key
openai.api_key = api_key

test_image_path = './test_1.jpg'
output_dir_path = './test-output'
image_process = ImageProcessingToolBoxes_det(image_path=test_image_path, output_dir_name=output_dir_path)
chat_client = AgentClient(api_key=api_key, toolbox_instance=image_process, debug=False)

object_description = 'the black car'
# 确定目标的四边框
_ = chat_client.agent_interaction(
    prompt=object_det_prompt(object_description=object_description)
)

