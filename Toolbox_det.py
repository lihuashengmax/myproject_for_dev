import os
from typing import List, Callable
import numpy as np
from Levenshtein import distance

import pyautogui

import cv2
import matplotlib.pyplot as plt
from io import BytesIO

from PIL import Image, ImageEnhance, ExifTags
from ImageProcessing import *

from Utils import pretty_print_content


def tool_doc(description):
    def decorator(func):
        func.tool_doc = description
        return func
    return decorator


class ImageProcessingToolBoxes_det:

    def __init__(self, image_path, output_dir_name):
        self.image_path = image_path
        self.output_dir_name = output_dir_name
        self.coor_list = []#记录coor的历史值

        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        self.height, self.width = img.shape[:2]  # 只取前两个维度，忽略通道数

        self.coor = None#线条当前位置
        self.side = None#当前正在找哪一条边框，可以为left or right or up or down
        self.min, self.max = None, None#二分查找的最大值与最小值

        self.history_messages = []  # Record of conversation history

    def check_history_messages(self):
        """
        Print the history messages in a formatted manner.
        This method is useful for reviewing the sequence of operations performed on the image.
        """
        pretty_print_content(self.history_messages)

    def add_history_messages(self, response):
        self.history_messages.append({"role": "assistant", "content": response})
        
        

    def check_if_finish(self):#判断是否为最终框线
        if len(self.coor_list) > 1 and self.coor_list[-1] == self.coor_list[-2]:
            return True
        else:
            return False
        
    def get_coor(self):#返回coor
        return self.coor


    @staticmethod
    def get_tool_docs(tools: List[Callable]):
        
        tool_docs = []
        for func in tools:
            if callable(func) and hasattr(func, 'tool_doc'):
                tool_docs += [{"type": "function", "function": func.tool_doc[0]}] 
        return tool_docs
    
    def get_function_mapping(self):
        """
        Get a mapping of function names to their corresponding method names.
        This method is useful for retrieving the mapping of function names to their corresponding methods.
        """
        tool_docs = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, 'tool_doc'):
                tool_docs[attr.tool_doc[0]["name"]] = attr.__name__
        return tool_docs
    
    def clear_move_history(self):
        self.coor_list = []
        self.coor = None#线条当前位置
        self.side = None#当前正在找哪一条边框，可以为left or right or up or down
        self.min, self.max = None, None#二分查找的最大值与最小值

        # self.history_messages = []

    def get_current_image_path(self):
        """Return the path of the most recently processed image
        This method is useful for retrieving the current state of the image after processing"""
        return self.image_path
    
    def initialize_coor(self, side):
        assert side == 'left' or side == 'right' or side == 'up' or side == 'down'
        self.side = side

        self.min = 0
        if self.side == 'left' or self.side == 'right':
            self.max = self.width
        elif self.side == 'up' or self.side == 'down':
            self.max = self.height
        self.coor = (self.min + self.max) // 2
        # self.coor = 0
        self.coor_list.append(self.coor)
        return self.draw_line()

    
    def draw_line(self):
        """在图像上按照coor与mode进行画线"""
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        line_width = 2
        if self.side == 'left' or self.side == 'right':
            # 绘制竖线（x=coor，起点 (coor, 0)，终点 (coor, height)）
            cv2.line(img, (self.coor, 0), (self.coor, self.height), (255, 0, 0), line_width)  # 蓝色
        elif self.side == 'up' or self.side == 'down':
            # 绘制横线（y=coor，起点 (0, coor)，终点 (width, coor)）
            cv2.line(img, (0, self.coor), (self.width, self.coor), (0, 0, 255), line_width)  # 红色
        # 保存图像
        file_name = self.side + '_' + str(len(self.coor_list)) + '.' + self.image_path.split(".")[-1]
        file_name = os.path.join(self.output_dir_name, file_name).replace("\\", "/")
        
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        cv2.imwrite(file_name, img)
        return file_name

    @tool_doc([
        {
            "name": "func_to_return_responses",
            "description": """
                A function for GPT to structure and return its responses. Instead of providing responses in the content, 
                GPT should use this function to encapsulate all responses within the response parameter. This ensures a 
                consistent format for response handling and processing.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": """
                            The complete response content from GPT. This should contain all information, explanations, 
                            or answers that GPT wants to communicate. The content should be properly formatted with 
                            appropriate line breaks and spacing for readability. Markdown formatting is supported.
                        """
                    },
                },
                "required": ["response"]
            }
        }
    ])
    # def func_to_return_responses(self, response):#二分法行不通
    #     assert response == '1' or response == '2'#1则coor应该变小 2则coor应该变大
    #     assert self.side == 'left' or self.side == 'right' or self.side == 'up' or self.side == 'down'
    #     if response == '1':#coor应该向左或者向上寻找
    #         self.max = self.coor
    #     elif response == '2':#coor应该向右或者向下寻找
    #         self.min = self.coor
    #     self.coor = (self.min + self.max) // 2
    #     self.coor_list.append(self.coor)

    def func_to_return_responses(self, response):#不用二分法 直接调整像素值

        # self.history_messages.append({"role": "assistant", "content": response})

        assert response == '1' or response == '2'#1则coor应该变小 2则coor应该变大
        assert self.side == 'left' or self.side == 'right' or self.side == 'up' or self.side == 'down'
        if response == '1':#coor应该向左或者向上寻找
            self.coor = max(self.coor - 10, 0)
        elif response == '2':#coor应该向右或者向下寻找
            self.coor = min(self.coor + 10, self.width)

        self.coor_list.append(self.coor)
        
