import base64
import os
import json

from PIL import Image, ImageEnhance, ExifTags
import openai

from Utils import base64_encode_image


class AgentClient:
    def __init__(self, api_key, model="gpt-4o-2024-11-20", toolbox_instance=None, debug=False):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.debug = debug
        self.toolbox_instance = toolbox_instance
        self.total_tokens = []

    def create_chat_completion(self, messages, tools, tool_choice="required", model=None, max_tokens=None):
        params = {
            "model": model or self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        completion = self.client.chat.completions.create(**params)

        tokens = completion.usage.total_tokens
        self.total_tokens.append(tokens)

        if self.debug:
            print(f"Parameters: {json.dumps(params, indent=2, ensure_ascii=False)}")

        return completion

    def agent_interaction(self, prompt):
        """
        Handles the interaction between the agent and the user.
        This is for the conversation without additional function calling, only `func_to_return_responses` is used.

        Args:
            system_prompt (str): The initial system message.
            user_prompt (str, optional): The user's input message. Defaults to None.
            provide_image (bool): Whether to include an image in the interaction. Defaults to True.
            run_tool (bool): Whether to execute a tool based on the completion. Defaults to True.

        Returns:
            If run_tool is False, returns the completion object.
            Otherwise, executes the parsed function call.
        """
        image_path = self.toolbox_instance.get_current_image_path()#获得图像文件位置

        ###############首先判断图像中是否有目标
        past_messages = None
        user_prompt = None
        messages = self.build_messages(prompt[0], user_prompt, image_path, past_messages)
        completion = self.create_chat_completion(messages, self.toolbox_instance.get_tool_docs([self.toolbox_instance.func_to_return_responses]), tool_choice="required")
        response = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["response"]
        assert response == '1' or response == '2'
        if response == '1':
            print('The initial judgement from the model is having the object.')
        elif response == '2':
            print('The initial judgement from the model is not having the object.')
            return
        ###############首先判断图像中是否有目标
        box = []
        ###############确定左框线
        self.toolbox_instance.clear_move_history()
        image_path = self.toolbox_instance.initialize_coor('left')
        while True:
            past_messages = None
            user_prompt = None
            messages = self.build_messages(prompt[1], user_prompt, image_path, past_messages)
            completion = self.create_chat_completion(messages, self.toolbox_instance.get_tool_docs([self.toolbox_instance.func_to_return_responses]), tool_choice="required")
            response = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["response"]
            print(response)
            self.parse_function_call(completion, self.toolbox_instance)
            ##判断是否终止循环
            finish = self.toolbox_instance.check_if_finish()
            if finish:
                box.append(self.toolbox_instance.get_coor())
                break
            image_path = self.toolbox_instance.draw_line()
        ###############确定左框线
            
        # ###############确定右框线
        # self.toolbox_instance.clear_move_history()
        # image_path = self.toolbox_instance.initialize_coor('right')
        # while True:
        #     past_messages = None
        #     user_prompt = None
        #     messages = self.build_messages(prompt[2], user_prompt, image_path, past_messages)
        #     completion = self.create_chat_completion(messages, self.toolbox_instance.get_tool_docs([self.toolbox_instance.func_to_return_responses]), tool_choice="required")
        #     response = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["response"]
        #     print(response)
        #     self.parse_function_call(completion, self.toolbox_instance)
        #     ##判断是否终止循环
        #     finish = self.toolbox_instance.check_if_finish()
        #     if finish:
        #         box.append(self.toolbox_instance.get_coor())
        #         break
        #     ##判断是否终止循环
        #     image_path = self.toolbox_instance.draw_line()
        # ###############确定右框线
            
        # ###############确定上框线
        # self.toolbox_instance.clear_move_history()
        # image_path = self.toolbox_instance.initialize_coor('up')
        # while True:
        #     past_messages = None
        #     user_prompt = None
        #     messages = self.build_messages(prompt[3], user_prompt, image_path, past_messages)
        #     completion = self.create_chat_completion(messages, self.toolbox_instance.get_tool_docs([self.toolbox_instance.func_to_return_responses]), tool_choice="required")
        #     response = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["response"]
        #     print(response)
        #     self.parse_function_call(completion, self.toolbox_instance)
        #     ##判断是否终止循环
        #     finish = self.toolbox_instance.check_if_finish()
        #     if finish:
        #         box.append(self.toolbox_instance.get_coor())
        #         break
        #     ##判断是否终止循环
        #     image_path = self.toolbox_instance.draw_line()
        # ###############确定上框线
            
        # ###############确定下框线
        # self.toolbox_instance.clear_move_history()
        # image_path = self.toolbox_instance.initialize_coor('down')
        # while True:
        #     past_messages = None
        #     user_prompt = None
        #     messages = self.build_messages(prompt[4], user_prompt, image_path, past_messages)
        #     completion = self.create_chat_completion(messages, self.toolbox_instance.get_tool_docs([self.toolbox_instance.func_to_return_responses]), tool_choice="required")
        #     response = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["response"]
        #     print(response)
        #     self.parse_function_call(completion, self.toolbox_instance)
        #     ##判断是否终止循环
        #     finish = self.toolbox_instance.check_if_finish()
        #     if finish:
        #         box.append(self.toolbox_instance.get_coor())
        #         break
        #     ##判断是否终止循环
        #     image_path = self.toolbox_instance.draw_line()
        # ###############确定下框线

        

        


    @staticmethod
    def build_image_message(image_path, messages: list = None):
        """
        Build a message containing an image encoded in Base64 format.

        Parameters:
        image_path (str): The path to the image file.
        messages (list, optional): A list of existing messages to append the image message to. Defaults to None.

        Returns:
        list: A list of messages including the image message.
        """
        if messages is None:
            return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_encode_image(image_path)}",}}]}]
        else:
            if messages[0]["role"] == "system":
                return messages[:1] + [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_encode_image(image_path)}",}}]}] + messages[1:]
            else:
                return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_encode_image(image_path)}",}}]}] + messages

    @staticmethod
    def build_messages(system_prompt: str, user_prompt: str = None, image_path: str = None, past_messages: list = None):
        """
        Build a list of messages for a chat system, optionally including a user prompt, an image, and past messages.

        Parameters:
        system_prompt (str): The system's initial message.
        user_prompt (str, optional): The user's message. Defaults to None.
        image_path (str, optional): The path to an image file to include in the messages. Defaults to None.
        past_messages (list, optional): A list of previous messages. Defaults to None.

        Returns:
        list: A list of messages including the system prompt, past messages (if provided), 
              user prompt (if provided), and image (if provided).
        """
        messages = [{"role": "system", "content": system_prompt}]

        if past_messages is not None:
            messages.extend(past_messages)
        
        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})
        
        if image_path is not None:
            messages = AgentClient.build_image_message(image_path, messages)
        
        return messages

    @staticmethod
    def parse_function_call(completion, instance):
        """
        Parse and call the function suggested by the API

        Parameters:
        completion (object): The API completion object
        instance (object): The instance of the class `ImageProcessingToolBoxes` to call the method on

        Returns:
        object: The result of the function call
        """
        function_mapping = instance.get_function_mapping()

        if completion.choices[0].finish_reason == "tool_calls":
            function_name = completion.choices[0].message.tool_calls[0].function.name
            arguments = completion.choices[0].message.tool_calls[0].function.arguments
            function_args = json.loads(arguments)
            
            if function_name in function_mapping:
                method_name = function_mapping[function_name]
                if hasattr(instance, method_name):
                    method = getattr(instance, method_name)
                    result = method(**function_args)
                    return result
                else:
                    raise AttributeError(f"The instance does not have a method named '{method_name}'.")
            else:
                raise ValueError(f"The function '{function_name}' is not defined in the function mapping.")
        return None


