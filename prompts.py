def object_det_prompt(object_description: str) -> list:
    prompt = [
    (
        "Is there " + object_description + " in the image?\n"
        "If the answer is yes, return 1.\n"
        "If the answer is no, return 2.\n"
    ),
    (
        "**Character**\n"
        "You are a professional object detection data annotator responsible for labeling objects in images using bounding boxes.\n"
        "**Background**\n"
        "In this image, there is " + object_description + " that you should annotate using a bounding box. Additionally, there is a vertical blue line in the image.\n"
        "**Ambition**\n"
        "Your objective is to judge horizontal relative position of " + object_description + " bounding box's left edge and the vertical blue line. Relative position in the horizontal direction should be described using left or right.\n"
        "**Task**\n"
        "1. Judge the relative position as follows: If the vertical blue line is to the left of " + object_description + " bounding box's left edge, return 2. If the vertical blue line is to the right of " + object_description + " bounding box's left edge, return 1.\n"
        "2. Only return the number 1 or 2 without any explanation or additional text.\n"
    ),
    (
        "Is the blue vertical line in the image to the right of " + object_description + "?\n"
        "If the answer is yes, return 1.\n"
        "If the answer is no, return 2.\n"
    ),
    (
        "Is the red horizontal line in the image above " + object_description + "?\n"
        "If the answer is yes, return 2.\n"
        "If the answer is no, return 1.\n"
    ),
    (
        "Is the red horizontal line in the image below " + object_description + "?\n"
        "If the answer is yes, return 1.\n"
        "If the answer is no, return 2.\n"
    ),
    ]
    return prompt

