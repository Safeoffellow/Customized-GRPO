import os
import json
import time
import argparse
import threading
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import jsonlines
import base64

writer_lock = threading.Lock()

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")

def filter_dict(d):
    return {k: v for k, v in d.items() if v <= 3}

def vlm_api(msg, max_retries=3, delay=2):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f'Your_key',
        'Content-Type': 'application/json',
    }

    data = {
        "messages": msg,
        "model": "gpt-4o-0806-global" # qwen2.5-vl-72b-instruct | qwen-vl-max
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            # print("Response:", response.json())
            # print("Response:", response)
            if response.status_code == 200:
                res = response.json()['choices'][0]['message']['content']
                if res:
                    return res  # 返回有效内容
                else:
                    retries += 1
            else:
                retries += 1

        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Request failed , error: {str(e)}")
        
        # 重试前的延迟
        time.sleep(delay)

    return ""  # 请求失败时返回空字符串，避免返回 None

def process_image(line, task):
    concept_prompt = """
        ### Task Definition
        You will be provided with an image generated based on reference image.
        As an experienced evaluator, your task is to evaluate the semantic consistency between the subject of the generated image and the reference image, according to the scoring criteria.

        ### Scoring Criteria
        You should compare only the subjects present in the two images, focusing on four basic visual features:
        1. Shape: Evaluate whether the outline, structure, and proportions of the subject in the generated image match those of the subject in the reference image. This includes the geometric shape of the subject, clarity of its edges, relative sizes, and the spatial relationships between various parts forming the subject. Ignore the background and any other elements that are not part of the main subject.
        2. Color: Compare the accuracy and consistency of the main colors of the subject in the generated image with those of the subject in the reference image. This includes the subject's saturation, hue, brightness, and the distribution of colors. Do not consider any colors that appear outside the subject itself.
        3. Texture: Focus on whether the generated image accurately captures the fine textures and surface details of the subject, without appearing blurry, and whether it meets the required level of realism and clarity. Excessive abstraction or stylization of the subject’s texture is unnecessary unless specifically indicated by the text prompt. The evaluation of texture should apply only to the subject, not to the background or unrelated areas.
        4. Facial Features: If the subject is a person or animal, closely examine whether the facial features of the subject in the generated image visually match those of the subject in the reference image. Pay particular attention to facial similarity, but do not evaluate faces or features unrelated to the primary subject.

        ### Scoring Range
        You need to give a specific integer score based on the comprehensive performance of the visual features above, ranging from 0 to 4:
        - Very Poor (0): No resemblance. The generated image's subject has no relation to the reference.
        - Poor (1): Minimal resemblance. The subject falls within the same broad category but differs significantly.
        - Fair (2): Moderate resemblance. The subject shows likeness to the reference with notable variances.
        - Good (3): Strong resemblance. The subject closely matches the reference with only minor discrepancies.
        - Excellent (4): Near-identical. The subject of the generated image is virtually indistinguishable from the reference.

        ### Input format
        Every time you will receive two images, the first image is a reference image, and the second image is the generated image.

        Please carefully review each image of the subject: {subject}.

        ### Output Format
        Score: [Your Score]

        You must adhere to the specified output format, which means that only the scores need to be output, excluding your analysis process.
    """
    concept_prompt_cot = """
        ### Task Definition
        You will be provided with an image generated based on reference image.
        As an experienced evaluator, your task is to evaluate the semantic consistency between the subject of the generated image and the reference image, according to the scoring criteria.

        ### Scoring Criteria
        You should compare only the subjects present in the two images, focusing on four basic visual features:
        1. Shape: Evaluate whether the outline, structure, and proportions of the subject in the generated image match those of the subject in the reference image. This includes the geometric shape of the subject, clarity of its edges, relative sizes, and the spatial relationships between various parts forming the subject. Ignore the background and any other elements that are not part of the main subject.
        2. Color: Compare the accuracy and consistency of the main colors of the subject in the generated image with those of the subject in the reference image. This includes the subject's saturation, hue, brightness, and the distribution of colors. Do not consider any colors that appear outside the subject itself.
        3. Texture: Focus on whether the generated image accurately captures the fine textures and surface details of the subject, without appearing blurry, and whether it meets the required level of realism and clarity. Excessive abstraction or stylization of the subject’s texture is unnecessary unless specifically indicated by the text prompt. The evaluation of texture should apply only to the subject, not to the background or unrelated areas.
        4. Facial Features: If the subject is a person or animal, closely examine whether the facial features of the subject in the generated image visually match those of the subject in the reference image. Pay particular attention to facial similarity, but do not evaluate faces or features unrelated to the primary subject.

        ### Scoring Range
        You need to give a specific integer score based on the comprehensive performance of the visual features above, ranging from 0 to 4:
        - Very Poor (0): No resemblance. The generated image's subject has no relation to the reference.
        - Poor (1): Minimal resemblance. The subject falls within the same broad category but differs significantly.
        - Fair (2): Moderate resemblance. The subject shows likeness to the reference with notable variances.
        - Good (3): Strong resemblance. The subject closely matches the reference with only minor discrepancies.
        - Excellent (4): Near-identical. The subject of the generated image is virtually indistinguishable from the reference.

        ### Input format
        Every time you will receive two images, the first image is a reference image, and the second image is the generated image.

        Please carefully review each image of the subject: {subject}. Before giving a score, please provide a brief analysis of the above evaluation criteria, which should be very concise and accurate.

        ### Output Format
        Analysis: [Your analysis]
        Score: [Your Score]
    """

    text_following = """
        ### Task Definition
        You will be provided with an image and text prompt.
        As an experienced evaluator, your task is to evaluate the semantic consistency between image and text prompt, according to the scoring criteria.

        ### Scoring Criteria
        When assessing the semantic consistency between an image and its accompanying text, it is crucial to consider how well the visual content of the image aligns with the textual description. This evaluation can be based on several key aspects:
        1. Relevance: Determine if the elements and subjects presented in the image directly relate to the core topics and concepts mentioned in the text. The image should reflect the main ideas or narratives described.
        2. Accuracy: Examine the image for the presence and correctness of specific details mentioned in the text. This includes the depiction of particular objects, settings, actions, or characteristics that the text describes.
        3. Completeness: Evaluate whether the image captures all the critical elements of the text. The image should not omit significant details that are necessary for the full understanding of the text's message.
        4. Context: Consider the context in which the text places the subject and whether the image accurately represents this setting. This includes the portrayal of the appropriate environment, interactions, and background elements that align with the text.

        ### Scoring Range
        Based on these criteria, a specific integer score from 0 to 4 can be assigned to determine the level of semantic consistency:
        - Very Poor (0): No correlation. The image does not reflect any of the key points or details of the text.
        - Poor (1): Weak correlation. The image addresses the text in a very general sense but misses most details and nuances.
        - Fair (2): Moderate correlation. The image represents the text to an extent but lacks several important details or contains some inaccuracies.
        - Good (3): Strong correlation. The image accurately depicts most of the information from the text with only minor omissions or inaccuracies.
        - Excellent (4): Near-perfect correlation. The image captures the text's content with high precision and detail, leaving out no significant information.

        ### Input format
        Every time you will receive a text prompt and an image.
        Text Prompt: {text}

        Please carefully review image and text prompt.

        ### Output Format
        Score: [Your Score]

        You must adhere to the specified output format, which means that only the scores need to be output, excluding your analysis process.
    """

    text_following_cot = """
        ### Task Definition
        You will be provided with an image and text prompt.
        As an experienced evaluator, your task is to evaluate the semantic consistency between image and text prompt, according to the scoring criteria.

        ### Scoring Criteria
        When assessing the semantic consistency between an image and its accompanying text, it is crucial to consider how well the visual content of the image aligns with the textual description. This evaluation can be based on several key aspects:
        1. Relevance: Determine if the elements and subjects presented in the image directly relate to the core topics and concepts mentioned in the text. The image should reflect the main ideas or narratives described.
        2. Accuracy: Examine the image for the presence and correctness of specific details mentioned in the text. This includes the depiction of particular objects, settings, actions, or characteristics that the text describes.
        3. Completeness: Evaluate whether the image captures all the critical elements of the text. The image should not omit significant details that are necessary for the full understanding of the text's message.
        4. Context: Consider the context in which the text places the subject and whether the image accurately represents this setting. This includes the portrayal of the appropriate environment, interactions, and background elements that align with the text.

        ### Scoring Range
        Based on these criteria, a specific integer score from 0 to 4 can be assigned to determine the level of semantic consistency:
        - Very Poor (0): No correlation. The image does not reflect any of the key points or details of the text.
        - Poor (1): Weak correlation. The image addresses the text in a very general sense but misses most details and nuances.
        - Fair (2): Moderate correlation. The image represents the text to an extent but lacks several important details or contains some inaccuracies.
        - Good (3): Strong correlation. The image accurately depicts most of the information from the text with only minor omissions or inaccuracies.
        - Excellent (4): Near-perfect correlation. The image captures the text's content with high precision and detail, leaving out no significant information.

        ### Input format
        Every time you will receive a text prompt and an image.
        Text Prompt: {text}

        Please carefully review image and text prompt. Before giving a score, please provide a brief analysis of the above evaluation criteria, which should be very concise and accurate.

        ### Output Format
        Analysis: [Your analysis]
        Score: [Your Score]
    """