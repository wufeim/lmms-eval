import base64
import io
import json
import os

import numpy as np
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.bench_3dsrbench.evals_3dsrbench import Evaluator_3DSRBench

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
GPT_SYS_PROMPT = "There are several options:"
GPT_EVAL_MODEL_NAME = "gpt-3.5-turbo-0125"
evaluator_3dsrbench = Evaluator_3DSRBench(sys_prompt=GPT_SYS_PROMPT, API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)

LABELS = ['A', 'B', 'C', 'D']
mapping = {
    'location': ['location_above', 'location_closer_to_camera', 'location_next_to'],
    'height': ['height_higher'],
    'orientation': ['orientation_in_front_of', 'orientation_on_the_left', 'orientation_viewpoint'],
    'multi_object': ['multi_object_closer_to', 'multi_object_facing', 'multi_object_viewpoint_towards_object', 'multi_object_parallel', 'multi_object_same_direction']}
types = ['height', 'location', 'orientation', 'multi_object']
subtypes = sum([mapping[k] for k in types], [])


def doc_to_visual_3dsrbench(doc):
    base64_string = doc['image']
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    return [image]


def doc_to_text_3dsrbench(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt} {doc['question']} {post_prompt}"


def process_results_3dsrbench(doc, results):
    model_response = results[0].strip()
    category = doc['category']
    if 'multi_object' in category:
        cate = 'multi_object'
    else:
        cate = category.split('_')[0]
    subcate = category.replace(cate+'_', '')
    data = {
        "3dsrbench_score": {
            "index": doc["index"],
            "qid": doc["qid"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "hint": '',
            "source": '3dsrbench',
            "split": 'test',
            "category": cate,
            "L2-category": subcate,
        }
    }
    option_candidate = ["A", "B", "C", "D"]
    for c in option_candidate:
        data["3dsrbench_score"][c] = doc.get(c, "nan")
    return data

def df_to_scores(df):
    df = df[['qid', 'category', 'l2-category', 'hit']]
    results = {}
    for i in range(len(df.index)):
        qid, cate, l2cate, hit = df.iloc[i].tolist()

        assert hit in [0, 1], df.iloc[i].tolist()

        if qid[-2] == '-':
            qid = qid[:-2]

        if qid in results:
            results[qid][0] = results[qid][0] * hit
        else:
            results[qid] = [hit, cate+'_'+l2cate]

    overall_acc = np.mean([results[k][0] for k in results])

    category_acc = {}
    for t in types:
        category_acc[t] = np.mean([results[k][0] for k in results if results[k][1] in mapping[t]])

    l2_category_acc = {}
    for t in subtypes:
        l2_category_acc[t] = np.mean([results[k][0] for k in results if results[k][1] == t])

    return overall_acc, category_acc, l2_category_acc


def aggregate_results_3dsrbench(results, args):
    results_df = evaluator_3dsrbench.eval_result(results, eval_method='openai')
    overall_acc, category_acc, l2_category_acc = df_to_scores(results_df)
    file = generate_submission_file("3dsrbench_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return {'3dsrbench_score': overall_acc * 100}
