''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schema import INPUT_SCHEMA


MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # Download input objects
    job_input['init_image'], job_input['mask'] = rp_download.download_files_from_urls(
        job['id'],
        [job_input.get('init_image', None), job_input.get('mask', None)]
    )  # pylint: disable=unbalanced-tuple-unpacking

    if validated_input['seed'] is None:
        validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

    outputs = MODEL.predict(
        prompt=validated_input["prompt"],
        width=validated_input['width'],
        height=validated_input['height'],
        num_outputs=validated_input['num_outputs'],
        num_inference_steps=validated_input['num_inference_steps'],
        guidance_scale=validated_input['guidance_scale'],
        seed=validated_input['seed']
    )

    job_output = []

    for index, output in enumerate(outputs):
        job_output.append({
            "image": output,
            "seed": validated_input['seed'] + index
        })

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output


runpod.serverless.start({"handler": run})
