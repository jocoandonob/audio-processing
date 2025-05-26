from setuptools import setup, find_packages

setup(
    name="sagemaker-audio-analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "numpy",
        "librosa",
        "soundfile",
        "scipy",
    ],
)