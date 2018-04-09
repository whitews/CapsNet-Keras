from setuptools import setup

setup(
    name='CapsNet-Keras',
    version='0.1',
    license='MIT',
    packages=['capsnet_keras'],
    package_data={'': []},
    description='Implementation of a Capsule Neural Network using Keras',
    install_requires=[
        'numpy',
        'matplotlib',
        'Pillow',
        'tensorflow',
        'keras'
    ],
)
