from setuptools import setup, find_packages

setup(
    name='satark',
    version='0.2.0',
    description='Headgear-Aware Crowd Counting (Simhastha / Kumbh Mela)',
    packages=find_packages(exclude=['scripts*', 'app*', 'tests*', 'configs*']),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0', 'torchvision>=0.15', 'numpy>=1.24',
        'scipy>=1.10', 'opencv-python>=4.7', 'matplotlib>=3.7',
        'Pillow>=10.0', 'Flask>=3.0', 'werkzeug>=3.0', 'PyYAML>=6.0',
    ],
)
