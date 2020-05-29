from pathlib import Path
from setuptools import setup, find_packages

README = Path("README.md").read_text()
with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="carla-birdeye-view",
    version="1.1.1",
    description="Bird-eye's view for CARLA simulator",
    keywords=["CARLA", "birdview", "bird-eye's view", "Reinforcement Learning", "RL"],
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/deepsense-ai/carla-birdeye-view",
    author="Michał Martyniak",
    author_email="michal.martyniak@linux.pl",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    include_package_data=True,
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    entry_points={"console_scripts": ["birdview-demo = carla_birdeye_view.__main__:main"]},
)
