from pathlib import Path
from setuptools import setup

REPO_ROOT = Path(__file__).parent
README = (REPO_ROOT / "README.md").read_text()
with open(REPO_ROOT / "requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="carla_birdeye_view",
    version="1.0",
    description="Bird-eye's view for CARLA simulator",
    keywords=["CARLA", "birdview", "bird-eye's view", "Reinforcement Learning", "RL"],
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/deepsense-ai/carla_birdeye_view",
    author="Michał Martyniak",
    author_email="michal.martyniak@linux.pl",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    install_requires=REQUIREMENTS,
    entry_points={"console_scripts": ["birdview-demo=birdview.__main__:main"]},
)
