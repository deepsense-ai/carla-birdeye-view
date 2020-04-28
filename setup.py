from pathlib import Path
from setuptools import setup

REPO_ROOT = Path(__file__).parent
README = (REPO_ROOT / "README.md").read_text()

setup(
    name="carla-bridview",
    version="1.0.0",
    description=TODO,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/realpython/reader",
    author="Real Python",
    author_email="office@realpython.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "birdview=birdview.__main__:main",
        ]
    },
)