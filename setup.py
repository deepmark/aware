import re
from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).parent
version = re.search(r'^__version__ = "(.+)"$', (here / "src/aware/__init__.py").read_text(), re.M).group(1)

setup(
    name="aware",
    version=version,
    description="AWARE: audio watermarking with adversarial resistance to edits",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="DeepMark Inc.",
    license="MIT",
    license_files=["LICENSE"],
    url="https://github.com/deepmark/aware",
    project_urls={
        "Paper": "https://arxiv.org/abs/2510.17512",
        "Issues": "https://github.com/deepmark/aware/issues",
    },
    keywords=["audio", "watermarking", "speech", "adversarial"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"aware": ["cards/*.yaml"]},
    python_requires=">=3.9,<3.13",
    install_requires=[
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "numpy==1.26.4",
        "librosa==0.9.2",
        "setuptools<82",  # librosa 0.9.2 imports pkg_resources, which setuptools 82 removed
        "soundfile==0.12.1",
        "pydantic==2.5.0",
        "matplotlib==3.7.2",
        "scikit-learn==1.5.0",
        "numba==0.59.0",
        "resampy==0.4.2",
        "tqdm==4.66.1",
        "pesq==0.0.4",
        "pyyaml==6.0.1",
        "pystoi==0.4.1",
        "webrtcvad==2.0.10"
    ]
)
