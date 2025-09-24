from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="indian-language-nlp",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Language Modeling for Underrepresented Indian Languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/indian-language-nlp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "gpu": [
            "torch-audio>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "indian-nlp-train=src.scripts.train:main",
            "indian-nlp-evaluate=src.scripts.evaluate:main",
            "indian-nlp-collect=src.scripts.collect_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["configs/*.yaml"],
    },
)
