from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="apias",
    version="0.1.16",
    author="Emanuele Sabetta",
    author_email="713559+Emasoft@users.noreply.github.com",
    description="AI powered API documentation scraper and converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Emasoft/apias",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "requests",
        "beautifulsoup4",
        "lxml",
        "tenacity",
        "playwright",
    ],
    entry_points={
        "console_scripts": [
            "apias=apias.apias:main",
        ],
    },
)
