from setuptools import setup, find_packages

setup(
    name="act",
    version="0.1.0",
    packages=find_packages(exclude=[]),
    author="HIROL",
    author_email="kjustdoitno1@gmail.com",  # 
    description="hirol act module",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/linkage/act",  # Replace with your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)

