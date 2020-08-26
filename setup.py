import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stroopwafel",
    version="0.0.1",
    author="Lokesh Khandelwal",
    author_email="lokesh.khandelwal92@gmail.com",
    description="A python package for STROOPWAFEL and GenAIS algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lokiysh/stroopwafel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy'
    ]
)