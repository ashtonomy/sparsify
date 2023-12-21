import setuptools

with open("README.md", "r", encoding="utf-8") as r:
    long_description = r.read()

setuptools.setup(
    name="sparsify",
    version="0.0.1",
    author="ashtonomy",
    author_email="tashtonwilliamson@gmail.com",
    description="Element-wise sparsification for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashtonomy/sparsify",
    classfiers= [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",  
    ],
    package_dir={"":"sparsify"}
    packages=setuptools.find_packages(where="sparsify"),
    python_requires=">=3.6"
)