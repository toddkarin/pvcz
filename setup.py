import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pvcz",
    version="0.0.3",
    author="toddkarin",
    author_email="pvtools.lbl@gmail.com",
    description="Photovoltaic climate zones and degradation stressors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toddkarin/pvcz",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy','pandas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)