import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mana",
    version="0.0.1",
    author="IISY at Beuth",
    author_email="iisy@beuth-hochschule.de",
    description="A Motion ANAlysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.beuth-hochschule.de/iisy/mana",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    install_requires=[
        'numpy', 'scikit-learn', 'plotly>=4.9', 'chart-studio>=1', 'kaleido',
        'transforms3d', 'matplotlib', 'opencv-python'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
