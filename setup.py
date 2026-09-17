from setuptools import setup, find_packages

setup(
    name="pathconfuse",
    version="1.0.0",
    description="Knowledge-Guided Multimodal Fusion for Cancer Prognostication under Missing Modalities and Cross-Modal Conflict",
    author="Wanjun Ma, Wenjun Li, Mengyun Yang, Xiwei Tang",
    author_email="wanjun@hnfnu.edu.cn, nudt_xiwei@126.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "lifelines>=0.27.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.6.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
