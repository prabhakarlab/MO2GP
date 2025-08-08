from setuptools import setup, find_packages

setup(
    name="shapealign",
    version="0.1.0",
    description="A package for preprocessing and embedding shape contours using FFT and PCA",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "scikit-image",
        "tqdm",
        "joblib",
    ],
    python_requires=">=3.7",
)
