from setuptools import setup, find_packages

setup(
    name="shapealign",
    version="0.1.0",
    description="A package for preprocessing and embedding shape contours using FFT and PCA",
    author="Lab of Systems Biology and Data Analytics, GIS, A*STAR, Singapore",
    author_email="Jagadish_Sankaran@gis.a-star.edu.sg",
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
