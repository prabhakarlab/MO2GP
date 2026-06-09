from setuptools import setup, find_packages

setup(
    name="mo2gp",
    version="0.1.0",
    description="Morphology O(2)-invariant General-purpose Projection (MO2GP), a highly discriminative, efficient and interpretable algorithm guaranteeing rotation- and reflection-invariant shape representation. MO2GP uses Fourier descriptors of complex-plane representations of contour position and regularity, with dimensionality reduction for noise reduction and scalability.",
    author="Lab of Systems Biology and Data Analytics, GIS, A*STAR, Singapore",
    author_email="Ignasius_Joanito_Irwan@a-star.edu.sg",
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
