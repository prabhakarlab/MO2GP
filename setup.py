from setuptools import setup, find_packages

setup(
    name="mo2gp",
    version="0.1.0",
    description="Morphology O(2)-invariant General-purpose Projection (MO2GP) algorithm.",
    author="Lab of Systems Biology and Data Analytics, GIS, A*STAR, Singapore",
    author_email="Ignasius_Joanito_Irwan@a-star.edu.sg",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mo2gp": ["data/*", "data/**/*"],
    },
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "scikit-image",
        "tqdm",
        "joblib",
        "pandas",
    ],
    python_requires=">=3.7",
)
