from setuptools import setup, find_packages

setup(
    name="NTkpp",
    version="0.0.2.4",
    author="Тихонов Иван",
    author_email="tihonovivan737@gmail.com",
    description="Простая библиотека для полносвязных нейронных сетей",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
)