from setuptools import find_packages, setup

setup(
    name="poisson-deconvolution",
    version="1.0.0",
    description="Recovering a discrete signal, modeled as a k-atomic uniform distribution, from a binned Poisson convolution model",
    url="https://github.com/dlitskevich/poisson-deconvolution",
    author="Danila Litskevich",
    author_email="danila.litskevich@gmail.com",
    license="BSD",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.13.0",
        "matplotlib==3.8.4",
    ],
)
