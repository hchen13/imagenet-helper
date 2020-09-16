import setuptools

setuptools.setup(
    name='imagenet-helper',
    version='1.0.0',
    author='Ethan Chen',
    packages=setuptools.find_packages(),
    package_data={
        "": ['metadata/*.txt', 'metadata/*.json']
    }
)