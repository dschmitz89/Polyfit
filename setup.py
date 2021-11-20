from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == '__main__':
    
    setup(
        name='polyfit',   
        version='1.0', 
        description='Constrained polynomial regression',
        url='https://github.com/dschmitz89/polyfit/',
        author='Daniel Schmitz',
        license='MIT',
        packages=['polyfit'],
        install_requires=[
            'numpy',
            'cvxpy',
            'scikit-learn'
        ],
        long_description=long_description,
        long_description_content_type='text/markdown',
        include_package_data=True,
        package_data={'': ['Example_Data.npz']}             
    )