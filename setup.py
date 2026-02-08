from setuptools import setup, find_packages

setup(
    name='mosaic_multigrid',
    version='1.5.0',
    description='Research-grade multi-agent gridworld environments for reproducible RL experiments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Abdulhamid Mousa',
    author_email='abdulhamid97mousa@gmail.com',
    url='https://github.com/Abdulhamid97Mousa/mosaic_multigrid',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=1.0.0',
        'numpy>=1.24.0',
        'aenum>=3.1.0',
        'numba>=0.58.0',
        'pygame>=2.5.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
