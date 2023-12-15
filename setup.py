from setuptools import setup

setup(
    name='fittingtools',
    version='0.1.3',
    description='Basic tools to generate confidence intervals and prediction confidence'
                'intervals based on scipy.optimize.least_squares',
    url='https://github.com/erickmartinez/fittingtools',
    author='Erick Martinez Loran',
    author_email='erickrmartinez@gmail.com',
    license='MIT License',
    packages=['fittingtools'],
    install_requires=[
        'numpy', 'scipy'

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ]
)