from setuptools import setup, Extension, find_packages

fastLA_module = Extension(
    'sparseklearn.fastLA._fastLA',
    sources=[
        'sparseklearn/fastLA/distances.c',
        'sparseklearn/fastLA/moments.c',
        'sparseklearn/fastLA/auxiliary.c'
    ]
)

setup(
    name='sparseklearn',
    version='0.1.4',
    url='http://github.com/EricKightley/sparseklearn',
    author='Eric Kightley',
    author_email='kightley.1@gmail.com',
    license='MIT',
    ext_modules = [fastLA_module],
    py_modules = ["fastLA"],
    packages = find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'scikit-learn',
        'pytest'
    ],
    zip_safe=False
)
