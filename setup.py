import setuptools

def find_version(path):
    for line in open(path):
        if line.startswith('__version__'):
            return line.split("=")[1].strip().replace("'", "").replace('"', '')
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="tarexp",
    version=find_version("./tarexp/__init__.py"), 
    author="Eugene Yang",
    author_email="eugene.yang@jhu.edu",
    description="A Python framework for Technology-Assisted Review experiments.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eugene-yang/tarexp",
    packages=setuptools.find_packages(include=['tarexp', 'tarexp.*']),
    install_requires=list(open('requirements.txt')),
    python_requires='>=3.7'
)