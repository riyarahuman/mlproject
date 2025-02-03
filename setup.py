from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Reads the requirements.txt file and returns a list of dependencies."""
    requirements = []
    try:
        with open(file_path, 'r') as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]
            if hyphen_e_dot in requirements:
                requirements.remove(hyphen_e_dot)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='riya',
    author_email='riyarahuman212@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
