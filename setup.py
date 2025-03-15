from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
            
    return requirements
            


setup(
    name="LA Crime Future Forecast",
    version="0.1",
    author="Arav Pandey",
    author_email="aravpandey3010@gmail.com",
    description="A project to forecast future crime in Los Angeles",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    
)
