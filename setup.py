from setuptools import setup, find_packages

requirements = [
    "logbook",
    "Flask==2.2.2",
    "virtualenv"
]

setup(
    name="ROAR",
    author="Janik Luechinger",
    author_email="janik.luechinger@uzh.ch",
    description="Master Thesis on Impact Optimization of Ransomware",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    # entry_points={
    #    "console_scripts": ["client=client.__main__:client"]  # copied from other repository, needs adjustments
    # }
)
