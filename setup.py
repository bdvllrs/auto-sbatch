from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements


def get_version():
    with open("version.txt", "r") as f:
        version = f.read().lstrip("\n")
    return version


setup(name='auto_sbatch',
      version=get_version(),
      install_requires=get_requirements(),
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'auto-sbatch=auto_sbatch.sbatch:main',
              'register-run=auto_sbatch.register_run:register_run'
          ]
      })
