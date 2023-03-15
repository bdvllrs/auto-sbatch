from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements


def get_version():
    with open("version.txt", "r") as f:
        version = f.read().lstrip("\n")
    return version

extra_requires = {
    "dev": ["pytest", "mock"]
}

setup(name='auto_sbatch',
      version=get_version(),
      install_requires=get_requirements(),
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'register-run=auto_sbatch.register_run:register_run'
          ]
      },
      extras_require=extra_requires,
)
