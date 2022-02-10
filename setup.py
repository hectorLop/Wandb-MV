from setuptools import setup

setup(
    name='wandb-mv',
    version='0.2.6',    
    description='Model Versioning using Weight & Biases',
    url='https://github.com/hectorLop/Wandb-MV',
    author='Hector Lopez',
    author_email='lopez.almazan.hector@gmail.com',
    license='MIT',
    py_modules=['wandb_mv'],
    packages=['wandb_mv'],
    install_requires=['wandb']
)
