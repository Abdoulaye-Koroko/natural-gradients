from setuptools import find_packages, setup

# Installation
config = {
    'name': 'Natural_gradient',
    'version': '0.1.0',
    'description': 'Natural gradient-based algorithm.',
    'author': 'Abdoulaye Koroko',
    'author_email': 'abdoulayekoroko@gmail.com',
    'packages': find_packages(),
    'zip_safe': True
}

setup(**config)