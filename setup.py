from setuptools import setup, find_packages


setup(
      name='my_custom_sklearn_transforms',
      version='1.0',
      description='''
            This is a sample python package for encapsulating custom
            tranforms from scikit-learn into Watson Machine Learning
      ''',
      url='https://github.com/vnderlev/sklearn_transforms/',
      author='Vanderlei Munhoz',
      author_email='vnderlev@protonmail.ch',
      license='BSD',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
            'catboost'
      ]
)
