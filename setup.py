from setuptools import setup

setup(
    name="minitorch",
    version="0.1",
    packages=[
        "minitorch"
    ],
    package_data={"minitorch": []},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
