from setuptools import find_packages, setup


setup(
    name="scu-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "server": ["fastapi>=0.100.0", "uvicorn", "torch", "transformers", "peft"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": ["scu=scu_api.cli.main:cli"],
    },
)
