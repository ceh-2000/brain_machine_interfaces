# Assignment 1

I investigate for a single dataset whether autonomous dynamics arise in the activity patterns of a monkey performing a delayed center-out reach task.

## Getting started
Prerequisites: Python 3.12+, pip3
1. Create a new virtual environment with `python3 -m venv venv` and activate it with `source venv/bin/activate`
2. Install requirements with `pip3 install -r requirements.txt`
3. Decide what distortion you would like. Set `should_distort = 0` for no distortion, `should_distort = 1` for inversion distortion, and `should_distort = 2` for time distortion. 
4. Run `python3 main.py`. Generated plots are saved in `outputs/`.