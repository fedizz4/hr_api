#!/bin/bash
pip install --upgrade pip
pip install wheel
pip install --only-binary :all: scikit-learn
pip install -r requirements.txt