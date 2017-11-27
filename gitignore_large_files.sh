#!/bin/sh
find . -size +1M | cat >> .gitignore
