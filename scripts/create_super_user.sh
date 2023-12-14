#!/bin/bash
cd c:\
cd Users\hezhenfan\AppData\Local\Programs\Python\.virtualenvs\dyborg\Scripts
.\activate
cd d:\
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
python create_super_user.py
deactivate