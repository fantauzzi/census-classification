#!/bin/bash
cd src || exit
uvicorn server:app --host=127.0.0.1
