#!/usr/bin/env bash

NAME=$1

if [ -z "$NAME" ]; then
  echo "Usage: new_project <project_name>"
  exit 1
fi

cp -r ../_template ../experiments/$NAME
echo "Created project: experiments/$NAME"
