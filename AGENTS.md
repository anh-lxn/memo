# AGENTS.md

## Project
MeMo membrane sensing project.

## Goal
Maintain and extend the Python codebase for:
- sensor acquisition from Raspberry Pi / ADS1115
- labeled data recording
- GUI / demonstrator
- ML training for position (x,y) and force (F)

## Current structure
- src/: Python source code
- data/: datasets
- models/: trained models
- doc/: documentation
- figs/: figures

## Working rules
- Prefer small, reviewable changes.
- Do not break existing GUI behavior.
- Keep hardware access and mock/simulated access separated.
- When changing code, explain which files were touched and why.
- Prefer adding mock data sources so development works without hardware.