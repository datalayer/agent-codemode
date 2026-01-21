SHELL=/bin/bash

.DEFAULT_GOAL := help

.PHONY: help agent agent-codemode clean-codemode

help:
	@echo "Targets:"
	@echo "  agent           Start the MCP agent CLI (stdio server)"
	@echo "  agent-codemode  Start the Agent Codemode agent CLI (stdio server)"
	@echo "  clean-codemode  Remove generated folder (two levels up)"

agent:
	python agent_cli.py

agent-codemode:
	python agent_cli.py --codemode

clean-codemode:
	rm -rf ../../generated
