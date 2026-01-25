SHELL=/bin/bash

.DEFAULT_GOAL := help

.PHONY: help agent agent-standard clean-codemode

help:
	@echo "Targets:"
	@echo "  agent           Start the Skills Agent CLI (codemode, default)"
	@echo "  agent-standard  Start the Agent CLI in standard MCP mode"
	@echo "  clean-codemode  Remove generated folder (two levels up)"

agent:
	python agent_cli.py

agent-standard:
	python agent_cli.py --standard

clean-codemode:
	rm -rf ../../generated
