# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY agent_codemode/* agent_codemode/

RUN pip install -e .

RUN pip uninstall -y pycrdt datalayer_pycrdt
RUN pip install datalayer_pycrdt==0.12.17

CMD ["python", "-m", "agent_codemode.server"]
