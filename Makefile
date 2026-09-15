# Makefile
#
# Common development shortcuts for this project. These targets wrap the
# underlying tools (`uv`, `prek`, `pytest`, `docker compose`) so contributors
# can use one consistent command surface.
#
# Frontend (Streamlit) đã được chuyển sang repository riêng ở v7.

# Use bash for recipe execution so shell behavior is consistent across targets.
SHELL = /bin/bash

# Mark all command names as phony so make does not confuse them with files of
# the same name in the repository.
.PHONY: help style restart clean docker-clean start_backend start_vectordb setup start_docker down test test-vcr update-vcr-tests test-e2e dashboard-configmap dashboard-apply metrics-scrape-apply

# Print a quick reference for the most common developer commands.
help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and git hooks"
	@echo "  make style          - Run code formatting/linting"
	@echo "  make test           - Run fast tests (unit + integration) with coverage"
	@echo "  make test-vcr       - Run VCR-marked tests"
	@echo "  make update-vcr-tests - Rewrite VCR recordings"
	@echo "  make test-e2e       - Run end-to-end tests"
	@echo "  make dashboard-configmap - Generate dashboard ConfigMap for Grafana sidecar"
	@echo "  make dashboard-apply     - Apply dashboard ConfigMap + ServiceMonitor to cluster"
	@echo "  make metrics-scrape-apply - Scrape metrics from Docker-hosted service (chirp3 pattern)"
	@echo "  make start_backend  - Start the FastAPI backend"
	@echo "  make start_docker   - Start all docker containers"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make docker-clean   - Docker system prune (dọn image/volume khi build fail)"

# Install backend dependencies and git hooks.
setup:
	uv sync
	prek install

# Run the repository's configured formatting, linting, and static checks.
style:
	prek run --all-files

# Run the default test suite while excluding slower or externally-dependent
# tests. Coverage output is written both to the terminal and htmlcov/.
test:
	uv run pytest -n auto -m "not vcr and not e2e" -v --cov=src/agent --cov-report=term --cov-report=html tests/

# Run tests that rely on checked-in VCR cassettes instead of live services.
test-vcr:
	uv run pytest -m "vcr" -v tests/

# Refresh VCR cassette recordings. Use this when expected API interactions have
# changed and the checked-in recordings need to be regenerated.
update-vcr-tests:
	uv run pytest --record-mode=rewrite -m "vcr" -v tests/

# Run live end-to-end tests. These may require credentials, network access, or
# other local setup not needed by the default test target.
test-e2e:
	RUN_LIVE_E2E=1 uv run pytest -m "e2e" -v tests/

# Start the FastAPI backend in reload mode for local development.
start_backend:
	uv run uvicorn agent.api:app --reload --port 8001

# Start only the vector database service needed by local backend workflows.
start_vectordb:
	docker compose -f ../qdrant_docker/docker-compose.yml up --build -d

# Start all services defined in docker-compose.yml in detached mode.
start_docker:
	docker compose up --build -d

# Rebuild and restart all Docker services from a clean compose state.
restart:
	docker compose down --remove-orphans
	docker compose up --build -d

# Stop Docker services and remove compose-managed orphan containers.
down:
	docker compose down --remove-orphans

# Dọn toàn bộ Docker image/volume/network khi build fail hoặc muốn khởi động lại sạch.
# Cảnh báo: xoá cả image/volume của project khác — chỉ chạy khi chắc chắn.
docker-clean:
	docker compose down --remove-orphans -v
	docker system prune -a --volumes -f

# Remove local caches, Python bytecode, coverage output, and macOS metadata.
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.DS_Store" -delete
	rm -rf .coverage* htmlcov/

## Sinh ConfigMap chứa dashboard JSON từ monitoring/dashboards/, gắn label để Grafana sidecar tự nhận
## (namespace "monitoring" là giả định — chỉnh theo cluster của tổ chức nếu khác)
dashboard-configmap:
	kubectl create configmap rag-retrieval-dashboard \
		--from-file=monitoring/dashboards/rag-retrieval-dashboard.json \
		-n monitoring --dry-run=client -o yaml \
		| kubectl label --local -f - -o yaml grafana_dashboard=1 \
		> monitoring/helm/dashboard-configmap.generated.yaml

## Apply ConfigMap + ServiceMonitor lên cluster (Grafana trung tâm sẽ tự pick up dashboard)
## CHÚ Ý: xác nhận namespace/label với đội hạ tầng trước (xem monitoring/README.md)
dashboard-apply: dashboard-configmap
	kubectl apply -f monitoring/helm/dashboard-configmap.generated.yaml
	kubectl apply -f monitoring/helm/servicemonitor.yaml -n monitoring

## Scrape metrics từ service chạy Docker trên host (pattern chirp3):
## Service không selector + Endpoints trỏ IP host + ServiceMonitor.
## CHÚ Ý: điền __NODE_IP__ trong monitoring/k8s/rag-retrieval-metrics-scrape.yaml trước
## (IP INTERNAL của server chạy Docker: kubectl get nodes -o wide).
## Dùng khi service CHƯA deploy lên k8s; khi đã deploy trong k8s thì bỏ target này.
metrics-scrape-apply:
	kubectl apply -f monitoring/k8s/rag-retrieval-metrics-scrape.yaml
