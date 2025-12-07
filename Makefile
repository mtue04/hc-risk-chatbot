.PHONY: build up down stop restart logs status psql reload seed load-tables load-features test-api test-chat

COMPOSE ?= docker compose
POSTGRES_USER ?= hc_admin
POSTGRES_DB ?= homecredit_db
TABLES_SQL ?= /docker-entrypoint-initdb.d/02_load_tables.sql
FEATURES_SQL ?= /docker-entrypoint-initdb.d/03_load_features.sql

# Build all services
build:
	$(COMPOSE) build

# Start all services
up:
	$(COMPOSE) up -d

# Start only database
up-db:
	$(COMPOSE) up -d postgres redis

# Stop all services
down:
	$(COMPOSE) down

# Stop without removing containers
stop:
	$(COMPOSE) stop

# Restart all services
restart:
	$(COMPOSE) down
	$(COMPOSE) up -d

# View logs
logs:
	$(COMPOSE) logs -f

logs-chatbot:
	$(COMPOSE) logs -f chatbot

logs-model:
	$(COMPOSE) logs -f model_serving

# Check status
status:
	$(COMPOSE) ps

# Database shell
psql:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

# Reload data
reload seed: load-tables load-features

load-tables:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $(TABLES_SQL)

load-features:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $(FEATURES_SQL)

# Test endpoints
test-api:
	@echo "Testing Model API..."
	curl -s http://localhost:8001/health | python -m json.tool
	@echo "\nTesting prediction..."
	curl -s http://localhost:8001/predict/applicant/100001 | python -m json.tool

test-chat:
	@echo "Testing Chatbot API..."
	curl -s http://localhost:8500/health | python -m json.tool

test-hypothetical:
	@echo "Testing hypothetical prediction..."
	curl -s -X POST http://localhost:8001/predict/hypothetical \
		-H "Content-Type: application/json" \
		-d '{"features": {"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 500000}}' | python -m json.tool

# Clean everything
clean:
	$(COMPOSE) down -v
	docker system prune -f
