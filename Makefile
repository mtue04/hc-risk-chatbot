.PHONY: build up down stop restart logs status psql reload seed load-tables load-features

COMPOSE ?= docker compose
POSTGRES_USER ?= hc_admin
POSTGRES_DB ?= homecredit_db
TABLES_SQL ?= /docker-entrypoint-initdb.d/02_load_tables.sql
FEATURES_SQL ?= /docker-entrypoint-initdb.d/03_load_features.sql

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d postgres

down:
	$(COMPOSE) down

stop:
	$(COMPOSE) stop

restart:
	$(COMPOSE) down
	$(COMPOSE) up -d postgres

logs:
	$(COMPOSE) logs -f postgres

status:
	$(COMPOSE) ps

psql:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

reload seed: load-tables load-features

load-tables:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $(TABLES_SQL)

load-features:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $(FEATURES_SQL)
