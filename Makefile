.PHONY: build up down stop restart logs status psql reload seed

COMPOSE ?= docker compose
POSTGRES_USER ?= hc_admin
POSTGRES_DB ?= homecredit_db
INIT_LOAD_SQL ?= /docker-entrypoint-initdb.d/02_load_tables.sql

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

reload seed:
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f $(INIT_LOAD_SQL)
