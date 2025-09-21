.PHONY: up down logs api train promote rebuild

up:
\tdocker compose up -d mlflow-ui api

down:
\tdocker compose down

logs:
\tdocker compose logs -f --tail=200

api:
\topen http://127.0.0.1:$(PORT)/docs

train:
\tdocker compose run --rm trainer

promote:
\tdocker compose run --rm promote

rebuild:
\tdocker compose build --no-cache
