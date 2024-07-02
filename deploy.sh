docker build -t memory77/app_food:latest .

docker push memory77/app_food:latest

az container create --resource-group RG_SOCCIOD --file deploy.yaml
