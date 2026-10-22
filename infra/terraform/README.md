# Terraform — AWS

Provisiona:

- ECR para a imagem da aplicação.
- ECS Cluster + Service Fargate para a API.
- Task definition do autopilot + regra EventBridge diária.
- S3 versionado e criptografado para artefatos.
- Secrets Manager para OpenAI e TypeSafe.
- CloudWatch Logs para API e autopilot.
- IAM roles de execução, task e EventBridge.

## Uso

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edite image_uri e allowed_cidr
terraform init
terraform apply
```

Depois:

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t <ecr_repository_url>:latest .
docker push <ecr_repository_url>:latest

aws secretsmanager put-secret-value \
  --secret-id akropoli-credit/openai \
  --secret-string '{"OPENAI_API_KEY":"..."}'
aws secretsmanager put-secret-value \
  --secret-id akropoli-credit/typesafe \
  --secret-string '{"TYPESAFE_API_KEY":"..."}'

aws ecs update-service --cluster akropoli-credit \
  --service akropoli-credit-api --force-new-deployment
```

O task definition referencia as chaves JSON (`:OPENAI_API_KEY::` e `:TYPESAFE_API_KEY::`), então o segredo deve ser um objeto JSON com essas chaves.

## Notas

- O exemplo usa a VPC default e IP público para simplificar. Em produção, use subnets privadas, ALB, WAF e VPC endpoints.
- A API não tem auto scaling no exemplo; adicione Application Auto Scaling por CPU/requisição.
- **Artefatos**: a API lê `artifacts/` local; o autopilot grava em S3 quando `MLOPS_ARTIFACTS_BUCKET` está definido. Como as tasks Fargate são efêmeras, para a API enxergar o último run em produção use EFS montado em ambas as tasks ou implemente leitura read-through do S3. O exemplo não provisiona EFS.
- O autopilot grava em S3 quando `MLOPS_ARTIFACTS_BUCKET` está definido (o Terraform já injeta `s3://<bucket>`).
- O estado do Terraform deve ir para backend remoto com lock (S3 + DynamoDB); não commite `terraform.tfstate`.
- Ajuste `allowed_cidr`; o default `0.0.0.0/0` é apenas para laboratório.
