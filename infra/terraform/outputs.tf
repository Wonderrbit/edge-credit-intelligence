output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "api_log_group" {
  value = aws_cloudwatch_log_group.api.name
}

output "autopilot_log_group" {
  value = aws_cloudwatch_log_group.autopilot.name
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}

output "openai_secret_arn" {
  value = aws_secretsmanager_secret.openai.arn
}

output "typesafe_secret_arn" {
  value = aws_secretsmanager_secret.typesafe.arn
}
