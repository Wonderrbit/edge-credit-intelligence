terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_caller_identity" "current" {}

locals {
  name         = var.project
  api_logs     = "/ecs/${local.name}/api"
  auto_logs    = "/ecs/${local.name}/autopilot"
  api_image    = var.image_uri
  auto_image   = var.image_uri
  artifact_uri = "s3://${aws_s3_bucket.artifacts.bucket}"
}

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${local.name}-artifacts-"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_secretsmanager_secret" "openai" {
  name                    = "${local.name}/openai"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "typesafe" {
  name                    = "${local.name}/typesafe"
  recovery_window_in_days = 7
}

resource "aws_cloudwatch_log_group" "api" {
  name              = local.api_logs
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "autopilot" {
  name              = local.auto_logs
  retention_in_days = 30
}

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn, aws_secretsmanager_secret.typesafe.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "${local.name}-execution-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task_s3" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "task_s3" {
  name   = "${local.name}-task-s3"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_s3.json
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "API do credit intelligence"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "api"
    image     = local.api_image
    essential = true
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    environment = [
      { name = "MLOPS_ENV", value = "production" },
      { name = "MLOPS_AUTONOMY", value = "full" },
      { name = "MLOPS_USE_JEV", value = "1" },
      { name = "MLOPS_USE_LLM", value = "1" },
      { name = "MLOPS_ARTIFACTS_BUCKET", value = local.artifact_uri }
    ]
    secrets = [
      { name = "OPENAI_API_KEY", valueFrom = "${aws_secretsmanager_secret.openai.arn}:OPENAI_API_KEY::" },
      { name = "TYPESAFE_API_KEY", valueFrom = "${aws_secretsmanager_secret.typesafe.arn}:TYPESAFE_API_KEY::" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = local.api_logs
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }
  }])
  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy.execution_secrets]
}

resource "aws_ecs_task_definition" "autopilot" {
  family                   = "${local.name}-autopilot"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "autopilot"
    image     = local.auto_image
    essential = true
    command   = ["python", "scripts/run_autopilot.py", "--mode", "full"]
    environment = [
      { name = "MLOPS_ENV", value = "production" },
      { name = "MLOPS_AUTONOMY", value = "full" },
      { name = "MLOPS_USE_JEV", value = "1" },
      { name = "MLOPS_USE_LLM", value = "1" },
      { name = "MLOPS_ARTIFACTS_BUCKET", value = local.artifact_uri }
    ]
    secrets = [
      { name = "OPENAI_API_KEY", valueFrom = "${aws_secretsmanager_secret.openai.arn}:OPENAI_API_KEY::" },
      { name = "TYPESAFE_API_KEY", valueFrom = "${aws_secretsmanager_secret.typesafe.arn}:TYPESAFE_API_KEY::" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = local.auto_logs
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "autopilot"
      }
    }
  }])
  depends_on = [aws_cloudwatch_log_group.autopilot, aws_iam_role_policy.execution_secrets]
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "events" {
  name               = "${local.name}-events"
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

data "aws_iam_policy_document" "events_run_task" {
  statement {
    actions   = ["ecs:RunTask"]
    resources = [aws_ecs_task_definition.autopilot.arn]
  }
  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
  }
}

resource "aws_iam_role_policy" "events_run_task" {
  name   = "${local.name}-events-run-task"
  role   = aws_iam_role.events.id
  policy = data.aws_iam_policy_document.events_run_task.json
}

resource "aws_cloudwatch_event_rule" "schedule" {
  name                = "${local.name}-daily"
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "autopilot" {
  rule     = aws_cloudwatch_event_rule.schedule.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.events.arn

  ecs_target {
    task_count          = 1
    task_definition_arn = aws_ecs_task_definition.autopilot.arn
    launch_type         = "FARGATE"

    network_configuration {
      subnets          = data.aws_subnets.default.ids
      security_groups  = [aws_security_group.service.id]
      assign_public_ip = true
    }
  }
  depends_on = [aws_iam_role_policy.events_run_task]
}

