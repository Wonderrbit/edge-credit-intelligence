variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "akropoli-credit"
}

variable "image_uri" {
  type        = string
  description = "URI da imagem no ECR, ex.: 123456789012.dkr.ecr.us-east-1.amazonaws.com/akropoli-credit:latest"
}

variable "task_cpu" {
  type    = string
  default = "2048"
}

variable "task_memory" {
  type    = string
  default = "8192"
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "allowed_cidr" {
  type        = string
  description = "CIDR autorizado a acessar a API na porta 8000."
  default     = "0.0.0.0/0"
}

variable "schedule_expression" {
  type        = string
  description = "Agendamento EventBridge do autopilot (UTC)."
  default     = "cron(0 9 * * ? *)"
}
