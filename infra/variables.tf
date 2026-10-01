variable "subscription_id" {
  description = "The Azure subscription to deploy into."
  type        = string
}

variable "alert_email" {
  description = "Where the budget alerts are sent."
  type        = string
}

variable "budget_usd" {
  description = "The monthly budget for the resource group. Alerts fire at 50, 80 and 100 percent of it, and at 100 percent forecast."
  type        = number
  default     = 5
}

variable "budget_start" {
  description = "The first day of the month the budget starts in (ISO 8601, midnight UTC). It cannot be in a past month."
  type        = string
  default     = "2026-10-01T00:00:00Z"
}

variable "location" {
  description = "The Azure region."
  type        = string
  default     = "westeurope"
}

variable "name" {
  description = "A short name for the resources."
  type        = string
  default     = "analyst-demo"
}

variable "image" {
  description = "The service image. A public GHCR image needs no credentials."
  type        = string
  default     = "ghcr.io/adaboranyilmaz/ai-data-analyst/api:main"
}

variable "registry_server" {
  description = "A private registry's server (an Azure Container Registry), if the image is not public."
  type        = string
  default     = null
}

variable "registry_username" {
  description = "The private registry's user name."
  type        = string
  default     = null
}

variable "registry_password" {
  description = "The private registry's password."
  type        = string
  default     = null
  sensitive   = true
}

variable "max_replicas" {
  description = "The most replicas the app scales out to. Replay mode holds no state, so more replicas only add capacity."
  type        = number
  default     = 2
}
