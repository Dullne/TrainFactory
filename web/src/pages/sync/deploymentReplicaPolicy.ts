interface DeploymentReplicaLike {
  replica_id: string
  status: string
  health_status: string
}

interface DeploymentLike {
  deployment_id: string
  inference_framework: string
  status: string
  replica_instances?: DeploymentReplicaLike[]
}

export function getHealthyDeploymentReplicas<T extends DeploymentReplicaLike>(
  deployment: { replica_instances?: T[] } | undefined,
): T[] {
  return (deployment?.replica_instances ?? []).filter(
    (replica) =>
      replica.status === 'running' && replica.health_status === 'HEALTHY',
  )
}

export function isSelectableSyncDeployment(deployment: DeploymentLike): boolean {
  return (
    (deployment.status === 'running' || deployment.status === 'degraded') &&
    (deployment.inference_framework === 'vllm' ||
      deployment.inference_framework === 'sglang') &&
    getHealthyDeploymentReplicas(deployment).length > 0
  )
}

export function getAutomaticHealthyReplicaId(
  deployment: DeploymentLike | undefined,
): string | undefined {
  const healthyReplicas = getHealthyDeploymentReplicas(deployment)
  return healthyReplicas.length === 1 ? healthyReplicas[0].replica_id : undefined
}

export function isHealthyDeploymentReplicaBinding(
  deployments: DeploymentLike[],
  deploymentId: string,
  replicaId: string,
): boolean {
  if (!deploymentId) return !replicaId
  const deployment = deployments.find((item) => item.deployment_id === deploymentId)
  if (!deployment || !isSelectableSyncDeployment(deployment) || !replicaId) return false
  return getHealthyDeploymentReplicas(deployment).some(
    (replica) => replica.replica_id === replicaId,
  )
}
