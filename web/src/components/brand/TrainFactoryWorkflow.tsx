import workflowUrl from '@/assets/brand/trainfactory-workflow.svg'

interface TrainFactoryWorkflowProps {
  alt: string
}

export function TrainFactoryWorkflow({ alt }: TrainFactoryWorkflowProps) {
  return (
    <img
      className="auth-workflow-art"
      src={workflowUrl}
      width={960}
      height={392}
      alt={alt}
      decoding="async"
      draggable={false}
    />
  )
}
