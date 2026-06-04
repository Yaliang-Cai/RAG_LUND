export interface EvalResult {
  score: number
  gap: string
}

export async function evaluateAnswer(
  workspaceId: string,
  query: string,
  answer: string,
): Promise<EvalResult> {
  const res = await fetch('/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workspace_id: workspaceId, query, answer }),
  })
  if (!res.ok) throw new Error('Evaluation failed')
  return res.json() as Promise<EvalResult>
}
