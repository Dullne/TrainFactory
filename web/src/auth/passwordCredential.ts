import type { AuthLoginRequest } from '@/services/api'

type PasswordCredentialConstructor = new (data: { id: string; password: string }) => Credential

type PasswordCredentialWindow = Window & {
  PasswordCredential?: PasswordCredentialConstructor
}

export async function offerLoginCredentialForSaving({
  username,
  password,
}: AuthLoginRequest): Promise<void> {
  const PasswordCredential = (window as PasswordCredentialWindow).PasswordCredential
  const credentialStore = navigator.credentials?.store

  if (typeof PasswordCredential !== 'function' || typeof credentialStore !== 'function') return

  try {
    const credential = new PasswordCredential({ id: username, password })
    await credentialStore.call(navigator.credentials, credential)
  } catch {
    // Saving is best-effort; unsupported browsers and user refusal must not block login.
  }
}
