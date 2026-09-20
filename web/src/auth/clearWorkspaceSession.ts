const WORKSPACE_PREFIXES = ['tf_training_draft:v1:', 'tf_list_workspace:v1:']

/** Clear account work only after the session has actually ended. */
export function clearWorkspaceSession() {
  try {
    const storage = window.sessionStorage
    const keys = Array.from({ length: storage.length }, (_, index) => storage.key(index))
    for (const key of keys) {
      if (key && WORKSPACE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
        storage.removeItem(key)
      }
    }
  } catch {
    // Memory-only drafts must also be cleared when browser storage is unavailable.
  }
  window.dispatchEvent(new Event('tf:workspace-session-cleared'))
}
