export const DEFAULT_RETURN_TO = '/training'

export function safeReturnTo(raw: string | null, origin: string): string {
  if (!raw || !raw.startsWith('/') || raw.startsWith('//') || raw.includes('\\')) {
    return DEFAULT_RETURN_TO
  }

  try {
    const decoded = decodeURIComponent(raw)
    if (decoded.startsWith('//') || decoded.includes('\\')) return DEFAULT_RETURN_TO

    const target = new URL(raw, origin)
    const decodedPath = decodeURIComponent(target.pathname)
    const loginComparablePath = decodedPath.replace(/\/+$/, '') || '/'

    if (
      target.origin !== origin ||
      target.pathname.startsWith('//') ||
      target.pathname.includes('\\') ||
      decodedPath.startsWith('//') ||
      decodedPath.includes('\\') ||
      /^\/login$/i.test(loginComparablePath)
    ) {
      return DEFAULT_RETURN_TO
    }

    const normalized = `${target.pathname}${target.search}${target.hash}`
    return normalized.startsWith('//') ? DEFAULT_RETURN_TO : normalized
  } catch {
    return DEFAULT_RETURN_TO
  }
}

export function currentReturnTo(
  location: Pick<Location, 'pathname' | 'search' | 'hash'>
): string {
  return `${location.pathname}${location.search}${location.hash}`
}

export function loginPathFor(returnTo: string): string {
  return `/login?redirect=${encodeURIComponent(returnTo)}`
}
