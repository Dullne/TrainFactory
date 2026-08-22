import crypto from 'node:crypto'

const hasOwn = (value, key) => Object.prototype.hasOwnProperty.call(value, key)
const invalidRunIdMessage = 'Playwright run ID generator must return a nonempty string'
const invalidMetadataMessage = 'Playwright runtime metadata is invalid'
const invalidMarkerMessage = 'Playwright run marker is invalid'
const managedSetupErrorMessage = 'Unable to verify the managed Playwright server identity'
const runMarkerName = 'train-factory-e2e-run'

export function resolvePlaywrightRuntime(env, makeRunId = crypto.randomUUID) {
  const hasExternalFlag = hasOwn(env, 'PLAYWRIGHT_USE_EXTERNAL_SERVER')
  const hasExternalURL = hasOwn(env, 'PLAYWRIGHT_BASE_URL')

  if (hasExternalFlag && env.PLAYWRIGHT_USE_EXTERNAL_SERVER !== '1') {
    throw new Error('PLAYWRIGHT_USE_EXTERNAL_SERVER must be unset or exactly 1')
  }
  if (hasExternalFlag !== hasExternalURL) {
    throw new Error(
      'External mode requires PLAYWRIGHT_USE_EXTERNAL_SERVER=1 and PLAYWRIGHT_BASE_URL together'
    )
  }

  const external = hasExternalFlag
  const externalURL = env.PLAYWRIGHT_BASE_URL

  if (external) {
    let parsed
    try {
      if (typeof externalURL !== 'string' || externalURL.length === 0) throw new Error()
      parsed = new URL(externalURL)
    } catch {
      throw new Error('PLAYWRIGHT_BASE_URL must be a valid HTTP(S) URL without credentials')
    }

    if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
      throw new Error('PLAYWRIGHT_BASE_URL must be an HTTP(S) URL without credentials')
    }

    return { external: true, baseURL: parsed.origin, runId: null }
  }

  const port = Number(env.PLAYWRIGHT_PORT ?? '41731')
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('PLAYWRIGHT_PORT must be an integer between 1 and 65535')
  }

  let runId
  try {
    runId = makeRunId()
  } catch {
    throw new Error(invalidRunIdMessage)
  }
  if (typeof runId !== 'string' || runId.trim().length === 0) {
    throw new Error(invalidRunIdMessage)
  }

  return {
    external: false,
    baseURL: `http://127.0.0.1:${port}`,
    port,
    runId,
  }
}

export function createWebServerConfig(runtime) {
  if (runtime.external) return undefined

  return {
    command: `npm run dev -- --host 127.0.0.1 --port ${runtime.port} --strictPort`,
    url: runtime.baseURL,
    env: {
      VITE_API_BASE_URL: '/api',
      VITE_AUTH_LOGOUT_TIMEOUT: '1000',
      VITE_E2E_RUN_ID: runtime.runId,
    },
    reuseExistingServer: false,
    timeout: 120000,
  }
}

export function resolvePlaywrightSetupMetadata(_metadata) {
  try {
    if (!isRecord(_metadata)) throw new Error()
    const runtime = _metadata.trainFactoryE2E
    if (!isRecord(runtime) || typeof runtime.external !== 'boolean') throw new Error()
    if (typeof runtime.baseURL !== 'string') throw new Error()

    const parsed = new URL(runtime.baseURL)
    if (
      !['http:', 'https:'].includes(parsed.protocol) ||
      parsed.username ||
      parsed.password ||
      runtime.baseURL !== parsed.origin
    ) {
      throw new Error()
    }

    if (runtime.external) {
      if (runtime.runId !== null) throw new Error()
      return { external: true, baseURL: parsed.origin, runId: null }
    }

    if (typeof runtime.runId !== 'string' || runtime.runId.trim().length === 0) {
      throw new Error()
    }
    return { external: false, baseURL: parsed.origin, runId: runtime.runId }
  } catch {
    throw new Error(invalidMetadataMessage)
  }
}

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const rawTextElements = new Set([
  'iframe',
  'noembed',
  'noframes',
  'plaintext',
  'script',
  'style',
  'textarea',
  'title',
  'xmp',
])
const inertElements = new Set(['noscript', 'template'])
const headRawTextElements = new Set(['noframes', 'script', 'style', 'title'])
const headVoidElements = new Set(['base', 'basefont', 'bgsound', 'link', 'meta'])

function isHTMLSpace(character) {
  return character === ' ' || character === '\t' || character === '\n' || character === '\f' || character === '\r'
}

function hasNonWhitespaceText(value) {
  for (const character of value) {
    if (!isHTMLSpace(character)) return true
  }
  return false
}

function findTagEnd(html, start) {
  let quote = null
  for (let index = start + 1; index < html.length; index += 1) {
    const character = html[index]
    if (quote) {
      if (character === quote) quote = null
    } else if (character === '"' || character === "'") {
      quote = character
    } else if (character === '>') {
      return index
    }
  }
  return -1
}

function readTagAttributes(source) {
  const attributes = []
  let cursor = 0
  let selfClosing = false

  while (cursor < source.length) {
    while (cursor < source.length && isHTMLSpace(source[cursor])) cursor += 1
    if (cursor === source.length) break

    if (source[cursor] === '/') {
      cursor += 1
      while (cursor < source.length && isHTMLSpace(source[cursor])) cursor += 1
      if (cursor !== source.length) throw new Error()
      selfClosing = true
      break
    }

    const nameStart = cursor
    while (
      cursor < source.length &&
      !isHTMLSpace(source[cursor]) &&
      !['=', '/', '"', "'", '<', '`'].includes(source[cursor])
    ) {
      cursor += 1
    }
    if (cursor === nameStart) throw new Error()
    const name = source.slice(nameStart, cursor).toLowerCase()

    while (cursor < source.length && isHTMLSpace(source[cursor])) cursor += 1
    let value = null
    if (source[cursor] === '=') {
      cursor += 1
      while (cursor < source.length && isHTMLSpace(source[cursor])) cursor += 1
      if (cursor === source.length) throw new Error()

      const quote = source[cursor]
      if (quote === '"' || quote === "'") {
        cursor += 1
        const valueStart = cursor
        while (cursor < source.length && source[cursor] !== quote) cursor += 1
        if (cursor === source.length) throw new Error()
        value = source.slice(valueStart, cursor)
        cursor += 1
      } else {
        const valueStart = cursor
        while (cursor < source.length && !isHTMLSpace(source[cursor])) {
          if (['"', "'", '<', '=', '`'].includes(source[cursor])) throw new Error()
          cursor += 1
        }
        if (cursor === valueStart) throw new Error()
        value = source.slice(valueStart, cursor)
      }
    }
    attributes.push({ name, value })
  }

  return { attributes, selfClosing }
}

function parseStartTag(tag) {
  const match = /^<([A-Za-z][A-Za-z0-9:_-]*)(?=[\t\n\f\r />])/.exec(tag)
  if (!match) return null

  try {
    const { attributes, selfClosing } = readTagAttributes(tag.slice(match[0].length, -1))
    return { name: match[1].toLowerCase(), attributes, selfClosing }
  } catch {
    return null
  }
}

function parseEndTag(tag) {
  const match = /^<\/([A-Za-z][A-Za-z0-9:_-]*)[\t\n\f\r ]*>$/.exec(tag)
  return match?.[1].toLowerCase() ?? null
}

function findRawTextEnd(html, lowerHTML, start, tagName) {
  const prefix = `</${tagName}`
  let cursor = lowerHTML.indexOf(prefix, start)

  while (cursor !== -1) {
    const boundary = html[cursor + prefix.length]
    if (boundary === '>' || isHTMLSpace(boundary)) {
      const end = findTagEnd(html, cursor)
      if (end === -1) return -1
      if (parseEndTag(html.slice(cursor, end + 1)) === tagName) return end
    }
    cursor = lowerHTML.indexOf(prefix, cursor + prefix.length)
  }
  return -1
}

function markerFromMetaAttributes(attributes) {
  const names = attributes.filter((attribute) => attribute.name === 'name')
  if (!names.some((attribute) => attribute.value === runMarkerName)) return undefined

  const contents = attributes.filter((attribute) => attribute.name === 'content')
  if (
    names.length !== 1 ||
    contents.length !== 1 ||
    typeof contents[0].value !== 'string' ||
    contents[0].value.trim().length === 0
  ) {
    throw new Error()
  }
  return contents[0].value
}

export function extractPlaywrightRunId(html) {
  try {
    if (typeof html !== 'string') throw new Error()
    const lowerHTML = html.toLowerCase()
    const markers = []
    const inertStack = []
    let inHead = false
    let headSeen = false
    let bodySeen = false
    let cursor = 0

    while (cursor < html.length) {
      const start = html.indexOf('<', cursor)
      const textEnd = start === -1 ? html.length : start
      if (
        inertStack.length === 0 &&
        (inHead || (!headSeen && !bodySeen)) &&
        hasNonWhitespaceText(html.slice(cursor, textEnd))
      ) {
        inHead = false
        bodySeen = true
      }
      if (start === -1) break

      if (html.startsWith('<!--', start)) {
        const commentEnd = html.indexOf('-->', start + 4)
        if (commentEnd === -1) break
        cursor = commentEnd + 3
        continue
      }

      const end = findTagEnd(html, start)
      if (end === -1) break
      const tag = html.slice(start, end + 1)
      cursor = end + 1

      if (tag.startsWith('<!') || tag.startsWith('<?')) {
        const allowedDoctype = /^<!doctype(?=[\t\n\f\r >])/i.test(tag)
        if (inHead || (!headSeen && !bodySeen && !allowedDoctype)) {
          inHead = false
          bodySeen = true
        }
        continue
      }

      if (tag.startsWith('</')) {
        const tagName = parseEndTag(tag)
        if (!tagName) continue

        if (!headSeen && !bodySeen) bodySeen = true

        if (inertStack.length > 0) {
          if (inertStack[inertStack.length - 1] === tagName) inertStack.pop()
          continue
        }
        if (tagName === 'head') {
          inHead = false
        } else if (inHead) {
          inHead = false
          bodySeen = true
        }
        continue
      }

      const parsed = parseStartTag(tag)
      if (!parsed) {
        if (inertStack.length === 0 && (inHead || (!headSeen && !bodySeen))) {
          throw new Error()
        }
        continue
      }
      const { name: tagName, attributes, selfClosing } = parsed

      if (!headSeen && !bodySeen && tagName !== 'html' && tagName !== 'head') {
        bodySeen = true
      }

      if (inertStack.length > 0) {
        if (inertElements.has(tagName) && !selfClosing) inertStack.push(tagName)
        if (rawTextElements.has(tagName) && !selfClosing) {
          const rawEnd = findRawTextEnd(html, lowerHTML, cursor, tagName)
          if (rawEnd === -1) break
          cursor = rawEnd + 1
        }
        continue
      }

      if (rawTextElements.has(tagName) && !selfClosing) {
        if (inHead && !headRawTextElements.has(tagName)) {
          inHead = false
          bodySeen = true
        }
        const rawEnd = findRawTextEnd(html, lowerHTML, cursor, tagName)
        if (rawEnd === -1) break
        cursor = rawEnd + 1
        continue
      }

      if (inertElements.has(tagName) && !selfClosing) {
        inertStack.push(tagName)
        continue
      }

      if (tagName === 'head') {
        if (!headSeen && !bodySeen) {
          headSeen = true
          inHead = true
        }
        continue
      }
      if (tagName === 'body') {
        bodySeen = true
        inHead = false
        continue
      }
      if (inHead && !headVoidElements.has(tagName)) {
        inHead = false
        bodySeen = true
        continue
      }
      if (tagName !== 'meta' || !inHead) continue

      const marker = markerFromMetaAttributes(attributes)
      if (marker !== undefined) markers.push(marker)
    }

    if (markers.length !== 1) throw new Error()
    return markers[0]
  } catch {
    throw new Error(invalidMarkerMessage)
  }
}

function defaultWrite(value) {
  process.stdout.write(value)
}

export async function runPlaywrightGlobalSetup(metadata, dependencies = {}) {
  const runtime = resolvePlaywrightSetupMetadata(metadata)
  const write = dependencies.write ?? defaultWrite

  if (runtime.external) {
    const safeOrigin = new URL(runtime.baseURL).origin
    write(
      `[TrainFactory E2E] external=true origin=${safeOrigin}; current source was not verified\n`
    )
    return
  }

  const fetchImpl = dependencies.fetch ?? globalThis.fetch
  try {
    const response = await fetchImpl(runtime.baseURL, { redirect: 'error' })
    if (
      !isRecord(response) ||
      response.ok !== true ||
      !Number.isInteger(response.status) ||
      response.status < 200 ||
      response.status >= 300 ||
      response.redirected !== false ||
      typeof response.url !== 'string'
    ) {
      throw new Error()
    }

    const responseURL = new URL(response.url)
    if (
      !['http:', 'https:'].includes(responseURL.protocol) ||
      responseURL.username ||
      responseURL.password ||
      responseURL.origin !== runtime.baseURL
    ) {
      throw new Error()
    }

    const html = await response.text()
    if (extractPlaywrightRunId(html) !== runtime.runId) throw new Error()
  } catch {
    throw new Error(managedSetupErrorMessage)
  }

  write(
    `[TrainFactory E2E] external=false origin=${runtime.baseURL}; current source run verified\n`
  )
}
